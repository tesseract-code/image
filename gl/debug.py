"""
gl_debug.py
===========
OpenGL debug output and diagnostics utilities for PyQt6/PyOpenGL applications.

Provides:
- Debug callback registration with safe GC anchoring.
- A diagnostic helper that queries implementation limits from an active context.

Minimum requirement: OpenGL 4.1
--------------------------------
GL 4.1 is the lowest version this module targets, which is the ceiling on
macOS (Apple froze its OpenGL support there before deprecating the API in
2018).

Debug output is **not** a GL 4.1 Core feature — it was promoted to Core only
in GL 4.3.  On a 4.1 / 4.2 context the driver must expose one of the
equivalent opt-in extensions for the callback API to be available:

* ``GL_KHR_debug``        — the cross-vendor standardised backport.
* ``GL_ARB_debug_output`` — the earlier ARB precursor; same semantics.

On GL 4.3+ drivers the unsuffixed Core entry-points are tried first; the
extension paths serve as fallbacks.  Either way the runtime behaviour is
identical.

macOS note
----------
Apple's GL 4.1 implementation deliberately omits both debug extensions, so
``enable_gl_debug_output`` exits early on Darwin rather than issuing
confusing driver errors.

Thread-safety:
    ``enable_gl_debug_output`` must be called from the thread that owns the GL
    context.  ``_GL_CALLBACK_REFS`` is written only during initialisation and
    read (implicitly) by the driver thereafter, so no explicit lock is needed
    as long as the caller respects that constraint.
"""

from __future__ import annotations

import logging
import platform
import threading
from typing import Any, Callable, NamedTuple

from PyQt6.QtGui import QOpenGLContext

from cross_platform.qt6_utils.image.gl.backend import GL, GLConfig

# ---------------------------------------------------------------------------
# Module logger
# ---------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------

class _SeverityMeta(NamedTuple):
    """Pairs a Python ``logging`` level with a human-readable log prefix."""
    level: int
    prefix: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Maps GL debug-severity tokens → (_SeverityMeta).
# Using a NamedTuple value avoids tuple-unpacking errors on lookup failures.
_SEVERITY_MAP: dict[int, _SeverityMeta] = {
    GL.GL_DEBUG_SEVERITY_HIGH: _SeverityMeta(logging.ERROR, "[GL ERROR]"),
    GL.GL_DEBUG_SEVERITY_MEDIUM: _SeverityMeta(logging.WARNING, "[GL WARN] "),
    GL.GL_DEBUG_SEVERITY_LOW: _SeverityMeta(logging.INFO, "[GL INFO] "),
    GL.GL_DEBUG_SEVERITY_NOTIFICATION: _SeverityMeta(logging.DEBUG,
                                                     "[GL NOTE] "),
}

# Keeps C-level callback objects alive for the lifetime of the process.
# PyOpenGL wraps ctypes function pointers in Python objects; if those objects
# are garbage-collected the driver retains a dangling pointer, causing a
# segfault on the next GL error.  This list is the canonical GC anchor.
#
# Protected by ``_CALLBACK_LOCK`` so multiple GL contexts (e.g. in tests) can
# each install their own callback without a race.
_GL_CALLBACK_REFS: list[Any] = []
_CALLBACK_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_gl_message(raw: bytes | Any) -> str:
    """
    Safely convert the raw message pointer supplied by the GL driver into a
    Python ``str``.

    The GLDEBUGPROC signature passes ``raw_message`` as a ``const GLchar *``
    (null-terminated).  PyOpenGL may surface this as:

    * ``bytes``  – the common case on most platforms / PyOpenGL versions.
    * A ctypes ``Array`` – seen on some Linux Mesa builds.
    * Any other object – defensive fallback via ``str()``.

    Args:
        raw: The raw message value received in the debug callback.

    Returns:
        A decoded UTF-8 string, with invalid byte sequences replaced.
    """
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")
    # ctypes arrays expose a ``value`` attribute for byte strings.
    value = getattr(raw, "value", None)
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    return str(raw)


# ---------------------------------------------------------------------------
# GL debug callback
# ---------------------------------------------------------------------------

def _gl_debug_callback(
        source: int,
        msg_type: int,
        msg_id: int,
        severity: int,
        length: int,  # noqa: ARG001 – present in GLDEBUGPROC; driver owns it
        raw_message: bytes | Any,
        user_param: Any,  # noqa: ARG001 – reserved for caller data; unused here
) -> None:
    """
    GLDEBUGPROC-compatible callback registered with the GL driver.

    The driver invokes this function for every debug event that passes the
    current message-control filter.  The ``length`` and ``user_param``
    arguments are required by the C signature but intentionally unused:
    ``length`` is redundant because ``raw_message`` is always null-terminated,
    and ``user_param`` was set to ``None`` at registration time.

    Args:
        source:      GL_DEBUG_SOURCE_* token identifying the event source.
        msg_type:    GL_DEBUG_TYPE_* token (e.g. ERROR, PERFORMANCE).
        msg_id:      Driver-assigned message identifier.
        severity:    GL_DEBUG_SEVERITY_* token used to select the log level.
        length:      Byte-length of ``raw_message`` (null-terminator excluded).
        raw_message: Null-terminated UTF-8 message string from the driver.
        user_param:  Opaque pointer supplied at callback registration (``None``).
    """
    meta = _SEVERITY_MAP.get(severity, _SeverityMeta(logging.DEBUG,
                                                     f"[GL {severity:#010x}]"))

    # Short-circuit: avoid decoding the message if the level is suppressed.
    if not logger.isEnabledFor(meta.level):
        return

    msg = _decode_gl_message(raw_message)
    logger.log(meta.level, "%s ID:%d | %s", meta.prefix, msg_id, msg)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def enable_gl_debug_output(*, synchronous: bool = True) -> None:
    """
    Register a GL debug callback using the best available entry-point.

    The minimum supported context is **OpenGL 4.1**.  Because debug output was
    only promoted to Core in GL 4.3, on a 4.1 or 4.2 context this function
    depends on the driver exposing at least one of the opt-in extensions.
    Entry-points are resolved in preference order:

    1. ``glDebugMessageCallback``    – Core, available on GL 4.3+.
    2. ``glDebugMessageCallbackKHR`` – ``GL_KHR_debug`` extension.
    3. ``glDebugMessageCallbackARB`` – ``GL_ARB_debug_output`` extension.

    The same priority applies to the matching ``glDebugMessageControl*``
    variant.  If none of the three can be resolved the function logs a warning
    and returns without error — debug output is a diagnostic aid, never a
    hard runtime requirement.

    This function is a no-op when:

    * :attr:`GLConfig.DEBUG_MODE` is ``False``.
    * The platform is macOS (Apple's frozen GL 4.1 omits the debug extensions).
    * No suitable entry-point is found at runtime.

    Must be called from the thread that owns the current GL context, typically
    inside an ``initializeGL()`` override.

    Args:
        synchronous: When ``True`` (default), enables
            ``GL_DEBUG_OUTPUT_SYNCHRONOUS`` so callbacks fire on the same
            thread and call-stack as the offending GL command.  This is
            essential for accurate stack traces but may reduce throughput.
            Set to ``False`` for release-candidate profiling runs.

    Raises:
        Does **not** raise; all errors are logged at ``ERROR`` level so that a
        misconfigured debug layer never crashes the application.
    """
    if not GLConfig.DEBUG_MODE:
        return

    # macOS caps at GL 4.1 and Apple never shipped the KHR_debug or
    # ARB_debug_output extensions on any macOS version before deprecating
    # OpenGL outright.  Attempting the calls produces opaque driver errors,
    # so we exit cleanly instead.
    if platform.system() == "Darwin":
        logger.warning(
            "OpenGL debug output is unavailable on macOS: Apple's GL 4.1 "
            "implementation does not expose GL_KHR_debug or "
            "GL_ARB_debug_output."
        )
        return

    try:
        # ------------------------------------------------------------------ #
        # 1. Resolve entry-points: Core (4.3+) → KHR extension → ARB ext    #
        # ------------------------------------------------------------------ #
        # On our GL 4.1 minimum baseline the Core path will often be absent;
        # KHR/ARB are the expected route on 4.1 and 4.2 contexts.
        # ``getattr`` with a default of ``None`` avoids AttributeError when
        # an entry-point is not exported by the current driver.
        callback_func: Callable[..., None] | None = (
                getattr(GL, "glDebugMessageCallback", None)
                or getattr(GL, "glDebugMessageCallbackKHR", None)
                or getattr(GL, "glDebugMessageCallbackARB", None)
        )
        control_func: Callable[..., None] | None = (
                getattr(GL, "glDebugMessageControl", None)
                or getattr(GL, "glDebugMessageControlKHR", None)
                or getattr(GL, "glDebugMessageControlARB", None)
        )

        if callback_func is None or control_func is None:
            logger.warning(
                "OpenGL debug output unavailable: no suitable entry-point found. "
                "On GL 4.1/4.2 the driver must expose GL_KHR_debug or "
                "GL_ARB_debug_output; on GL 4.3+ the Core functions should be "
                "present.  Check driver/extension support."
            )
            return

        logger.info(
            "Initialising GL debug output via %s / %s.",
            callback_func.__name__,
            control_func.__name__,
        )

        # ------------------------------------------------------------------ #
        # 2. Enable capabilities                                              #
        # ------------------------------------------------------------------ #
        GL.glEnable(GL.GL_DEBUG_OUTPUT)

        if synchronous:
            # Synchronous mode: callbacks are fired inline with the offending
            # call, preserving the full Python call-stack in tracebacks.
            GL.glEnable(GL.GL_DEBUG_OUTPUT_SYNCHRONOUS)

        # ------------------------------------------------------------------ #
        # 3. Wrap and anchor the callback                                     #
        # ------------------------------------------------------------------ #
        # GLDEBUGPROC is ABI-compatible with the KHR/ARB suffixed typedefs;
        # PyOpenGL does not expose separate ctypes types for them.
        c_callback = GL.GLDEBUGPROC(_gl_debug_callback)

        # Thread-safe GC anchor: multiple GL contexts in the same process
        # (common in test-suites) can each call this function independently.
        with _CALLBACK_LOCK:
            _GL_CALLBACK_REFS.append(c_callback)

        # ------------------------------------------------------------------ #
        # 4. Register callback and enable all message categories              #
        # ------------------------------------------------------------------ #
        callback_func(c_callback, None)

        # GL_DONT_CARE for source / type / severity → receive everything.
        # count=0 + ids=None → the filter applies to all message IDs.
        control_func(
            GL.GL_DONT_CARE,  # source
            GL.GL_DONT_CARE,  # type
            GL.GL_DONT_CARE,  # severity
            0,  # count
            None,  # ids
            GL.GL_TRUE,  # enabled
        )

        logger.info("GL debug output successfully enabled (synchronous=%s).",
                    synchronous)

    except Exception:
        # Never let debug-layer setup crash the application.
        logger.error("Failed to enable OpenGL Debug Output.", exc_info=True)


def diagnose_gl_limits() -> dict[str, Any]:
    """
    Query and log key OpenGL implementation limits from the active context.

    Intended for use inside ``initializeGL()`` during development / QA to
    surface driver capability mismatches early.  Call only when a GL context
    is current on the calling thread.

    Queried limits
    --------------
    ========================== ================================ ==========
    Key                        GL token                         Since
    ========================== ================================ ==========
    max_vertex_attribs         GL_MAX_VERTEX_ATTRIBS            GL 2.0
    max_texture_size           GL_MAX_TEXTURE_SIZE              GL 1.0
    max_texture_image_units    GL_MAX_TEXTURE_IMAGE_UNITS       GL 2.0
    max_vertex_uniform_comps   GL_MAX_VERTEX_UNIFORM_COMPONENTS GL 2.0
    max_fragment_uniform_comps GL_MAX_FRAGMENT_UNIFORM_COMPONENTS GL 2.0
    max_varying_components     GL_MAX_VARYING_COMPONENTS        GL 3.2 *
    max_viewport_dims          GL_MAX_VIEWPORT_DIMS             GL 1.0
    ========================== ================================ ==========

    ``*`` ``GL_MAX_VARYING_FLOATS`` was deprecated in GL 3.2; this function
    uses ``GL_MAX_VARYING_COMPONENTS`` which supersedes it.

    Returns:
        A ``dict`` mapping limit names (``str``) to their integer values, or
        ``{"error": <message>}`` when no context is current or a GL call
        fails.

    Note:
        This function uses the module-level ``GL`` import (from
        ``cross_platform.qt6_utils.image.gl.backend``) for consistency.
        A local ``from OpenGL import GL`` import in the original code created
        a shadowing bug where the two GL namespaces could diverge.
    """
    # Guard: a stray call before the context is made current produces
    # confusing GL errors rather than a clean diagnostic message.
    context = QOpenGLContext.currentContext()
    if context is None:
        logger.warning(
            "diagnose_gl_limits() called without an active GL context; "
            "skipping query."
        )
        return {"error": "No active OpenGL context"}

    try:
        limits: dict[str, Any] = {
            # Vertex pipeline
            "max_vertex_attribs": int(
                GL.glGetIntegerv(GL.GL_MAX_VERTEX_ATTRIBS)),
            "max_vertex_uniform_comps": int(
                GL.glGetIntegerv(GL.GL_MAX_VERTEX_UNIFORM_COMPONENTS)),

            # Fragment pipeline
            "max_fragment_uniform_comps": int(
                GL.glGetIntegerv(GL.GL_MAX_FRAGMENT_UNIFORM_COMPONENTS)),

            # Texturing
            "max_texture_size": int(GL.glGetIntegerv(GL.GL_MAX_TEXTURE_SIZE)),
            "max_texture_image_units": int(
                GL.glGetIntegerv(GL.GL_MAX_TEXTURE_IMAGE_UNITS)),

            # Interpolation
            # GL_MAX_VARYING_FLOATS was deprecated in GL 3.2; the successor
            # token is GL_MAX_VARYING_COMPONENTS (alias in core 3.2+).
            "max_varying_components": int(
                GL.glGetIntegerv(GL.GL_MAX_VARYING_COMPONENTS)),

            # Rasterisation
            "max_viewport_dims": list(
                GL.glGetIntegerv(GL.GL_MAX_VIEWPORT_DIMS)),
        }

        # ------------------------------------------------------------------ #
        # Structured diagnostic log block                                     #
        # ------------------------------------------------------------------ #
        _SEP = "=" * 60
        logger.info(_SEP)
        logger.info("OPENGL IMPLEMENTATION LIMITS")
        logger.info(_SEP)
        for key, value in limits.items():
            logger.info("  %-30s %s", key, value)
        logger.info(_SEP)

        return limits

    except Exception:
        logger.error("Failed to query GL limits.", exc_info=True)
        return {"error": "GL query failed – see log for details"}

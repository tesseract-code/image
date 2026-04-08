"""
gl_version.py
=============
Utilities for querying and inspecting the active OpenGL context version.

Intended to be called once at application startup (inside or immediately after
``initializeGL``) to log driver information and enforce the minimum supported
version.

Minimum supported version: OpenGL 4.1
    All callers should validate the returned version against this floor before
    proceeding with resource allocation.  The ``check_minimum_version`` helper
    encapsulates that check with a structured log and a typed exception.
"""

from __future__ import annotations

import logging
import re
from typing import TypedDict

from cross_platform.qt6_utils.image.gl.backend import GL
from cross_platform.qt6_utils.image.gl.error import GLInitializationError

__all__ = [
    "OpenGLInfo",
    "get_gl_version",
    "get_gl_info",
    "check_minimum_version",
]

logger = logging.getLogger(__name__)

# Minimum OpenGL version required by this package.
_MIN_GL_MAJOR = 4
_MIN_GL_MINOR = 1


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class OpenGLInfo(TypedDict):
    """
    Structured snapshot of the active OpenGL context's identity strings.

    All fields are populated from ``glGetString`` queries.  The ``major`` and
    ``minor`` fields are parsed integers; all other fields are the raw
    driver-supplied strings decoded from UTF-8.

    Fields:
        major:                Parsed major version (e.g. ``4``).
        minor:                Parsed minor version (e.g. ``1``).
        version_str:          Full ``GL_VERSION`` string from the driver.
        vendor:               GPU vendor name (``GL_VENDOR``).
        renderer:             GPU model name (``GL_RENDERER``).
        shading_lang_version: GLSL version string (``GL_SHADING_LANGUAGE_VERSION``).
    """
    major: int
    minor: int
    version_str: str
    vendor: str
    renderer: str
    shading_lang_version: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_gl_string(enum_val: int) -> str:
    """
    Query a ``glGetString`` enum and return a decoded Python ``str``.

    Args:
        enum_val: A ``GL_*`` string-query enum token (e.g. ``GL.GL_VENDOR``).

    Returns:
        The driver-supplied string decoded from UTF-8, or ``"Unknown"`` when
        the driver returns ``NULL`` (which is valid for some enums on
        unsupported extensions).
    """
    raw = GL.glGetString(enum_val)
    if raw is None:
        return "Unknown"
    if isinstance(raw, (bytes, bytearray)):
        return raw.decode("utf-8", errors="replace")
    return str(raw)


# Compiled once at import time; matches "4.1", "4.1.0", "4.1 NVIDIA …" etc.
_VERSION_RE = re.compile(r"^(\d+)\.(\d+)")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_gl_version() -> tuple[int, int]:
    """
    Return the major and minor version of the active OpenGL context.

    Queries ``GL_VERSION``, which the driver must return as a string of the
    form ``"<major>.<minor>[.<release>] [<vendor-specific info>]"`` (OpenGL
    2.0 spec §6.1.5).  Only the first two numeric components are extracted.

    Returns:
        ``(major, minor)`` as a pair of ``int``, e.g. ``(4, 1)``.

    Raises:
        GLInitializationError: If no GL context is current (``glGetString``
            returns ``None``), or if the version string cannot be parsed.
            This is preferred over ``RuntimeError`` so callers that already
            catch ``GLError`` subclasses handle this uniformly.

    Note:
        Must be called from the thread that owns the current GL context.
    """
    version_str = _decode_gl_string(GL.GL_VERSION)

    # _decode_gl_string returns "Unknown" only when glGetString returns NULL.
    # That happens when no context is current — a hard precondition failure.
    if version_str == "Unknown":
        raise GLInitializationError(
            "glGetString(GL_VERSION) returned NULL: no OpenGL context is "
            "current on this thread."
        )

    match = _VERSION_RE.search(version_str)
    if not match:
        raise GLInitializationError(
            "Failed to parse OpenGL version string: %r.  "
            "Expected format '<major>.<minor>[.<release>] [<vendor info>]'."
            % version_str
        )

    major = int(match.group(1))
    minor = int(match.group(2))
    logger.debug("OpenGL version detected: %d.%d (raw: %r)", major, minor, version_str)
    return major, minor


def get_gl_info() -> OpenGLInfo:
    """
    Return a structured snapshot of the active OpenGL context's identity.

    Queries all six driver identity strings in a single call.  Intended for
    startup logging and diagnostics; not suitable for hot paths.

    Returns:
        An :class:`OpenGLInfo` ``TypedDict`` populated from ``glGetString``
        queries.

    Raises:
        GLInitializationError: Propagated from :func:`get_gl_version` when no
            context is current or the version string is unparseable.

    Note:
        Unlike the original implementation, this function does **not** return
        a sentinel ``OpenGLInfo`` on failure.  A sentinel with ``major=0,
        minor=0`` is indistinguishable from a successful query on a very old
        driver, and silently swallowing the error defers the failure to a
        later, harder-to-diagnose call site.
    """
    # get_gl_version raises GLInitializationError on any failure — let it
    # propagate unchanged so the caller gets a typed, actionable exception.
    major, minor = get_gl_version()

    info: OpenGLInfo = {
        "major":                major,
        "minor":                minor,
        "version_str":          _decode_gl_string(GL.GL_VERSION),
        "vendor":               _decode_gl_string(GL.GL_VENDOR),
        "renderer":             _decode_gl_string(GL.GL_RENDERER),
        "shading_lang_version": _decode_gl_string(GL.GL_SHADING_LANGUAGE_VERSION),
    }

    logger.info(
        "OpenGL context info — version: %s | vendor: %s | renderer: %s | GLSL: %s",
        info["version_str"],
        info["vendor"],
        info["renderer"],
        info["shading_lang_version"],
    )

    return info


def check_minimum_version(
    required_major: int = _MIN_GL_MAJOR,
    required_minor: int = _MIN_GL_MINOR,
) -> tuple[int, int]:
    """
    Assert that the active context meets the minimum required GL version.

    Calls :func:`get_gl_version` and compares against ``required_major`` /
    ``required_minor``.  Logs the detected version at ``INFO`` level on
    success and at ``ERROR`` level on failure.

    Args:
        required_major: Minimum acceptable major version (default: ``4``).
        required_minor: Minimum acceptable minor version (default: ``1``).

    Returns:
        ``(major, minor)`` of the active context on success.

    Raises:
        GLInitializationError: If the detected version is below the required
            minimum, or if :func:`get_gl_version` raises (no context /
            unparseable string).

    Example::

        # Enforce the package minimum at widget initialisation time:
        major, minor = check_minimum_version()
        logger.info("GL version OK: %d.%d", major, minor)
    """
    major, minor = get_gl_version()   # raises GLInitializationError on failure

    # Compare as a tuple so (4, 2) > (4, 1) and (5, 0) > (4, 1) both hold.
    if (major, minor) < (required_major, required_minor):
        raise GLInitializationError(
            "OpenGL %d.%d is required but the active context reports %d.%d.  "
            "Update your GPU drivers or use a machine with a compatible GPU."
            % (required_major, required_minor, major, minor)
        )

    logger.info(
        "OpenGL version check passed: %d.%d >= %d.%d",
        major, minor, required_major, required_minor,
    )
    return major, minor
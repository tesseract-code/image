"""
backend.py
=============
OpenGL backend configuration and context initialization.

Applies PyOpenGL performance and correctness flags (``ERROR_CHECKING``,
``ERROR_LOGGING``, ``ERROR_ON_COPY``) before any GL import, then exposes a
``GLConfig`` singleton that is populated with runtime-detected capabilities
by `initialize_context`.

Usage
-----
Call `initialize_context` exactly once after the OpenGL context and
window have been created.  All other modules should import ``GL``, ``GLU``,
and config from here rather than importing PyOpenGL directly::

    from image.gl.backend import GL, GLU, GL_CONFIGS, initialize_context

    initialize_context()   # call once at startup

    if GL_CONFIGS["default"].USE_IMMUTABLE_STORAGE:
        GL.glTexStorage2D(...)

Environment
-----------
``GL_DEBUG_MODE``
    Set to ``"1"`` to enable PyOpenGL error checking, error logging, and the
    KHR_debug callback via `enable_gl_debug_output`.  Defaults to
    ``"0"``.  Has no effect on macOS, where the Apple GL stack does not
    support KHR_debug callbacks.

Requires
--------
Python >= 3.10, PyOpenGL, OpenGL >= 4.1
"""

import logging
import os
import platform
import sys

from image.gl.config import GL_CONFIGS, GLConfig

# Warn if PyOpenGL was imported before this module had a chance to set the
# ERROR_CHECKING / ERROR_LOGGING flags.  Once GL is imported those flags are
# read and baked in — setting them afterward has no effect.
if "OpenGL.GL" in sys.modules:
    logging.getLogger(__name__).warning(
        "OpenGL.GL was imported before GLConfig could apply optimizations. "
        "PyOpenGL error checking may still be active."
    )

import OpenGL

_DEBUG_ENV: bool = os.environ.get("GL_DEBUG_MODE", "0") == "1"


if _DEBUG_ENV:
    # Both flags must be set together: ERROR_CHECKING enables the per-call
    # glGetError drain; ERROR_LOGGING routes those errors through Python's
    # logging system rather than printing to stderr.
    OpenGL.ERROR_CHECKING = True
    OpenGL.ERROR_LOGGING = True
    # ERROR_ON_COPY catches accidental Python-object copies of GPU data
    OpenGL.ERROR_ON_COPY = True
else:
    # Disable in non-debug builds to eliminate per-call glGetError overhead.
    OpenGL.ERROR_CHECKING = False
    OpenGL.ERROR_LOGGING = False
    OpenGL.ERROR_ON_COPY = False

try:
    from OpenGL import GL as _GL
    from OpenGL import GLU as _GLU
except ImportError as e:
    raise ImportError(
        "PyOpenGL not found. Install it with: pip install PyOpenGL"
    ) from e

_context_initialized: bool = False

def initialize_context() -> None:
    """
    Detect runtime GL capabilities and populate ``GL_CONFIGS["default"]``.

    Must be called exactly once after the OpenGL context is made current
    (i.e. after the Qt window is shown and ``initializeGL`` has run).
    Subsequent calls are safe but redundant — the config is overwritten with
    the same values.

    Detected capabilities
    ---------------------
    ``USE_IMMUTABLE_STORAGE``
        ``True`` when ``glTexStorage2D`` is available, either via GL 4.2+
        Core or via the ``GL_ARB_texture_storage`` extension on GL 4.1.
        Always ``False`` on macOS (Apple's GL 4.1 stack omits the extension).

    Side effects
    ------------
    On non-macOS platforms with ``GL_DEBUG_MODE=1``, calls
    `~image.gl.debug.enable_gl_debug_output`
    after the config is written so the KHR_debug callback is active for all
    subsequent GL calls.
    """
    global _context_initialized
    if _context_initialized:
        return
    _context_initialized = True
    
    logger = logging.getLogger("GLBackend")

    # macOS caps at GL 4.1 and Apple never shipped GL_ARB_texture_storage.
    # Skip all capability detection and return early — no debug callback either,
    # since Apple's stack does not expose KHR_debug on any version.
    if platform.system() == "Darwin":
        GL_CONFIGS["default"] = GLConfig(USE_IMMUTABLE_STORAGE=False)
        logger.info(
            "macOS detected: immutable texture storage unavailable (GL capped at 4.1)"
        )
        # Note: GL_DEBUG_MODE is intentionally ignored on macOS.  The warning
        # that would belong here is not emitted because the early return makes
        # this path unambiguous — debug output simply does not exist on this
        # platform.
        return

    try:
        version_str = _GL.glGetString(_GL.GL_VERSION)
        if version_str:
            # Version strings take the form "4.6.0 <vendor info>".
            # Split on whitespace first to strip vendor suffixes, then on '.'
            # to isolate major and minor components.
            numeric_part = version_str.decode().split()[0].split(".")
            major_ver = int(numeric_part[0])
            minor_ver = int(numeric_part[1]) if len(numeric_part) > 1 else 0
        else:
            # glGetString returned NULL — no context or very old driver.
            major_ver, minor_ver = 3, 0

        has_immutable = False

        if major_ver >= 4:
            # Core Profile (GL 3.1+) forbids glGetString(GL_EXTENSIONS).
            # Enumerate extensions by index instead.
            try:
                num_exts = int(_GL.glGetIntegerv(_GL.GL_NUM_EXTENSIONS))
                extensions = {
                    _GL.glGetStringi(_GL.GL_EXTENSIONS, i).decode()
                    for i in range(num_exts)
                }
                is_42_or_later = (major_ver > 4) or (major_ver == 4 and minor_ver >= 2)
                if "GL_ARB_texture_storage" in extensions or is_42_or_later:
                    has_immutable = True

            except Exception as e:
                logger.warning("Could not enumerate GL extensions: %s", e)

        else:
            # Compatibility / legacy profile below GL 4.0: fall back to the
            # deprecated GL_EXTENSIONS string.
            try:
                ext_str = _GL.glGetString(_GL.GL_EXTENSIONS)
                if ext_str and b"GL_ARB_texture_storage" in ext_str:
                    has_immutable = True
            except Exception as e:
                logger.debug("Legacy extension string unavailable: %s", e)

        GL_CONFIGS["default"] = GLConfig(USE_IMMUTABLE_STORAGE=has_immutable)

        if has_immutable:
            logger.info("GL Strategy: Immutable Storage (Optimized)")
        else:
            logger.info("GL Strategy: Mutable Storage (Legacy)")

    except Exception as e:
        logger.error("Context capability check failed: %s", e)
        GL_CONFIGS["default"] = GLConfig(USE_IMMUTABLE_STORAGE=False)

    # Enable the KHR_debug callback now that the config is written and the
    # context is confirmed live.  GLConfig.DEBUG_MODE is set by the config
    # layer, not by _DEBUG_ENV, so operators can enable GL debug output
    # without reloading the backend.
    if GL_CONFIGS["default"].DEBUG_MODE:
        logger.info(
            "GL Debug Mode: ENABLED (PyOpenGL checking: %s)", OpenGL.ERROR_CHECKING
        )
        from image.gl.debug import enable_gl_debug_output
        enable_gl_debug_output()
    else:
        logger.info("GL Debug Mode: DISABLED (High Performance)")


GL = _GL
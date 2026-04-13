"""OpenGL context and surface configuration utilities for 2D image processing.

Provides safe context management and Qt surface format configuration
optimized for GPU image processing operations.
"""

import logging
from contextlib import contextmanager
from typing import Generator

from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from cross_platform.qt6_utils.image.gl.error import GLError

logger = logging.getLogger(__name__)

__all__ = [
    "get_surface_format",
    "gl_context",
]


def get_surface_format(
        debug: bool = False,
        vsync: bool = False,
        gl_version: tuple[int, int] = (4, 1),
) -> QSurfaceFormat:
    """Configure Qt Surface Format for OpenGL context.

    This function configures Qt's OpenGL context creation parameters.
    It must use Qt's QSurfaceFormat API as it's part of Qt's OpenGL
    initialization, not a direct OpenGL call.

    Optimizations for 2D image processing:
    - Disables depth buffer (saves GPU memory bandwidth)
    - Disables MSAA (high-res image processing doesn't benefit)
    - Enables double buffering (flicker-free rendering)

    Args:
        debug: Enable OpenGL debug context (GL_KHR_debug).
               Useful for development; adds performance overhead.
        vsync: Enable vertical sync (swap interval).
               Controls frame rate synchronization.
               Set False for maximum throughput in processing.
        gl_version: Target OpenGL version (major, minor).
                   - (4, 1): Minimum supported, widest compatibility
                   - (4, 5): DSA support, improved performance
                   - (4, 6): Latest features
                   Defaults to (4, 1).

    Returns:
        QSurfaceFormat configured for optimal 2D rendering

    Raises:
        ImportError: If Qt binding (PyQt6/PySide6) not found
    """
    if QSurfaceFormat is None:
        raise ImportError("Qt binding (PyQt6/PySide6) not found.")

    fmt = QSurfaceFormat()

    # Request OpenGL version
    fmt.setVersion(*gl_version)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)

    # Optimization: Disable depth buffer for 2D-only rendering
    # Saves GPU memory bandwidth and cache pressure
    fmt.setDepthBufferSize(0)

    # Keep stencil for potential clipping/masking operations
    fmt.setStencilBufferSize(8)

    # MSAA: Disable for high-res image processing
    # MSAA is costly for fill-rate; image processing doesn't need antialiasing
    fmt.setSamples(0)

    # Double buffering is essential for flicker-free rendering
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)

    # VSync Control
    # 0 = immediate (no vsync, maximum throughput)
    # 1 = vsync (60 FPS typical)
    # -1 = adaptive vsync (if supported)
    fmt.setSwapInterval(1 if vsync else 0)

    if debug:
        fmt.setOption(QSurfaceFormat.FormatOption.DebugContext)
        logger.debug(
            f"GL debug context enabled (version {gl_version[0]}.{gl_version[1]})")

    return fmt


@contextmanager
def gl_context(
        widget: QOpenGLWidget,
        operation: str,
) -> Generator[None, None, None]:
    """Safely manage OpenGL context activation.

    Ensures the OpenGL context is made current before the operation
    and properly released afterward, even if an exception occurs.

    Args:
        widget: The QOpenGLWidget to make current
        operation: Description of the operation (for logging/errors).
                  Example: "texture upload", "shader compilation"

    Raises:
        GLError: If widget is invalid or context operations fail
    """
    if not widget.isValid():
        raise GLError(f"Widget invalid during '{operation}'")

    try:
        logger.debug(f"Making GL context current for: {operation}")
        widget.makeCurrent()
    except Exception as e:
        raise GLError(
            f"Failed to make context current during '{operation}': {e}") from e

    try:
        yield
    except Exception as e:
        logger.error(f"Error during '{operation}': {e}")
        raise
    finally:
        try:
            logger.debug(f"Releasing GL context after: {operation}")
            widget.doneCurrent()
        except Exception as e:
            logger.warning(f"Error releasing context after '{operation}': {e}")

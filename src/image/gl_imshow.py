import logging
import sys
from typing import Optional

import numpy as np
from PyQt6.QtCore import QCoreApplication, Qt
from PyQt6.QtGui import QGuiApplication, QSurfaceFormat
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout

from pycore.log.ctx import ContextAdapter
from image.gl.backend import GL
from image.gl.utils import get_surface_format
from image.gl.view import GLFrameViewer
from image.pipeline.stats import (
    get_frame_stats, FrameStats)
from image.settings.base import ImageSettings
from image.settings.pixels import PixelFormat

logger = ContextAdapter(logging.getLogger(__name__), {})

_APP_INSTANCE = None
_ACTIVE_WINDOWS = []


# noinspection PyPep8Naming
class GLImageShow(QWidget):
    def __init__(self,
                 parent: Optional[QWidget] = None):
        """
        Args:
            X: Input image data.
            fmt: Explicit ImageFormat. If None, inferred from X.
            vmin: Minimum intensity (0.0 for OpenGL).
            vmax: Maximum intensity (1.0 for OpenGL).
            cmap: Colormap name (e.g., 'viridis', 'gray'). Ignored for RGB images.
            title: Window title.
        """
        super().__init__(parent=parent)
        self.settings = ImageSettings()
        self.metadata: FrameStats | None = None
        self._deffered_frame: np.ndarray | None = None
        self.viewer = GLFrameViewer(settings=self.settings,
                                    monitor_performance=False,
                                    parent=self)

        self._setup_ui()

    def _setup_ui(self):
        QVBoxLayout(self)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.layout().setSpacing(0)
        self.layout().addWidget(self.viewer)
        self.setWindowTitle('GL imshow')

    def set_data(self,
                 X: np.ndarray,
                 fmt: Optional['PixelFormat'] = None,
                 vmin: float | None = None,
                 vmax: float | None = None,
                 cmap: str = "viridis",
                 title: str = "GL imshow"):

        if fmt is None:
            fmt = PixelFormat.infer_from_shape(X.shape) or PixelFormat.RGB

        is_scalar_input = (X.ndim == 2) or (X.ndim == 3 and X.shape[2] == 1)
        use_colormap = False

        norm_vmin = vmin
        norm_vmax = vmax

        if is_scalar_input:
            fmt = PixelFormat.MONOCHROME
            use_colormap = True

            # Auto-scale contrast if user didn't provide range
            if norm_vmin is None:
                norm_vmin = self.metadata.dmin
            if norm_vmax is None:
                norm_vmax = self.metadata.dmax
        else:
            if norm_vmin is None: norm_vmin = 0.0
            if norm_vmax is None: norm_vmax = 1.0

        self.settings.format = fmt
        self.settings.norm_vmin = norm_vmin
        self.settings.norm_vmax = norm_vmax
        self.settings.colormap_enabled = use_colormap
        self.settings.colormap_name = cmap
        self.settings.colormap_reverse = bool("_r" in cmap)

        self.setWindowTitle(title)
        self.metadata: FrameStats = get_frame_stats(image=X)
        if self.viewer.is_initialized:
            logger.debug("Uploadind image to pinned buffer")
            h, w = X.shape[:2]
            dtype = X.dtype
            pbo, pbo_buf = self.viewer.request_pinned_buffer(width=w,
                                                             height=h,
                                                             pixel_fmt=fmt,
                                                             dtype=dtype)
            self.viewer.write_to_pinned_buffer(pbo_array=pbo_buf,
                                               image=X,
                                               pixel_fmt=fmt)
            self.viewer.present_pinned(pbo_object=pbo,
                                       metadata=self.metadata,
                                       width=w,
                                       height=h,
                                       img_fmt=fmt,
                                       dtype=dtype)
            self.viewer.repaint()
            if GL and hasattr(GL, "glFinish"):
                logger.debug("Syncing GPU ")
                GL.glFinish()
            return

        self._deffered_frame = X
        self.update()

    def keyPressEvent(self, event):
        """
        Handle key press events.
        Closes the window when 'Esc' or 'Q' is pressed.
        """
        if event.key() == Qt.Key.Key_Escape or event.key() == Qt.Key.Key_Q:
            self.close()
            event.accept()
        else:
            super().keyPressEvent(event)

    def showEvent(self, event):
        """
        Triggered when the window is shown.
        OpenGL widgets usually require visibility before painting/context creation.
        """

        # Upload texture data to GPU
        if self._deffered_frame is not None:
            self.viewer.present(self._deffered_frame, self.metadata,
                                self.settings.format)
            GL.glFinish()
            self._deffered_frame = None
        super().showEvent(event)

    def closeEvent(self, event):
        self.viewer.cleanup()
        return super().closeEvent(event)


# noinspection PyPep8Naming
def imshow(
        X: np.ndarray,
        title: str = "Image",
        cmap: str = "gray",
        fmt: Optional['PixelFormat'] = None,
        vmin: float | None = None,
        vmax: float | None = None,
        block: bool = True
) -> QWidget:
    """
    Drop-in replacement for plt.imshow / cv2.imshow using Custom OpenGL.

    Args:
        X: The numpy array (2D or 3D).
        title: Window title.
        cmap: Colormap ('viridis', 'gray', 'magma', 'plasma').
              Defaults to 'viridis'. Ignored if X is RGB/BGR.
        fmt: Explicit ImageFormat (optional).
        vmin: Minimum value for normalization.
        vmax: Maximum value for normalization.
        block: If True, runs the Qt Event loop (blocks).
               If False, shows window and returns immediately.

    Returns:
        The QMainWindow instance.
    """
    global _APP_INSTANCE, _ACTIVE_WINDOWS

    if X is None:
        raise ValueError("Input X cannot be None")

    if _APP_INSTANCE is None:
        _APP_INSTANCE = QGuiApplication.instance() or QCoreApplication.instance()
        if _APP_INSTANCE is None:
            _APP_INSTANCE = QApplication(sys.argv)

    QSurfaceFormat.setDefaultFormat(get_surface_format())

    window = GLImageShow()
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    window.setAttribute(Qt.WidgetAttribute.WA_DontCreateNativeAncestors, True)
    window.setAttribute(Qt.WidgetAttribute.WA_NativeWindow, True)
    window.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)
    window.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

    height, width = X.shape[:2]
    screen_size = _APP_INSTANCE.primaryScreen().size()
    max_w, max_h = screen_size.width(), screen_size.height()
    display_w = min(width, max_w)
    display_h = min(height, max_h)

    if display_w > 0 and display_h > 0:
        window.resize(display_w, display_h)
    else:
        window.resize(800, 600)

    window.show()
    window.set_data(X=X,
                    fmt=fmt,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    title=title)
    window.update()

    # Prevent Python GC from killing the window object immediately
    _ACTIVE_WINDOWS.append(window)

    def _cleanup(w):
        if w in _ACTIVE_WINDOWS:
            _ACTIVE_WINDOWS.remove(w)

    window.destroyed.connect(lambda: _cleanup(window))

    if block:
        _APP_INSTANCE.exec()

    return window

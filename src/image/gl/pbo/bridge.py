"""
pbo_qt_bridge.py
================
Qt-specific bridge between pbo_core and QOpenGLWidget.

This is the **only** file in the PBO stack that imports PyQt6.
pbo_core.py is entirely framework-agnostic.

Classes
-------
QtWidgetBridge
    Implements the WidgetBridge protocol for a QOpenGLWidget.
    Handles HiDPI scaling, context validity, and safe teardown detection.

QtPBOBridge  (QObject)
    Owns one PBODownloadManager and one QtWidgetBridge.
    Converts the numpy FrameCallback into a pyqtSignal(QImage).
    Wires widget.destroyed → manager.cleanup automatically.

Upload usage
------------
PBOUploadManager lives entirely in pbo_core.py and requires no Qt bridge —
instantiate it directly inside initializeGL and call cleanup() from the
widget's destructor or closeEvent.

Download usage
--------------
    # in initializeGL():
    self._pbo = QtPBOBridge(self)
    self._pbo.initialize()
    self._pbo.imageReady.connect(self.on_image)

    # in resizeGL(w, h):
    self._pbo.on_resize(w, h)

    # in paintGL() — very last line:
    self._pbo.capture_now()

    # to trigger a framebuffer capture:
    self._pbo.request_capture()

    # to trigger a texture capture:
    self._pbo.request_texture_capture(tex_id, width, height)
"""

from __future__ import annotations

import logging

import numpy as np

from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui  import QImage
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from image.gl.pbo.core import PBODownloadManager
from qtcore.reference import has_qt_cpp_binding

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# QtWidgetBridge
# ---------------------------------------------------------------------------
class QtWidgetBridge:
    """
    Implements WidgetBridge for a QOpenGLWidget.

    physical_width / physical_height multiply logical pixels by
    devicePixelRatio so the PBO is always sized to the real framebuffer —
    critical on HiDPI / Retina displays where the ratio is typically 2.0.
    """

    def __init__(self, widget: QOpenGLWidget) -> None:
        self._widget = widget

    # -- WidgetBridge protocol ------------------------------------------

    def physical_width(self) -> int:
        return round(self._widget.width() * self._widget.devicePixelRatio())

    def physical_height(self) -> int:
        return round(self._widget.height() * self._widget.devicePixelRatio())

    def schedule_update(self) -> None:
        self._widget.update()

    def make_current(self) -> None:
        self._widget.makeCurrent()

    def done_current(self) -> None:
        self._widget.doneCurrent()

    def is_context_valid(self) -> bool:
        """
        Return True only when both the C++ widget and its GL context exist.
        Wrapped in RuntimeError to handle the deletion race window.
        """
        try:
            if not has_qt_cpp_binding(self._widget):
                return False
            ctx = self._widget.context()
            return ctx is not None and ctx.isValid()
        except RuntimeError:
            return False


# ---------------------------------------------------------------------------
# QtPBOBridge
# ---------------------------------------------------------------------------
class QtPBOBridge(QObject):
    """
    Qt adapter for PBODownloadManager.

    Owns a PBODownloadManager and a QtWidgetBridge.  Converts the numpy
    FrameCallback into a typed pyqtSignal(QImage).  Wires widget.destroyed
    to manager.cleanup automatically.

    imageReady emits a QImage in Format_RGBA8888 — fully detached from GPU
    memory and safe to display, save, or pass to another thread.

    For uploads, use PBOUploadManager directly from pbo_core — no Qt wrapper
    is needed because the upload manager has no widget lifecycle dependencies.
    """

    imageReady = pyqtSignal(QImage)

    def __init__(self, widget: QOpenGLWidget) -> None:
        super().__init__(widget)
        self._widget  = widget
        self._bridge  = QtWidgetBridge(widget)
        self._manager = PBODownloadManager(
            bridge         = self._bridge,
            on_frame_ready = self._on_frame_ready,
        )
        # Automatic cleanup when the widget's C++ side is destroyed.
        widget.destroyed.connect(self._manager.cleanup)

    # ------------------------------------------------------------------
    # Public interface — thin delegation to PBODownloadManager
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Call once inside initializeGL(). Context is already current."""
        self._manager.initialize()

    def on_resize(self, w: int = 0, h: int = 0) -> None:
        """
        Call from resizeGL(w, h).

        Arguments are forwarded but ignored by the manager — physical size
        is always queried directly from the bridge so HiDPI scaling is
        handled correctly.
        """
        self._manager.on_resize(w, h)

    def capture_now(self) -> None:
        """
        Call from paintGL() as the very last statement after all rendering.
        Drives the two-frame async pipeline.
        """
        self._manager.capture_now()

    def request_capture(self) -> None:
        """Request a framebuffer screenshot. imageReady fires asynchronously."""
        self._manager.request_capture()

    def request_texture_capture(
        self, texture_id: int, width: int, height: int
    ) -> None:
        """
        Request a texture download. imageReady fires asynchronously.
        Must be called while the GL context is current.
        """
        self._manager.request_texture_capture(texture_id, width, height)

    def cleanup(self) -> None:
        """Manually release GPU resources. Safe to call multiple times."""
        self._manager.cleanup()

    # expose read-only dimensions for callers that need them
    @property
    def capture_width(self) -> int:
        return self._manager.width

    @property
    def capture_height(self) -> int:
        return self._manager.height

    # ------------------------------------------------------------------
    # Private: numpy → QImage
    # ------------------------------------------------------------------

    def _on_frame_ready(
        self, arr: np.ndarray, width: int, height: int
    ) -> None:
        """
        Receives a (H×W×4) uint8 RGBA array from the core, wraps it in a
        QImage, and emits imageReady.

        arr is already:
          - top-left origin (vertically flipped from OpenGL convention)
          - C-contiguous (safe for QImage's internal pointer arithmetic)
          - fully independent of GPU memory

        QImage.copy() detaches the image from arr's buffer so the array can
        be collected without affecting the emitted QImage lifetime.
        """
        qimage = QImage(
            arr.data,
            width,
            height,
            QImage.Format.Format_RGBA8888,
        ).copy()
        self.imageReady.emit(qimage)
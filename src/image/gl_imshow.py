"""
GL-accelerated image viewer
"""

from __future__ import annotations

import logging
from functools import partial
from typing import Final, NamedTuple, Optional, Callable

import numpy as np
from PyQt6.QtCore import Qt, QCoreApplication, pyqtSlot, QDir, QFileInfo
from PyQt6.QtGui import QSurfaceFormat, QScreen, QImage
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QMessageBox, QFileDialog,
)

from image.gl.backend import GL
from image.gl.utils import get_surface_format
from image.gl.view import GLFrameViewer
from image.gui.overlay.toolbar import GLToolbar
from image.pipeline.stats import get_frame_stats, FrameStats
from image.settings.base import ImageSettings
from image.settings.pixels import PixelFormat
from pycore.log.ctx import ContextAdapter
from qtcore.app import Application

__all__ = ["imshow", "GLImageShow"]

logger = ContextAdapter(logging.getLogger(__name__), {})

APP_NAME: Final[str] = "GL Image Show"
ORG_NAME: Final[str] = "The Little Engine That Could"
APP_VERSION: Final[str] = "0.1.0"

_ZOOM_STEP: Final[float] = 1.25  # multiply / divide per click
_ZOOM_MIN: Final[float] = 0.05

_DEFAULT_WINDOW_SIZE: Final[tuple[int, int]] = (800, 600)


class _SaveFormat(NamedTuple):
    """Associates a Qt format tag with its file suffix."""
    qt_tag: str  # e.g. "PNG"
    suffix: str  # e.g. ".png"


_FILTER_TO_FORMAT: dict[str, _SaveFormat] = {
    "PNG": _SaveFormat("PNG", ".png"),
    "JPEG": _SaveFormat("JPEG", ".jpg"),
    "BMP": _SaveFormat("BMP", ".bmp"),
}
_EXTENSION_TO_FORMAT: dict[str, _SaveFormat] = {
    "png": _SaveFormat("PNG", ".png"),
    "jpg": _SaveFormat("JPEG", ".jpg"),
    "jpeg": _SaveFormat("JPEG", ".jpg"),
    "bmp": _SaveFormat("BMP", ".bmp"),
}
_DEFAULT_SAVE_FORMAT: Final[_SaveFormat] = _SaveFormat("PNG", ".png")

_SAVE_DIALOG_FILTERS: Final[list[str]] = [
    "PNG Image (*.png)",
    "JPEG Image (*.jpg *.jpeg)",
    "BMP Image (*.bmp)",
    "All Files (*)",
]


def _detect_save_format(selected_filter: str, file_path: str) -> _SaveFormat:
    """
    Resolve a _SaveFormat from the dialog's selected filter string,
    falling back to the file extension, then to PNG.
    """
    for key, fmt in _FILTER_TO_FORMAT.items():
        if key in selected_filter:
            return fmt

    ext = QFileInfo(file_path).suffix().lower()
    return _EXTENSION_TO_FORMAT.get(ext, _DEFAULT_SAVE_FORMAT)


class _AppManager:
    """
    Application lifecycle manager
    """

    def __init__(self) -> None:
        self._app: QApplication | QCoreApplication | None = None
        self._surface_format_set: bool = False
        self._active_windows: list[GLImageShow] = []

    def ensure_app(self) -> QApplication | QCoreApplication:
        """
        Return the managed QApplication, creating it if necessary.
        Respects any QApplication already owned by the caller.
        Surface format is configured exactly once.
        """
        if self._app is not None:
            return self._app

        existing = QApplication.instance()
        if existing is not None:
            self._app = existing
        else:
            self._app = Application(
                app_name=APP_NAME,
                org_name=ORG_NAME,
                app_version=APP_VERSION,
            )
            self._app.show_splash(min_display_ms=500)

        # Global GL mutation — must happen before any surface is created.
        if not self._surface_format_set:
            QSurfaceFormat.setDefaultFormat(get_surface_format())
            self._surface_format_set = True

        return self._app

    def register(self, window: GLImageShow) -> None:
        """Pin *window* against the garbage collector for the session."""
        self._active_windows.append(window)

    def unregister(self, window: GLImageShow) -> None:
        """Remove *window* from the registry when it is destroyed."""
        try:
            self._active_windows.remove(window)
        except ValueError:
            pass  # already removed; safe to ignore


_app_manager = _AppManager()


class GLImageShow(QWidget):
    """OpenGL-accelerated image viewer widget."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent=parent)
        self._frame_stats: FrameStats | None = None
        self._deferred_frame: np.ndarray | None = None
        self._pending_capture: bool = False
        self._toolbar: GLToolbar | None = None
        self.viewer: GLFrameViewer | None = None

        self._settings = ImageSettings()
        self._setup_ui()
        self._wire_ui()

    def _wire_ui(self) -> None:
        """Wire up the UI."""
        self.viewer.image_ready.connect(self._on_capture_ready)
        # Wire toolbar signals
        self._toolbar.zoom_in_requested.connect(self._zoom_in)
        self._toolbar.zoom_out_requested.connect(self._zoom_out)
        self._toolbar.zoom_reset_requested.connect(self._zoom_reset)
        self._toolbar.capture_requested.connect(self._on_capture_requested)

    def _setup_toolbar(self) -> None:
        """Set up the toolbar."""
        self._toolbar = GLToolbar(parent=self)

        if self._toolbar is None:
            return

        self._settings.update_setting("zoom_enabled", True)
        self._toolbar.raise_()

    def _setup_ui(self) -> None:
        """Set up the UI elements."""
        self.viewer = GLFrameViewer(settings=self._settings, parent=self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.viewer)

        self._setup_toolbar()

    @pyqtSlot()
    def _zoom_in(self) -> None:
        self._apply_zoom(_ZOOM_STEP)

    @pyqtSlot()
    def _zoom_out(self) -> None:
        self._apply_zoom(1.0 / _ZOOM_STEP)

    @pyqtSlot()
    def _zoom_reset(self) -> None:
        """Restore 1:1 pixel mapping."""
        if self._settings.zoom_enabled:
            # [B4] Use float literal — the settings contract expects float.
            self._settings.update_setting("zoom", 1.0)

    def _apply_zoom(self, factor: float) -> None:
        if self._settings.zoom_enabled:
            new_zoom = max(_ZOOM_MIN, self._settings.zoom * factor)
            self._settings.update_setting("zoom", new_zoom)

    @pyqtSlot()
    def _on_capture_requested(self) -> None:
        """Initiate a capture."""
        if self.viewer:
            self.viewer.request_capture()
            self._pending_capture = True

    @pyqtSlot(QImage)
    def _on_capture_ready(self, image: QImage) -> None:
        """Save the captured QImage to a user-chosen file."""
        if not self._pending_capture:
            return

        if image.isNull():
            QMessageBox.warning(
                self, "Capture Failed",
                "No image data received from the viewer.",
            )
            return

        self._pending_capture = False

        file_path, selected_filter = QFileDialog.getSaveFileName(
            parent=self,
            caption="Save Captured Image",
            directory=QDir.homePath(),
            filter=";;".join(_SAVE_DIALOG_FILTERS),
            initialFilter=_SAVE_DIALOG_FILTERS[0],
        )

        if not file_path:
            return  # User canceled

        save_fmt = _detect_save_format(selected_filter, file_path)

        if not QFileInfo(file_path).suffix():
            file_path += save_fmt.suffix

        if image.save(file_path, save_fmt.qt_tag):
            QMessageBox.information(
                self, "Capture Saved", f"Screenshot saved to:\n{file_path}",
            )
        else:
            QMessageBox.critical(
                self, "Save Error",
                f"Failed to save screenshot to:\n{file_path}",
            )

    @staticmethod
    def _infer_format(
            X: np.ndarray,
            fmt: Optional[PixelFormat],
    ) -> PixelFormat:
        """Determine pixel format from shape or user hint."""
        if fmt:
            return fmt
        inferred = PixelFormat.infer_from_shape(X.shape)
        return inferred if inferred is not None else PixelFormat.RGB

    def _resolve_norm_and_format(
            self,
            X: np.ndarray,
            fmt: PixelFormat,
            vmin: float | None,
            vmax: float | None,
    ) -> tuple[PixelFormat, bool, float, float]:
        """
        Compute the effective format, colormap flag, and normalization range.

        Returns:
            (fmt, use_colormap, norm_vmin, norm_vmax)
        """
        is_scalar = X.ndim == 2 or (X.ndim == 3 and X.shape[2] == 1)
        if is_scalar:
            fmt = PixelFormat.MONOCHROME
            use_cmap = True
            if vmin is None:
                vmin = (
                    self._frame_stats.dmin if self._frame_stats
                    else float(X.min())
                )
            if vmax is None:
                vmax = (
                    self._frame_stats.dmax if self._frame_stats
                    else float(X.max())
                )
        else:
            use_cmap = False
            vmin = 0.0 if vmin is None else vmin
            vmax = 1.0 if vmax is None else vmax

        return fmt, use_cmap, vmin, vmax

    def _update_normalization_settings(
            self,
            fmt: PixelFormat,
            vmin: float,
            vmax: float,
            use_cmap: bool,
            cmap: str,
    ) -> None:
        """Persist computed normalization and colormap settings."""
        self._settings.update_setting("format", fmt)
        self._settings.update_setting("norm_vmin", vmin)
        self._settings.update_setting("norm_vmax", vmax)
        self._settings.update_setting("colormap_enabled", use_cmap)
        self._settings.update_setting("colormap_name", cmap)
        self._settings.update_setting("colormap_reverse", cmap.endswith("_r"))

    def _upload_to_gpu_and_present(
            self,
            X: np.ndarray,
            fmt: PixelFormat,
    ) -> None:
        """Upload image to pinned buffer, present, and synchronise GL."""
        if not (isinstance(self.viewer, GLFrameViewer) and self._frame_stats):
            return

        h, w = X.shape[:2]
        dtype = X.dtype
        pbo, pbo_buf = self.viewer.request_pinned_buffer(
            width=w, height=h, pixel_fmt=fmt, dtype=dtype,
        )
        self.viewer.write_to_pinned_buffer(
            pbo_array=pbo_buf, image=X, pixel_fmt=fmt,
        )
        self.viewer.present_pinned(
            pbo_object=pbo,
            metadata=self._frame_stats,
            width=w,
            height=h,
            img_fmt=fmt,
            dtype=dtype,
        )
        self.viewer.repaint()
        self._gl_finish()

    def _queue_deferred_frame(self, X: np.ndarray) -> None:
        """Store frame for deferred upload and schedule a repaint."""
        self._deferred_frame = X
        self.update()

    def _gl_finish(self) -> None:
        if GL and hasattr(GL, "glFinish"):
            logger.debug("Syncing GPU")
            GL.glFinish()

    def set_data(
            self,
            X: np.ndarray,
            fmt: Optional[PixelFormat] = None,
            vmin: float | None = None,
            vmax: float | None = None,
            cmap: str = "gray",
            title: str = "GL imshow",
    ) -> None:
        """
        Load new image data, resolve normalization, and display.

        Args:
            X:     2-D or 3-D numpy array.
            fmt:   Explicit PixelFormat; inferred from shape when omitted.
            vmin:  Lower normalization bound (scalar images only).
            vmax:  Upper normalization bound (scalar images only).
            cmap:  Colormap name (ignored for RGB/BGR arrays).
            title: Window title.
        """
        if X is None:
            raise ValueError("Input array X cannot be None")

        self.setWindowTitle(title)
        self._frame_stats = get_frame_stats(image=X)

        fmt = self._infer_format(X, fmt)
        fmt, use_cmap, norm_vmin, norm_vmax = self._resolve_norm_and_format(
            X, fmt, vmin, vmax,
        )
        self._update_normalization_settings(
            fmt, norm_vmin, norm_vmax, use_cmap, cmap,
        )

        if self.viewer is not None and self.viewer.is_initialized:
            self._upload_to_gpu_and_present(X, fmt)
        else:
            self._queue_deferred_frame(X)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._toolbar is not None:
            self._toolbar.reposition()

    def keyPressEvent(self, event) -> None:
        # Dispatch table
        _key_actions: dict[Qt.Key, Callable] = {
            Qt.Key.Key_Escape: self.close,
            Qt.Key.Key_Q: self.close,
            Qt.Key.Key_Plus: self._zoom_in,
            Qt.Key.Key_Equal: self._zoom_in,
            Qt.Key.Key_Minus: self._zoom_out,
            Qt.Key.Key_1: self._zoom_reset,
        }
        action = _key_actions.get(event.key())
        if action is not None:
            action()
            event.accept()
        else:
            super().keyPressEvent(event)

    def showEvent(self, event) -> None:
        if (
                self._deferred_frame is not None
                and self._deferred_frame.size > 0
                and self.viewer is not None
                and self._frame_stats is not None
        ):
            self.viewer.present(
                self._deferred_frame,
                self._frame_stats,
                self._settings.format,
            )
            self._gl_finish()
            self._deferred_frame = None

        super().showEvent(event)

        if self._toolbar is not None:
            self._toolbar.reposition()
            self._toolbar.raise_()

    def closeEvent(self, event) -> None:
        if self.viewer is not None:
            self.viewer.cleanup()
            self.viewer = None
        super().closeEvent(event)


def _window_size(
        app: QApplication | QCoreApplication,
        img_w: int,
        img_h: int,
) -> tuple[int, int]:
    """
    Clamp image dimensions to the available screen area (logical pixels).

    Falls back to ``_DEFAULT_WINDOW_SIZE`` when no screen is available.
    """
    # [D3] Magic numbers replaced by the named constant.
    screen: QScreen | None = (
        app.primaryScreen() if isinstance(app, QApplication) else None
    )
    if screen is None:
        return _DEFAULT_WINDOW_SIZE

    available = screen.availableSize()
    scale = screen.devicePixelRatio()
    max_w = int(available.width() / scale)
    max_h = int(available.height() / scale)

    w = min(img_w, max_w) if img_w > 0 else _DEFAULT_WINDOW_SIZE[0]
    h = min(img_h, max_h) if img_h > 0 else _DEFAULT_WINDOW_SIZE[1]
    return w, h


def _configure_window(window: GLImageShow) -> None:
    """
    [D6] Apply standard Qt widget attributes to *window*.

    Extracted from ``imshow`` so the attributes are named and documented
    in one place, and ``imshow`` itself stays focused on orchestration.
    """
    # WA_DeleteOnClose      — Qt frees the C++ object when the window closes.
    # WA_OpaquePaintEvent   — Skip background clear; GL fills every pixel.
    # WA_NoSystemBackground — Suppress the OS-level background flicker.
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    window.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
    window.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)


def imshow(
        X: np.ndarray,
        title: str = "Image",
        cmap: str = "gray",
        fmt: Optional[PixelFormat] = None,
        vmin: float | None = None,
        vmax: float | None = None,
        block: bool = True,
) -> GLImageShow:
    """
    Drop-in replacement for ``plt.imshow`` / ``cv2.imshow`` using OpenGL.

    Args:
        X:     The numpy array (2-D or 3-D).
        title: Window title.
        cmap:  Colormap name ('viridis', 'gray', 'magma', 'plasma').
               Ignored when *X* is RGB/BGR.
        fmt:   Explicit ``PixelFormat``; inferred from shape when omitted.
        vmin:  Minimum value for normalization.
        vmax:  Maximum value for normalization.
        block: If ``True``, enter the Qt event loop (blocks until all windows
               are closed).  If ``False``, show and return immediately.

    Returns:
        The ``GLImageShow`` window instance.
    """
    if X is None:
        raise ValueError("Input array X cannot be None")

    app = _app_manager.ensure_app()

    window = GLImageShow()
    _configure_window(window)

    img_h, img_w = X.shape[:2]
    window.resize(*_window_size(app, img_w, img_h))

    window.set_data(X=X, fmt=fmt, vmin=vmin, vmax=vmax, cmap=cmap, title=title)
    window.show()

    if isinstance(app, Application):
        app.finish_splash(window)

    _app_manager.register(window)

    window.destroyed.connect(partial(_app_manager.unregister, window))

    if block:
        app.exec()

    return window

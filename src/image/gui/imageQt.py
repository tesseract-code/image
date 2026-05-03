from typing import Optional

import logging
import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt, QRectF
from PyQt6.QtGui import QPainter, QColor, QPixmap, QImage
from PyQt6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QWidget, QFrame)

from pycore.log.ctx import with_logger

LOG = logging.getLogger(__name__)


@with_logger
class GraphicsImageView(QGraphicsView):
    """QGraphicsView for displaying images.

    Parameters
    ----------
    parent : Optional[QWidget], optional
        Parent widget, by default None.
    """

    zoomChanged = pyqtSignal(float)

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent)

        self._pixmap_item: Optional[QGraphicsPixmapItem] = None
        self._zoom_factor = 1.0
        self._smooth_scaling = False

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._setup_view()

    def _setup_view(self):
        """Configure view for high-fidelity image display."""
        self.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setInteractive(False)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setBackgroundBrush(QColor(32, 32, 32))

    def set_image_from_array(self, array: np.ndarray,
                             magnification: int = 1) -> bool:
        """Set image from numpy array with optional magnification.

        Parameters
        ----------
        array : np.ndarray
            Input image array. Can be grayscale (2D) or RGB (3D with 3 channels).
        magnification : int, optional
            Integer magnification factor, by default 1.

        Returns
        -------
        bool
            True if the image was set successfully, False otherwise.
        """
        try:
            self._logger.debug(
                f"SETTING CROP IMAGE - min: {np.min(array):.6f}, max: {np.max(array):.6f}, shape: {array.shape}, dtype: {array.dtype}")

            if array.ndim == 3 and array.shape[2] == 3:
                self._logger.debug("Detected RGB image")
                if array.dtype == np.float32 and array.min() >= 0 and array.max() <= 1.0:
                    self._logger.debug(
                        "Image appears to be normalized float32 RGB - converting directly to uint8")

            self._logger.debug(array)
            qimage = GraphicsImageView._array_to_qimage(array)
            if qimage is None:
                return False

            if magnification > 1:
                qimage = qimage.scaled(
                    qimage.width() * magnification,
                    qimage.height() * magnification,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.FastTransformation
                )

            pixmap = QPixmap.fromImage(qimage)
            return self.set_pixmap(pixmap)

        except Exception as e:
            raise e
            return False

    def set_pixmap(self, pixmap: QPixmap) -> bool:
        """Set the pixmap to display and resize view to match exactly.

        Parameters
        ----------
        pixmap : QPixmap
            Pixmap to display.

        Returns
        -------
        bool
            True on success, False on failure.
        """
        try:
            if self._pixmap_item is not None:
                self._scene.removeItem(self._pixmap_item)

            self._pixmap_item = QGraphicsPixmapItem(pixmap)
            self._pixmap_item.setTransformationMode(
                Qt.TransformationMode.FastTransformation)
            self._scene.addItem(self._pixmap_item)
            self._scene.setSceneRect(QRectF(pixmap.rect()))
            self.resetTransform()
            self.setFixedSize(pixmap.size())
            return True

        except Exception as e:
            self._logger.error(f"Error setting pixmap: {e}")
            return False

    @staticmethod
    def _array_to_qimage(array: np.ndarray) -> Optional[QImage]:
        """Convert numpy array to QImage efficiently.

        Parameters
        ----------
        array : np.ndarray
            Input array, 2D grayscale or 3D RGB.

        Returns
        -------
        Optional[QImage]
            QImage object or None if conversion fails.
        """
        try:
            if not array.flags['C_CONTIGUOUS']:
                array = np.ascontiguousarray(array)

            height, width = array.shape[:2]

            if array.ndim == 3 and array.shape[2] == 3:
                LOG.debug("Processing RGB image in crop widget")

                if array.dtype == np.float32:
                    if array.min() >= 0 and array.max() <= 1.0:
                        LOG.debug("Converting float32 [0,1] RGB to uint8 [0,255]")
                        array_uint8 = (array * 255).astype(np.uint8)
                    else:
                        LOG.debug("Normalizing float32 RGB to uint8")
                        array_normalized = (array - array.min()) / (
                                    array.max() - array.min())
                        array_uint8 = (array_normalized * 255).astype(np.uint8)
                else:
                    array_uint8 = GraphicsImageView._normalize_to_uint8(
                        array)

                bytes_per_line = 3 * width
                qimage = QImage(
                    array_uint8.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format.Format_RGB888
                )

                unique_colors = np.unique(array_uint8.reshape(-1, 3), axis=0)
                LOG.debug(f"Final RGB unique colors count: {len(unique_colors)}")
                if len(unique_colors) <= 10:
                    LOG.debug(f"Final RGB colors: {unique_colors}")

            elif array.ndim == 2:
                LOG.debug("Processing grayscale image in crop widget")

                if _is_binary_image(array):
                    LOG.debug("Detected binary image, using fixed normalization")
                    array_uint8 = GraphicsImageView._binary_to_uint8(
                        array)
                else:
                    if array.dtype != np.uint8:
                        array_uint8 = GraphicsImageView._normalize_to_uint8(
                            array)
                    else:
                        array_uint8 = array

                bytes_per_line = width
                qimage = QImage(
                    array_uint8.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format.Format_Grayscale8
                )

            else:
                LOG.debug(f"Unsupported array dimensions: {array.ndim}")
                return None

            LOG.debug(
                f"Created QImage: {qimage.width()}x{qimage.height()}, format: {qimage.format()}")
            return qimage.copy()

        except Exception as e:
            LOG.debug(f"Error converting array to QImage: {e}")
            return None

    @staticmethod
    def _binary_to_uint8(array: np.ndarray) -> np.ndarray:
        """Convert binary (0/1) image to uint8 without interpolation artifacts.

        Parameters
        ----------
        array : np.ndarray
            Input binary image.

        Returns
        -------
        np.ndarray
            uint8 array with values 0 or 255.
        """
        result = np.zeros_like(array, dtype=np.uint8)

        threshold = 0.1
        mask = array > threshold
        result[mask] = 255

        LOG.debug(
            f"Binary conversion: {np.sum(mask)} white pixels, {np.sum(~mask)} black pixels")
        return result

    @staticmethod
    def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
        """Normalize array values to uint8 range [0, 255].

        Parameters
        ----------
        array : np.ndarray
            Input array of any numeric type.

        Returns
        -------
        np.ndarray
            Array normalized to uint8.
        """
        if array.ndim == 3:
            LOG.debug("Warning: _normalize_to_uint8 called for RGB array")
            if array.dtype == np.float32 and array.min() >= 0 and array.max() <= 1.0:
                return (array * 255).astype(np.uint8)
            else:
                array_normalized = (array - array.min()) / (
                            array.max() - array.min())
                return (array_normalized * 255).astype(np.uint8)

        arr_min, arr_max = array.min(), array.max()
        LOG.debug(f"Normalizing grayscale - min: {arr_min:.6f}, max: {arr_max:.6f}")

        if arr_max <= arr_min:
            LOG.debug("Constant image, returning zeros")
            return np.zeros_like(array, dtype=np.uint8)

        if _is_binary_image(array) or _is_binary_like(array):
            LOG.debug("Image is binary-like, using binary conversion")
            return GraphicsImageView._binary_to_uint8(array)

        normalized = (array - arr_min) / (arr_max - arr_min)
        result = (normalized * 255).astype(np.uint8)

        unique_result = np.unique(result)
        LOG.debug(f"Normalized to uint8 - unique values: {len(unique_result)}")
        return result


    def clear_image(self):
        """Clear the displayed image and reset to minimum size."""
        if self._pixmap_item is not None:
            self._scene.removeItem(self._pixmap_item)
            self._pixmap_item = None
        self._scene.clear()

        self.setMinimumSize(200, 200)
        self.setMaximumSize(16777215, 16777215)

    def fit_in_view(self):
        """Fit the image in the view while maintaining aspect ratio."""
        if self._pixmap_item is None:
            return

        self.resetTransform()

        scene_rect = self._scene.sceneRect()
        viewport_rect = self.viewport().rect()

        x_scale = viewport_rect.width() / scene_rect.width()
        y_scale = viewport_rect.height() / scene_rect.height()

        scale_factor = min(x_scale, y_scale)

        self.scale(scale_factor, scale_factor)
        self.centerOn(scene_rect.center())

        self._zoom_factor = 1.0
        self.zoomChanged.emit(self._zoom_factor)

    def set_zoom(self, factor: float):
        """Set absolute zoom factor.

        Parameters
        ----------
        factor : float
            Zoom factor (1.0 = fit in view, >1 = zoom in, <1 = zoom out).
        """
        if factor <= 0:
            return

        self.resetTransform()
        self.scale(factor, factor)
        self._zoom_factor = factor
        self.zoomChanged.emit(factor)

    def zoom_in(self, factor: float = 1.25):
        """Zoom in by a factor.

        Parameters
        ----------
        factor : float, optional
            Multiplication factor, by default 1.25.
        """
        self.set_zoom(self._zoom_factor * factor)

    def zoom_out(self, factor: float = 1.25):
        """Zoom out by a factor.

        Parameters
        ----------
        factor : float, optional
            Division factor, by default 1.25.
        """
        self.set_zoom(self._zoom_factor / factor)

    def wheelEvent(self, event):
        """Disable mouse wheel zooming."""
        event.ignore()

    def _array_to_qimage(array: np.ndarray) -> Optional[QImage]:
        """Convert numpy array to QImage with proper handling of edge cases.

        Parameters
        ----------
        array : np.ndarray
            Input array, 2D grayscale or 3D RGB.

        Returns
        -------
        Optional[QImage]
            QImage object or None if conversion fails.
        """
        try:
            if not array.flags['C_CONTIGUOUS']:
                array = np.ascontiguousarray(array)

            height, width = array.shape[:2]
            LOG.debug(
                f"Converting to QImage - shape: {array.shape}, dtype: {array.dtype}")

            if array.ndim == 3 and array.shape[2] == 3:
                LOG.debug("Processing as RGB image")

                if array.dtype == np.float32:
                    LOG.debug(
                        f"Float32 RGB - actual range: [{array.min():.6f}, {array.max():.6f}]")

                    if array.max() < 0.01:
                        LOG.debug(
                            "Very dark image detected - using special handling")
                        scaled_array = array * (1.0 / max(array.max(), 0.001))
                        array_uint8 = (
                                    np.clip(scaled_array, 0, 1) * 255).astype(
                            np.uint8)
                    else:
                        array_uint8 = (np.clip(array, 0, 1) * 255).astype(
                            np.uint8)

                elif array.dtype == np.uint8:
                    array_uint8 = array
                else:
                    array_normalized = (array - array.min()) / (
                                array.max() - array.min())
                    array_uint8 = (array_normalized * 255).astype(np.uint8)

                bytes_per_line = 3 * width

                unique_colors = np.unique(array_uint8.reshape(-1, 3), axis=0)
                LOG.debug(f"Final uint8 colors - count: {len(unique_colors)}")
                if len(unique_colors) <= 5:
                    LOG.debug(f"All colors: {unique_colors}")

                qimage = QImage(
                    array_uint8.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format.Format_RGB888
                )
                return qimage.copy()

            elif array.ndim == 2:
                LOG.debug("Processing as grayscale image")
                return GraphicsImageView._grayscale_to_qimage(array)

            else:
                LOG.debug(f"Unsupported array shape: {array.shape}")
                return None

        except Exception as e:
            LOG.debug(f"Error in _array_to_qimage: {e}")
            return None

    @staticmethod
    def _grayscale_to_qimage(array: np.ndarray) -> QImage:
        """Convert grayscale array to QImage with proper handling.

        Parameters
        ----------
        array : np.ndarray
            2D grayscale image.

        Returns
        -------
        QImage
            Resulting QImage.
        """
        height, width = array.shape

        if array.dtype != np.uint8:
            if array.min() == array.max():
                LOG.debug("Uniform grayscale image - creating solid color")
                if array.min() == 0:
                    array_uint8 = np.zeros((height, width), dtype=np.uint8)
                else:
                    value = int(np.clip(array.min(), 0, 1) * 255)
                    array_uint8 = np.full((height, width), value,
                                          dtype=np.uint8)
            else:
                array_normalized = (array - array.min()) / (
                            array.max() - array.min())
                array_uint8 = (array_normalized * 255).astype(np.uint8)
        else:
            array_uint8 = array

        bytes_per_line = width
        qimage = QImage(
            array_uint8.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_Grayscale8
        )
        return qimage.copy()

    @staticmethod
    def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
        """Normalize array values to uint8 range [0, 255].

        Parameters
        ----------
        array : np.ndarray
            Input array of any numeric type.

        Returns
        -------
        np.ndarray
            Array normalized to uint8.
        """
        arr_min, arr_max = array.min(), array.max()

        if arr_max <= arr_min:
            return np.zeros_like(array, dtype=np.uint8)

        normalized = (array - arr_min) / (arr_max - arr_min)
        return (normalized * 255).astype(np.uint8)


def _is_binary_image(array: np.ndarray) -> bool:
    """Check if array represents a binary image (only 0 and 1 values).

    Parameters
    ----------
    array : np.ndarray
        Input array.

    Returns
    -------
    bool
        True if binary, False otherwise.
    """
    unique_vals = np.unique(array)
    return len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1]))


def _is_binary_like(array: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Check if array is essentially binary (values very close to 0 and 1).

    Parameters
    ----------
    array : np.ndarray
        Input array.
    tolerance : float, optional
        Tolerance for closeness to 0 or 1, by default 1e-6.

    Returns
    -------
    bool
        True if binary-like, False otherwise.
    """
    unique_vals = np.unique(array)
    if len(unique_vals) > 2:
        return False

    for val in unique_vals:
        if not (abs(val) < tolerance or abs(val - 1) < tolerance):
            return False
    return True
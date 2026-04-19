from typing import Optional

import numpy as np
from PyQt6.QtCore import pyqtSignal, Qt, QRectF
from PyQt6.QtGui import QPainter, QColor, QPixmap, QImage
from PyQt6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QWidget, QFrame)

from pycore.log.ctx import with_logger


@with_logger
class GraphicsImageView(QGraphicsView):
    """QGraphicsView for displaying images."""

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
        """Set image from numpy array with optional magnification."""
        try:
            print(
                f"SETTING CROP IMAGE - min: {np.min(array):.6f}, max: {np.max(array):.6f}, shape: {array.shape}, dtype: {array.dtype}")

            # Debug: Check if this is already a colormapped image
            if array.ndim == 3 and array.shape[2] == 3:
                print("Detected RGB image (likely colormapped)")
                # Check if it's already in [0,1] range (typical for colormapped images)
                if array.dtype == np.float32 and array.min() >= 0 and array.max() <= 1.0:
                    print(
                        "Image appears to be normalized float32 RGB - converting directly to uint8")

            print(array)
            qimage = GraphicsImageView._array_to_qimage(array)
            if qimage is None:
                return False

            # Apply magnification
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
        """Set the pixmap to display and resize view to match exactly."""
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
        """Convert numpy array to QImage efficiently."""
        try:
            # Ensure C-contiguous for direct memory access
            if not array.flags['C_CONTIGUOUS']:
                array = np.ascontiguousarray(array)

            height, width = array.shape[:2]

            # Handle RGB images (likely already colormapped)
            if array.ndim == 3 and array.shape[2] == 3:
                print("Processing RGB image in crop widget")

                # If it's already float32 in [0,1] range, convert directly to uint8
                if array.dtype == np.float32:
                    # Check if values are in [0,1] range
                    if array.min() >= 0 and array.max() <= 1.0:
                        print("Converting float32 [0,1] RGB to uint8 [0,255]")
                        array_uint8 = (array * 255).astype(np.uint8)
                    else:
                        # Normalize to [0,1] first, then to uint8
                        print("Normalizing float32 RGB to uint8")
                        array_normalized = (array - array.min()) / (
                                    array.max() - array.min())
                        array_uint8 = (array_normalized * 255).astype(np.uint8)
                else:
                    # For non-float types, use standard normalization
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

                # Debug: Check the final image
                unique_colors = np.unique(array_uint8.reshape(-1, 3), axis=0)
                print(f"Final RGB unique colors count: {len(unique_colors)}")
                if len(unique_colors) <= 10:
                    print(f"Final RGB colors: {unique_colors}")

            # Grayscale image
            elif array.ndim == 2:
                print("Processing grayscale image in crop widget")

                # Use fixed normalization for binary images to avoid precision issues
                if _is_binary_image(array):
                    print("Detected binary image, using fixed normalization")
                    array_uint8 = GraphicsImageView._binary_to_uint8(
                        array)
                else:
                    # Convert to uint8 if needed with standard normalization
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
                print(f"Unsupported array dimensions: {array.ndim}")
                return None

            print(
                f"Created QImage: {qimage.width()}x{qimage.height()}, format: {qimage.format()}")
            return qimage.copy()

        except Exception as e:
            print(f"Error converting array to QImage: {e}")
            return None

    @staticmethod
    def _binary_to_uint8(array: np.ndarray) -> np.ndarray:
        """Convert binary (0/1) image to uint8 without interpolation artifacts."""
        result = np.zeros_like(array, dtype=np.uint8)

        # Find all values that are clearly not zero
        threshold = 0.1
        mask = array > threshold
        result[mask] = 255

        print(
            f"Binary conversion: {np.sum(mask)} white pixels, {np.sum(~mask)} black pixels")
        return result

    @staticmethod
    def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
        """Normalize array values to uint8 range [0, 255]."""
        # For RGB images, we handle them separately in _array_to_qimage
        if array.ndim == 3:
            # This shouldn't be called for RGB images anymore, but handle gracefully
            print("Warning: _normalize_to_uint8 called for RGB array")
            if array.dtype == np.float32 and array.min() >= 0 and array.max() <= 1.0:
                return (array * 255).astype(np.uint8)
            else:
                array_normalized = (array - array.min()) / (
                            array.max() - array.min())
                return (array_normalized * 255).astype(np.uint8)

        # For grayscale images
        arr_min, arr_max = array.min(), array.max()
        print(f"Normalizing grayscale - min: {arr_min:.6f}, max: {arr_max:.6f}")

        if arr_max <= arr_min:
            print("Constant image, returning zeros")
            return np.zeros_like(array, dtype=np.uint8)

        # Check if this is essentially a binary image
        if _is_binary_image(array) or _is_binary_like(array):
            print("Image is binary-like, using binary conversion")
            return GraphicsImageView._binary_to_uint8(array)

        # Standard normalization for continuous images
        normalized = (array - arr_min) / (arr_max - arr_min)
        result = (normalized * 255).astype(np.uint8)

        unique_result = np.unique(result)
        print(f"Normalized to uint8 - unique values: {len(unique_result)}")
        return result


    def clear_image(self):
        """Clear the displayed image and reset to minimum size."""
        if self._pixmap_item is not None:
            self._scene.removeItem(self._pixmap_item)
            self._pixmap_item = None
        self._scene.clear()

        # Reset to minimum size when cleared
        self.setMinimumSize(200, 200)
        self.setMaximumSize(16777215, 16777215)  # QWIDGETSIZE_MAX

    def fit_in_view(self):
        """Fit the image in the view while maintaining aspect ratio."""
        if self._pixmap_item is None:
            return

            # Reset transform first
        self.resetTransform()

        # Get dimensions
        scene_rect = self._scene.sceneRect()
        viewport_rect = self.viewport().rect()

        # Calculate scale factors
        x_scale = viewport_rect.width() / scene_rect.width()
        y_scale = viewport_rect.height() / scene_rect.height()

        # Use the smaller scale to maintain aspect ratio
        scale_factor = min(x_scale, y_scale)

        self.scale(scale_factor, scale_factor)
        self.centerOn(scene_rect.center())

        self._zoom_factor = 1.0
        self.zoomChanged.emit(self._zoom_factor)

    def set_zoom(self, factor: float):
        """Set absolute zoom factor.

        Args:
            factor: Zoom factor (1.0 = fit in view, >1 = zoom in, <1 = zoom out)
        """
        if factor <= 0:
            return

        # Reset transform and apply new zoom
        self.resetTransform()
        self.scale(factor, factor)
        self._zoom_factor = factor
        self.zoomChanged.emit(factor)

    def zoom_in(self, factor: float = 1.25):
        """Zoom in by a factor."""
        self.set_zoom(self._zoom_factor * factor)

    def zoom_out(self, factor: float = 1.25):
        """Zoom out by a factor."""
        self.set_zoom(self._zoom_factor / factor)

    def wheelEvent(self, event):
        """Disable mouse wheel zooming."""
        event.ignore()

    def _array_to_qimage(array: np.ndarray) -> Optional[QImage]:
        """Convert numpy array to QImage with proper handling of edge cases."""
        try:
            if not array.flags['C_CONTIGUOUS']:
                array = np.ascontiguousarray(array)

            height, width = array.shape[:2]
            print(
                f"Converting to QImage - shape: {array.shape}, dtype: {array.dtype}")

            # Handle RGB images
            if array.ndim == 3 and array.shape[2] == 3:
                print("Processing as RGB image")

                # For float32 images in [0,1] range
                if array.dtype == np.float32:
                    print(
                        f"Float32 RGB - actual range: [{array.min():.6f}, {array.max():.6f}]")

                    # Check if we have very low values that might cause issues
                    if array.max() < 0.01:  # Very dark image
                        print(
                            "Very dark image detected - using special handling")
                        # Scale up low values to make them visible
                        scaled_array = array * (1.0 / max(array.max(), 0.001))
                        array_uint8 = (
                                    np.clip(scaled_array, 0, 1) * 255).astype(
                            np.uint8)
                    else:
                        # Normal conversion
                        array_uint8 = (np.clip(array, 0, 1) * 255).astype(
                            np.uint8)

                elif array.dtype == np.uint8:
                    array_uint8 = array
                else:
                    # Convert other types
                    array_normalized = (array - array.min()) / (
                                array.max() - array.min())
                    array_uint8 = (array_normalized * 255).astype(np.uint8)

                bytes_per_line = 3 * width

                # Debug the final uint8 array
                unique_colors = np.unique(array_uint8.reshape(-1, 3), axis=0)
                print(f"Final uint8 colors - count: {len(unique_colors)}")
                if len(unique_colors) <= 5:
                    print(f"All colors: {unique_colors}")

                qimage = QImage(
                    array_uint8.data,
                    width,
                    height,
                    bytes_per_line,
                    QImage.Format.Format_RGB888
                )
                return qimage.copy()

            # Handle grayscale images
            elif array.ndim == 2:
                print("Processing as grayscale image")
                return GraphicsImageView._grayscale_to_qimage(array)

            else:
                print(f"Unsupported array shape: {array.shape}")
                return None

        except Exception as e:
            print(f"Error in _array_to_qimage: {e}")
            return None

    @staticmethod
    def _grayscale_to_qimage(array: np.ndarray) -> QImage:
        """Convert grayscale array to QImage with proper handling."""
        height, width = array.shape

        # Convert to uint8
        if array.dtype != np.uint8:
            if array.min() == array.max():
                print("Uniform grayscale image - creating solid color")
                # Create a solid image
                if array.min() == 0:
                    array_uint8 = np.zeros((height, width), dtype=np.uint8)
                else:
                    value = int(np.clip(array.min(), 0, 1) * 255)
                    array_uint8 = np.full((height, width), value,
                                          dtype=np.uint8)
            else:
                # Normal normalization
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

        Args:
            array: Input array of any numeric type

        Returns:
            Array normalized to uint8
        """
        arr_min, arr_max = array.min(), array.max()

        if arr_max <= arr_min:
            return np.zeros_like(array, dtype=np.uint8)

        # Normalize to [0, 1] then scale to [0, 255]
        normalized = (array - arr_min) / (arr_max - arr_min)
        return (normalized * 255).astype(np.uint8)


def _is_binary_image(array: np.ndarray) -> bool:
    """Check if array represents a binary image (only 0 and 1 values)."""
    unique_vals = np.unique(array)
    return len(unique_vals) <= 2 and np.all(np.isin(unique_vals, [0, 1]))


def _is_binary_like(array: np.ndarray, tolerance: float = 1e-6) -> bool:
    """Check if array is essentially binary (values very close to 0 and 1)."""
    unique_vals = np.unique(array)
    if len(unique_vals) > 2:
        return False

    for val in unique_vals:
        if not (abs(val) < tolerance or abs(val - 1) < tolerance):
            return False
    return True
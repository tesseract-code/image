import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal, pyqtSlot


class ROIManager(QtCore.QObject):
    """Manages ROI (Region of Interest) operations and crop extraction."""

    # Signals
    roiClicked = pyqtSignal(object)
    roiChanged = pyqtSignal(object)
    roiCropChanged = pyqtSignal(np.ndarray, object)  # numpy array, ROI bounds

    def __init__(self, overlay_view, image_data_callback, image_dims_callback):
        super().__init__(overlay_view)
        self.overlay_view = overlay_view
        self._get_image_data = image_data_callback
        self._get_image_dims = image_dims_callback

        self._setup_connections()

    def _setup_connections(self):
        """Setup signal forwarding and monitoring."""
        # Forward signals from overlay view
        self.overlay_view.roiClicked.connect(self._on_roi_clicked)
        self.overlay_view.roiChanged.connect(self._on_roi_changed)

    @pyqtSlot(object)
    def _on_roi_clicked(self, roi):
        """Handle ROI selection."""
        self.roiClicked.emit(roi)
        self._emit_crop_for_roi(roi)

    @pyqtSlot(object)
    def _on_roi_changed(self, roi):
        """Handle ROI geometry changes (move/resize)."""
        self.roiChanged.emit(roi)

        # Emit crop update if this is the active ROI
        if roi == self.overlay_view.active_roi:
            self._emit_crop_for_roi(roi)

    def _emit_crop_for_roi(self, roi):
        """Extract and emit crop for specified ROI."""
        if roi is None:
            return

        crop_array, roi_bounds = self.get_roi_crop(roi)
        if crop_array is not None and roi_bounds is not None:
            self.roiCropChanged.emit(crop_array, roi_bounds)

    def get_active_roi_crop(self):
        """
        Extract the image area under the active ROI as a numpy array.

        Returns:
            tuple: (numpy_array, roi_bounds_dict) or (None, None) if no active ROI
                   roi_bounds_dict contains {'x', 'y', 'width', 'height'} in image coords
        """
        active_roi = self.overlay_view.active_roi
        if not active_roi:
            return None, None

        return self.get_roi_crop(active_roi)

    def get_roi_crop(self, roi):
        """
        Extract the image area under a specific ROI.

        Args:
            roi: ROI graphics item

        Returns:
            tuple: (numpy_array, roi_bounds_dict) or (None, None)
        """
        if not roi:
            return None, None

        # Get ROI bounds in image coordinates
        roi_bounds = self._get_roi_image_bounds(roi)
        if not roi_bounds:
            return None, None

        # Extract image data
        image_data = self._get_image_data()
        if image_data is None:
            return None, None

        crop_array = self._extract_image_region(
            image_data,
            roi_bounds['x'],
            roi_bounds['y'],
            roi_bounds['width'],
            roi_bounds['height']
        )


        if crop_array is None:
            return None, None

        return crop_array, roi_bounds

    def _get_roi_image_bounds(self, roi):
        """
        Get the bounding rectangle of an ROI in image pixel coordinates.

        Args:
            roi: ROI graphics item

        Returns:
            dict: {'x', 'y', 'width', 'height'} or None
        """
        if not roi:
            return None

        try:
            # Get ROI's bounding rect in local coordinates
            local_bounds = roi.boundingRect()

            # Map to scene coordinates (which equal image coordinates)
            scene_bounds = roi.mapRectToScene(local_bounds)

            # Clamp to image dimensions
            image_width, image_height = self._get_image_dims()

            x = max(0, int(scene_bounds.x()))
            y = max(0, int(scene_bounds.y()))
            width = int(scene_bounds.width())
            height = int(scene_bounds.height())

            # Ensure bounds are within image
            if image_width > 0:
                width = min(width, image_width - x)
            if image_height > 0:
                height = min(height, image_height - y)

            if width <= 0 or height <= 0:
                return None

            return {
                'x': x,
                'y': y,
                'width': width,
                'height': height
            }

        except Exception as e:
            print(f"Error getting ROI bounds: {e}")
            return None

    def _extract_image_region(self, image_data, x, y, width, height):
        """
        Extract a rectangular region from image data.

        Args:
            image_data: numpy array
            x, y: top-left corner
            width, height: region dimensions

        Returns:
            numpy array or None
        """
        try:
            # Handle different image dimensions
            if len(image_data.shape) == 2:
                # Grayscale
                return image_data[y:y + height, x:x + width].copy()
            elif len(image_data.shape) == 3:
                # RGB/RGBA
                return image_data[y:y + height, x:x + width, :].copy()
            else:
                return None

        except (IndexError, ValueError) as e:
            print(f"Error extracting image region: {e}")
            return None

    # ========== ROI Creation API (delegate to overlay_view) ==========

    def add_ellipse_roi(self, rect=None, pos=None):
        """Add an ellipse ROI to the view."""
        return self.overlay_view.addEllipseROI(rect, pos)

    def add_rect_roi(self, rect=None, pos=None):
        """Add a rectangular ROI to the view."""
        return self.overlay_view.addRectROI(rect, pos)

    def add_line_roi(self, line=None, pos=None):
        """Add a line ROI to the view."""
        return self.overlay_view.addLineROI(line, pos)

    def add_polygon_roi(self, polygon=None, pos=None):
        """Add a polygon ROI to the view."""
        return self.overlay_view.addPolygonROI(polygon, pos)

    def remove_roi(self, roi):
        """Remove a specific ROI from the view."""
        self.overlay_view.removeROI(roi)

    def clear_rois(self):
        """Remove all ROIs from the view."""
        self.overlay_view.clearROIs()

    def get_rois(self):
        """Get list of all ROIs in the view."""
        return self.overlay_view.getROIs()

    def set_active_roi(self, roi):
        """Set the active ROI."""
        self.overlay_view.setActiveROI(roi)
        if roi:
            self._emit_crop_for_roi(roi)

    def get_active_roi(self):
        """Get the currently active ROI."""
        return self.overlay_view.active_roi

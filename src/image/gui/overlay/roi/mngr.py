import logging
from typing import Optional, Protocol, Tuple, runtime_checkable, Callable

import numpy as np
from PyQt6 import QtCore
from PyQt6.QtCore import pyqtSignal, pyqtSlot, QTimer

logger = logging.getLogger(__name__)


@runtime_checkable
class ROIOverlayView(Protocol):
    active_roi: object

    roiClicked: pyqtSignal
    roiChanged: pyqtSignal

    def addEllipseROI(self, rect=None, pos=None): ...

    def addRectROI(self, rect=None, pos=None): ...

    def addLineROI(self, line=None, pos=None): ...

    def addPolygonROI(self, polygon=None, pos=None): ...

    def removeROI(self, roi) -> None: ...

    def clearROIs(self) -> None: ...

    def getROIs(self) -> list: ...

    def setActiveROI(self, roi) -> None: ...


class ROIManager(QtCore.QObject):
    """Manages ROI (Region of Interest) operations and crop extraction.

    Args:
        overlay_view: A view object satisfying the ROIOverlayView protocol.
            Must expose roiClicked/roiChanged signals and an active_roi
            attribute, plus the standard ROI mutation methods.
        image_data_callback: Callable returning the current image as a numpy
            array with shape (height, width[, channels]), pre-flipped so that
            row 0 is the top of the image.  Returns None when no image is
            loaded.
        image_dims_callback: Callable returning (width, height) as ints, or
            None when no image is loaded.
    """

    # Signals
    roiClicked = pyqtSignal(object)
    roiChanged = pyqtSignal(object)
    roiCropChanged = pyqtSignal(np.ndarray, object)  # numpy array, ROI bounds
    roiCropFailed = pyqtSignal(object)  # FIX 6: emit on failure

    def __init__(
            self,
            overlay_view: ROIOverlayView,
            image_data_callback: Callable,
            image_dims_callback: Callable,
            parent=None
    ):
        super().__init__(parent)
        self.overlay_view = overlay_view
        self._get_image_data = image_data_callback
        self._get_image_dims = image_dims_callback

        self._setup_connections()

    def _setup_connections(self):
        """Setup signal forwarding and monitoring."""
        self.overlay_view.roiClicked.connect(self._on_roi_clicked)
        self.overlay_view.roiChanged.connect(self._on_roi_changed)

    @pyqtSlot(object)
    def _on_roi_clicked(self, roi):
        """Handle ROI selection."""
        self.roiClicked.emit(roi)
        # FIX 8: Skip redundant crop emission when roi is None (deselect).
        if roi is not None:
            self._emit_crop_for_roi(roi)

    @pyqtSlot(object)
    def _on_roi_changed(self, roi):
        """Handle ROI geometry changes (move/resize)."""
        self.roiChanged.emit(roi)

        # FIX 4: Use `is` for identity comparison on QGraphicsItem instances.
        # `==` may invoke an unexpected __eq__ or always return False.
        if roi is self.overlay_view.active_roi:
            self._emit_crop_for_roi(roi)

    def _emit_crop_for_roi(self, roi):
        """Extract and emit crop for the specified ROI.

        Emits roiCropChanged on success and roiCropFailed on failure so that
        callers are never left silently waiting for a crop that will not come.
        """
        if roi is None:
            return

        crop_array, roi_bounds = self.get_roi_crop(roi)
        if crop_array is not None and roi_bounds is not None:
            self.roiCropChanged.emit(crop_array, roi_bounds)
        else:
            # FIX 6: Emit failure signal and log so problems are diagnosable.
            logger.debug("Crop extraction failed for ROI %r", roi)
            self.roiCropFailed.emit(roi)

    def get_active_roi_crop(self) -> Tuple[
        Optional[np.ndarray], Optional[dict]]:
        """
        Extract the image area under the active ROI as a numpy array.

        Returns:
            tuple: (numpy_array, roi_bounds_dict) or (None, None) if no active
                   ROI or no image is loaded.
                   roi_bounds_dict contains {'x', 'y', 'width', 'height'} in
                   image pixel coordinates (row 0 = top of image).
        """
        active_roi = self.overlay_view.active_roi
        if not active_roi:
            return None, None

        return self.get_roi_crop(active_roi)

    def get_roi_crop(self, roi) -> Tuple[Optional[np.ndarray], Optional[dict]]:
        """
        Extract the image area under a specific ROI.

        Args:
            roi: ROI graphics item.

        Returns:
            tuple: (numpy_array, roi_bounds_dict) or (None, None).
        """
        if not roi:
            return None, None

        roi_bounds = self._get_roi_image_bounds(roi)
        if not roi_bounds:
            return None, None

        image_data = self._get_image_data()
        if image_data is None:
            return None, None

        crop_array = self._extract_image_region(
            image_data,
            roi_bounds['x'],
            roi_bounds['y'],
            roi_bounds['width'],
            roi_bounds['height'],
        )

        if crop_array is None:
            return None, None

        return crop_array, roi_bounds

    def _get_roi_image_bounds(self, roi) -> Optional[dict]:
        """
        Get the bounding rectangle of an ROI in image pixel coordinates.

        Scene coordinates are assumed to match image pixel coordinates, with
        row 0 at the top of the image (the image data callback pre-flips GL
        textures so this invariant holds).

        Args:
            roi: ROI graphics item.

        Returns:
            dict with keys 'x' (column), 'y' (row), 'width', 'height', or
            None if the bounds cannot be determined or fall outside the image.
        """
        if not roi:
            return None

        # FIX 1: Guard against _get_image_dims() returning None before
        # unpacking, preventing a TypeError when no image is loaded.
        dims = self._get_image_dims()
        if dims is None:
            return None
        image_width, image_height = dims

        try:
            local_bounds = roi.boundingRect()
            scene_bounds = roi.mapRectToScene(local_bounds)

            # FIX 3: Clamp x and y to valid column/row indices on both ends
            # before computing clamped width/height.  Without the upper clamp
            # an out-of-bounds origin produced a negative extent that was only
            # caught accidentally by the width <= 0 check below.
            x = min(max(0, int(scene_bounds.x())), image_width - 1)
            y = min(max(0, int(scene_bounds.y())), image_height - 1)
            width = min(int(scene_bounds.width()), image_width - x)
            height = min(int(scene_bounds.height()), image_height - y)

            if width <= 0 or height <= 0:
                return None

            return {'x': x, 'y': y, 'width': width, 'height': height}

        # FIX 10: Narrow the exception to types that can legitimately arise
        # from Qt geometry calls and int conversion; let programming errors
        # (e.g. wrong callback signature) propagate so they are not hidden.
        except (TypeError, ValueError):
            logger.exception("Error getting ROI bounds for %r", roi)
            return None

    def _extract_image_region(
            self,
            image_data: np.ndarray,
            x: int,
            y: int,
            width: int,
            height: int,
    ) -> Optional[np.ndarray]:
        """
        Extract a rectangular region from image data.

        Args:
            image_data: numpy array with shape (height, width[, channels]).
                        Row 0 is the top of the image.
            x: Left column of the region (axis 1).
            y: Top row of the region (axis 0).
            width: Number of columns to extract.
            height: Number of rows to extract.

        Returns:
            numpy array copy of the region, or None on error.

        Note on axis order:
            image_data is indexed as [row, col], so `y` selects rows (axis 0)
            and `x` selects columns (axis 1).  This matches a standard
            (height, width[, channels]) layout.
        """
        # FIX 2: Clarified that y→rows (axis 0), x→cols (axis 1) in the
        # docstring above so the axis semantics are unambiguous for callers
        # working with pre-flipped GL texture data.
        try:
            if image_data.ndim == 2:
                # Grayscale: (height, width)
                return image_data[y:y + height, x:x + width].copy()
            elif image_data.ndim == 3:
                # RGB / RGBA: (height, width, channels)
                return image_data[y:y + height, x:x + width, :].copy()
            else:
                logger.error(
                    "Unsupported image ndim %d; expected 2 or 3",
                    image_data.ndim,
                )
                return None

        # FIX 10: Narrow catch to errors that numpy slicing can raise.
        except (IndexError, ValueError):
            logger.exception(
                "Error extracting image region x=%d y=%d w=%d h=%d",
                x, y, width, height,
            )
            return None

    # ========== ROI Creation API (delegate to overlay_view) ==========
    # FIX 12: These are intentionally thin delegation wrappers. They exist to
    # give OverlayStack a stable, manager-owned surface for future logic (e.g.
    # tracking, validation) without breaking callers. If no logic is ever
    # added, consider collapsing them and having OverlayStack call overlay_view
    # directly.

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

    def remove_roi(self, roi) -> None:
        """Remove a specific ROI from the view."""
        self.overlay_view.removeROI(roi)

    def clear_rois(self) -> None:
        """Remove all ROIs from the view."""
        self.overlay_view.clearROIs()

    def get_rois(self) -> list:
        """Get list of all ROIs in the view."""
        return self.overlay_view.getROIs()

    def set_active_roi(self, roi) -> None:
        """Set the active ROI and emit its crop.

        The crop emission is deferred by one event-loop iteration to ensure
        the view has finished updating active_roi before we read it back.
        """
        self.overlay_view.setActiveROI(roi)
        # FIX 7: Defer crop emission so the view's internal state (active_roi)
        # is guaranteed to reflect the new ROI before _emit_crop_for_roi runs.
        if roi is not None:
            QTimer.singleShot(0, lambda: self._emit_crop_for_roi(roi))

    def get_active_roi(self):
        """Get the currently active ROI."""
        return self.overlay_view.active_roi

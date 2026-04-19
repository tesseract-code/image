import logging
import timeit
from typing import Optional, Tuple

import numpy as np
from PyQt6 import QtCore, QtWidgets, QtGui
from PyQt6.QtCore import pyqtSignal, pyqtSlot, Qt, QTimer, QRectF, QSize
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QGraphicsScene, QFrame

from cross_platform.dev.utils.create_image import create_rgb_checkered
from image.gl.utils import get_surface_format
from image.gl.view import (GLFrameViewer)
from image.gui.overlay.view import (
    SynchableGraphicsView)
from image.pipeline.stats import get_frame_stats
from image.settings.base import ImageSettings
from image.settings.pixels import PixelFormat
from image.gui.overlay.crosshair.mngr import CrosshairManager
from image.gui.overlay.roi.mngr import ROIManager
from image.gui.overlay.sync import (TransformSynchronizer)
from image.gui.overlay.tooltip import TooltipManager
from pycore.log.ctx import with_logger

logger = logging.getLogger(__name__)


@with_logger
class OverlayStack(QFrame):
    """
    High-performance layered image viewer combining OpenGL rendering with
    transparent graphics overlay for crosshairs and ROIs.

    Architecture:
    - Scene coordinates: Always match viewport pixel dimensions
    - Image coordinates: Actual image pixel dimensions
    - Transform: Maps from scene to visible image area
    """

    # Forward signals from managers
    image_changed = pyqtSignal(object)
    transform_changed = pyqtSignal()
    roiClicked = pyqtSignal(object)
    roiChanged = pyqtSignal(object)
    roiCropChanged = pyqtSignal(np.ndarray, object)

    mouse_pos_changed = pyqtSignal(int, int)

    _req_sync_view = pyqtSignal()

    def __init__(self, settings, parent=None):
        super().__init__(parent)
        self.settings = settings

        # Window state tracking
        self._last_window_state: QtCore.Qt.WindowState | None = None
        self._global_pos = None
        self._scene_pos = None
        self._pending_resize = False

        self._setup_ui()

    def _setup_transparent_overlay(self):
        """Configure the overlay view to be transparent and optimized"""
        # Create scene
        scene = QGraphicsScene()
        scene.setSceneRect(0, 0, self.width(), self.height())

        # Top layer: Transparent graphics view
        self.overlay = SynchableGraphicsView(scene=scene, parent=self)

        self.overlay.setup_view_for_images()
        self.overlay.enableScrollBars(False)

        self.overlay.setAutoFillBackground(False)
        self.overlay.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground)
        self.overlay.setFrameShape(QtWidgets.QFrame.Shape.NoFrame)
        self.overlay.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.overlay.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.overlay.setMouseTracking(True)
        self.overlay.setViewportUpdateMode(
            QtWidgets.QGraphicsView.ViewportUpdateMode.SmartViewportUpdate)

        self.overlay.setRenderHint(
            QtGui.QPainter.RenderHint.Antialiasing, False)
        self.overlay.setOptimizationFlag(
            QtWidgets.QGraphicsView.OptimizationFlag.DontSavePainterState,
            True
        )
        self.overlay.setOptimizationFlag(
            QtWidgets.QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing,
            True)

    def _setup_managers(self):
        """Initialize all manager objects."""
        # Tooltip manager
        self.tooltip_manager = TooltipManager(self.overlay)

        # Crosshair manager
        self.crosshair_manager = CrosshairManager(self.overlay)
        self.crosshair_manager.fix_scene_reference(self.overlay.scene())

        # ROI manager
        self.roi_manager = ROIManager(
            self.overlay,
            lambda: np.flip(self._get_img_data(), 0),
            self._get_img_dims
        )
        # Forward ROI signals
        self.roi_manager.roiClicked.connect(self.roiClicked)
        self.roi_manager.roiChanged.connect(self.roiChanged)
        self.roi_manager.roiCropChanged.connect(self.roiCropChanged)

        # Transform synchronizer
        self.sync_manager = TransformSynchronizer(
            self.gl_viewer,
            self.overlay,
            self.settings
        )
        self.sync_manager.transformSynced.connect(self.transform_changed)

        # Setup GL viewer connections
        self.gl_viewer.glError.connect(self._on_gl_error)

        # Install event filter
        self.overlay.viewport().installEventFilter(self)

    def _setup_ui(self):
        """Create the layered widget structure"""
        self.gl_viewer = GLFrameViewer(self.settings)
        self.gl_viewer.setParent(self)
        self.gl_viewer.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)

        self._setup_transparent_overlay()
        self._position_layers()
        self._setup_managers()
        self._initialize_overlays()
        self._req_sync_view.connect(self._sync_view)

    def _position_layers(self):
        """Position both layers to fill the widget"""
        geometry = self.rect()
        self.gl_viewer.setGeometry(geometry)
        self.overlay.setGeometry(geometry)
        self.overlay.raise_()

    def _get_img_data(self) -> Optional[np.ndarray]:
        if self.gl_viewer.has_data:
            return self.gl_viewer.data

    def _get_img_dims(self) -> Optional[Tuple[int, int]]:
        if self.gl_viewer.has_data:
            width, height = self.gl_viewer.data.shape[:2]
            return width, height

    def _initialize_overlays(self):
        """Initialize overlays after the view is fully set up"""
        if self.overlay.scene():
            self._update_scene_rect_to_viewport(force=True)
            self.overlay.viewport().update()
            self.overlay.raise_()

    def _update_scene_rect_to_viewport(self, force=False):
        """Update the scene rect to match the current viewport size."""
        if not self.overlay.scene():
            return

        viewport_rect = self.overlay.viewport().rect()
        scene_rect = self.overlay.sceneRect()

        if (force or
                abs(viewport_rect.width() - scene_rect.width()) > 1 or
                abs(viewport_rect.height() - scene_rect.height()) > 1):
            self.overlay.scene().setSceneRect(
                0, 0, viewport_rect.width(), viewport_rect.height())
            self.crosshair_manager.update_dimensions(
                viewport_rect.width(), viewport_rect.height())

    @pyqtSlot()
    def _sync_view(self):
        self.sync_manager.with_sync_disabled(
            lambda: self._reset_view_on_load())
        self.crosshair_manager.update_dimensions(self._image_width,
                                                 self._image_height)

    @pyqtSlot(object)
    def _on_gl_image_updated(self, metadata):
        """Handle image updates from GL viewer"""
        if (metadata and hasattr(metadata, 'width') and
                hasattr(metadata, 'height')):
            self._image_width = metadata.width
            self._image_height = metadata.height
            # Reset view
            self._sync_view()

        self.image_changed.emit(metadata)

    def _reset_view_on_load(self):
        """Reset view when new image loads."""
        self.settings.update_setting('zoom', 1.0)
        self.settings.update_setting('pan_x', 0.0)
        self.settings.update_setting('pan_y', 0.0)
        self.overlay.resetTransform()
        self.overlay.centerOn(self._image_width / 2,
                              self._image_height / 2)

    @pyqtSlot()
    def _on_gl_error(self, error_msg):
        """Handle GL errors"""
        print(f"GL Error: {error_msg}")

    def changeEvent(self, event):
        """Handle window state changes"""
        if event.type() == QtCore.QEvent.Type.WindowStateChange:
            if self.window():
                new_state = self.window().windowState()
                if new_state != self._last_window_state:
                    self._last_window_state = new_state
                    QTimer.singleShot(100, self._handle_window_state_change)
        super().changeEvent(event)

    def _handle_window_state_change(self):
        """Handle window state changes"""
        self._position_layers()
        self._update_scene_rect_to_viewport(force=True)

    def resizeEvent(self, event):
        """Handle resize"""
        super().resizeEvent(event)

        print("OVERLAY RESIZE event")

        self._position_layers()

        if self._pending_resize:
            return

        self._pending_resize = True
        self._deferred_resize_handling()

    def _deferred_resize_handling(self):
        """Handle resize after delay"""
        self._pending_resize = False
        self._update_scene_rect_to_viewport(force=True)

        if self.sync_manager.is_enabled() and self.gl_viewer.has_data:
            self.sync_manager.sync_gl_to_overlay()

        self.overlay.viewport().update()

    def eventFilter(self, obj, event):
        """Forward wheel events and handle mouse events"""
        if obj == self.overlay.viewport():
            if event.type() == QtCore.QEvent.Type.Wheel:
                self.gl_viewer.setAttribute(
                    Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
                QtWidgets.QApplication.sendEvent(self.gl_viewer, event)
                self.gl_viewer.setAttribute(
                    Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
                return False
            elif event.type() == QtCore.QEvent.Type.MouseMove:
                overlay_pos = event.pos()
                is_valid_pos = self.overlay.rect().contains(overlay_pos)
                if is_valid_pos and self.gl_viewer.has_data:
                    scene_pos = self.overlay.mapToScene(overlay_pos)
                    global_pos = self.overlay.mapToGlobal(overlay_pos)

                    self._global_pos = global_pos
                    self._scene_pos = scene_pos
                    self.mouse_pos_changed.emit(int(scene_pos.x()),
                                                    int(scene_pos.y()))
                return False

        return super().eventFilter(obj, event)

    def update_tooltip(self,  x:int, y:int, value):
        if x == int(self._scene_pos.x()) and y == int(self._scene_pos.y()):
            self.tooltip_manager.update_tooltip(x, y, value, self._global_pos)

    # ========== Public API ==========

    def update_image(self, numpy_array: np.ndarray):
        """Update the displayed image"""
        self.gl_viewer.present(numpy_array)
        self._image_width = numpy_array.shape[0]
        self._image_height = numpy_array.shape[1]

    def set_sync_enabled(self, enabled: bool):
        """Enable/disable transform synchronization"""
        self.sync_manager.set_enabled(enabled)

    def get_gl_viewer(self):
        """Access the GL viewer directly"""
        return self.gl_viewer

    def get_overlay_view(self):
        """Access the overlay graphics view directly"""
        return self.overlay

    # ========== Crosshair API (delegated to CrosshairManager) ==========

    def enable_tracking_crosshair(self, enable=True):
        """Enable mouse tracking crosshair"""
        self.crosshair_manager.enable_tracking_crosshair(enable)

    def enable_center_crosshair(self, enable=True):
        """Enable fixed center crosshair"""
        self.crosshair_manager.enable_center_crosshair(enable)

    def lock_center_crosshair(self, lock=True):
        """Lock center crosshair to view center"""
        self.crosshair_manager.lock_center_crosshair(lock)

    def set_center_crosshair_length(self, length):
        """Set center crosshair arm length"""
        self.crosshair_manager.set_center_crosshair_length(length)

    def set_crosshair_color(self, color):
        """Set crosshair color"""
        self.crosshair_manager.set_crosshair_color(color)

    # ========== ROI API (delegated to ROIManager) ==========

    def addEllipseROI(self, rect=None, pos=None):
        """Add an ellipse ROI"""
        return self.roi_manager.add_ellipse_roi(rect, pos)

    def addRectROI(self, rect=None, pos=None):
        """Add a rectangular ROI"""
        return self.roi_manager.add_rect_roi(rect, pos)

    def addLineROI(self, line=None, pos=None):
        """Add a line ROI"""
        return self.roi_manager.add_line_roi(line, pos)

    def addPolygonROI(self, polygon=None, pos=None):
        """Add a polygon ROI"""
        return self.roi_manager.add_polygon_roi(polygon, pos)

    def removeROI(self, roi):
        """Remove a specific ROI"""
        self.roi_manager.remove_roi(roi)

    def clearROIs(self):
        """Remove all ROIs"""
        self.roi_manager.clear_rois()

    def getROIs(self):
        """Get list of all ROIs"""
        return self.roi_manager.get_rois()

    def setActiveROI(self, roi):
        """Set the active ROI"""
        self.roi_manager.set_active_roi(roi)

    def get_active_roi_crop(self):
        """
        Extract the image area under the active ROI.

        Returns:
            tuple: (numpy_array, roi_bounds_dict) or (None, None)
        """
        return self.roi_manager.get_active_roi_crop()

    # ========== View Control API ==========

    def fit_to_scene(self):
        """Fit view to show entire image"""
        if self.gl_viewer.has_data:
            width, height = self.gl_viewer.data.shape[:2]
            self.overlay.resetTransform()

            self.overlay.centerOn(width / 2,
                                  height / 2)
            self.sync_manager.sync_overlay_to_gl()

    def reset_view(self):
        """Reset view to default"""
        self.settings.update_setting('zoom', 1.0)
        self.settings.update_setting('pan_x', 0.0)
        self.settings.update_setting('pan_y', 0.0)
        self.settings.update_setting('rotation', 0.0)
        self.overlay.resetTransform()
        if self._image_width > 0:
            self.overlay.centerOn(self._image_width / 2,
                                  self._image_height / 2)

    def cleanup(self):
        """Clean up resources"""
        self.tooltip_manager.cleanup()
        self.sync_manager.cleanup()
        self.gl_viewer.cleanup()

    def verify_overlay_setup(self):
        """Verify overlay setup and return diagnostic info"""
        info = {
            'scene_exists': self.overlay.scene() is not None,
            'crosshair_overlay_exists': hasattr(self.overlay,
                                                'crosshair_overlay'),
            'scene_items_count': 0,
            'image_dimensions': (self._image_width, self._image_height),
            'overlay_geometry': (self.overlay.x(), self.overlay.y(),
                                 self.overlay.width(),
                                 self.overlay.height()),
            'gl_geometry': (self.gl_viewer.x(), self.gl_viewer.y(),
                            self.gl_viewer.width(), self.gl_viewer.height()),
            'overlay_is_visible': self.overlay.isVisible(),
            'overlay_on_top': self.overlay.parent() == self,
            'sync_enabled': self.sync_manager.is_enabled(),
            'tooltip_initialized': self.tooltip_manager is not None,
            'roi_count': len(self.roi_manager.get_rois()),
            'active_roi': self.roi_manager.get_active_roi() is not None,
        }

        if self.overlay.scene():
            scene = self.overlay.scene()
            info['scene_items_count'] = len(scene.items())
            info['scene_rect'] = (scene.sceneRect().x(), scene.sceneRect().y(),
                                  scene.sceneRect().width(),
                                  scene.sceneRect().height())

            if hasattr(self.overlay, 'crosshair_overlay'):
                overlay = self.overlay.crosshair_overlay
                info['crosshair_scene_ref'] = overlay.scene is not None

                if hasattr(overlay, 'h_tracking_line'):
                    info[
                        'tracking_in_scene'] = overlay.h_tracking_line.scene() is scene
                    info[
                        'tracking_visible'] = overlay.h_tracking_line.isVisible()

                if hasattr(overlay, 'center_crosshair'):
                    info[
                        'center_in_scene'] = overlay.center_crosshair.scene() is scene
                    info[
                        'center_visible'] = overlay.center_crosshair.isVisible()

        return info


def main():
    app = QtWidgets.QApplication([])
    QSurfaceFormat.setDefaultFormat(get_surface_format())

    settings = ImageSettings()
    widget = OverlayStack(settings=settings)
    widget.show()

    s = timeit.default_timer()
    arr = create_rgb_checkered()
    processing_t = (timeit.default_timer() - s) * 1000
    meta = get_frame_stats(arr, processing_t)

    widget.gl_viewer.present(arr, meta, PixelFormat.RGB)
    widget.enable_tracking_crosshair(True)
    widget.enable_center_crosshair(True)
    widget.addRectROI(QRectF(0, 0, 127, 127))
    widget.setFixedSize(QSize(1024, 1024))
    widget.overlay.setFixedSize(QSize(1024, 1024))
    widget.fit_to_scene()
    app.exec()


if __name__ == '__main__':
    logger.root.setLevel(logging.CRITICAL)
    main()

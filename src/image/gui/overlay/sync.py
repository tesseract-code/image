"""Manager classes for LayeredImageView components."""

from PyQt6 import QtCore
from PyQt6.QtCore import QTimer, pyqtSignal, pyqtSlot


class TransformSynchronizer(QtCore.QObject):
    """Synchronizes transforms between GL viewer and overlay view."""

    transformSynced = pyqtSignal()

    def __init__(self, gl_viewer, overlay_view, settings):
        super().__init__()
        self.gl_viewer = gl_viewer
        self.overlay_view = overlay_view
        self.settings = settings

        self._updating = False
        self._enabled = True

        # Timers
        self._scroll_timer = QTimer()
        self._scroll_timer.setSingleShot(True)
        self._scroll_timer.timeout.connect(self._handle_debounced_scroll)

        self._setup_connections()

    def _setup_connections(self):
        """Setup signal connections."""
        self.overlay_view.transform_changed.connect(self._on_overlay_changed)
        self.overlay_view.scrollChanged.connect(self._on_overlay_scroll)
        self.settings.changed.connect(self._on_settings_changed)

    def set_enabled(self, enabled):
        """Enable/disable synchronization."""
        self._enabled = enabled

    def is_enabled(self):
        """Check if synchronization is enabled."""
        return self._enabled

    @pyqtSlot()
    def _on_overlay_changed(self):
        """Handle overlay transform changes."""
        if self._updating or not self._enabled:
            return

        self._updating = True
        try:
            self.sync_overlay_to_gl()
            self.transformSynced.emit()
        finally:
            self._updating = False

    @pyqtSlot()
    def _on_overlay_scroll(self):
        """Handle scroll with debouncing."""
        if self._updating or not self._enabled:
            return
        self._scroll_timer.start(50)

    def _handle_debounced_scroll(self):
        """Handle debounced scroll events."""
        if self._updating or not self._enabled:
            return

        self._updating = True
        try:
            self.sync_overlay_to_gl()
        finally:
            self._updating = False

    @pyqtSlot()
    def _on_settings_changed(self):
        """Handle settings changes."""
        if self._updating or not self._enabled:
            return

        self._updating = True
        try:
            self.sync_gl_to_overlay()
        finally:
            self._updating = False

    def sync_overlay_to_gl(self):
        """Convert QGraphicsView transform to GL viewer parameters."""
        if not self.gl_viewer.has_data:
            return

        image_width, image_height = self.gl_viewer.data.shape[:2]

        if image_width == 0 or image_height == 0:
            return

        transform = self.overlay_view.transform()
        view_rect = self.overlay_view.viewport().rect()

        # Get current scale (zoom)
        scale = transform.m11()

        # Calculate visible area in scene coordinates
        top_left_scene = self.overlay_view.mapToScene(0, 0)
        bottom_right_scene = self.overlay_view.mapToScene(view_rect.width(),
                                                          view_rect.height())
        visible_rect_scene = QtCore.QRectF(top_left_scene, bottom_right_scene)
        visible_center_scene = visible_rect_scene.center()

        # Convert to normalized coordinates
        scene_width = self.overlay_view.sceneRect().width()
        scene_height = self.overlay_view.sceneRect().height()

        if scene_width > 0 and scene_height > 0:
            norm_x = visible_center_scene.x() / scene_width
            norm_y = visible_center_scene.y() / scene_height
        else:
            norm_x = 0.5
            norm_y = 0.5

        # Convert to GL coordinates
        pan_x = (norm_x - 0.5) * 2.0
        pan_y = -(norm_y - 0.5) * 2.0

        # Update settings
        self.settings.update_setting('zoom', scale)
        self.settings.update_setting('pan_x', pan_x)
        self.settings.update_setting('pan_y', pan_y)

    def sync_gl_to_overlay(self):
        """Convert GL viewer parameters to QGraphicsView transform."""
        image_width = self.gl_viewer._frame_data.shape[
            0] if self.gl_viewer._frame_data is not None else 0
        image_height = self.gl_viewer._frame_data.shape[
            1] if self.gl_viewer._frame_data is not None else 0

        if image_width == 0 or image_height == 0:
            return

        zoom = self.settings.zoom
        pan_x = self.settings.pan_x
        pan_y = self.settings.pan_y

        # Convert to normalized coordinates
        norm_x = (pan_x / 2.0) + 0.5
        norm_y = (-pan_y / 2.0) + 0.5

        # Convert to scene coordinates
        scene_rect = self.overlay_view.sceneRect()
        scene_center_x = norm_x * scene_rect.width()
        scene_center_y = norm_y * scene_rect.height()

        # Apply transform
        self.overlay_view.resetTransform()
        self.overlay_view.scale(zoom, zoom)
        self.overlay_view.centerOn(scene_center_x, scene_center_y)

    def with_sync_disabled(self, func):
        """Context helper to temporarily disable sync."""
        was_enabled = self._enabled
        self._enabled = False
        self._updating = True
        try:
            func()
        finally:
            self._enabled = was_enabled
            self._updating = False

    def cleanup(self):
        """Clean up resources."""
        if self._scroll_timer.isActive():
            self._scroll_timer.stop()

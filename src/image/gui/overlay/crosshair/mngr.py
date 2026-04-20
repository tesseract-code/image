from PyQt6 import QtCore


class CrosshairManager(QtCore.QObject):
    """Manages crosshair overlay interactions."""

    def __init__(self, overlay_view):
        super().__init__(overlay_view)
        self.overlay_view = overlay_view

    def update_dimensions(self, width, height):
        """Update crosshair dimensions."""
        if not hasattr(self.overlay_view, 'crosshair_overlay'):
            return

        overlay = self.overlay_view.crosshair_overlay
        if hasattr(overlay, 'update_geometry'):
            overlay.update_geometry(width, height)

    def fix_scene_reference(self, scene):
        """Fix CrosshairOverlay scene reference after scene is set."""
        if not hasattr(self.overlay_view, 'crosshair_overlay'):
            return

        overlay = self.overlay_view.crosshair_overlay
        overlay.scene = scene

        # Re-add items to the correct scene if they weren't added
        if overlay.scene:
            if hasattr(overlay,
                       'h_tracking_line') and overlay.h_tracking_line.scene() is None:
                overlay.scene.addItem(overlay.h_tracking_line)
            if hasattr(overlay,
                       'v_tracking_line') and overlay.v_tracking_line.scene() is None:
                overlay.scene.addItem(overlay.v_tracking_line)
            if hasattr(overlay,
                       'center_crosshair') and overlay.center_crosshair.scene() is None:
                overlay.scene.addItem(overlay.center_crosshair)

    # Delegate methods to overlay_view
    def enable_tracking_crosshair(self, enable=True):
        self.overlay_view.enable_tracking_crosshair(enable)

    def enable_center_crosshair(self, enable=True):
        self.overlay_view.enable_center_crosshair(enable)

    def lock_center_crosshair(self, lock=True):
        self.overlay_view.lock_center_crosshair(lock)

    def set_center_crosshair_length(self, length):
        self.overlay_view.set_center_crosshair_length(length)

    def set_crosshair_color(self, color):
        self.overlay_view.set_crosshair_color(color)

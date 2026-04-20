from PyQt6.QtCore import QObject, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtGui import QPen
from PyQt6.QtWidgets import QGraphicsLineItem, QGraphicsItemGroup

from cross_platform.qt6_utils.qtgui.src.qtgui.color_picker import to_qcolor
from pycore.log.ctx import with_logger


@with_logger
class CrosshairOverlay(QObject):
    """Manages crosshair overlay items for the graphics view"""

    def __init__(self, view):
        super().__init__()
        self.view = view
        self.scene = view.scene()

        # Track scene dimensions
        self.scene_width = 0
        self.scene_height = 0
        if self.scene:
            rect = self.scene.sceneRect()
            self.scene_width = rect.width()
            self.scene_height = rect.height()

        # Crosshair items
        self.h_tracking_line = None
        self.v_tracking_line = None
        self.center_crosshair = None

        # Crosshair states
        self.tracking_enabled = False
        self.center_enabled = False
        self.center_locked = False

        # Crosshair properties
        self.color = QColor(Qt.GlobalColor.white)
        self.center_length = 20  # Length of center crosshair arms (in pixels)

        # Store last mouse position for redraws after geometry changes
        self.last_mouse_scene_pos = None

        self.create_tracking_crosshair()
        self.create_center_crosshair()

    def create_tracking_crosshair(self):
        """Create the full XY tracking crosshair that spans the entire view"""
        # Horizontal line
        self.h_tracking_line = QGraphicsLineItem()
        # Vertical line
        self.v_tracking_line = QGraphicsLineItem()

        # Set pen properties (cosmetic = always 1 pixel regardless of zoom)
        pen = QPen(self.color)
        pen.setCosmetic(True)
        pen.setWidth(1)

        self.h_tracking_line.setPen(pen)
        self.v_tracking_line.setPen(pen)

        # Set high Z-value to ensure it's on top but below ROIs
        self.h_tracking_line.setZValue(900)
        self.v_tracking_line.setZValue(900)

        # Make invisible initially
        self.h_tracking_line.setVisible(False)
        self.v_tracking_line.setVisible(False)

        if self.scene:
            self.scene.addItem(self.h_tracking_line)
            self.scene.addItem(self.v_tracking_line)

    def create_center_crosshair(self):
        """Create the fixed plus sign at scene center"""
        # Create group for center crosshair
        self.center_crosshair = QGraphicsItemGroup()

        # Horizontal line
        h_line = QGraphicsLineItem()
        # Vertical line
        v_line = QGraphicsLineItem()

        # Set pen properties
        pen = QPen(self.color)
        pen.setCosmetic(True)
        pen.setWidth(1)

        h_line.setPen(pen)
        v_line.setPen(pen)

        # Add lines to group
        self.center_crosshair.addToGroup(h_line)
        self.center_crosshair.addToGroup(v_line)

        # Set Z-value below tracking crosshair
        self.center_crosshair.setZValue(899)

        # Make invisible initially
        self.center_crosshair.setVisible(False)

        if self.scene:
            self.scene.addItem(self.center_crosshair)

        # Update center crosshair position
        self.update_center_crosshair()

    def update_geometry(self, width, height):
        """
        Update stored scene dimensions when scene rect changes.
        This is called by LayeredImageView when the viewport resizes.
        """
        old_width = self.scene_width
        old_height = self.scene_height

        self.scene_width = width
        self.scene_height = height

        # Only trigger updates if dimensions actually changed
        if abs(old_width - width) > 1 or abs(old_height - height) > 1:
            self._logger.debug(
                f"CrosshairOverlay: Geometry updated to {width:.0f}x{height:.0f}")

            # Notify Qt that geometry is changing
            if self.h_tracking_line:
                self.h_tracking_line.prepareGeometryChange()
            if self.v_tracking_line:
                self.v_tracking_line.prepareGeometryChange()
            if self.center_crosshair:
                self.center_crosshair.prepareGeometryChange()

            # Update crosshairs with new dimensions
            self.update_crosshairs()

    def update_crosshairs(self):
        """Update all crosshairs after scene or transform changes"""
        # Update tracking crosshair if enabled and we have a last position
        if self.tracking_enabled and self.last_mouse_scene_pos:
            self._update_tracking_lines(self.last_mouse_scene_pos)

        # Update center crosshair if enabled
        if self.center_enabled:
            self.update_center_crosshair()

    def update_tracking_crosshair(self, mouse_pos):
        """Update tracking crosshair to span entire scene at mouse position"""
        if not self.tracking_enabled or not self.h_tracking_line.isVisible():
            return

        scene_pos = self.view.mapToScene(mouse_pos)
        self.last_mouse_scene_pos = scene_pos
        self._update_tracking_lines(scene_pos)

    def _update_tracking_lines(self, scene_pos):
        """Internal method to update tracking line geometry"""
        if not self.scene:
            return

        # Get scene boundaries
        scene_rect = self.scene.sceneRect()
        left = scene_rect.left()
        right = scene_rect.right()
        top = scene_rect.top()
        bottom = scene_rect.bottom()

        # Constrain mouse position to scene boundaries
        x = max(left, min(right, scene_pos.x()))
        y = max(top, min(bottom, scene_pos.y()))

        # Update horizontal line (span entire width at mouse Y)
        self.h_tracking_line.setLine(left, y, right, y)
        # Update vertical line (span entire height at mouse X)
        self.v_tracking_line.setLine(x, top, x, bottom)

    def update_center_crosshair(self):
        """Update center crosshair position at scene center"""
        if not self.center_crosshair or not self.scene:
            return

        # Get scene center
        scene_rect = self.scene.sceneRect()
        center_x = scene_rect.center().x()
        center_y = scene_rect.center().y()

        # Crosshair arm length (in scene coordinates = pixels since scene matches viewport)
        half_length = self.center_length / 2

        # Get the lines from the group
        child_items = self.center_crosshair.childItems()
        if len(child_items) < 2:
            return

        h_line = child_items[0] if isinstance(child_items[0],
                                              QGraphicsLineItem) else None
        v_line = child_items[1] if isinstance(child_items[1],
                                              QGraphicsLineItem) else None

        if h_line and v_line:
            # Notify that geometry is changing
            self.center_crosshair.prepareGeometryChange()

            # Update horizontal line (short arm)
            h_line.setLine(center_x - half_length, center_y,
                           center_x + half_length, center_y)
            # Update vertical line (short arm)
            v_line.setLine(center_x, center_y - half_length,
                           center_x, center_y + half_length)

    def set_tracking_enabled(self, enabled):
        """Enable/disable tracking crosshair"""
        self.tracking_enabled = enabled
        if self.h_tracking_line and self.v_tracking_line:
            self.h_tracking_line.setVisible(enabled)
            self.v_tracking_line.setVisible(enabled)
            if not enabled:
                # Clear the lines when disabled
                self.h_tracking_line.setLine(0, 0, 0, 0)
                self.v_tracking_line.setLine(0, 0, 0, 0)
                self.last_mouse_scene_pos = None

    def set_center_enabled(self, enabled):
        """Enable/disable center crosshair"""
        self.center_enabled = enabled
        if self.center_crosshair:
            self.center_crosshair.setVisible(enabled)
            if enabled:
                self.update_center_crosshair()

    def set_center_locked(self, locked):
        """Lock center crosshair to scene center (updates when view moves)"""
        self.center_locked = locked
        if locked and self.center_enabled:
            self.update_center_crosshair()

    def set_center_length(self, length):
        """Set the length of center crosshair arms (in pixels)"""
        self.center_length = length
        if self.center_enabled:
            self.update_center_crosshair()

    def set_color(self, color):
        """Set crosshair color"""
        self.color = to_qcolor(color)

        # Update tracking crosshair color
        pen = QPen(self.color)
        pen.setCosmetic(True)
        pen.setWidth(1)

        if self.h_tracking_line and self.v_tracking_line:
            self.h_tracking_line.setPen(pen)
            self.v_tracking_line.setPen(pen)

        # Update center crosshair color
        if self.center_crosshair:
            for item in self.center_crosshair.childItems():
                if isinstance(item, QGraphicsLineItem):
                    item.setPen(pen)

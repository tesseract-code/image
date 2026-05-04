__all__ = ['BaseROI', 'EllipseROI', 'RectROI', 'LineROI', 'PolygonROI']

from typing import List, Optional

from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import QPen
from PyQt6.QtWidgets import (QGraphicsEllipseItem, QGraphicsRectItem,
                             QGraphicsLineItem, QGraphicsPolygonItem,
                             QGraphicsItem, QGraphicsView)

from qtgui.joystick import compute_joystick_displacement

def _create_roi_pen(high_dpi: bool = False) -> QPen:
    """Create the standard ROI pen."""
    pen = QPen(Qt.GlobalColor.green)
    pen.setCosmetic(True)
    if high_dpi:
        pen.setWidth(max(2, pen.width()))
    return pen


class BaseROI(QGraphicsItem):
    """Base mixin with common functionality for all ROIs."""

    def __init__(self,
                 viewer: QGraphicsView,
                 parent: Optional[QGraphicsItem]= None):
        super().__init__(parent=parent)
        self._viewer = viewer
        self.setPen(_create_roi_pen())
        self.setFlags(
            QGraphicsEllipseItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsEllipseItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsEllipseItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

    def _get_scene_bounds(self) -> QRectF | None:
        """Get scene rect if valid, else None."""
        scene_rect = self._viewer.sceneRect()
        return None if scene_rect.isNull() else scene_rect

    def _clamp_rect_to_scene(self, rect: QRectF,
                             offset: QPointF = QPointF(0, 0)) -> QRectF:
        """
        Clamp a rectangle to stay within the viewer's scene rect.

        Args:
            rect: Rectangle in local coordinates
            offset: Item position offset to convert to scene coordinates
        """
        scene_rect = self._get_scene_bounds()
        if scene_rect is None:
            return rect

        # Convert to scene coordinates
        scene_left = rect.left() + offset.x()
        scene_top = rect.top() + offset.y()
        scene_right = rect.right() + offset.x()
        scene_bottom = rect.bottom() + offset.y()

        # Calculate corrections needed
        dx = dy = 0.0
        if scene_left < scene_rect.left():
            dx = scene_rect.left() - scene_left
        elif scene_right > scene_rect.right():
            dx = scene_rect.right() - scene_right

        if scene_top < scene_rect.top():
            dy = scene_rect.top() - scene_top
        elif scene_bottom > scene_rect.bottom():
            dy = scene_rect.bottom() - scene_bottom

        if dx != 0 or dy != 0:
            return rect.translated(dx, dy)
        return rect

    def _clamp_position_to_scene(self, new_pos: QPointF) -> QPointF:
        """Clamp position to ensure ROI stays within scene bounds."""
        scene_rect = self._get_scene_bounds()
        if scene_rect is None:
            return new_pos

        bounding = self.rect()
        new_left = new_pos.x()
        new_right = new_left + bounding.width()
        new_top = new_pos.y()
        new_bottom = new_top + bounding.height()

        dx = dy = 0.0
        if new_left < scene_rect.left():
            dx = scene_rect.left() - new_pos.x()
        elif new_right > scene_rect.right():
            dx = scene_rect.right() - new_right

        if new_top < scene_rect.top():
            dy = scene_rect.top() - new_top
        elif new_bottom > scene_rect.bottom():
            dy = scene_rect.bottom() - new_bottom

        return new_pos + QPointF(dx, dy)

    def mousePressEvent(self, event):
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._viewer.handleRoiClicked(self)

    def itemChange(self, change, value):
        if change == QGraphicsLineItem.GraphicsItemChange.ItemPositionChange:
            value = self._clamp_position_to_scene(value)
        elif change == QGraphicsLineItem.GraphicsItemChange.ItemPositionHasChanged:
            self._viewer.handleRoiChanged(self)
        return super().itemChange(change, value)

    def move_by_joystick_moves(self, moves: List["JoystickMove"]) -> None:
        """
        Move the ROI by applying a batched list of joystick moves.

        Args:
            moves: List of JoystickMove objects to apply
        """
        dx, dy = compute_joystick_displacement(moves)
        if dx == 0 and dy == 0:
            return

        current_x = self.pos().x()
        current_y = self.pos().y()
        clamped_pos = self._clamp_position_to_scene(QPointF(current_x + dx,
                                                            current_y + dy))
        self.setPos(clamped_pos)
        self._viewer.handleRoiChanged(self)


class EllipseROI(QGraphicsEllipseItem, BaseROI):
    def __init__(self, viewer: "ImageViewer", rect: QRectF | None = None):
        QGraphicsEllipseItem.__init__(self)
        BaseROI.__init__(self, viewer=viewer)
        if rect is not None:
            self.setRect(rect)


class RectROI(QGraphicsRectItem, BaseROI):
    def __init__(self, viewer: "ImageViewer", rect: QRectF | None = None):
        QGraphicsRectItem.__init__(self)
        BaseROI.__init__(self, viewer=viewer)
        if rect is not None:
            self.setRect(rect)

class LineROI(QGraphicsLineItem, BaseROI):
    def __init__(self, viewer: "ImageViewer", line=None):
        super().__init__()
        if line is not None:
            self.setLine(line)
        self._viewer = viewer
        self.setPen(_create_roi_pen())
        self.setFlags(
            QGraphicsLineItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsLineItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsLineItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )


class PolygonROI(QGraphicsPolygonItem, BaseROI):
    def __init__(self, viewer: QGraphicsView, polygon=None):
        super().__init__()
        if polygon is not None:
            self.setPolygon(polygon)
        self._viewer = viewer
        self.setPen(_create_roi_pen())
        self.setFlags(
            QGraphicsPolygonItem.GraphicsItemFlag.ItemIsSelectable |
            QGraphicsPolygonItem.GraphicsItemFlag.ItemIsMovable |
            QGraphicsPolygonItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )

"""
SynchableGraphicsView
"""

from PyQt6 import QtWidgets, QtGui
from PyQt6.QtCore import QPointF
from PyQt6.QtCore import QRectF
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPen
from PyQt6.QtWidgets import QGraphicsView

from image.gui.overlay.crosshair.crosshair import CrosshairOverlay
from image.gui.overlay.roi.roi import EllipseROI, RectROI, LineROI, PolygonROI


class SynchableGraphicsView(QGraphicsView):
    """
    QGraphicsView that can synchronize panning & zooming.

    Key behavior:
    - Scene rect matches viewport size (managed by parent)
    - ROIs positioned in image coordinates (not scene coordinates)
    - View transform maps scene to image space
    """

    # Signals for external communication
    transform_changed = pyqtSignal()
    scrollChanged = pyqtSignal()
    wheelNotches = pyqtSignal(float)
    roiClicked = pyqtSignal(object)
    roiChanged = pyqtSignal(object)

    def __init__(self, scene=None, parent=None):
        super().__init__(parent)

        if scene:
            self.setScene(scene)

        self._handDrag = False
        self.clearTransformChanges()
        self.connectSbarSignals(self.scrollChanged)

        # ROI management - Simple list, no position tracking needed
        self._roi_items = []
        self.active_roi = None

        # Crosshair overlay
        self.crosshair_overlay = CrosshairOverlay(self)

        # Enable mouse tracking for crosshair
        self.setMouseTracking(True)
        # Track mouse state
        self._mouse_pressed = False

        self.setup_view_for_images()

        # Connect signals for crosshair updates
        self.scrollChanged.connect(self.on_scroll_for_crosshair)
        self.transform_changed.connect(self.on_transform_for_crosshair)

    def handleRoiClicked(self, roi):
        """Handle ROI click events from ROI items"""
        self.setActiveROI(roi)
        self.roiClicked.emit(roi)

    def handleRoiChanged(self, roi):
        """Handle ROI change events from ROI items"""
        self.roiChanged.emit(roi)

    def setup_view_for_images(self):
        """Configure the view for optimal image display"""
        # Remove borders and make background transparent
        self.setFrameStyle(0)
        self.setStyleSheet(
            "QGraphicsView { border: 0px; background: transparent; }")

        # Disable scrollbars by default for clean image display
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def resizeEvent(self, event):
        """
        Handle resize WITHOUT auto-fitting to scene.
        Scene rect management is handled by parent (LayeredImageView).

        ROIs don't need position updates because they're in image space,
        not scene space. The view transform handles the mapping.
        """
        super().resizeEvent(event)

        # Update center crosshair if locked
        if self.crosshair_overlay.center_locked:
            pass
            # self.crosshair_overlay.update_center_crosshair()

    def on_scroll_for_crosshair(self):
        """Update center crosshair when scrolling if locked"""
        if self.crosshair_overlay.center_locked:
            pass
            # self.crosshair_overlay.update_center_crosshair()

    def on_transform_for_crosshair(self):
        """Update center crosshair when transforming (zooming) if locked"""
        if self.crosshair_overlay.center_locked:
            pass
            # self.crosshair_overlay.update_center_crosshair()

    def mouseMoveEvent(self, event):
        """Handle mouse movement for tracking crosshair"""
        # Update tracking crosshair if mouse is not pressed and tracking is enabled
        if not self._mouse_pressed and self.crosshair_overlay.tracking_enabled:
            self.crosshair_overlay.update_tracking_crosshair(event.pos())

        # Call parent implementation
        super().mouseMoveEvent(event)

    def mousePressEvent(self, event):
        """Handle mouse press - hide tracking crosshair during interaction"""
        self._mouse_pressed = True

        # Hide tracking crosshair during interaction
        if self.crosshair_overlay.tracking_enabled:
            self.crosshair_overlay.h_tracking_line.setVisible(False)
            self.crosshair_overlay.v_tracking_line.setVisible(False)

        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release - restore tracking crosshair"""
        self._mouse_pressed = False

        # Restore tracking crosshair if enabled
        if self.crosshair_overlay.tracking_enabled:
            self.crosshair_overlay.h_tracking_line.setVisible(True)
            self.crosshair_overlay.v_tracking_line.setVisible(True)
            self.crosshair_overlay.update_tracking_crosshair(event.pos())

        super().mouseReleaseEvent(event)

    def enterEvent(self, event):
        """Show tracking crosshair when mouse enters view"""
        if self.crosshair_overlay.tracking_enabled and not self._mouse_pressed:
            self.crosshair_overlay.h_tracking_line.setVisible(True)
            self.crosshair_overlay.v_tracking_line.setVisible(True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        """Hide tracking crosshair when mouse leaves view"""
        if self.crosshair_overlay.tracking_enabled:
            self.crosshair_overlay.h_tracking_line.setVisible(False)
            self.crosshair_overlay.v_tracking_line.setVisible(False)
        super().leaveEvent(event)

    # ========== Crosshair Management ==========

    def enable_tracking_crosshair(self, enable=True):
        """Enable or disable the mouse tracking crosshair (full XY lines)"""
        self.setMouseTracking(enable)
        self.crosshair_overlay.set_tracking_enabled(enable)

    def enable_center_crosshair(self, enable=True):
        """Enable or disable the fixed center crosshair (plus sign)"""
        self.crosshair_overlay.set_center_enabled(enable)

    def lock_center_crosshair(self, lock=True):
        """Lock the center crosshair to update with view changes"""
        self.crosshair_overlay.set_center_locked(lock)

    def set_center_crosshair_length(self, length):
        """Set the length of center crosshair arms (plus sign)"""
        self.crosshair_overlay.set_center_length(length)

    def set_crosshair_color(self, color):
        """Set crosshair color (QColor or Qt.GlobalColor)"""
        self.crosshair_overlay.set_color(color)

    # ========== Scrollbar Signal Management ==========

    def connectSbarSignals(self, slot):
        """Connect scrollbar signals"""
        sbar = self.horizontalScrollBar()
        sbar.valueChanged.connect(slot)
        sbar.rangeChanged.connect(slot)

        sbar = self.verticalScrollBar()
        sbar.valueChanged.connect(slot)
        sbar.rangeChanged.connect(slot)

    def disconnectSbarSignals(self):
        """Disconnect scrollbar signals"""
        sbar = self.horizontalScrollBar()
        try:
            sbar.valueChanged.disconnect()
            sbar.rangeChanged.disconnect()
        except TypeError:
            pass

        sbar = self.verticalScrollBar()
        try:
            sbar.valueChanged.disconnect()
            sbar.rangeChanged.disconnect()
        except TypeError:
            pass

    # ========== Scroll State Management ==========

    @property
    def scrollState(self):
        """Get current scroll state as normalized (0-1) position"""
        viewport_center = self.viewport().rect().center()
        center_point = self.mapToScene(viewport_center)

        scene_rect = self.sceneRect()
        center_width = center_point.x() - scene_rect.left()
        center_height = center_point.y() - scene_rect.top()
        scene_width = scene_rect.width()
        scene_height = scene_rect.height()

        scene_width_percent = center_width / scene_width if scene_width != 0 else 0
        scene_height_percent = center_height / scene_height if scene_height != 0 else 0
        return (scene_width_percent, scene_height_percent)

    @scrollState.setter
    def scrollState(self, state):
        """Set scroll state from normalized (0-1) position"""
        scene_width_percent, scene_height_percent = state
        x = scene_width_percent * self.sceneRect().width() + self.sceneRect().left()
        y = scene_height_percent * self.sceneRect().height() + self.sceneRect().top()
        self.centerOn(x, y)

    # ========== Zoom Factor Management ==========

    @property
    def zoomFactor(self):
        """Get current zoom factor"""
        return self.transform().m11()

    @zoomFactor.setter
    def zoomFactor(self, newZoomFactor):
        """Set zoom factor"""
        currentZoom = self.zoomFactor
        if currentZoom != 0:
            scaleFactor = newZoomFactor / currentZoom
            self.scale(scaleFactor, scaleFactor)

    # ========== Event Handlers ==========

    def wheelEvent(self, wheelEvent):
        """Handle wheel events for zooming"""
        if wheelEvent.modifiers() & Qt.KeyboardModifier.ControlModifier:
            delta = wheelEvent.angleDelta().y()
            self.wheelNotches.emit(delta / 240.0)
            wheelEvent.accept()
        else:
            super().wheelEvent(wheelEvent)

    def keyReleaseEvent(self, keyEvent):
        """Ignore key release events"""
        keyEvent.ignore()

    # ========== ROI Management ==========
    #
    # CRITICAL: ROIs are positioned in IMAGE COORDINATES.
    # - When you add an ROI at pos (100, 100), it stays at image pixel (100, 100)
    # - The view transform handles mapping from scene to image space
    # - ROIs don't need position updates when scene rect changes
    #

    def addEllipseROI(self, rect=None, pos=None):
        """
        Add an ellipse ROI.

        Args:
            rect: QRectF in IMAGE coordinates
            pos: QPointF in IMAGE coordinates (additional offset)
        """
        roi = EllipseROI(self, rect)
        if pos:
            roi.setPos(pos)
        if self.scene():
            self.scene().addItem(roi)

        self._roi_items.append(roi)
        return roi

    def addRectROI(self, rect=None, pos=None):
        """
        Add a rectangle ROI.

        Args:
            rect: QRectF in IMAGE coordinates
            pos: QPointF in IMAGE coordinates (additional offset)
        """
        roi = RectROI(self, rect)
        if pos:
            roi.setPos(pos)
        if self.scene():
            self.scene().addItem(roi)

        self._roi_items.append(roi)
        return roi

    def addLineROI(self, line=None, pos=None):
        """
        Add a line ROI.

        Args:
            line: QLineF in IMAGE coordinates
            pos: QPointF in IMAGE coordinates (additional offset)
        """
        roi = LineROI(self, line)
        if pos:
            roi.setPos(pos)
        if self.scene():
            self.scene().addItem(roi)

        self._roi_items.append(roi)
        return roi

    def addPolygonROI(self, polygon=None, pos=None):
        """
        Add a polygon ROI.

        Args:
            polygon: QPolygonF in IMAGE coordinates
            pos: QPointF in IMAGE coordinates (additional offset)
        """
        roi = PolygonROI(self, polygon)
        if pos:
            roi.setPos(pos)
        if self.scene():
            self.scene().addItem(roi)

        self._roi_items.append(roi)
        return roi

    def removeROI(self, roi):
        """Remove an ROI"""
        if roi in self._roi_items:
            self._roi_items.remove(roi)
        if self.scene():
            self.scene().removeItem(roi)

    def clearROIs(self):
        """Clear all ROIs"""
        for roi in self._roi_items[:]:
            if self.scene():
                self.scene().removeItem(roi)
        self._roi_items.clear()

    def getROIs(self):
        """Get list of all ROIs"""
        return self._roi_items.copy()

    def setActiveROI(self, roi):
        """Set the active ROI"""
        if self.active_roi:
            self.active_roi.setPen(QPen(Qt.GlobalColor.green))
        self.active_roi = roi
        if roi:
            roi.setPen(QPen(Qt.GlobalColor.cyan))

    def get_roi_image_coordinates(self, roi):
        """
        Get ROI position and geometry in IMAGE coordinates.
        This is what you should save/load.

        Returns:
            dict with 'pos' (QPointF) and type-specific geometry
        """
        result = {
            'pos': QPointF(roi.pos()),
            'type': type(roi).__name__
        }

        if hasattr(roi, 'rect'):
            result['rect'] = QRectF(roi.rect())
        elif hasattr(roi, 'line'):
            result['line'] = roi.line()
        elif hasattr(roi, 'polygon'):
            result['polygon'] = roi.polygon()

        return result

    # ========== Transform Management ==========

    def checkTransformChanged(self):
        """Check if transform has changed significantly"""
        delta = 0.0001
        result = False

        def different(t, u):
            if u == 0.0:
                d = abs(t - u)
            else:
                d = abs((t - u) / u)
            return d > delta

        t = self.transform()
        u = self._transform

        if (different(t.m11(), u.m11()) or
                different(t.m22(), u.m22()) or
                different(t.m12(), u.m12()) or
                different(t.m21(), u.m21()) or
                different(t.m31(), u.m31()) or
                different(t.m32(), u.m32())):
            self._transform = t
            self.transform_changed.emit()
            result = True
        return result

    def clearTransformChanges(self):
        """Clear transform change tracking"""
        self._transform = self.transform()

    # ========== Scroll Helper Methods ==========

    def scrollToTop(self):
        """Scroll to top"""
        self.verticalScrollBar().setValue(self.verticalScrollBar().minimum())

    def scrollToBottom(self):
        """Scroll to bottom"""
        self.verticalScrollBar().setValue(self.verticalScrollBar().maximum())

    def scrollToBegin(self):
        """Scroll to beginning (left)"""
        self.horizontalScrollBar().setValue(
            self.horizontalScrollBar().minimum())

    def scrollToEnd(self):
        """Scroll to end (right)"""
        self.horizontalScrollBar().setValue(
            self.horizontalScrollBar().maximum())

    def centerView(self):
        """Center the view"""
        vbar = self.verticalScrollBar()
        hbar = self.horizontalScrollBar()
        vbar.setValue((vbar.maximum() + vbar.minimum()) // 2)
        hbar.setValue((hbar.maximum() + hbar.minimum()) // 2)

    # ========== Feature Control ==========

    def enableScrollBars(self, enable):
        """Enable or disable scrollbars"""
        if enable:
            self.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded)
            self.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        else:
            self.setHorizontalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            self.setVerticalScrollBarPolicy(
                Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

    def enableHandDrag(self, enable):
        """Enable or disable hand drag mode"""
        if enable:
            if not self._handDrag:
                self._handDrag = True
                self.setDragMode(
                    QtWidgets.QGraphicsView.DragMode.ScrollHandDrag)
        else:
            if self._handDrag:
                self._handDrag = False
                self.setDragMode(QtWidgets.QGraphicsView.DragMode.NoDrag)

    # ========== Debug Methods ==========

    def dumpTransform(self, t, padding=""):
        """Dump transform matrix for debugging"""
        print(f"{padding}{t.m11():5.3f} {t.m12():5.3f} {t.m13():5.3f}")
        print(f"{padding}{t.m21():5.3f} {t.m22():5.3f} {t.m23():5.3f}")
        print(f"{padding}{t.m31():5.3f} {t.m32():5.3f} {t.m33():5.3f}")

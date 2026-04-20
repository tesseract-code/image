from PyQt6 import QtCore
from PyQt6.QtCore import QTimer, Qt, QPoint
from PyQt6.QtWidgets import QLabel


class TooltipManager(QtCore.QObject):
    """Manages tooltip display for image coordinates and values."""

    def __init__(self, parent):
        super().__init__(parent)

        self._label = QLabel(parent)
        self._label.setWindowFlags(
            Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        self._label.setStyleSheet("""
                    QLabel {
                        background-color: rgba(0, 0, 0, 180); 
                        color: white;
                        padding: 4px 8px;
                        border-radius: 3px;
                        font-size: 11px;
                        border: 1px solid rgba(255, 255, 255, 100);
                    }
                """)
        self._label.hide()
        self._label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._label.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)

        self._hide_timer = QTimer()
        self._hide_timer.setSingleShot(True)
        self._hide_timer.timeout.connect(self._hide_label)

        self._last_update_time = 0
        self._throttle_ms = 16  # ~60fps

    def update_tooltip(self, x, y, value, global_pos):
        """
        Update tooltip content and position, then show it.

        Args:
            x: X coordinate to display
            y: Y coordinate to display
            value: Value at coordinates (or None)
            global_pos: QPoint for tooltip position in global coordinates
        """
        # Throttle updates
        # current_time = QtCore.QTime.currentTime().msecsSinceStartOfDay()
        # if current_time - self._last_update_time < self._throttle_ms:
        #     return
        # self._last_update_time = current_time

        # Update label text
        tooltip_text = f"X: {x:.1f}, Y: {y:.1f}"
        if value is not None:
            tooltip_text += f"\nValue: {value}"

        self._label.setText(tooltip_text)
        self._label.adjustSize()

        # Position tooltip
        self._label.move(global_pos + QPoint(15, 15))

        # Show and reset auto-hide timer
        if not self._label.isVisible():
            self._label.show()

        self._hide_timer.start(1000)  # Auto-hide after 1 second

    def _hide_label(self):
        """Internal hide callback for timer."""
        if self._label:
            self._label.hide()

    def cleanup(self):
        """Clean up resources."""
        if self._hide_timer.isActive():
            self._hide_timer.stop()

        if self._label:
            self._label.hide()
            self._label.deleteLater()
            self._label = None

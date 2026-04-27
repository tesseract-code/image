from __future__ import annotations

from typing import Callable, Optional

from PyQt6.QtCore import Qt, pyqtSignal, QEvent
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QToolButton, QSizePolicy, QFrame,
)

from qtgui.pixmap import colorize_pixmap  # external, assumed working

# ----------------------------------------------------------------------
# Constants – easily adjustable, can be moved to a config class
# ----------------------------------------------------------------------
TOOLBAR_BG = QColor(20, 20, 20, 200)
TOOLBAR_BORDER = QColor(255, 255, 255, 30)
BTN_HOVER_BG = QColor(255, 255, 255, 22)
BTN_PRESS_BG = QColor(255, 255, 255, 40)

DEFAULT_BTN_SIZE = 28
DEFAULT_MARGIN = 10
DEFAULT_CORNER_RADIUS = 8


class _Separator(QFrame):
    """Thin vertical line used between button groups."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.VLine)
        self.setFixedSize(1, 18)
        self.setStyleSheet("background: rgba(255,255,255,0.15); border: none;")


class GLToolbar(QWidget):
    """
    Semi-transparent floating toolbar that attaches to a parent widget
    and stays anchored at the top‑right corner.
    """
    zoom_in_requested = pyqtSignal()
    zoom_out_requested = pyqtSignal()
    zoom_reset_requested = pyqtSignal()
    capture_requested = pyqtSignal()

    def __init__(self,
                 parent: QWidget,
                 margin: int = DEFAULT_MARGIN,
                 btn_size: int = DEFAULT_BTN_SIZE) -> None:
        super().__init__(parent)
        self._margin = margin
        self._btn_size = btn_size

        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        self._layout = QHBoxLayout(self)
        self._layout.setContentsMargins(6, 5, 6, 5)
        self._layout.setSpacing(4)

        # Store created buttons for potential later removal
        self._buttons: list[QToolButton] = []

        self._build_default_buttons()
        self._apply_stylesheet()
        self.reposition()

        # Watch parent resize events to keep the toolbar anchored
        if parent:
            parent.installEventFilter(self)

    def add_button(self,
                   icon: QIcon | None = None,
                   text: str = "",
                   tooltip: str = "",
                   slot: Optional[Callable] = None) -> QToolButton:
        """
        Add a custom button to the toolbar (appended to the right).
        Returns the created QToolButton for further customization.
        """
        btn = QToolButton()
        if icon:
            btn.setIcon(icon)
        if text:
            btn.setText(text)
        if tooltip:
            btn.setToolTip(tooltip)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedSize(self._btn_size, self._btn_size)
        if slot:
            btn.clicked.connect(slot)

        self._layout.addWidget(btn)
        self._buttons.append(btn)
        self._update_button_styles()
        self.reposition()
        return btn

    def add_separator(self) -> None:
        """Add a visual separator."""
        self._layout.addWidget(_Separator(self))
        self.reposition()

    def reposition(self) -> None:
        """Re‑anchor the toolbar to the top‑right corner of its parent."""
        parent = self.parent()
        if not parent:
            return
        self.adjustSize()
        x = parent.width() - self.width() - self._margin
        self.move(x, self._margin)

    def eventFilter(self, obj: QWidget, event: QEvent) -> bool:
        if obj is self.parent() and event.type() == QEvent.Type.Resize:
            self.reposition()
        return super().eventFilter(obj, event)

    def paintEvent(self, event) -> None:  # type: ignore[override]
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(),
                            DEFAULT_CORNER_RADIUS, DEFAULT_CORNER_RADIUS)
        painter.fillPath(path, TOOLBAR_BG)
        painter.setPen(TOOLBAR_BORDER)
        painter.drawPath(path)

    def _build_default_buttons(self) -> None:
        """Create the standard zoom and capture buttons."""
        # Icons are loaded once – can be overridden later.
        zoom_in_icon = self._create_icon("line-icons:zoom-in.svg")
        zoom_out_icon = self._create_icon("line-icons:zoom-out.svg")
        capture_icon = self._create_icon("line-icons:camera-lens.svg")

        self._btn_zoomin = self._icon_btn(zoom_in_icon, "Zoom in")
        self._btn_zoomout = self._icon_btn(zoom_out_icon, "Zoom out")
        self._btn_reset = self._label_btn("1:1", "Reset to 1:1 zoom")
        self._capture_btn = self._icon_btn(capture_icon, "Screenshot")

        for w in (self._btn_zoomin, self._btn_zoomout,
                  _Separator(self), self._btn_reset,
                  _Separator(self), self._capture_btn):
            self._layout.addWidget(w)
            if isinstance(w, QToolButton):
                self._buttons.append(w)

        self._btn_zoomin.clicked.connect(self.zoom_in_requested)
        self._btn_zoomout.clicked.connect(self.zoom_out_requested)
        self._btn_reset.clicked.connect(self.zoom_reset_requested)
        self._capture_btn.clicked.connect(self.capture_requested)

    def _icon_btn(self, icon: QIcon, tip: str) -> QToolButton:
        btn = QToolButton()
        btn.setIcon(icon)
        btn.setToolTip(tip)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedSize(self._btn_size, self._btn_size)
        return btn

    def _label_btn(self, label: str, tip: str) -> QToolButton:
        btn = QToolButton()
        btn.setText(label)
        btn.setToolTip(tip)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setFixedSize(max(self._btn_size, 36), self._btn_size)
        return btn

    def _create_icon(self, resource_path: str) -> QIcon:
        """Load an SVG or PNG icon and color it white."""
        pixmap = QPixmap(resource_path)
        colored = colorize_pixmap(pixmap, QColor("white"))
        return QIcon(colored)

    def _apply_stylesheet(self) -> None:
        """Generate the stylesheet dynamically using current button size."""
        css = f"""
        QToolButton {{
            color: #d8d8d8;
            background: transparent;
            border: 0.5px solid rgba(255,255,255,0.13);
            border-radius: 5px;
            font-size: 14px;
            padding: 0px 6px;
            min-width: {self._btn_size}px;
            min-height: {self._btn_size}px;
            max-width: {self._btn_size}px;
            max-height: {self._btn_size}px;
        }}
        QToolButton:hover  {{ background: rgba(255,255,255,0.09); }}
        QToolButton:pressed {{ background: rgba(255,255,255,0.16); }}
        """
        self.setStyleSheet(css)

    def _update_button_styles(self) -> None:
        """Re‑apply stylesheet after adding new buttons."""
        self._apply_stylesheet()

from __future__ import annotations

from typing import TYPE_CHECKING

from PIL.ImageQt import QPixmap
from PyQt6.QtCore import Qt, QSize, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QIcon
from PyQt6.QtWidgets import (
    QWidget, QHBoxLayout, QToolButton, QSizePolicy, QFrame,
)

from qtgui.pixmap import colorize_pixmap

if TYPE_CHECKING:
    pass

_TOOLBAR_BG = QColor(20, 20, 20, 200)
_TOOLBAR_BORDER = QColor(255, 255, 255, 30)
_SEP_COLOR = QColor(255, 255, 255, 38)
_BTN_HOVER_BG = QColor(255, 255, 255, 22)
_BTN_PRESS_BG = QColor(255, 255, 255, 40)

_TOOLBAR_CSS = """
QToolButton {{
    color: #d8d8d8;
    background: transparent;
    border: 0.5px solid rgba(255,255,255,0.13);
    border-radius: 5px;
    font-size: 14px;
    padding: 0px 6px;
    min-width: {btn_w}px;
    min-height: {btn_h}px;
    max-width:  {btn_w}px;
    max-height: {btn_h}px;
}}
QToolButton:hover  {{ background: rgba(255,255,255,0.09); }}
QToolButton:pressed {{ background: rgba(255,255,255,0.16); }}

QComboBox {{
    color: #c8c8c8;
    background: transparent;
    border: 0.5px solid rgba(255,255,255,0.13);
    border-radius: 5px;
    font-size: 11px;
    padding: 0px 6px;
    min-height: {btn_h}px;
    min-width: 72px;
}}
QComboBox::drop-down {{ border: none; width: 14px; }}
QComboBox QAbstractItemView {{
    color: #d0d0d0;
    background: #222222;
    border: 0.5px solid rgba(255,255,255,0.15);
    border-radius: 5px;
    selection-background-color: rgba(255,255,255,0.12);
    font-size: 11px;
    outline: none;
}}
"""

_BTN_W, _BTN_H = 28, 28


class _Separator(QFrame):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.Shape.VLine)
        self.setFixedSize(1, 18)
        self.setStyleSheet("background: rgba(255,255,255,0.15); border: none;")


class GLToolbar(QWidget):
    """
    Semi-transparent floating toolbar that overlays the top-right corner of a
    GLImageShow widget.

    Signals
    -------
    zoom_in_requested  — emitted when '+' is clicked
    zoom_out_requested — emitted when '−' is clicked
    zoom_reset_requested — emitted when '1:1' is clicked
    """

    zoom_in_requested = pyqtSignal()
    zoom_out_requested = pyqtSignal()
    zoom_reset_requested = pyqtSignal()

    def __init__(self,
                 parent: QWidget,
                 margin: int = 10) -> None:
        super().__init__(parent)
        self._margin = margin
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground)
        self.setStyleSheet(_TOOLBAR_CSS.format(btn_w=_BTN_W, btn_h=_BTN_H))
        self.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        self._build_ui()
        self._reposition()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 5, 6, 5)
        layout.setSpacing(4)

        self._btn_zoomin = self._icon_btn(
            QIcon(
                colorize_pixmap(
                    QPixmap("line-icons:zoom-in.svg"),
                    QColor("white")
                )
            ), "Zoom in")
        self._btn_zoomout = self._icon_btn(QIcon(
            colorize_pixmap(
                QPixmap("line-icons:zoom-out.svg"),
                QColor("white")
            )
        ), "Zoom out")
        self._btn_reset = self._label_btn("1:1", "Reset to 1:1 zoom")

        for w in (
                self._btn_zoomin,
                self._btn_zoomout,
                _Separator(self),
                self._btn_reset,
        ):
            layout.addWidget(w)

        self.adjustSize()

        self._btn_zoomin.clicked.connect(self.zoom_in_requested)
        self._btn_zoomout.clicked.connect(self.zoom_out_requested)
        self._btn_reset.clicked.connect(self.zoom_reset_requested)

    @staticmethod
    def _icon_btn(icon: QIcon, tip: str) -> QToolButton:
        btn = QToolButton()
        btn.setIcon(icon)
        btn.setToolTip(tip)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        return btn

    @staticmethod
    def _label_btn(label: str, tip: str) -> QToolButton:
        btn = QToolButton()
        btn.setText(label)
        btn.setToolTip(tip)
        btn.setCursor(Qt.CursorShape.PointingHandCursor)
        btn.setStyleSheet(
            "min-width: 28px; max-width: 36px;"
        )
        return btn

    def _reposition(self) -> None:
        if self.parent() is None:
            return
        p: QWidget = self.parent()  # type: ignore[assignment]
        self.adjustSize()
        x = p.width() - self.width() - self._margin
        self.move(x, self._margin)

    def paintEvent(self, event) -> None:  # noqa: N802
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)
        path = QPainterPath()
        path.addRoundedRect(0, 0, self.width(), self.height(), 8, 8)
        p.fillPath(path, _TOOLBAR_BG)
        p.setPen(_TOOLBAR_BORDER)
        p.drawPath(path)

    def sizeHint(self) -> QSize:  # noqa: N802
        return self.minimumSizeHint()

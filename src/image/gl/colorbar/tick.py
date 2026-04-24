"""
tick.py
=======
Software-rendered tick marks and labels for the colorbar widget.

Because OpenGL font rendering on macOS is unreliable (no native text
rasteriser, no subpixel hinting), ticks and labels are drawn into an
off-screen :class:`QImage` with a :class:`QPainter` and then blitted to
the widget surface in a single ``drawImage`` call.  This gives crisp,
platform-consistent text at the cost of one CPU-side rasterisation pass
per frame — acceptable for a static or near-static colorbar.

Coordinate conventions
-----------------------
* **Vertical** colorbar: position 1.0 → top of ``cb_rect``,
  position 0.0 → bottom.  This matches the OpenGL texture convention where
  U=1.0 is the "high" end of the LUT.
* **Horizontal** colorbar: position 0.0 → left, 1.0 → right.

Typical usage::

    config = TickConfig(count=7, length=5.0)
    renderer = TickRenderer(config, vmin=0.0, vmax=1.0)

    ticks = renderer.compute_ticks()
    renderer.render(painter, widget_rect, cb_rect,
                    Qt.Orientation.Vertical, TickPosition.END, ticks)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final

from PyQt6.QtCore import QPointF, QRect, QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QFontMetricsF, QImage, QPainter, QPen

__all__ = ["TickPosition", "TickConfig", "TickData", "TickRenderer"]

# ---------------------------------------------------------------------------
# Layout constants
# ---------------------------------------------------------------------------

# Extra physical pixels added to each image axis to prevent antialiased
# glyphs from being clipped at the buffer edge.
_IMAGE_EDGE_PADDING: Final[int] = 4

# Width of the invisible bounding rectangle passed to QPainter.drawText.
# Large enough to accommodate any realistic tick label; alignment flags
# position the text within it.
_LABEL_RECT_WIDTH: Final[int] = 200

# Minimum pixel clearance between a clamped label and the widget boundary.
_LABEL_EDGE_MARGIN: Final[int] = 2


# ---------------------------------------------------------------------------
# Public data types
# ---------------------------------------------------------------------------

class TickPosition(Enum):
    """
    Which side of the colorbar gradient the ticks protrude from.

    +-------------+-----------------+-------------------+
    | Orientation | ``START``       | ``END``           |
    +=============+=================+===================+
    | Vertical    | Left of bar     | Right of bar      |
    +-------------+-----------------+-------------------+
    | Horizontal  | Above bar       | Below bar         |
    +-------------+-----------------+-------------------+
    """
    START = auto()
    END   = auto()


@dataclass(frozen=True, slots=True)
class TickConfig:
    """
    Immutable configuration for tick appearance.

    Parameters
    ----------
    count:
        Total number of ticks including both endpoints.  Must be ≥ 1.
    length:
        Tick line length in logical (device-independent) pixels.
        Must be > 0.
    spacing:
        Gap between the end of the tick line and the nearest edge of the
        label bounding box, in logical pixels.
    font_family:
        Font family name passed to :class:`QFont`.
    font_size:
        Point size for tick labels.
    color:
        Any string accepted by :class:`QColor` — named colours, hex codes,
        or stylesheet expressions such as ``"palette(text)"``.

    Raises
    ------
    ValueError
        If ``count < 1`` or ``length <= 0``.
    """

    count:       int   = 5
    length:      float = 5.0
    spacing:     float = 3.0
    font_family: str   = "Arial"
    font_size:   int   = 9
    color:       str   = "palette(text)"

    def __post_init__(self) -> None:
        if self.count < 1:
            raise ValueError(f"TickConfig.count must be ≥ 1; got {self.count}")
        if self.length <= 0:
            raise ValueError(
                f"TickConfig.length must be > 0; got {self.length}"
            )


@dataclass(slots=True)
class TickData:
    """
    Computed data for a single tick mark.

    Attributes
    ----------
    position:
        Normalised position along the colorbar in ``[0.0, 1.0]``.
        0.0 is the low end (left / bottom), 1.0 is the high end (right / top).
    value:
        The data value at this tick (used only for display, not layout).
    label:
        Pre-formatted string to render next to the tick line.
    is_endpoint:
        ``True`` for the first and last ticks; may be used by renderers to
        apply special styling or clamping logic.
    """

    position:    float
    value:       float
    label:       str
    is_endpoint: bool


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class TickRenderer:
    """
    Rasterises tick marks and labels into a :class:`QImage` and blits the
    result to a :class:`QPainter` target.

    The two-pass approach (off-screen image → blit) sidesteps macOS OpenGL
    text quality issues.  The image is recreated on every :meth:`render`
    call and is not cached — the colorbar is static most of the time so the
    allocation cost is negligible compared with the GPU frame budget.

    Parameters
    ----------
    config:
        Immutable tick appearance settings.
    vmin:
        Data value mapped to position 0.0 (low end of the gradient).
    vmax:
        Data value mapped to position 1.0 (high end of the gradient).
    """

    def __init__(self, config: TickConfig, vmin: float, vmax: float) -> None:
        self.config = config
        self.vmin   = vmin
        self.vmax   = vmax

        self._font = QFont(config.font_family, config.font_size)
        self._font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
        self._fm   = QFontMetricsF(self._font)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_ticks(self) -> list[TickData]:
        """
        Generate evenly-spaced :class:`TickData` records between
        :attr:`vmin` and :attr:`vmax`.

        Returns an empty list when ``config.count < 2`` (a single tick
        cannot define a meaningful interval).

        The label format is chosen by the magnitude of the data range:

        +---------------------+----------+
        | Range               | Format   |
        +=====================+==========+
        | 0                   | ``:.2f`` |
        +---------------------+----------+
        | > 1 000 or < 0.001  | ``:.1e`` |
        +---------------------+----------+
        | < 1                 | ``:.3f`` |
        +---------------------+----------+
        | otherwise           | ``:.2f`` |
        +---------------------+----------+

        Returns
        -------
        list[TickData]
            Records in ascending position order (0.0 → 1.0).
        """
        count = self.config.count
        if count < 2:
            return []

        data_range = abs(self.vmax - self.vmin)

        match data_range:
            case 0:
                fmt = ".2f"
            case r if r > 1_000 or r < 0.001:
                fmt = ".1e"
            case r if r < 1:
                fmt = ".3f"
            case _:
                fmt = ".2f"

        ticks: list[TickData] = []
        for i in range(count):
            pos = i / (count - 1)
            val = self.vmin + pos * (self.vmax - self.vmin)
            ticks.append(
                TickData(
                    position=pos,
                    value=val,
                    label=format(val, fmt),
                    is_endpoint=(i == 0 or i == count - 1),
                )
            )
        return ticks

    def render(
        self,
        target_painter: QPainter,
        widget_rect: QRect,
        cb_rect: QRectF,
        orientation: Qt.Orientation,
        position: TickPosition,
        ticks: list[TickData],
    ) -> None:
        """
        Rasterise ``ticks`` and blit them to ``target_painter``.

        The method allocates a temporary :class:`QImage` sized to
        ``widget_rect`` (scaled by the device pixel ratio), draws all ticks
        into it, and then blits the result at the origin of
        ``target_painter``.  The image is transparent everywhere no tick or
        label is drawn.

        Parameters
        ----------
        target_painter:
            Active :class:`QPainter` for the widget surface.  Must remain
            active for the duration of this call.
        widget_rect:
            Full bounding rectangle of the widget in logical pixels.
            Used to size the off-screen buffer and to clamp label positions.
        cb_rect:
            Bounding rectangle of the gradient strip in logical pixels.
            Tick lines are anchored to its edges.
        orientation:
            Whether the gradient runs horizontally or vertically.
        position:
            Which side of the gradient strip the ticks protrude from.
        ticks:
            Pre-computed tick data from :meth:`compute_ticks`.
        """
        if not ticks:
            return

        dpr = target_painter.device().devicePixelRatio()

        phys_w = math.ceil(widget_rect.width()  * dpr) + _IMAGE_EDGE_PADDING
        phys_h = math.ceil(widget_rect.height() * dpr) + _IMAGE_EDGE_PADDING

        if phys_w < 1 or phys_h < 1:
            return

        image = QImage(phys_w, phys_h, QImage.Format.Format_ARGB32_Premultiplied)
        image.setDevicePixelRatio(dpr)
        image.fill(0)  # Fully transparent

        img_painter = QPainter(image)
        try:
            self._setup_painter(img_painter)
            for tick in ticks:
                match orientation:
                    case Qt.Orientation.Vertical:
                        self._draw_vert(img_painter, widget_rect, cb_rect, position, tick)
                    case _:
                        self._draw_horz(img_painter, widget_rect, cb_rect, position, tick)
        finally:
            img_painter.end()

        target_painter.drawImage(0, 0, image)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _setup_painter(self, painter: QPainter) -> None:
        """
        Apply shared render hints, font, and pen to ``painter``.

        Antialiasing is disabled for geometry (keeps tick lines pixel-sharp)
        but enabled for text (subpixel hinting via Qt's rasteriser).
        """
        painter.setRenderHint(QPainter.RenderHint.Antialiasing,     False)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing,  True)
        painter.setFont(self._font)

        pen = QPen(QColor(self.config.color))
        pen.setWidthF(1.0)
        painter.setPen(pen)

    def _draw_vert(
        self,
        painter: QPainter,
        widget_rect: QRect,
        cb_rect: QRectF,
        pos: TickPosition,
        tick: TickData,
    ) -> None:
        """
        Draw one tick line and its label for a **vertical** colorbar.

        Position 1.0 maps to the top edge of ``cb_rect``; 0.0 maps to the
        bottom.  The half-pixel offset (``+ 0.5``) aligns the line to a
        physical pixel centre, preventing double-width blur on integer
        boundaries.

        Label Y is clamped so no glyph falls outside ``widget_rect``.
        """
        # Map normalised position to a physical Y, snapped to pixel centre.
        available_h = cb_rect.height() - 1
        exact_y     = cb_rect.top() + (1.0 - tick.position) * available_h
        line_y      = round(exact_y) + 0.5

        tick_len = self.config.length
        spacing  = self.config.spacing

        match pos:
            case TickPosition.END:
                p1      = QPointF(cb_rect.right(), line_y)
                p2      = QPointF(cb_rect.right() + tick_len, line_y)
                text_x  = p2.x() + spacing
                h_align = Qt.AlignmentFlag.AlignLeft
            case _:  # START
                p1      = QPointF(cb_rect.left(), line_y)
                p2      = QPointF(cb_rect.left() - tick_len, line_y)
                text_x  = p2.x() - spacing
                h_align = Qt.AlignmentFlag.AlignRight

        painter.drawLine(p1, p2)

        txt_h  = self._fm.boundingRect(tick.label).height()
        text_y = int(round(line_y - txt_h / 2.0))

        # Clamp so the label stays within the widget boundary.
        bottom_limit = widget_rect.height() - int(txt_h) - _LABEL_EDGE_MARGIN
        text_y       = max(0, min(text_y, bottom_limit))

        align = h_align | Qt.AlignmentFlag.AlignVCenter

        match pos:
            case TickPosition.END:
                label_rect = QRect(int(text_x), text_y, _LABEL_RECT_WIDTH, int(txt_h + 4))
            case _:
                label_rect = QRect(int(text_x) - _LABEL_RECT_WIDTH, text_y, _LABEL_RECT_WIDTH, int(txt_h + 4))

        painter.drawText(label_rect, align, tick.label)

    def _draw_horz(
        self,
        painter: QPainter,
        widget_rect: QRect,
        cb_rect: QRectF,
        pos: TickPosition,
        tick: TickData,
    ) -> None:
        """
        Draw one tick line and its label for a **horizontal** colorbar.

        Position 0.0 maps to the left edge of ``cb_rect``; 1.0 maps to
        the right.  Label X is clamped so no glyph overflows the widget.
        """
        available_w = cb_rect.width() - 1
        exact_x     = cb_rect.left() + tick.position * available_w
        line_x      = round(exact_x) + 0.5

        tick_len = self.config.length
        spacing  = self.config.spacing

        match pos:
            case TickPosition.END:
                p1     = QPointF(line_x, cb_rect.bottom())
                p2     = QPointF(line_x, cb_rect.bottom() + tick_len)
                text_y = p2.y() + spacing
            case _:  # START
                p1     = QPointF(line_x, cb_rect.top())
                p2     = QPointF(line_x, cb_rect.top() - tick_len)
                text_y = p2.y() - spacing - self._fm.height()

        painter.drawLine(p1, p2)

        txt_w  = self._fm.horizontalAdvance(tick.label)
        txt_h  = self._fm.height()
        text_x = int(round(line_x - txt_w / 2.0))

        # Clamp so the label stays within the widget boundary.
        right_limit = widget_rect.width() - int(txt_w) - _LABEL_EDGE_MARGIN
        text_x      = max(0, min(text_x, right_limit))

        # Extra width in the bounding rect guards against glyph clipping on
        # fonts that report a narrower advance than their ink bounds.
        label_rect = QRect(text_x, int(text_y), int(txt_w + 10), int(txt_h + 4))
        painter.drawText(
            label_rect,
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop,
            tick.label,
        )

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"<TickRenderer count={self.config.count} "
            f"vmin={self.vmin} vmax={self.vmax}>"
        )
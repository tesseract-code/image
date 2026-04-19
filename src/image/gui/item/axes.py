import numpy as np
from PyQt6.QtCore import QSize
from PyQt6.QtGui import QPainter, QFontMetrics, QPainterPath
from PyQt6.QtWidgets import QWidget, QSizePolicy


class HorizontalAxis(QWidget):
    """
    High-performance horizontal axis with numpy optimizations and caching.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._min = 0.0
        self._max = 100.0
        self._tick_count = 5
        self._label_precision = 2
        self._flipped = False

        # Cache
        self._cached_ticks = None
        self._cached_labels = None
        self._cached_size_hint = None
        self._cache_valid = False

        self.setSizePolicy(QSizePolicy.Policy.Expanding,
                           QSizePolicy.Policy.Fixed)

    def setRange(self, min_val, max_val):
        if self._min != min_val or self._max != max_val:
            self._min = min_val
            self._max = max_val
            self._invalidate_cache()

    def setTickCount(self, count):
        count = max(2, count)
        if self._tick_count != count:
            self._tick_count = count
            self._invalidate_cache()

    def setLabelPrecision(self, precision):
        if self._label_precision != precision:
            self._label_precision = precision
            self._invalidate_cache()

    def setFlipped(self, flipped):
        if self._flipped != flipped:
            self._flipped = flipped
            self.update()

    def _invalidate_cache(self):
        self._cache_valid = False
        self._cached_ticks = None
        self._cached_labels = None
        self._cached_size_hint = None
        self.updateGeometry()
        self.update()

    def _ensure_cache(self):
        """Compute and cache tick positions and labels once."""
        if self._cache_valid:
            return

        # Use numpy for vectorized tick calculation
        if self._tick_count <= 1 or self._min == self._max:
            self._cached_ticks = np.array(
                [self._min, self._max] if self._tick_count >= 2 else [
                    self._min])
        else:
            self._cached_ticks = np.linspace(self._min, self._max,
                                             self._tick_count)

        # Pre-format all labels
        self._cached_labels = [f"{val:.{self._label_precision}f}" for val in
                               self._cached_ticks]

        self._cache_valid = True

    def sizeHint(self):
        if self._cached_size_hint is not None:
            return self._cached_size_hint

        self._ensure_cache()
        font_metrics = QFontMetrics(self.font())

        # Vectorized height calculation
        max_height = max((font_metrics.boundingRect(label).height()
                          for label in self._cached_labels), default=0)

        required_height = 1 + 5 + max_height + 5
        self._cached_size_hint = QSize(100, required_height)
        return self._cached_size_hint

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing,
                              False)  # Disable AA for speed

        self._ensure_cache()

        width = self.width()
        axis_y = 0

        # Draw axis line
        painter.drawLine(0, axis_y, width, axis_y)

        # Vectorized position calculation
        if self._flipped:
            x_positions = width - (self._cached_ticks - self._min) / (
                    self._max - self._min) * width
        else:
            x_positions = (self._cached_ticks - self._min) / (
                    self._max - self._min) * width

        # Batch draw ticks using QPainterPath
        path = QPainterPath()
        tick_bottom = axis_y + 5

        for x_pos in x_positions:
            path.moveTo(x_pos, axis_y)
            path.lineTo(x_pos, tick_bottom)

        painter.drawPath(path)

        # Draw labels
        font_metrics = painter.fontMetrics()
        max_width = width

        for x_pos, label in zip(x_positions, self._cached_labels):
            label_rect = font_metrics.boundingRect(label)
            label_width = label_rect.width()

            # Fast clamping
            label_x = x_pos - label_width * 0.5
            if label_x < 0:
                label_x = 0
            elif label_x + label_width > max_width:
                label_x = max_width - label_width

            label_y = tick_bottom + label_rect.height()
            painter.drawText(int(label_x), int(label_y), label)


class VerticalAxis(QWidget):
    """
    High-performance vertical axis with numpy optimizations and caching.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._min = 0.0
        self._max = 100.0
        self._tick_count = 5
        self._label_precision = 2
        self._flipped = False

        # Cache
        self._cached_ticks = None
        self._cached_labels = None
        self._cached_size_hint = None
        self._cache_valid = False

        self.setSizePolicy(QSizePolicy.Policy.Fixed,
                           QSizePolicy.Policy.Expanding)

    def setRange(self, min_val, max_val):
        if self._min != min_val or self._max != max_val:
            self._min = min_val
            self._max = max_val
            self._invalidate_cache()

    def setTickCount(self, count):
        count = max(2, count)
        if self._tick_count != count:
            self._tick_count = count
            self._invalidate_cache()

    def setLabelPrecision(self, precision):
        if self._label_precision != precision:
            self._label_precision = precision
            self._invalidate_cache()

    def setFlipped(self, flipped):
        if self._flipped != flipped:
            self._flipped = flipped
            self.update()

    def _invalidate_cache(self):
        self._cache_valid = False
        self._cached_ticks = None
        self._cached_labels = None
        self._cached_size_hint = None
        self.updateGeometry()
        self.update()

    def _ensure_cache(self):
        """Compute and cache tick positions and labels once."""
        if self._cache_valid:
            return

        # Use numpy for vectorized tick calculation
        if self._tick_count <= 1 or self._min == self._max:
            self._cached_ticks = np.array(
                [self._min, self._max] if self._tick_count >= 2 else [
                    self._min])
        else:
            self._cached_ticks = np.linspace(self._min, self._max,
                                             self._tick_count)

        # Pre-format all labels
        self._cached_labels = [f"{val:.{self._label_precision}f}" for val in
                               self._cached_ticks]

        self._cache_valid = True

    def sizeHint(self):
        if self._cached_size_hint is not None:
            return self._cached_size_hint

        self._ensure_cache()
        font_metrics = QFontMetrics(self.font())

        # Vectorized width calculation
        max_width = max((font_metrics.boundingRect(label).width()
                         for label in self._cached_labels), default=0)

        required_width = max_width + 5 + 5
        self._cached_size_hint = QSize(required_width, 100)
        return self._cached_size_hint

    def minimumSizeHint(self):
        return self.sizeHint()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing,
                              False)  # Disable AA for speed

        self._ensure_cache()

        width = self.width()
        height = self.height()
        axis_x = width - 1

        # Draw axis line
        painter.drawLine(axis_x, 0, axis_x, height)

        # Vectorized position calculation
        if self._flipped:
            y_positions = height - (self._cached_ticks - self._min) / (
                    self._max - self._min) * height
        else:
            y_positions = (self._cached_ticks - self._min) / (
                    self._max - self._min) * height

        # Batch draw ticks using QPainterPath
        path = QPainterPath()
        tick_left = axis_x - 5

        for y_pos in y_positions:
            path.moveTo(tick_left, y_pos)
            path.lineTo(axis_x, y_pos)

        painter.drawPath(path)

        # Draw labels
        font_metrics = painter.fontMetrics()

        for y_pos, label in zip(y_positions, self._cached_labels):
            label_rect = font_metrics.boundingRect(label)
            label_x = max(0, axis_x - 10 - label_rect.width())
            label_y = y_pos + label_rect.height() * 0.33

            # Fast vertical clamping
            label_height = label_rect.height()
            label_top = label_y - label_height

            if label_top < 0:
                label_y = label_height
            elif label_y > height:
                label_y = height

            painter.drawText(int(label_x), int(label_y), label)

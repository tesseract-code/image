"""
widget.py
=========
OpenGL-accelerated colorbar widget with software-rendered tick labels.

Architecture
------------
The widget is split into two rendering passes that share the same
:class:`QPainter` target:

1. **GL pass** (``beginNativePainting`` … ``endNativePainting``)
   :class:`GradientRenderer` draws the LUT gradient strip via a fullscreen
   quad on an OpenGL 4.1 Core context.  Qt suspends its own GL state around
   this block so native calls are safe.

2. **Software pass** (plain QPainter)
   :class:`TickRenderer` rasterises tick lines and labels into a transparent
   :class:`QImage` and blits it over the GL output.  This sidesteps macOS
   OpenGL text quality issues entirely.

Configuration is immutable (:class:`ColorBarConfig`).  Every public setter
rebuilds the frozen dataclass so there is a single, auditable source of truth
for widget state.  ``update()`` is called by every setter; callers never need
to trigger repaints manually.

OpenGL requirements
-------------------
* Version ≥ 4.1 Core (compatible with macOS default Metal translation layer)
* ``GL_BLEND`` with standard alpha compositing
* No depth test, no face culling (2-D quad only)

Lifecycle note
--------------
``__del__`` intentionally does **nothing**: Qt destroys the GL context before
the Python garbage collector runs, so any GL call inside ``__del__`` will
segfault.  Call :meth:`cleanup` explicitly (or connect it to the
``aboutToQuit`` signal) to release GPU resources safely.

Typical usage::

    bar = ColorbarWidget()
    bar.set_colormap("plasma")
    bar.set_range(vmin=-1.0, vmax=1.0)
    bar.set_orientation(Qt.Orientation.Horizontal)
    layout.addWidget(bar)

    # On application exit:
    bar.cleanup()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, replace
from typing import Final

import numpy as np
from PyQt6.QtCore import Qt, QRectF, QSize
from PyQt6.QtGui import QFont, QFontMetricsF, QPainter, QSurfaceFormat
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QSizePolicy, QWidget

from image.gl.backend import GL
from image.gl.colorbar.gradient import GradientRenderer
from image.gl.colorbar.tick import (
    TickConfig,
    TickPosition,
    TickRenderer,
)
from image.gl.errors import GLError, gl_error_check
from image.gl.types import (GLbitfield, GLenum, GLfloat,
                            GLint)
from image.model.cmap import ColormapModel
from pycore.log.ctx import ContextAdapter

__all__ = ["ColorBarConfig", "ColorbarWidget"]

logger = ContextAdapter(logging.getLogger(__name__), {})

# ---------------------------------------------------------------------------
# Layout / sizing constants
# ---------------------------------------------------------------------------

# Minimum gradient strip width / height in logical pixels.
_MIN_BAR_WIDTH: Final[int] = 10

# Minimum tick count enforced by set_tick_count().
_MIN_TICK_COUNT: Final[int] = 2

# Fallback size hint dimensions when no ticks are available.
_FALLBACK_SIZE_HINT: Final[int] = 20

# Padding added to the long axis of sizeHint beyond the measured label size.
_SIZE_HINT_PADDING: Final[int] = 10

# Default long-axis size hint (the axis the caller is expected to stretch).
_DEFAULT_LONG_AXIS: Final[int] = 200

# OpenGL version requested (4.1 Core — highest available on macOS).
_GL_VERSION: Final[tuple[int, int]] = (4, 1)
_GL_MSAA_SAMPLES: Final[int] = 4
_GL_ALPHA_BITS: Final[int] = 8


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class ColorBarConfig:
    """
    Immutable snapshot of all colorbar layout and appearance settings.

    Being frozen means every mutation goes through a ``dataclasses.replace``
    call, which produces a new instance.  This makes state transitions
    explicit and avoids partial-update bugs.

    Parameters
    ----------
    width:
        Thickness of the gradient strip in logical pixels (the axis
        perpendicular to the gradient direction).  Must be > 0.
    orientation:
        ``Vertical``  → gradient runs top-to-bottom.
        ``Horizontal`` → gradient runs left-to-right.
    tick_position:
        Which side of the strip ticks protrude from.
        See :class:`TickPosition` for the per-orientation meaning.
    tick_config:
        Appearance settings for tick marks and labels.

    Raises
    ------
    ValueError
        If ``width ≤ 0``.
    """

    width: int = 30
    orientation: Qt.Orientation = Qt.Orientation.Vertical
    tick_position: TickPosition = TickPosition.END
    tick_config: TickConfig = field(default_factory=TickConfig)

    def __post_init__(self) -> None:
        if self.width <= 0:
            raise ValueError(
                f"ColorBarConfig.width must be > 0; got {self.width}"
            )


# ---------------------------------------------------------------------------
# Widget
# ---------------------------------------------------------------------------

class ColorbarWidget(QOpenGLWidget):
    """
    A hardware-accelerated colorbar with pixel-perfect software tick labels.

    The gradient is rendered via OpenGL; tick marks and labels are composited
    on top using a :class:`QPainter` / :class:`QImage` proxy to guarantee
    legible, hinted text on every platform.

    Parameters
    ----------
    parent:
        Optional Qt parent widget.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        # Immutable config — replaced wholesale on every setter call.
        self._config = ColorBarConfig()

        # Data range for tick label formatting.
        self._vmin: float = 0.0
        self._vmax: float = 1.0

        # Sub-systems — constructed here; GL resources allocated in initializeGL.
        self._cmap_cache = ColormapModel()
        self._gradient_renderer = GradientRenderer(self._cmap_cache)

        self._setup_widget()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_widget(self) -> None:
        """
        Configure widget attributes and request an OpenGL 4.1 Core context.

        ``WA_AlwaysStackOnTop`` ensures the colorbar renders above sibling
        widgets that share the same window.  ``WA_TranslucentBackground``
        allows the transparent GL clear colour to show through the Qt
        compositor.
        """
        self._update_size_policy()
        self.setAttribute(Qt.WidgetAttribute.WA_AlwaysStackOnTop)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        fmt = QSurfaceFormat.defaultFormat()
        fmt.setVersion(*_GL_VERSION)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setSamples(_GL_MSAA_SAMPLES)
        fmt.setAlphaBufferSize(_GL_ALPHA_BITS)
        self.setFormat(fmt)

    def _update_size_policy(self) -> None:
        """
        Set the size policy so the bar expands along its gradient axis.

        +-------------+----------+------------------+
        | Orientation | Fixed    | Expanding        |
        +=============+==========+==================+
        | Vertical    | Width    | Height           |
        +-------------+----------+------------------+
        | Horizontal  | Height   | Width            |
        +-------------+----------+------------------+
        """
        match self._config.orientation:
            case Qt.Orientation.Vertical:
                self.setSizePolicy(
                    QSizePolicy.Policy.Fixed,
                    QSizePolicy.Policy.MinimumExpanding,
                )
            case _:
                self.setSizePolicy(
                    QSizePolicy.Policy.MinimumExpanding,
                    QSizePolicy.Policy.Fixed,
                )

    # ------------------------------------------------------------------
    # GL lifecycle
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:
        """
        Compile shaders and allocate GPU resources.

        Called once by Qt after the GL context becomes current.  All
        subsequent GL work happens in :meth:`paintGL`.

        Raises
        ------
        GLInitializationError
            Propagated from :meth:`GradientRenderer.initialize` if shader
            compilation or resource allocation fails.
        """
        logger.debug("ColorbarWidget: initializeGL")
        self._gradient_renderer.initialize()

    def resizeGL(self, w: int, h: int) -> None:
        """
        Handle viewport resize events.

        The viewport is set explicitly in :meth:`paintGL` (accounting for
        the device pixel ratio), so no work is needed here.  The override
        exists to document that omission intentionally.
        """

    def paintGL(self) -> None:
        """
        Execute the two-pass rendering pipeline.

        Pass 1 — OpenGL gradient (native painting block):
            Sets blend state, clears to transparent, and delegates to
            :class:`GradientRenderer`.

        Pass 2 — Software ticks (QPainter):
            Creates a :class:`TickRenderer`, computes tick positions, and
            blits them over the GL output.

        Any GL error raised by the gradient renderer is caught, logged, and
        swallowed so a transient GPU hiccup does not crash the widget.
        """
        painter = QPainter(self)

        # ---- Pass 1: OpenGL gradient ----------------------------------------
        painter.beginNativePainting()
        try:
            self._paint_gl_gradient()
        except GLError:
            logger.exception("ColorbarWidget: GL error during gradient render")
        finally:
            painter.endNativePainting()

        # ---- Pass 2: software ticks -----------------------------------------
        self._paint_ticks(painter)

        painter.end()

    def _paint_gl_gradient(self) -> None:
        """
        Configure GL state and delegate to :class:`GradientRenderer`.

        The viewport is set in *physical* pixels (logical size × DPR) so
        the gradient fills the widget exactly on high-DPI displays.
        """
        dpr = self.devicePixelRatio()
        viewport_w = GLint(int(self.width() * dpr))
        viewport_h = GLint(int(self.height() * dpr))

        GL.glViewport(0, 0, viewport_w, viewport_h)

        # Standard alpha compositing.
        with gl_error_check("ColorbarWidget blend state", GLError):
            GL.glEnable(GLenum(GL.GL_BLEND))
            GL.glBlendFunc(
                GLenum(GL.GL_SRC_ALPHA),
                GLenum(GL.GL_ONE_MINUS_SRC_ALPHA),
            )
            GL.glDisable(GLenum(GL.GL_DEPTH_TEST))
            GL.glDisable(GLenum(GL.GL_CULL_FACE))

        # Clear to fully transparent so Qt can composite the ticks on top.
        with gl_error_check("ColorbarWidget clear", GLError):
            GL.glClearColor(
                GLfloat(0.0), GLfloat(0.0), GLfloat(0.0), GLfloat(0.0)
            )
            GL.glClear(GLbitfield(GL.GL_COLOR_BUFFER_BIT))

        self._gradient_renderer.render(
            self._compute_layout_rect(),
            (self.width(), self.height()),
            self._config.orientation,
        )

    def _paint_ticks(self, painter: QPainter) -> None:
        """
        Build a :class:`TickRenderer`, compute tick data, and blit to ``painter``.

        A fresh :class:`TickRenderer` is constructed each frame because
        ``vmin`` / ``vmax`` may change between frames and the renderer is
        lightweight (no GPU resources).
        """
        tick_renderer = TickRenderer(
            self._config.tick_config, self._vmin, self._vmax
        )
        ticks = tick_renderer.compute_ticks()
        tick_renderer.render(
            painter,
            self.rect(),
            self._compute_layout_rect(),
            self._config.orientation,
            self._config.tick_position,
            ticks,
        )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _compute_layout_rect(self) -> QRectF:
        """
        Return the bounding rectangle of the gradient strip in logical pixels.

        The strip is ``config.width`` pixels thick and is positioned on the
        opposite side of the widget from the ticks:

        +-------------+------------------+-------------------+
        | Orientation | ``END`` ticks    | ``START`` ticks   |
        +=============+==================+===================+
        | Vertical    | strip on left    | strip on right    |
        +-------------+------------------+-------------------+
        | Horizontal  | strip on top     | strip on bottom   |
        +-------------+------------------+-------------------+
        """
        w, h = float(self.width()), float(self.height())
        bar_w = float(self._config.width)

        match self._config.orientation, self._config.tick_position:
            case Qt.Orientation.Vertical, TickPosition.END:
                return QRectF(0, 0, bar_w, h)
            case Qt.Orientation.Vertical, _:
                return QRectF(w - bar_w, 0, bar_w, h)
            case _, TickPosition.END:
                return QRectF(0, 0, w, bar_w)
            case _:
                return QRectF(0, h - bar_w, w, bar_w)

    def sizeHint(self) -> QSize:
        """
        Return a size hint that accommodates the gradient strip, ticks, and labels.

        The *short* axis is computed from the strip width plus the tick
        overhang plus the widest/tallest label.  The *long* axis defaults to
        :data:`_DEFAULT_LONG_AXIS` (the caller's layout manager is expected
        to stretch it).
        """
        renderer = TickRenderer(self._config.tick_config, self._vmin,
                                self._vmax)
        ticks = renderer.compute_ticks()

        if not ticks:
            max_label_w = float(_FALLBACK_SIZE_HINT)
            max_label_h = float(_FALLBACK_SIZE_HINT)
        else:
            fm = QFontMetricsF(
                QFont(
                    self._config.tick_config.font_family,
                    self._config.tick_config.font_size,
                )
            )
            max_label_w = max(fm.horizontalAdvance(t.label) for t in ticks)
            max_label_h = fm.height()

        tick_overhang = (
                self._config.tick_config.length + self._config.tick_config.spacing
        )

        match self._config.orientation:
            case Qt.Orientation.Vertical:
                short_axis = int(
                    self._config.width + tick_overhang + max_label_w + _SIZE_HINT_PADDING
                )
                return QSize(short_axis, _DEFAULT_LONG_AXIS)
            case _:
                short_axis = int(
                    self._config.width + tick_overhang + max_label_h + _SIZE_HINT_PADDING
                )
                return QSize(_DEFAULT_LONG_AXIS, short_axis)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """
        Release all GPU resources held by subcomponents.

        Must be called explicitly before the application exits — connect it
        to ``QApplication.aboutToQuit`` or call it from the parent widget's
        ``closeEvent``.  Never call from ``__del__``: the GL context is
        guaranteed to be dead by the time the garbage collector runs.
        """
        logger.debug("ColorbarWidget: cleanup")
        self.makeCurrent()
        try:
            self._gradient_renderer.cleanup()
        finally:
            self._cmap_cache.clear()
            self.doneCurrent()

    def __del__(self) -> None:
        # Intentionally empty.  See module docstring "Lifecycle note".
        pass

    # ------------------------------------------------------------------
    # Public API — colormap
    # ------------------------------------------------------------------

    def set_colormap(self, name: str, *, reverse: bool = False) -> None:
        """
        Switch to a named colormap.

        Parameters
        ----------
        name:
            Key recognised by :class:`ColormapModel`.
        reverse:
            When ``True`` the gradient runs high-to-low.
        """
        self._gradient_renderer.set_colormap(name, reverse=reverse)
        self.update()

    def set_colormap_data(self, data: np.ndarray) -> None:
        """
        Override the colormap with a raw RGB array.

        Parameters
        ----------
        data:
            ``(N, 3)`` or ``(N, 4)`` array of ``uint8`` or ``float32``
            values.  See :meth:`GradientRenderer.set_colormap_data` for
            full validation rules.
        """
        self._gradient_renderer.set_colormap_data(data)
        self.update()

    # ------------------------------------------------------------------
    # Public API — data range
    # ------------------------------------------------------------------

    def set_range(self, vmin: float, vmax: float) -> None:
        """
        Set the data range mapped to the low and high ends of the colormap.

        Parameters
        ----------
        vmin:
            Value at the low end (position 0.0) of the gradient.
        vmax:
            Value at the high end (position 1.0) of the gradient.
        """
        self._vmin, self._vmax = vmin, vmax
        self.update()

    # ------------------------------------------------------------------
    # Public API — layout
    # ------------------------------------------------------------------

    def set_orientation(self, orientation: Qt.Orientation) -> None:
        """
        Switch between horizontal and vertical gradient orientation.

        Triggers a geometry update so the parent layout recalculates its
        allocation.
        """
        self._config = replace(self._config, orientation=orientation)
        self._update_size_policy()
        self.updateGeometry()

    def set_colorbar_width(self, width: int) -> None:
        """
        Set the thickness of the gradient strip in logical pixels.

        Values below :data:`_MIN_BAR_WIDTH` are silently clamped.
        """
        self._config = replace(self._config, width=max(_MIN_BAR_WIDTH, width))
        self.updateGeometry()

    # ------------------------------------------------------------------
    # Public API — ticks
    # ------------------------------------------------------------------

    def set_tick_position(self, position: TickPosition) -> None:
        """
        Move ticks to the opposite side of the gradient strip.

        See :class:`TickPosition` for the per-orientation interpretation.
        """
        self._config = replace(self._config, tick_position=position)
        self.update()

    def set_tick_count(self, count: int) -> None:
        """
        Set the number of tick marks (including both endpoints).

        Values below :data:`_MIN_TICK_COUNT` are silently clamped.
        """
        self._update_tick_config(count=max(_MIN_TICK_COUNT, count))
        self.update()

    def _update_tick_config(self, **kwargs: object) -> None:
        """
        Rebuild :class:`TickConfig` from the current config plus ``kwargs``.

        Uses ``dataclasses.replace`` so all unspecified fields are preserved
        without manual forwarding of every attribute.

        Parameters
        ----------
        **kwargs:
            Any subset of :class:`TickConfig` field names and their new values.
        """
        new_tick_cfg = replace(self._config.tick_config, **kwargs)
        self._config = replace(self._config, tick_config=new_tick_cfg)

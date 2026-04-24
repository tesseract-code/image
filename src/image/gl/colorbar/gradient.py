"""
gradient.py
===========
Renderer for the colorbar gradient strip.

``GradientRenderer`` is the single public entry-point for the colorbar
rendering pipeline.  It owns and orchestrates three lower-level GL objects:

* :class:`~quad.QuadGeometry`     — VAO/VBO for the screen-aligned quad
* :class:`~texture.Texture2D`     — 1-D LUT texture (1 × N RGBA strip)
* :class:`~program.ShaderProgramManager` — compiled GLSL shader pair

Coordinate conventions
----------------------
All geometry is expressed in **Normalised Device Coordinates** (NDC),
where (-1, -1) is the bottom-left corner of the viewport and (1, 1) is
the top-right.  Qt widget coordinates (origin top-left, y increases
downward) are converted at the last moment in :meth:`_compute_vertices`.

Texture mapping
---------------
The LUT texture is a 1-D strip sampled along the **U** axis only;
the V component is always 0.5 (the texel centre of a height-1 texture).

+-------------+----------------------------------+
| Orientation | Gradient direction               |
+=============+==================================+
| Horizontal  | left (U=0.0) → right (U=1.0)    |
+-------------+----------------------------------+
| Vertical    | top  (U=1.0) → bottom (U=0.0)   |
+-------------+----------------------------------+

Typical usage::

    renderer = GradientRenderer(colormap_cache)
    renderer.initialize()
    renderer.set_colormap("plasma", reverse=False)

    # Inside paintGL:
    renderer.render(rect, canvas_size=(w, h), orientation=Qt.Orientation.Horizontal)

    # Teardown:
    renderer.cleanup()
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from PyQt6.QtCore import QRectF, Qt

from image.gl.backend import initialize_context
from image.gl.colorbar.quad import QuadGeometry
from image.gl.colorbar.texture import Texture1D
from image.gl.errors import (
    GLError,
    GLInitializationError,
    GLUploadError,
    gl_error_check,
)
from image.gl.program import ShaderProgramManager
from image.gl.shaders import SHADERS
from image.gl.types import GLTexture
from image.model.cmap import ColormapModel

if TYPE_CHECKING:
    pass  # Reserved for forward-reference type aliases if needed.

__all__ = ["GradientRenderer"]

logger = logging.getLogger(__name__)

# Sentinel for the fixed V-coordinate used when sampling a 1-D LUT strip.
_LUT_V_CENTRE: float = 0.5


class GradientRenderer:
    """
    Orchestrates the texture, geometry, and shader for the colorbar gradient.

    The renderer is intentionally stateful: changes to the colormap or its
    data set a ``_texture_dirty`` flag, deferring the GPU upload to the next
    :meth:`render` call.  This avoids redundant uploads when multiple
    properties change between frames.

    Parameters
    ----------
    cache:
        A :class:`ColormapModel` instance used to look up pre-built LUT
        arrays by name.  The renderer holds a reference but never mutates it.

    Attributes
    ----------
    is_initialized : bool
        ``True`` after a successful :meth:`initialize` call.
    """

    def __init__(self, cache: ColormapModel) -> None:
        self._cache = cache

        # GL sub-components — none perform any GL calls at construction time.
        self._shader = ShaderProgramManager()
        self._quad = QuadGeometry()
        self._texture = Texture1D()

        # Dirty flag: upload deferred to the next render() call.
        self._texture_dirty: bool = True

        # Colormap state.
        self._current_cmap: str = "viridis"
        self._reverse: bool = False
        self._manual_data: np.ndarray | None = None

        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Compile shaders and allocate GPU resources for all sub-components.

        Must be called once from the GL thread that owns the current context,
        after the context has been made current.  Calling :meth:`render`
        before this method raises a :class:`GLInitializationError`.

        Raises
        ------
        GLInitializationError
            If shader compilation/linking, VAO/VBO allocation, or texture
            object creation fails.
        """
        if self._initialized:
            return

        logger.debug("GradientRenderer: initializing sub-components")
        initialize_context()

        with gl_error_check("GradientRenderer shader init", GLInitializationError):
            self._shader.initialize(
                vertex_path=SHADERS["colorbar_vertex"],
                fragment_path=SHADERS["colorbar_fragment"],
            )

        self._quad.initialize()    # raises GLInitializationError on failure
        self._texture.initialize() # raises GLInitializationError on failure

        self._initialized = True
        logger.debug("GradientRenderer: ready")

    def cleanup(self) -> None:
        """
        Release all GPU resources held by sub-components.

        Safe to call more than once and safe to call if :meth:`initialize`
        was never called.  Errors during deletion are logged but not raised
        so that cleanup can proceed even when the GL context is degraded.
        """
        if not self._initialized:
            return

        self._initialized = False
        logger.debug("GradientRenderer: releasing GPU resources")

        # Reverse initialisation order: shader → texture → quad.
        for component, name in (
            (self._shader,  "shader"),
            (self._texture, "texture"),
            (self._quad,    "quad"),
        ):
            try:
                component.cleanup()
            except Exception:
                logger.exception(
                    "GradientRenderer: exception while cleaning up %s", name
                )

    # ------------------------------------------------------------------
    # Colormap API
    # ------------------------------------------------------------------

    def set_colormap(self, name: str, *, reverse: bool = False) -> None:
        """
        Switch to a named colormap from the :class:`ColormapModel` cache.

        Clears any manually-supplied data set via :meth:`set_colormap_data`
        and schedules a GPU texture upload on the next :meth:`render` call.

        Parameters
        ----------
        name:
            Key passed to :meth:`ColormapModel.get_lut`.
        reverse:
            When ``True``, the LUT is sampled in reverse order so that the
            gradient runs high-to-low rather than low-to-high.
        """
        if name == self._current_cmap and reverse == self._reverse and self._manual_data is None:
            return  # No-op: state has not changed.

        self._current_cmap = name
        self._reverse = reverse
        self._manual_data = None
        self._texture_dirty = True
        logger.debug(
            "GradientRenderer: colormap set to '%s' (reverse=%s)", name, reverse
        )

    def set_colormap_data(self, rgb_data: np.ndarray) -> None:
        """
        Override the colormap with a raw RGB array, bypassing the cache.

        Useful when the caller has already computed a custom LUT (e.g. from
        a user-defined transfer function) and needs to push it directly.

        Parameters
        ----------
        rgb_data:
            A ``(N, 3)`` or ``(N, 4)`` array.  May be ``uint8`` (values
            0–255) or a floating-point dtype (values 0.0–1.0, which are
            automatically scaled to ``uint8``).  Only the first three
            channels (RGB) are retained; alpha is discarded.

        Raises
        ------
        ValueError
            If ``rgb_data`` is not 2-D or has fewer than 3 columns.
        """
        data = np.asarray(rgb_data)

        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(
                f"set_colormap_data expects a (N, ≥3) array; "
                f"got shape {data.shape}"
            )

        if data.dtype != np.uint8:
            data = (np.clip(data, 0.0, 1.0) * 255).astype(np.uint8)

        # Keep only the RGB channels; store as a contiguous C array so that
        # the subsequent texture upload never needs an implicit copy.
        self._manual_data = np.ascontiguousarray(data[:, :3])
        self._texture_dirty = True
        logger.debug(
            "GradientRenderer: manual LUT set (%d entries)", self._manual_data.shape[0]
        )

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(
        self,
        rect: QRectF,
        canvas_size: tuple[float, float],
        orientation: Qt.Orientation,
    ) -> None:
        """
        Upload any pending texture data and issue the quad draw call.

        This method is the per-frame entry point.  It is a no-op when the
        canvas has zero area, so callers do not need to guard against
        degenerate ``QRectF`` or resize events.

        Parameters
        ----------
        rect:
            The sub-region of the widget (in Qt pixel coordinates, origin
            top-left) that the gradient should fill.
        canvas_size:
            ``(width, height)`` of the full GL viewport in pixels.  Used to
            convert ``rect`` into NDC.
        orientation:
            ``Qt.Orientation.Horizontal`` or ``Qt.Orientation.Vertical``.
            Controls which screen axis maps to the LUT's U coordinate.

        Raises
        ------
        GLInitializationError
            If called before :meth:`initialize`.
        GLUploadError
            If the deferred texture upload fails.
        GLError
            If the draw call itself reports a GL error.
        """
        if not self._initialized:
            raise GLInitializationError(
                "GradientRenderer.render() called before initialize()"
            )

        w, h = canvas_size
        if w <= 0 or h <= 0:
            return

        if self._texture_dirty:
            self._update_texture()

        verts = self._compute_vertices(rect, w, h, orientation)
        self._quad.update_vertices(verts)

        with self._shader:
            self._texture.bind(0)
            self._quad.draw()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _update_texture(self) -> None:
        """
        Resolve the active LUT and push it to the GPU.

        Prefers :attr:`_manual_data` when set; otherwise delegates to the
        :class:`ColormapModel` cache.

        Raises
        ------
        GLUploadError
            Propagated from :meth:`Texture2D.upload_rgb`.
        """
        lut: np.ndarray = (
            self._manual_data
            if self._manual_data is not None
            else self._cache.get_lut(self._current_cmap, self._reverse)
        )

        logger.debug(
            "GradientRenderer: uploading LUT (%d × %d, dtype=%s)",
            lut.shape[0],
            lut.shape[1] if lut.ndim > 1 else 1,
            lut.dtype,
        )

        self._texture.upload_rgb(lut, width=lut.shape[0])
        self._texture_dirty = False

    @staticmethod
    def _compute_vertices(
        rect: QRectF,
        w: float,
        h: float,
        orientation: Qt.Orientation,
    ) -> np.ndarray:
        """
        Convert a Qt pixel rectangle into an interleaved ``[x, y, u, v]``
        vertex array in NDC space.

        The quad is wound as a ``GL_TRIANGLE_FAN`` starting at the top-left
        vertex (index 0)::

            TL(0) ── TR(1)
             │     ╲   │
            BL(3) ── BR(2)

        Parameters
        ----------
        rect:
            Source rectangle in Qt pixel coordinates (y increases downward).
        w, h:
            Viewport dimensions in pixels; used as the NDC normalisation
            divisor.
        orientation:
            Controls which screen axis drives the LUT U coordinate.

        Returns
        -------
        np.ndarray
            Flat ``(16,)`` ``float32`` array: 4 vertices × ``[x, y, u, v]``.
        """
        # Qt → NDC conversion: x ∈ [0, w] → [-1, 1],  y ∈ [0, h] → [1, -1]
        ndc_l = (rect.left()   / w) * 2.0 - 1.0
        ndc_r = (rect.right()  / w) * 2.0 - 1.0
        ndc_t = 1.0 - (rect.top()    / h) * 2.0
        ndc_b = 1.0 - (rect.bottom() / h) * 2.0

        v = _LUT_V_CENTRE  # V is always the texel centre of a 1-row LUT.

        match orientation:
            case Qt.Orientation.Vertical:
                # Screen Y (top→bottom) maps to U (1.0→0.0) so that the
                # "high" end of the colormap sits at the top of the bar.
                return np.array(
                    [ndc_l, ndc_t, 1.0, v,
                     ndc_r, ndc_t, 1.0, v,
                     ndc_r, ndc_b, 0.0, v,
                     ndc_l, ndc_b, 0.0, v],
                    dtype=np.float32,
                )
            case _:
                # Horizontal (default): screen X (left→right) → U (0.0→1.0).
                return np.array(
                    [ndc_l, ndc_t, 0.0, v,
                     ndc_r, ndc_t, 1.0, v,
                     ndc_r, ndc_b, 1.0, v,
                     ndc_l, ndc_b, 0.0, v],
                    dtype=np.float32,
                )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """``True`` after :meth:`initialize` succeeds and before :meth:`cleanup`."""
        return self._initialized

    @property
    def current_colormap(self) -> str:
        """Name of the active named colormap (irrelevant when manual data is set)."""
        return self._current_cmap

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        status = (
            f"cmap='{self._current_cmap}', reverse={self._reverse}, "
            f"dirty={self._texture_dirty}"
            if self._initialized
            else "uninitialized"
        )
        return f"<GradientRenderer {status}>"
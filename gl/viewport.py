"""
gl_viewport.py
==============
View transformation management for pixel-perfect image rendering.

:class:`ViewManager` owns the two matrices consumed by the image shader:

* **projection** — orthographic, maps pixel coordinates to normalised device
  coordinates (NDC).  Rebuilt on every :meth:`~ViewManager.handle_resize`.
* **transform** — encodes pan, zoom, and rotation applied to the fullscreen
  quad.  Rebuilt on every state change via :meth:`~ViewManager.update_transform`.

Coordinate conventions
-----------------------
* Viewport origin is **bottom-left** (OpenGL convention).
* Pan and zoom-centre arguments are expressed in **viewport pixel units**.
* Rotation is stored and accepted in **degrees**.
* ``QMatrix4x4.data()`` returns elements in **column-major** order.
  :meth:`~ViewManager.get_projection_data` and
  :meth:`~ViewManager.get_transform_data` return the data as a flat
  ``(16,)`` ``float32`` array in that same column-major layout, which is
  what ``glUniformMatrix4fv(..., GL_FALSE, ...)`` expects.

Transform order
---------------
Operations are applied right-to-left in the vertex shader
(``gl_Position = projection * transform * position``):

1. Scale the NDC quad ``(±1)`` to image pixel dimensions.
2. Rotate around the image centre.
3. Apply user zoom.
4. Apply user pan.
5. Translate the image centre to the viewport centre.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PyQt6.QtGui import QMatrix4x4

from cross_platform.qt6_utils.image.gl.error import GLInitializationError

__all__ = ["ViewManager"]

logger = logging.getLogger(__name__)

# Rotation values below this threshold (degrees) are treated as zero to avoid
# submitting a rotation matrix for imperceptibly small angles.
_ROTATION_EPSILON = 0.001

# Fraction of the viewport used when fitting an image — 5 % padding on each
# side prevents the image touching the viewport edge.
_FIT_PADDING = 0.95


class ViewManager:
    """
    Manages the projection and transform matrices for image rendering.

    Maintains user-controlled pan, zoom, and rotation state and rebuilds
    the two GPU matrices whenever any of those values change.

    All matrix data is stored as :class:`~PyQt6.QtGui.QMatrix4x4` internally
    and converted to ``numpy`` arrays on demand by
    :meth:`get_projection_data` and :meth:`get_transform_data`.

    Attributes:
        zoom_level: Current zoom factor (``1.0`` = one image pixel per
                    viewport pixel).
        pan_x:      Horizontal pan offset in viewport pixel units.
        pan_y:      Vertical pan offset in viewport pixel units.
        rotation:   Current rotation in degrees (positive = counter-clockwise).
        viewport_w: Current viewport width in pixels.
        viewport_h: Current viewport height in pixels.
        image_w:    Width of the image being displayed, in pixels.
        image_h:    Height of the image being displayed, in pixels.
    """

    def __init__(self) -> None:
        self.projection: QMatrix4x4 = QMatrix4x4()
        self.transform:  QMatrix4x4 = QMatrix4x4()

        # User-controlled view state
        self.zoom_level: float = 1.0
        self.pan_x:      float = 0.0
        self.pan_y:      float = 0.0
        self.rotation:   float = 0.0   # degrees

        # Viewport and image dimensions — initialised to 1 to avoid
        # divide-by-zero in update_transform before the first resize event.
        self.viewport_w: int = 1
        self.viewport_h: int = 1
        self.image_w:    int = 1
        self.image_h:    int = 1

    # ------------------------------------------------------------------
    # Dimension setters
    # ------------------------------------------------------------------

    def set_image_size(self, width: int, height: int) -> None:
        """
        Update the image dimensions and rebuild the transform matrix.

        Args:
            width:  Image width in pixels.  Must be > 0.
            height: Image height in pixels.  Must be > 0.

        Raises:
            GLInitializationError: If either dimension is non-positive.
        """
        if width <= 0 or height <= 0:
            raise GLInitializationError(
                "Image dimensions must be positive; received (%d, %d)." % (width, height)
            )
        self.image_w = width
        self.image_h = height
        self.update_transform()

    def handle_resize(self, width: int, height: int) -> None:
        """
        React to a viewport resize event.

        Rebuilds the orthographic projection so that one unit in NDC
        corresponds to one viewport pixel, then recentres the image.

        Args:
            width:  New viewport width in pixels.  Must be > 0.
            height: New viewport height in pixels.  Must be > 0.

        Raises:
            GLInitializationError: If either dimension is non-positive, which
                would produce a degenerate (singular) projection matrix.
        """
        if width <= 0 or height <= 0:
            raise GLInitializationError(
                "Viewport dimensions must be positive; received (%d, %d). "
                "Call handle_resize only after the widget has a valid size."
                % (width, height)
            )

        self.viewport_w = width
        self.viewport_h = height

        # Orthographic projection: maps (0, width) × (0, height) → NDC.
        # Origin at bottom-left, matching OpenGL's default convention.
        self.projection.setToIdentity()
        self.projection.ortho(0, width, 0, height, -1, 1)

        self.update_transform()

    # ------------------------------------------------------------------
    # Interactive controls
    # ------------------------------------------------------------------

    def handle_pan(self, dx: float, dy: float) -> None:
        """
        Accumulate a pan delta in viewport pixel units.

        Args:
            dx: Horizontal delta (positive = right).
            dy: Vertical delta (positive = up, OpenGL convention).
        """
        self.pan_x += dx
        self.pan_y += dy
        self.update_transform()

    def handle_zoom(
        self,
        factor: float,
        center_x: Optional[float] = None,
        center_y: Optional[float] = None,
    ) -> None:
        """
        Multiply the current zoom level by ``factor``.

        When a centre point is supplied the pan is adjusted so that the
        pixel under the cursor stays stationary after the zoom.

        Args:
            factor:   Zoom multiplier.  Values > 1 zoom in; values in
                      ``(0, 1)`` zoom out.  Must be positive and non-zero.
            center_x: Optional horizontal zoom pivot in viewport pixel
                      coordinates.
            center_y: Optional vertical zoom pivot in viewport pixel
                      coordinates (bottom-left origin).

        Raises:
            GLInitializationError: If ``factor`` is zero or negative, which
                would invert or zero the zoom level and leave the view
                unrecoverable without a :meth:`reset_view` call.
        """
        if factor <= 0.0:
            raise GLInitializationError(
                "Zoom factor must be positive; received %r.  "
                "Use reset_view() to restore the default zoom level." % factor
            )

        if center_x is not None and center_y is not None:
            # Keep the viewport point (center_x, center_y) fixed in image
            # space by compensating the pan for the scale change.
            # dx/dy is the cursor offset from the current image centre.
            dx = center_x - self.viewport_w / 2.0 - self.pan_x
            dy = center_y - self.viewport_h / 2.0 - self.pan_y
            self.pan_x += dx * (1.0 - factor)
            self.pan_y += dy * (1.0 - factor)

        self.zoom_level *= factor
        self.update_transform()

    def handle_rotation(self, angle_degrees: float) -> None:
        """
        Set the absolute rotation angle.

        Args:
            angle_degrees: Target rotation in degrees.  Positive values rotate
                           counter-clockwise.  Any finite float is accepted;
                           values are not clamped or normalised.
        """
        self.rotation = angle_degrees
        self.update_transform()

    # ------------------------------------------------------------------
    # Preset views
    # ------------------------------------------------------------------

    def reset_view(self) -> None:
        """Reset pan, zoom, and rotation to their default (identity) values."""
        self.zoom_level = 1.0
        self.pan_x      = 0.0
        self.pan_y      = 0.0
        self.rotation   = 0.0
        self.update_transform()

    def fit_to_viewport(self) -> None:
        """
        Zoom the image to fill the viewport with :data:`_FIT_PADDING` margin.

        Selects the smaller of the horizontal and vertical scale factors so
        that the entire image is always visible.  Pan and rotation are reset
        to zero.

        A no-op when the image dimensions have not been set (both are still
        at their default value of ``1``).

        The ``_FIT_PADDING`` factor (``0.95``) adds 5 % breathing room on
        each axis so the image never touches the viewport edge.
        """
        if self.image_w <= 0 or self.image_h <= 0:
            return

        scale_x = self.viewport_w / self.image_w
        scale_y = self.viewport_h / self.image_h
        self.zoom_level = min(scale_x, scale_y) * _FIT_PADDING

        self.pan_x    = 0.0
        self.pan_y    = 0.0
        self.rotation = 0.0
        self.update_transform()

    # ------------------------------------------------------------------
    # Matrix construction
    # ------------------------------------------------------------------

    def update_transform(self) -> None:
        """
        Rebuild the transform matrix from the current pan / zoom / rotation state.

        The operations are composed left-to-right (each call post-multiplies),
        which means they are applied right-to-left in the vertex shader:

        1. **Scale** the NDC quad ``(±1)`` to image pixel dimensions.
        2. **Rotate** around the image centre (skipped for angles < :data:`_ROTATION_EPSILON`).
        3. **Zoom** by the user-controlled scale factor.
        4. **Pan** by the user-controlled offset.
        5. **Translate** the image centre to the viewport centre.

        Called automatically by all state-mutating methods; callers rarely
        need to invoke this directly.
        """
        self.transform.setToIdentity()

        # Step 5 — move the image centre to the centre of the viewport.
        self.transform.translate(self.viewport_w / 2.0, self.viewport_h / 2.0, 0)

        # Step 4 — apply user pan in viewport pixel units.
        self.transform.translate(self.pan_x, self.pan_y, 0)

        # Step 3 — apply user zoom.
        self.transform.scale(self.zoom_level, self.zoom_level, 1)

        # Step 2 — rotate around the image centre.
        if abs(self.rotation) > _ROTATION_EPSILON:
            self.transform.rotate(self.rotation, 0, 0, 1)

        # Step 1 — scale the NDC quad (±1 → 2 units wide) to image dimensions.
        self.transform.scale(self.image_w / 2.0, self.image_h / 2.0, 1)

    # ------------------------------------------------------------------
    # Matrix data accessors
    # ------------------------------------------------------------------

    def get_projection_data(self) -> np.ndarray:
        """
        Return the projection matrix as a flat ``float32`` array for GPU upload.

        ``QMatrix4x4.data()`` returns the 16 elements in **column-major**
        order, which is the memory layout that ``glUniformMatrix4fv`` with
        ``GL_FALSE`` (do not transpose) expects.  The array is returned flat
        rather than reshaped to ``(4, 4)`` to preserve that layout — numpy's
        row-major reshape would reinterpret the columns as rows, producing the
        mathematical transpose.

        Returns:
            ``ndarray`` of shape ``(16,)`` and dtype ``float32``.
        """
        return np.array(self.projection.data(), dtype=np.float32)

    def get_transform_data(self) -> np.ndarray:
        """
        Return the transform matrix as a flat ``float32`` array for GPU upload.

        See :meth:`get_projection_data` for the rationale behind returning a
        flat ``(16,)`` array rather than a ``(4, 4)`` one.

        Returns:
            ``ndarray`` of shape ``(16,)`` and dtype ``float32``.
        """
        return np.array(self.transform.data(), dtype=np.float32)
"""
texture.py
==========
Manages a 1-D LUT texture for the colorbar gradient renderer.

``Texture2D`` wraps a single OpenGL 2-D texture object configured for
1-row RGB LUT sampling.  Despite the "2D" name the texture is logically
1-D: it has a fixed height of 1 and is sampled along U only, with
``GL_CLAMP_TO_EDGE`` preventing any border bleeding.

Design notes
------------
* **Lazy init**: construction is free of GL side-effects; all driver calls
  are deferred to :meth:`initialize`.
* **Dirty-state caching**: :meth:`bind` caches the active unit per instance
  so :meth:`unbind` can restore state without an extra ``glGetIntegerv``.
* **Class-level debug registry**: ``_bound_textures`` lets the caller
  inspect which texture name is bound to each unit without a round-trip to
  the driver.  It is a *debug* facility; production code should not rely on
  its accuracy across threads.
* **No ``__del__``**: finaliser-based cleanup is unreliable in CPython
  (GC order, dead GL context at interpreter shutdown) and silently wrong in
  PyPy.  Callers must use :meth:`cleanup` or the context-manager protocol.

Typical usage::

    with Texture2D() as tex:
        tex.upload_rgb(lut, width=256)
        tex.bind(unit=0)
        quad.draw()
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import ClassVar, Final

import numpy as np

from image.gl.backend import GL
from image.gl.errors import (
    GLInitializationError,
    GLTextureError,
    GLUploadError,
    gl_error_check,
)
from image.gl.types import GLTexture

__all__ = ["Texture1D"]

logger = logging.getLogger(__name__)

# Fixed height for a 1-D LUT strip.  A value > 1 is never needed here.
_LUT_HEIGHT: Final[int] = 1


class Texture1D:
    """
    Owns and manages a single OpenGL 2-D texture for 1-D LUT sampling.

    The texture is always configured with:

    * ``GL_RGB`` internal format (24 bpp, no wasted alpha channel)
    * ``GL_LINEAR`` min/mag filters (smooth gradient interpolation)
    * ``GL_CLAMP_TO_EDGE`` on both axes (no border bleed for a LUT)
    * ``GL_UNPACK_ALIGNMENT = 1`` during upload (RGB rows are not 4-byte aligned)

    Attributes
    ----------
    _bound_textures : ClassVar[dict[int, int]]
        Class-level map of ``unit → texture_name`` maintained by
        :meth:`bind`.  Useful for debugging; treat as read-only.
    """

    # Class-level registry: unit index → GL texture name currently bound there.
    # Populated by bind(); not authoritative across threads or external binds.
    _bound_textures: ClassVar[dict[int, int]] = {}

    # Cache the maximum texture unit count once per class (not per instance)
    # to avoid a glGetIntegerv round-trip on every bind() call.
    _max_texture_units: ClassVar[int | None] = None

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._tex_id: GLTexture = GLTexture(0)
        self._initialized: bool = False
        # Remember which unit this texture was last bound to so unbind()
        # can restore state without an extra driver query.
        self._active_unit: int | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Allocate the OpenGL texture object.

        Idempotent: subsequent calls after a successful first call return
        immediately.  Must be called from the thread that owns the active
        GL context.

        Raises
        ------
        GLInitializationError
            If ``glGenTextures`` reports a GL error.
        """
        if self._initialized:
            return

        with gl_error_check("Texture2D glGenTextures", GLInitializationError):
            self._tex_id = GLTexture(int(GL.glGenTextures(1)))

        self._initialized = True
        logger.debug("Texture2D: allocated tex_id=%d", self._tex_id)

    def cleanup(self) -> None:
        """
        Delete the texture object and reset all state.

        Idempotent and exception-safe: deletion errors are logged but never
        re-raised so that cleanup can proceed inside ``__exit__`` while
        another exception is already propagating.

        Note
        ----
        The class-level ``_bound_textures`` registry is *not* cleared here
        because the caller may have multiple ``Texture2D`` instances on
        different units.  Stale entries are harmless debug artefacts.
        """
        if not self._initialized:
            return

        self._initialized = False
        tex_id = self._tex_id
        self._tex_id = GLTexture(0)
        self._active_unit = None

        if tex_id:
            try:
                GL.glDeleteTextures(1, np.array([tex_id], dtype=np.uint32))
                logger.debug("Texture2D: deleted tex_id=%d", tex_id)
            except Exception:
                logger.exception(
                    "Texture2D: exception while deleting tex_id=%d", tex_id
                )

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload_rgb(self, data: np.ndarray, *, width: int,
                   height: int = _LUT_HEIGHT) -> None:
        """
        Upload ``(N, 3)`` ``uint8`` RGB pixel data to the GPU texture.

        The texture is (re-)allocated with ``glTexImage2D`` on every call, so
        ``width`` can change between uploads without needing explicit teardown.
        Callers that only need to *update* an existing same-size texture should
        call :meth:`upload_rgb_sub` instead (not yet implemented).

        Parameters
        ----------
        data:
            A ``(N, 3)`` or ``(N * height, 3)`` ``uint8`` array of RGB values.
            The array must be C-contiguous; use ``np.ascontiguousarray`` before
            calling if in doubt.
        width:
            Number of texels along the U axis.  For a 1-D LUT this equals
            ``data.shape[0]``.
        height:
            Number of texel rows.  Defaults to 1 (1-D LUT mode).

        Raises
        ------
        GLInitializationError
            If called before :meth:`initialize`.
        TypeError
            If ``data.dtype`` is not ``uint8``.
        ValueError
            If ``data`` is not 2-D, does not have exactly 3 columns, or its
            total element count does not match ``width * height * 3``.
        GLUploadError
            If any GL call during the upload reports an error.
        """
        if not self._initialized:
            raise GLInitializationError(
                "Texture2D.upload_rgb() called before initialize()"
            )

        # --- Validate -------------------------------------------------------
        if data.dtype != np.uint8:
            raise TypeError(
                f"Texture2D.upload_rgb expects uint8 data; got {data.dtype}"
            )
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(
                f"Texture2D.upload_rgb expects shape (N, 3); got {data.shape}"
            )
        expected_elements = width * height * 3
        if data.size != expected_elements:
            raise ValueError(
                f"Data size mismatch: width={width}, height={height} requires "
                f"{expected_elements} elements; got {data.size}"
            )

        logger.debug(
            "Texture2D: uploading RGB LUT — tex_id=%d, %d×%d",
            self._tex_id, width, height,
        )

        with gl_error_check("Texture2D upload_rgb", GLUploadError):
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_id)

            # RGB rows of a LUT are commonly 3 bytes wide, which is not
            # 4-byte aligned.  Setting UNPACK_ALIGNMENT=1 prevents the driver
            # from reading garbage padding bytes at the end of each row.
            GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)

            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                0,  # mip level
                GL.GL_RGB,  # internal format
                width,
                height,
                0,  # border (must be 0)
                GL.GL_RGB,  # source format
                GL.GL_UNSIGNED_BYTE,
                data,
            )

            # Sampling parameters — set after upload so they apply to the
            # newly allocated storage.
            for pname, param in (
                    (GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR),
                    (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR),
                    (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE),
                    (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE),
            ):
                GL.glTexParameteri(GL.GL_TEXTURE_2D, pname, param)

            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    # ------------------------------------------------------------------
    # Bind / Unbind
    # ------------------------------------------------------------------

    def bind(self, unit: int = 0) -> None:
        """
        Activate the given texture unit and bind this texture to it.

        Parameters
        ----------
        unit:
            Zero-based texture unit index.  Valid range is
            ``[0, GL_MAX_TEXTURE_IMAGE_UNITS − 1]``; the upper bound is
            queried from the driver and cached on the first call.

        Raises
        ------
        GLInitializationError
            If called before :meth:`initialize`.
        ValueError
            If ``unit`` is outside the driver-reported valid range.
        GLTextureError
            If ``glActiveTexture`` or ``glBindTexture`` reports a GL error.
        """
        if not self._initialized:
            raise GLInitializationError(
                "Texture2D.bind() called before initialize()"
            )

        # Lazy-query the maximum unit count and cache it at class level.
        if Texture1D._max_texture_units is None:
            Texture1D._max_texture_units = int(
                GL.glGetIntegerv(GL.GL_MAX_TEXTURE_IMAGE_UNITS)
            )

        max_units: int = Texture1D._max_texture_units
        if not (0 <= unit < max_units):
            raise ValueError(
                f"Texture unit {unit} out of range [0, {max_units - 1}]"
            )

        with gl_error_check(f"Texture2D bind unit={unit}", GLTextureError):
            GL.glActiveTexture(GL.GL_TEXTURE0 + unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self._tex_id)

        self._active_unit = unit
        Texture1D._bound_textures[unit] = int(self._tex_id)

    def unbind(self) -> None:
        """
        Unbind this texture from the unit it was last bound to.

        No-op if :meth:`bind` has not been called.  Restores the unit to
        the GL null texture (name 0) without disturbing other units.
        """
        if self._active_unit is None:
            return

        with gl_error_check("Texture2D unbind", GLTextureError):
            GL.glActiveTexture(GL.GL_TEXTURE0 + self._active_unit)
            GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        Texture1D._bound_textures.pop(self._active_unit, None)
        self._active_unit = None

    # ------------------------------------------------------------------
    # Debug helpers
    # ------------------------------------------------------------------

    @classmethod
    def get_bound_texture(cls, unit: int = 0) -> int | None:
        """
        Return the GL texture name most recently bound to ``unit`` by any
        ``Texture2D`` instance, or ``None`` if the unit is unoccupied.

        This is a *debug* helper backed by the in-process registry; it does
        not query the driver and will be stale if external code rebinds the
        unit directly.
        """
        return cls._bound_textures.get(unit)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """``True`` after :meth:`initialize` succeeds and before :meth:`cleanup`."""
        return self._initialized

    @property
    def tex_id(self) -> GLTexture:
        """The raw GL texture name, or ``GLTexture(0)`` when unallocated."""
        return self._tex_id

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> Texture1D:
        """
        Call :meth:`initialize` and return ``self``.

        Example::

            with Texture2D() as tex:
                tex.upload_rgb(lut, width=256)
                tex.bind(0)
        """
        self.initialize()
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
    ) -> bool:
        """Release GPU resources; never suppresses exceptions."""
        self.cleanup()
        return False

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        if not self._initialized:
            return "<Texture2D uninitialized>"
        unit_info = (
            f", bound_to_unit={self._active_unit}"
            if self._active_unit is not None
            else ""
        )
        return f"<Texture2D tex_id={self._tex_id}{unit_info}>"

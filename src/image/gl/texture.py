"""
gl_texture.py
=============
Texture allocation, upload, binding, and lifecycle management for PyOpenGL.

Platform notes
--------------
macOS (Apple GL 4.1)
    ``GL_BGRA`` must be used as the transfer format to avoid a colour-channel
    swap artifact.  `get_platform_gl_spec` handles this automatically.

Immutable storage (``glTexStorage2D``)
    Used when ``GLConfig.USE_IMMUTABLE_STORAGE`` is ``True`` (GL 4.2+ or
    ``GL_ARB_texture_storage``).  Falls back to ``glTexImage2D`` transparently.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import sys
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Generator, Optional, Union

import numpy as np

from cross_platform.qt6_utils.image.gl.backend import GL
from cross_platform.qt6_utils.image.gl.config import get_gl_config
from cross_platform.qt6_utils.image.gl.error import (
    GLError,
    GLMemoryError,
    GLTextureError,
)
from cross_platform.qt6_utils.image.gl.types import (
    GLenum,
    GLHandle,
    GLint,
    GLsizei,
    GLTexture,
    GLBuffer,
)
from cross_platform.qt6_utils.image.pipeline.metadata import FrameStats

__all__ = [
    "TextureUploadPayload",
    "TextureSpec",
    "TextureState",
    "TextureManager",
    "SwizzleMode",
    "get_platform_gl_spec",
    "ensure_format_compatibility",
    "alloc_texture_storage",
    "bind_texture",
]

logger = logging.getLogger(__name__)

# Sentinel handle used to unbind a texture target.
_NO_TEXTURE = GLTexture(GLHandle(0))


# ---------------------------------------------------------------------------
# Transfer payload
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class TextureUploadPayload:
    """
    Immutable descriptor for a single PBO→Texture transfer operation.

    Produced by the upload pipeline and consumed by
    `~cross_platform.qt6_utils.image.gl.frame_viewer.GLFrameViewer._upload_frame`.

    Attributes:
        meta:               Frame metadata emitted with ``frame_changed``.
        pbo_id:             Handle of the PBO holding the pixel data.
        width:              Frame width in pixels.
        height:             Frame height in pixels.
        gl_format:          GL base-format token (transfer format).
        gl_internal_format: GL internal-format token (GPU storage format).
        gl_type:            GL element-type token.
        data:               CPU-side array (unpinned path only; ``None`` for
                            pinned uploads where the PBO already contains data).
        is_pinned:          ``True`` when the PBO was pre-filled by the caller
                            (pinned/zero-copy path).
    """

    meta:               FrameStats
    pbo_id:             GLBuffer
    width:              GLsizei
    height:             GLsizei
    gl_format:          GLenum
    gl_internal_format: GLenum
    gl_type:            GLenum
    data:               Optional[Any] = None
    is_pinned:          bool          = False


# ---------------------------------------------------------------------------
# Format descriptor
# ---------------------------------------------------------------------------

@dataclass
class TextureSpec:
    """
    Describes the GL format tokens required for a specific image layout.

    Used by `get_platform_gl_spec`, `TextureManager.allocate_from_spec`,
    and `TextureManager.upload_image` to keep format decisions centralised.

    Attributes:
        internal_format: GL internal-format token (GPU storage).
        fmt:             GL base-format token (transfer / sampling format).
        type:            GL element-type token.
        swizzle_needed:  When ``True``, the R and B channels must be swapped
                         before upload (required for ``GL_BGRA`` on macOS).
    """

    internal_format: GLenum
    fmt:             GLenum
    type:            GLenum
    swizzle_needed:  bool


def get_platform_gl_spec() -> TextureSpec:
    """
    Return the appropriate `TextureSpec` for the current platform.

    On macOS, Apple's GL 4.1 driver requires ``GL_BGRA`` as the transfer
    format when working with RGBA textures.  On all other platforms ``GL_RGBA``
    is used directly.

    Returns:
        A `TextureSpec` suitable for the current OS.
    """
    if sys.platform == "darwin":
        # Apple's driver performs the B↔R swap internally when GL_BGRA is
        # used; the resulting GPU texture samples as RGBA in the shader.
        return TextureSpec(
            internal_format=GLenum(GL.GL_RGBA8),
            fmt=GLenum(GL.GL_BGRA),
            type=GLenum(GL.GL_UNSIGNED_BYTE),
            swizzle_needed=True,
        )
    return TextureSpec(
        internal_format=GLenum(GL.GL_RGBA8),
        fmt=GLenum(GL.GL_RGBA),
        type=GLenum(GL.GL_UNSIGNED_BYTE),
        swizzle_needed=False,
    )


# ---------------------------------------------------------------------------
# Format compatibility
# ---------------------------------------------------------------------------

def _validate_image_for_spec(
    image_data: np.ndarray,
    spec: TextureSpec,
    in_place: bool,
) -> np.ndarray:
    """
    Validate and coerce ``image_data`` to the dtype required by ``spec``.

    Args:
        image_data: Source array.  Must be 2-D (greyscale) or 3-D (channel).
        spec:       Target format descriptor.
        in_place:   When ``True``, dtype conversion is done without an extra
                    copy if the underlying memory is writable.

    Returns:
        The (possibly converted) array.

    Raises:
        GLTextureError: If ``image_data`` is ``None``, has unsupported
            dimensionality, or has an invalid channel count.
    """
    if image_data is None:
        raise GLTextureError("image_data must not be None.")

    if image_data.ndim not in (2, 3):
        raise GLTextureError(
            "Expected a 2-D (greyscale) or 3-D (channelled) array; "
            "received shape %s." % (image_data.shape,)
        )

    if image_data.ndim == 3 and image_data.shape[-1] not in (1, 3, 4):
        raise GLTextureError(
            "Expected 1, 3, or 4 channels in the last axis; "
            "received %d." % image_data.shape[-1]
        )

    # Coerce dtype to match the GL element type in the spec.
    if spec.type == GL.GL_UNSIGNED_BYTE and image_data.dtype != np.uint8:
        image_data = image_data.astype(np.uint8, copy=not in_place)
    elif spec.type == GL.GL_FLOAT and image_data.dtype != np.float32:
        image_data = image_data.astype(np.float32, copy=not in_place)

    return image_data


def ensure_format_compatibility(
    image_data: np.ndarray,
    spec: TextureSpec,
    in_place: bool = True,
) -> np.ndarray:
    """
    Coerce an image array to be compatible with ``spec`` for GPU upload.

    Handles dtype conversion, C-contiguity enforcement, and the B↔R channel
    swap required when `~TextureSpec.swizzle_needed` is ``True``.

    Swizzle strategy
    ----------------
    * In-place (``in_place=True``): reshape to ``(N, channels)``, swap column
      0 and column 2 using a temporary copy of column 0.  Zero allocation
      overhead for the common case.
    * Out-of-place (``in_place=False``): fancy-index reorder and ``.copy()``.

    Args:
        image_data: Source image.  Must be 2-D or 3-D.
        spec:       Target format descriptor.
        in_place:   Whether to prefer in-place operations (default ``True``).

    Returns:
        A compatible, C-contiguous array ready for ``glTexSubImage2D``.

    Raises:
        GLTextureError: Propagated from `_validate_image_for_spec`.
    """
    image_data = _validate_image_for_spec(image_data, spec, in_place)

    # Enforce C-contiguity first.  If the swizzle is needed and the array is
    # non-contiguous, apply the channel reorder at the same time as the copy
    # to avoid a redundant allocation.
    if not image_data.flags["C_CONTIGUOUS"]:
        if (
            spec.swizzle_needed
            and image_data.ndim == 3
            and image_data.shape[-1] >= 3
        ):
            new_order = [2, 1, 0, 3] if image_data.shape[-1] == 4 else [2, 1, 0]
            return np.ascontiguousarray(image_data[..., new_order])
        return np.ascontiguousarray(image_data)

    # Array is already contiguous.  If no swizzle is needed, return it as-is.
    if not spec.swizzle_needed:
        return image_data

    # Swizzle: swap R (channel 0) and B (channel 2).
    if image_data.ndim == 3 and image_data.shape[-1] >= 3:
        channels = image_data.shape[-1]
        if in_place:
            # Reshape to (N, C) to operate on individual colour channels.
            # A temporary copy of channel 0 is required to avoid clobbering
            # before channel 2 is written.
            flat    = image_data.reshape(-1, channels)
            tmp     = flat[:, 0].copy()
            flat[:, 0] = flat[:, 2]
            flat[:, 2] = tmp
            return image_data
        else:
            order = [2, 1, 0, 3] if channels == 4 else [2, 1, 0]
            return image_data[..., order].copy()

    return image_data


# ---------------------------------------------------------------------------
# Storage allocation
# ---------------------------------------------------------------------------

def alloc_texture_storage(
    target:     GLenum,
    width:      GLsizei,
    height:     GLsizei,
    gl_fmt:     GLenum,
    gl_int_fmt: GLenum,
    gl_type:    GLenum,
    data:       Any,
    levels:     GLint = GLint(1),
) -> None:
    """
    Allocate GPU texture storage and optionally upload initial pixel data.

    Selects the allocation path based on runtime driver capability:

    * **Immutable** (``glTexStorage2D`` + ``glTexSubImage2D``) — used when
      ``GLConfig.USE_IMMUTABLE_STORAGE`` is ``True``.  Preferred: the driver
      can validate the format once and cache internal state.
    * **Mutable** (``glTexImage2D``) — fallback for GL < 4.2 without the
      ``GL_ARB_texture_storage`` extension, and always used on macOS.

    ``GL_UNPACK_ALIGNMENT`` is set to ``1`` before every upload to ensure
    tightly-packed rows are read correctly regardless of image width.

    Args:
        target:     GL texture target token (e.g. ``GL_TEXTURE_2D``).
        width:      Texture width in pixels.
        height:     Texture height in pixels.
        gl_fmt:     Base-format token for the pixel data layout.
        gl_int_fmt: Internal-format token for GPU storage.
        gl_type:    Element-type token for the pixel data.
        data:       Pixel data to upload, or ``None`` to allocate without
                    uploading (storage is uninitialised).
        levels:     Number of mip-map levels (default ``1``).

    Raises:
        GLTextureError: If storage allocation or pixel upload fails.
        GLMemoryError:  If the driver reports ``GL_OUT_OF_MEMORY``.
    """
    # Alignment 1 disables row-padding, which is safe for any width.
    GL.glPixelStorei(GLenum(GL.GL_UNPACK_ALIGNMENT), GLint(1))

    use_immutable = get_gl_config().USE_IMMUTABLE_STORAGE

    try:
        if use_immutable:
            # Immutable path: allocate once, upload separately.
            # If the driver rejects glTexStorage2D (e.g. an internal format
            # it doesn't support at this level count), fall through to mutable.
            try:
                GL.glTexStorage2D(target, levels, gl_int_fmt, width, height)
                if data is not None:
                    GL.glTexSubImage2D(
                        target, 0, 0, 0, width, height, gl_fmt, gl_type, data
                    )
                return
            except (GLError, AttributeError) as e:
                # GLError: driver rejected the format/size combination.
                # AttributeError: glTexStorage2D is not available despite the
                # config flag — degrade gracefully to the mutable path.
                logger.warning(
                    "glTexStorage2D failed (%s); falling back to glTexImage2D.", e
                )

        # Mutable path: combined allocation + upload in one call.
        GL.glTexImage2D(
            target, 0, gl_int_fmt, width, height, 0, gl_fmt, gl_type, data
        )

    except GLError as e:
        # Re-raise as a typed texture error so callers can catch narrowly.
        raise GLTextureError(
            "Texture storage allocation failed (%dx%d, internal_fmt=0x%x): %s"
            % (width, height, gl_int_fmt, e)
        ) from e
    except MemoryError as e:
        raise GLMemoryError(
            "GPU out of memory allocating %dx%d texture." % (width, height)
        ) from e


# ---------------------------------------------------------------------------
# Binding context manager
# ---------------------------------------------------------------------------

@contextlib.contextmanager
def bind_texture(
    texture_id: GLTexture,
    unit:       Optional[GLint] = None,
    target:     GLenum = GLenum(GL.GL_TEXTURE_2D),
) -> Generator[None, None, None]:
    """
    Context manager that binds ``texture_id`` and restores the previous binding.

    Saves the current ``GL_TEXTURE_BINDING_2D`` (for ``GL_TEXTURE_2D`` targets)
    before binding and restores it on exit, even when the block raises.

    Args:
        texture_id: Texture handle to bind.
        unit:       When provided, activates ``GL_TEXTURE0 + unit`` before
                    binding and again on exit.
        target:     Texture target (default ``GL_TEXTURE_2D``).

    Yields:
        ``None`` — the body of the ``with`` block.
    """
    previous_binding: Optional[GLHandle] = None

    try:
        # Record the current binding so we can restore it on exit.
        # glGetIntegerv (not glGetInteger — that function does not exist in PyOpenGL).
        if target == GL.GL_TEXTURE_2D:
            prev = GL.glGetIntegerv(GLenum(GL.GL_TEXTURE_BINDING_2D))
            if prev is not None:
                previous_binding = GLHandle(int(prev))

        if unit is not None:
            GL.glActiveTexture(GLenum(GL.GL_TEXTURE0 + int(unit)))

        GL.glBindTexture(target, texture_id)
        yield

    finally:
        # Re-activate the same unit so the restore targets the correct slot.
        if unit is not None:
            GL.glActiveTexture(GLenum(GL.GL_TEXTURE0 + int(unit)))

        restore = (
            GLTexture(previous_binding)
            if previous_binding is not None
            else _NO_TEXTURE
        )
        GL.glBindTexture(target, restore)


# ---------------------------------------------------------------------------
# Swizzle mode
# ---------------------------------------------------------------------------

@unique
class SwizzleMode(Enum):
    """
    Channel-swizzle presets applied via ``GL_TEXTURE_SWIZZLE_RGBA``.

    Members
    -------
    GRAY
        Routes the red channel to all three colour outputs (``R → RGB``) and
        sets alpha to 1.  Used for single-channel (greyscale / ``GL_RED``)
        textures so the shader samples them as white-on-black rather than
        red-on-black.
    RGB
        Identity mapping — no swizzle.  Used for RGB and RGBA textures.
    BGR
        Swaps red and blue.  Retained for completeness; in practice the
        B↔R swap is handled on the CPU by `ensure_format_compatibility`
        before upload.
    """

    GRAY = (GL.GL_RED,  GL.GL_RED,   GL.GL_RED,  GL.GL_ONE)
    RGB  = (GL.GL_RED,  GL.GL_GREEN, GL.GL_BLUE, GL.GL_ALPHA)
    BGR  = (GL.GL_BLUE, GL.GL_GREEN, GL.GL_RED,  GL.GL_ALPHA)

    def as_gl_params(self) -> ctypes.Array:
        """
        Return the swizzle values as a ``ctypes.c_int[4]`` array.

        This is the form expected by ``glTexParameteriv``.

        Returns:
            A 4-element ctypes integer array containing the four GL swizzle
            tokens in ``(R, G, B, A)`` order.
        """
        # ctypes.c_int is used here rather than GL.GLint: GL.GLint is a
        # PyOpenGL typedef that is not always a usable ctypes array factory
        # across all platforms and PyOpenGL versions.
        return (ctypes.c_int * 4)(*self.value)


# ---------------------------------------------------------------------------
# Per-texture state
# ---------------------------------------------------------------------------

@dataclass(slots=True, frozen=False)
class TextureState:
    """
    Per-texture bookkeeping maintained by `TextureManager`.

    Attributes:
        renderer_id:     GL texture handle.
        width:           Allocated width in pixels (``0`` until
                         `~TextureManager.allocate` is called).
        height:          Allocated height in pixels.
        internal_format: GL internal-format token used at allocation time.
        is_allocated:    ``True`` after the first successful
                         `~TextureManager.allocate` call.
        current_swizzle: The last `SwizzleMode` written to the driver,
                         or ``None`` when the swizzle has not been set.
    """

    renderer_id:     GLTexture
    width:           GLsizei           = GLsizei(0)
    height:          GLsizei           = GLsizei(0)
    internal_format: GLenum            = GLenum(GL.GL_R32F)
    is_allocated:    bool              = False
    current_swizzle: Optional[SwizzleMode] = None


# ---------------------------------------------------------------------------
# Texture manager
# ---------------------------------------------------------------------------

class TextureManager:
    """
    Keyed pool of OpenGL textures with allocation, upload, and binding helpers.

    Each texture is identified by an arbitrary string or integer key.  The
    manager tracks per-texture state (`TextureState`) and provides
    scoped binding via context managers.

    Lifetime
    --------
    Call `cleanup` explicitly before destroying the GL context.
    `__del__` attempts a best-effort cleanup but is not reliable when
    the context may already be gone at GC time.
    """

    def __init__(self) -> None:
        self._textures: dict[Union[str, int], TextureState] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """
        Delete all managed textures and free GPU memory.

        Safe to call multiple times.  Errors during deletion are caught and
        logged rather than raised, because cleanup should never abort
        application shutdown.
        """
        ids_to_delete = [t.renderer_id for t in self._textures.values()]
        if ids_to_delete:
            try:
                GL.glDeleteTextures(
                    GLsizei(len(ids_to_delete)),
                    np.array(ids_to_delete, dtype=np.uint32),
                )
            except Exception as e:
                logger.error("Error during bulk texture cleanup: %s", e)
        self._textures.clear()

    def __del__(self) -> None:
        """
        Best-effort cleanup when the manager is garbage-collected.

        Silently skips if the GL context is no longer available.  Callers
        should call `cleanup` explicitly rather than relying on this.
        """
        try:
            self.cleanup()
        except Exception:
            # The GL context may be gone; any driver call will fail.
            # Suppress all errors so the GC does not produce spurious
            # tracebacks during interpreter shutdown.
            pass

    # ------------------------------------------------------------------
    # Creation and deletion
    # ------------------------------------------------------------------

    def create_texture(self, key: Union[str, int]) -> GLTexture:
        """
        Generate a new GL texture object and register it under ``key``.

        If ``key`` already exists the existing handle is returned without
        generating a new texture, and a warning is logged.

        Args:
            key: Unique identifier for this texture.

        Returns:
            The GL texture handle as :data:`GLTexture`.
        """
        if key in self._textures:
            logger.warning(
                "Texture key '%s' already exists — returning existing handle.",
                key,
            )
            return self._textures[key].renderer_id

        raw_id      = GL.glGenTextures(1)
        renderer_id = GLTexture(raw_id)
        self._textures[key] = TextureState(renderer_id=renderer_id)
        return renderer_id

    def delete_texture(self, key: Union[str, int]) -> None:
        """
        Delete the GL texture registered under ``key`` and remove it from the pool.

        A no-op when ``key`` is not present.

        Args:
            key: Texture identifier.
        """
        if key not in self._textures:
            return

        state = self._textures.pop(key)
        try:
            GL.glDeleteTextures(
                GLsizei(1), np.array([state.renderer_id], dtype=np.uint32)
            )
        except Exception as e:
            logger.error("Failed to delete texture '%s': %s", key, e)

    # ------------------------------------------------------------------
    # State accessors
    # ------------------------------------------------------------------

    def get_id(self, key: Union[str, int]) -> GLTexture:
        """
        Return the GL handle for ``key``.

        Args:
            key: Texture identifier.

        Returns:
            GL texture handle.

        Raises:
            GLTextureError: If ``key`` is not registered.
        """
        if key not in self._textures:
            raise GLTextureError(
                "Texture '%s' not found in manager.  "
                "Call create_texture() first." % key
            )
        return self._textures[key].renderer_id

    def get_state(self, key: Union[str, int]) -> Optional[TextureState]:
        """
        Return the `TextureState` for ``key``, or ``None`` if absent.

        Args:
            key: Texture identifier.

        Returns:
            `TextureState` when registered, ``None`` otherwise.
        """
        return self._textures.get(key)

    # ------------------------------------------------------------------
    # Binding
    # ------------------------------------------------------------------

    @staticmethod
    def activate(unit: GLint = GLint(0)) -> None:
        """
        Activate texture unit ``unit`` if it is not already active.

        Avoids a redundant ``glActiveTexture`` call by querying
        ``GL_ACTIVE_TEXTURE`` first.  Unit indices start at ``0``
        (``GL_TEXTURE0``).

        Args:
            unit: Zero-based texture unit index.
        """
        # glGetIntegerv — not glGetInteger (does not exist in PyOpenGL).
        current  = GL.glGetIntegerv(GLenum(GL.GL_ACTIVE_TEXTURE))
        expected = GL.GL_TEXTURE0 + int(unit)
        if int(current) != expected:
            GL.glActiveTexture(GLenum(expected))

    def bind(self, key: Union[str, int], unit: GLint = GLint(0)) -> None:
        """
        Activate texture unit ``unit`` and bind the texture for ``key``.

        Args:
            key:  Texture identifier.
            unit: Zero-based texture unit index.

        Raises:
            KeyError: If ``key`` is not registered (use `get_state`
                to check before calling).
        """
        state = self._textures[key]
        self.activate(unit=unit)
        GL.glBindTexture(GLenum(GL.GL_TEXTURE_2D), state.renderer_id)

    def unbind(self, unit: GLint = GLint(0)) -> None:
        """
        Activate texture unit ``unit`` and unbind any texture from ``GL_TEXTURE_2D``.

        Args:
            unit: Zero-based texture unit index.
        """
        self.activate(unit=unit)
        GL.glBindTexture(GLenum(GL.GL_TEXTURE_2D), _NO_TEXTURE)

    def bound(
        self,
        key:  Union[str, int],
        unit: GLint = GLint(0),
    ) -> contextlib.AbstractContextManager:
        """
        Return a context manager that binds the texture for ``key``.

        Delegates to `bind_texture`, which saves and restores the
        previous binding.

        Args:
            key:  Texture identifier.
            unit: Zero-based texture unit index.

        Returns:
            A context manager (see `bind_texture`).

        Raises:
            GLTextureError: If ``key`` is not registered.
        """
        if key not in self._textures:
            raise GLTextureError(
                "Texture '%s' not found; cannot create a binding context." % key
            )
        renderer_id = self._textures[key].renderer_id
        return bind_texture(renderer_id, unit=unit, target=GLenum(GL.GL_TEXTURE_2D))

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------

    def allocate(
        self,
        key:        Union[str, int],
        data:       Any,
        width:      GLsizei,
        height:     GLsizei,
        levels:     GLint   = GLint(1),
        gl_int_fmt: GLenum  = GLenum(GL.GL_R32F),
        gl_fmt:     GLenum  = GLenum(GL.GL_RED),
        gl_type:    GLenum  = GLenum(GL.GL_FLOAT),
    ) -> None:
        """
        Allocate GPU storage for ``key`` and optionally upload ``data``.

        Updates the `TextureState` fields (``width``, ``height``,
        ``internal_format``, ``is_allocated``) after a successful call.

        Args:
            key:        Texture identifier (must already be registered via
                        `create_texture`).
            data:       Pixel data to upload, or ``None`` for uninitialised storage.
            width:      Texture width in pixels.
            height:     Texture height in pixels.
            levels:     Mip-map level count (default ``1``).
            gl_int_fmt: GL internal-format token (default ``GL_R32F``).
            gl_fmt:     GL base-format token (default ``GL_RED``).
            gl_type:    GL element-type token (default ``GL_FLOAT``).

        Raises:
            GLTextureError: If ``key`` is not registered, or if the driver
                rejects the storage allocation.
            GLMemoryError:  If the driver reports out-of-memory.
        """
        if key not in self._textures:
            raise GLTextureError(
                "Texture '%s' must be created via create_texture() "
                "before it can be allocated." % key
            )

        state                = self._textures[key]
        state.width          = width
        state.height         = height
        state.internal_format = gl_int_fmt

        with self.bound(key, unit=GLint(0)):
            alloc_texture_storage(
                target=GLenum(GL.GL_TEXTURE_2D),
                width=width,
                height=height,
                gl_fmt=gl_fmt,
                gl_int_fmt=gl_int_fmt,
                gl_type=gl_type,
                data=data,
                levels=levels,
            )

        state.is_allocated = True

    def allocate_from_spec(
        self,
        key:    Union[str, int],
        data:   Any,
        width:  GLsizei,
        height: GLsizei,
        spec:   TextureSpec,
        levels: GLint = GLint(1),
    ) -> None:
        """
        Allocate storage and apply the correct swizzle for ``spec``.

        Convenience wrapper around `allocate` that uses
        `~TextureSpec.internal_format`, `~TextureSpec.fmt`, and
        `~TextureSpec.type` from ``spec``, then calls
        `update_swizzle` to configure the shader-facing channel mapping.

        Args:
            key:    Texture identifier.
            data:   Pixel data or ``None``.
            width:  Texture width in pixels.
            height: Texture height in pixels.
            spec:   Platform texture spec from `get_platform_gl_spec`.
            levels: Mip-map level count.

        Raises:
            GLTextureError: Propagated from `allocate`.
            GLMemoryError:  Propagated from `allocate`.
        """
        self.allocate(
            key=key,
            data=data,
            width=width,
            height=height,
            levels=levels,
            gl_int_fmt=spec.internal_format,
            gl_fmt=spec.fmt,
            gl_type=spec.type,
        )
        with self.bound(key, unit=GLint(0)):
            self.update_swizzle(key, spec.fmt)

    # ------------------------------------------------------------------
    # Upload
    # ------------------------------------------------------------------

    def upload(
        self,
        key:     Union[str, int],
        data:    np.ndarray,
        format_: GLenum = GLenum(GL.GL_RED),
        type_:   GLenum = GLenum(GL.GL_FLOAT),
    ) -> None:
        """
        Upload new pixel data into an already-allocated texture.

        The texture must have been allocated via `allocate` or
        `allocate_from_spec` before this method is called.

        ``GL_UNPACK_ALIGNMENT`` is set to ``1`` to accommodate tightly-packed
        rows of any width.

        Args:
            key:     Texture identifier.
            data:    Pixel data array.  Must be C-contiguous and compatible
                     with the format and type tokens.
            format_: GL base-format token.
            type_:   GL element-type token.

        Raises:
            GLTextureError: If the texture is not allocated, or if the driver
                rejects the upload.
        """
        if key not in self._textures:
            raise GLTextureError("Texture '%s' not found." % key)

        state = self._textures[key]
        if not state.is_allocated:
            raise GLTextureError(
                "Texture '%s' has not been allocated.  "
                "Call allocate() before upload()." % key
            )

        with self.bound(key, unit=GLint(0)):
            GL.glPixelStorei(GLenum(GL.GL_UNPACK_ALIGNMENT), GLint(1))
            try:
                GL.glTexSubImage2D(
                    GLenum(GL.GL_TEXTURE_2D), GLint(0), GLint(0), GLint(0),
                    state.width, state.height,
                    format_, type_, data,
                )
            except GLError as e:
                raise GLTextureError(
                    "glTexSubImage2D failed for texture '%s': %s" % (key, e)
                ) from e

    def upload_image(
        self,
        key:      Union[str, int],
        image:    np.ndarray,
        spec:     TextureSpec,
        in_place: bool = True,
    ) -> np.ndarray:
        """
        Prepare ``image`` for the platform format and upload it.

        Calls `ensure_format_compatibility` to handle dtype conversion
        and channel swizzle, then uploads via `upload`, and finally
        updates the shader swizzle mask via `update_swizzle`.

        Args:
            key:      Texture identifier.
            image:    Source image (``HxW`` or ``HxWxC``).
            spec:     Platform texture spec.
            in_place: Whether to prefer in-place channel operations.

        Returns:
            The (possibly modified) array that was uploaded.

        Raises:
            GLTextureError: Propagated from `upload`.
        """
        prepared = ensure_format_compatibility(image, spec, in_place=in_place)
        self.upload(key, prepared, format_=spec.fmt, type_=spec.type)
        with self.bound(key, unit=GLint(0)):
            self.update_swizzle(key, spec.fmt)
        return prepared

    # ------------------------------------------------------------------
    # Swizzle
    # ------------------------------------------------------------------

    def update_swizzle(
        self,
        key:       Union[str, int],
        gl_format: GLenum,
    ) -> None:
        """
        Set the ``GL_TEXTURE_SWIZZLE_RGBA`` parameters for ``key``.

        Selects `SwizzleMode.GRAY` for single-channel formats and
        `SwizzleMode.RGB` for multi-channel formats.  A no-op when the
        swizzle mode has not changed since the last call, avoiding a redundant
        driver round-trip.

        The texture must be bound before this method is called.

        Args:
            key:       Texture identifier.
            gl_format: The GL base-format token that was used for upload.

        Raises:
            GLTextureError: If ``key`` is not registered.
        """
        if key not in self._textures:
            raise GLTextureError(
                "Cannot update swizzle: texture '%s' not found." % key
            )

        state = self._textures[key]

        # GL_LUMINANCE is a compatibility-profile token absent from Core 3.1+.
        # Guard with getattr so the comparison works on both profiles.
        gl_luminance   = getattr(GL, "GL_LUMINANCE", None)
        scalar_formats = {int(GL.GL_RED), int(GL.GL_R32F)}
        if gl_luminance is not None:
            scalar_formats.add(int(gl_luminance))

        new_mode = (
            SwizzleMode.GRAY if int(gl_format) in scalar_formats else SwizzleMode.RGB
        )

        if state.current_swizzle is new_mode:
            # Swizzle unchanged — skip the driver call.
            return

        GL.glTexParameteriv(
            GLenum(GL.GL_TEXTURE_2D),
            GLenum(GL.GL_TEXTURE_SWIZZLE_RGBA),
            new_mode.as_gl_params(),
        )
        state.current_swizzle = new_mode

    # ------------------------------------------------------------------
    # Sampling parameters
    # ------------------------------------------------------------------

    @staticmethod
    def set_sampling_mode(
        min_filter:      GLenum = GLenum(GL.GL_NEAREST),
        mag_filter:      GLenum = GLenum(GL.GL_NEAREST),
        wrap_s:          GLenum = GLenum(GL.GL_CLAMP_TO_EDGE),
        wrap_t:          GLenum = GLenum(GL.GL_CLAMP_TO_EDGE),
        generate_mipmaps: bool  = False,
    ) -> None:
        """
        Set the filter and wrap parameters on the currently bound texture.

        Must be called while the target texture is bound.

        When ``generate_mipmaps`` is ``True``, ``glGenerateMipmap`` is called
        first and the filter mode is upgraded to the matching mip-map variant:
        ``GL_LINEAR → GL_LINEAR_MIPMAP_LINEAR``,
        ``GL_NEAREST → GL_NEAREST_MIPMAP_NEAREST``.

        Args:
            min_filter:      Minification filter (default ``GL_NEAREST``).
            mag_filter:      Magnification filter (default ``GL_NEAREST``).
            wrap_s:          Horizontal wrap mode (default ``GL_CLAMP_TO_EDGE``).
            wrap_t:          Vertical wrap mode   (default ``GL_CLAMP_TO_EDGE``).
            generate_mipmaps: When ``True``, generate and enable mip-maps.
        """
        if generate_mipmaps:
            # Upgrade the minification filter to its mip-map equivalent so
            # the driver actually samples the generated mip chain.
            if min_filter == GL.GL_LINEAR:
                min_filter = GLenum(GL.GL_LINEAR_MIPMAP_LINEAR)
            elif min_filter == GL.GL_NEAREST:
                min_filter = GLenum(GL.GL_NEAREST_MIPMAP_NEAREST)
            GL.glGenerateMipmap(GLenum(GL.GL_TEXTURE_2D))

        GL.glTexParameteri(
            GLenum(GL.GL_TEXTURE_2D), GLenum(GL.GL_TEXTURE_MIN_FILTER), GLint(min_filter)
        )
        GL.glTexParameteri(
            GLenum(GL.GL_TEXTURE_2D), GLenum(GL.GL_TEXTURE_MAG_FILTER), GLint(mag_filter)
        )
        GL.glTexParameteri(
            GLenum(GL.GL_TEXTURE_2D), GLenum(GL.GL_TEXTURE_WRAP_S), GLint(wrap_s)
        )
        GL.glTexParameteri(
            GLenum(GL.GL_TEXTURE_2D), GLenum(GL.GL_TEXTURE_WRAP_T), GLint(wrap_t)
        )
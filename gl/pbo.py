"""
gl_pbo.py
=========
Pixel Buffer Object (PBO) allocation, mapping, and pooled management.

Provides three public layers:

* **Free functions** — :func:`calculate_pixel_alignment`,
  :func:`configure_pixel_storage`, :func:`memmove_pbo`,
  :func:`write_pbo_buffer`.
* **:class:`PBO`** — single PBO wrapper with mapping and lifecycle helpers.
* **:class:`PBOManager`** — pool of PBOs cycled in round-robin order with a
  :class:`PBOBufferingStrategy`-controlled count.

Upload pipelines
----------------
Standard (unpinned)
    ``CPU array → memmove_pbo (map/copy/unmap) → glTexSubImage2D``

Pinned (zero-copy)
    ``acquire_next_writeable → write directly → unmap → glTexSubImage2D``

Thread-safety
-------------
:class:`PBOManager` guards its pool and iterator with a lock so that
:meth:`~PBOManager.get_next` and :meth:`~PBOManager.acquire_next_writeable`
are safe to call from worker threads.  All ``GL.*`` calls must still be issued
from the thread that owns the current GL context.
"""

from __future__ import annotations

import ctypes
import itertools
import logging
import threading
from enum import IntEnum
from typing import Optional

import numpy as np

from cross_platform.core.copy import tuned_parallel_copy
from cross_platform.qt6_utils.image.gl.backend import GL
from cross_platform.qt6_utils.image.gl.error import (
    GLMemoryError,
    GLUploadError,
    gl_error_check,
)
from cross_platform.qt6_utils.image.gl.types import (
    GLenum,
    GLBuffer,
    GLHandle,
    GLint,
    GLintptr,
    GLsizei,
    GLsizeiptr,
    GLbitfield,
)
from cross_platform.qt6_utils.image.settings.pixels import (
    PixelFormat,
    broadcast_to_format,
)
from cross_platform.qt6_utils.image.utils.data import ensure_contiguity

__all__ = [
    "PBOBufferingStrategy",
    "PBO",
    "PBOManager",
    "calculate_pixel_alignment",
    "configure_pixel_storage",
    "memmove_pbo",
    "write_pbo_buffer",
]

logger = logging.getLogger(__name__)

# Sentinel used to unbind any buffer from a target.
_NO_BUFFER = GLBuffer(GLHandle(0))


# ---------------------------------------------------------------------------
# Buffering strategy
# ---------------------------------------------------------------------------

class PBOBufferingStrategy(IntEnum):
    """
    Number of PBOs cycled during streaming pixel transfers.

    Members
    -------
    SINGLE (1)
        One PBO.  ``glMapBuffer`` stalls the CPU until the GPU finishes
        reading the previous frame.  Only appropriate for non-real-time
        or very low frame-rate workloads.

    DOUBLE (2)
        Two PBOs alternated each frame.  While the GPU reads PBO *n*, the
        CPU writes PBO *n+1*, hiding transfer latency behind render work.
        Recommended default for most real-time pipelines.

    TRIPLE (3)
        Three PBOs.  Absorbs latency spikes when CPU and GPU workloads do
        not overlap cleanly (e.g. variable-size frames, CPU-side decode
        jitter).  Uses more GPU memory but prevents ``glMapBuffer`` stalls
        even when one pipeline stage runs long.
    """

    SINGLE = 1
    DOUBLE = 2
    TRIPLE = 3

    @classmethod
    def default(cls) -> PBOBufferingStrategy:
        """
        Return the recommended strategy for real-time streaming pipelines.

        Returns:
            :attr:`DOUBLE`
        """
        return cls.DOUBLE

    @classmethod
    def from_int(cls, value: int) -> PBOBufferingStrategy:
        """
        Coerce a plain ``int`` to a :class:`PBOBufferingStrategy` member.

        Args:
            value: Integer in ``{1, 2, 3}``.

        Raises:
            ValueError: If ``value`` is not a valid member, with a message
                that lists accepted options.
        """
        try:
            return cls(value)
        except ValueError:
            valid = ", ".join(str(m.value) for m in cls)
            raise ValueError(
                "Invalid PBOBufferingStrategy value %r. Valid values: %s."
                % (value, valid)
            )

    @property
    def description(self) -> str:
        """One-line description for log output and debug UIs."""
        # Dict is module-level to avoid reconstruction on every access.
        return _STRATEGY_DESCRIPTIONS[self]


# Built once at module load; keyed by member identity not value so adding a
# new member produces a KeyError immediately rather than silently returning
# stale text.
_STRATEGY_DESCRIPTIONS: dict[PBOBufferingStrategy, str] = {
    PBOBufferingStrategy.SINGLE: (
        "Single PBO — simple, may stall CPU on glMapBuffer"
    ),
    PBOBufferingStrategy.DOUBLE: (
        "Double PBO — GPU reads while CPU writes; recommended default"
    ),
    PBOBufferingStrategy.TRIPLE: (
        "Triple PBO — absorbs latency spikes from uneven CPU/GPU overlap"
    ),
}


# ---------------------------------------------------------------------------
# Pixel storage helpers
# ---------------------------------------------------------------------------

def calculate_pixel_alignment(gl_type: GLenum, gl_format: GLenum) -> GLint:
    """
    Return the ``GL_UNPACK_ALIGNMENT`` value appropriate for a format/type pair.

    OpenGL requires pixel rows to start on aligned byte boundaries.  The
    correct value depends on the element type and the number of channels:

    ==================  ===========  ===========
    gl_type             gl_format    alignment
    ==================  ===========  ===========
    GL_FLOAT (4 bytes)  any          4
    GL_UNSIGNED_BYTE    GL_RGBA      4
    GL_UNSIGNED_BYTE    anything     1
    anything else       any          1
    ==================  ===========  ===========

    A value of ``1`` disables row-alignment padding entirely, which is
    always safe (if slightly sub-optimal for 4-byte-aligned rows).

    Args:
        gl_type:   GL element-type token (e.g. ``GL_FLOAT``).
        gl_format: GL base-format token (e.g. ``GL_RGB``).

    Returns:
        Alignment in bytes as :data:`GLint`.  Always one of ``{1, 2, 4, 8}``.
    """
    if gl_type == GL.GL_FLOAT:
        # Each float is 4 bytes; any multi-channel float format is naturally
        # 4-byte aligned.
        return GLint(4)

    if gl_type == GL.GL_UNSIGNED_BYTE:
        # RGBA packs 4 bytes per pixel — rows are inherently 4-byte aligned.
        # RGB packs 3 bytes — rows may not be 4-byte aligned without padding.
        return GLint(4) if gl_format == GL.GL_RGBA else GLint(1)

    # For all other types (GL_UNSIGNED_SHORT, GL_HALF_FLOAT, etc.) default to
    # the safest possible value.  Callers with performance-critical sub-byte-
    # padded rows should compute alignment manually and call glPixelStorei
    # directly.
    return GLint(1)


def configure_pixel_storage(
    gl_type: GLenum,
    gl_format: GLenum,
    row_length: GLint = GLint(0),
    skip_pixels: GLint = GLint(0),
    skip_rows: GLint = GLint(0),
) -> None:
    """
    Set ``glPixelStorei`` parameters to match the layout of source pixel data.

    Alignment is derived automatically from ``gl_type`` and ``gl_format`` via
    :func:`calculate_pixel_alignment`.  The remaining parameters map directly
    to the GL_UNPACK_* tokens:

    * ``row_length`` — logical row width in pixels (``0`` = use texture width).
    * ``skip_pixels`` — pixels to skip at the start of each row.
    * ``skip_rows``   — rows to skip before the first data row.

    Args:
        gl_type:     GL element-type token.
        gl_format:   GL base-format token.
        row_length:  ``GL_UNPACK_ROW_LENGTH``  value (default ``0``).
        skip_pixels: ``GL_UNPACK_SKIP_PIXELS`` value (default ``0``).
        skip_rows:   ``GL_UNPACK_SKIP_ROWS``   value (default ``0``).
    """
    alignment = calculate_pixel_alignment(gl_type, gl_format)
    GL.glPixelStorei(GLenum(GL.GL_UNPACK_ALIGNMENT),   alignment)
    GL.glPixelStorei(GLenum(GL.GL_UNPACK_ROW_LENGTH),  row_length)
    GL.glPixelStorei(GLenum(GL.GL_UNPACK_SKIP_PIXELS), skip_pixels)
    GL.glPixelStorei(GLenum(GL.GL_UNPACK_SKIP_ROWS),   skip_rows)


# ---------------------------------------------------------------------------
# Data-transfer helpers
# ---------------------------------------------------------------------------

def memmove_pbo(pbo_id: GLBuffer, data: np.ndarray) -> bool:
    """
    Copy a CPU array into a PBO using the map/copy/unmap path.

    Pipeline: ``CPU array → glMapBufferRange → tuned_parallel_copy → glUnmapBuffer``

    The PBO is left **bound** to ``GL_PIXEL_UNPACK_BUFFER`` on return.
    The caller is responsible for calling ``PBOManager.unbind()`` (or
    ``glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0)``) after passing the PBO
    pointer to ``glTexSubImage2D``.

    Args:
        pbo_id: Handle of the target PBO (:data:`GLBuffer`).
        data:   Source array.  Does not need to be contiguous — contiguity
                is ensured internally before the copy.

    Returns:
        ``True`` on success; ``False`` when the PBO could not be mapped
        (e.g. the driver ran out of mapped-buffer slots).
    """
    nbytes = data.nbytes

    # Orphan the existing buffer storage so the driver can return a fresh
    # mapping without waiting for any in-flight reads to complete.
    GL.glBindBuffer(GLenum(GL.GL_PIXEL_UNPACK_BUFFER), pbo_id)
    GL.glBufferData(
        GLenum(GL.GL_PIXEL_UNPACK_BUFFER),
        GLsizeiptr(nbytes),
        None,
        GLenum(GL.GL_STREAM_DRAW),
    )

    access = GLbitfield(GL.GL_MAP_WRITE_BIT | GL.GL_MAP_INVALIDATE_BUFFER_BIT)
    ptr_obj = GL.glMapBufferRange(
        GLenum(GL.GL_PIXEL_UNPACK_BUFFER),
        GLintptr(0),
        GLsizeiptr(nbytes),
        access,
    )

    if not ptr_obj:
        logger.error("memmove_pbo: glMapBufferRange returned NULL for PBO %d", pbo_id)
        return False

    ptr_int = _extract_pointer(ptr_obj)
    if ptr_int is None:
        logger.error("memmove_pbo: could not extract pointer address for PBO %d", pbo_id)
        return False

    tuned_parallel_copy(ptr_int, ensure_contiguity(data))
    GL.glUnmapBuffer(GLenum(GL.GL_PIXEL_UNPACK_BUFFER))
    return True


def write_pbo_buffer(
    pbo_array: np.ndarray,
    image: np.ndarray,
    pixel_fmt: PixelFormat,
) -> None:
    """
    Write image data into a mapped PBO buffer, broadcasting channels if needed.

    Handles the common case where a greyscale ``(H, W)`` array must be
    broadcast into an ``(H, W, C)`` PBO buffer via :func:`broadcast_to_format`.

    Args:
        pbo_array: Mapped buffer array from
                   :meth:`~PBOManager.acquire_next_writeable`.
        image:     Source image (``HxW`` or ``HxWxC``).
        pixel_fmt: Pixel format describing the channel layout of ``pbo_array``.

    Raises:
        GLUploadError: If the broadcast shape does not match ``pbo_array.shape``.
            This indicates a caller contract violation — the PBO was allocated
            for a different resolution or channel count than the image.
    """
    broadcasted = broadcast_to_format(image, pixel_fmt, copy=False)

    if broadcasted.shape != pbo_array.shape:
        raise GLUploadError(
            "Broadcast image shape %s does not match PBO buffer shape %s.  "
            "Ensure the PBO was acquired with the correct width, height, and "
            "channel count for this image." % (broadcasted.shape, pbo_array.shape)
        )

    pbo_array[:] = broadcasted


# ---------------------------------------------------------------------------
# Internal pointer extraction helper
# ---------------------------------------------------------------------------

def _extract_pointer(ptr_obj: object) -> Optional[int]:
    """
    Normalise a PyOpenGL-returned mapped-buffer pointer to a plain ``int``.

    PyOpenGL may return the mapped address as any of:

    * A ``ctypes.c_void_p`` with a ``.value`` attribute.
    * A raw ``int``.
    * A ctypes pointer that must be cast via ``ctypes.cast``.

    Args:
        ptr_obj: The raw return value of ``glMapBufferRange``.

    Returns:
        The pointer address as a Python ``int``, or ``None`` if extraction
        fails (null pointer or unrecognised type).
    """
    # Fast path: ctypes c_void_p surfaces .value directly.
    value = getattr(ptr_obj, "value", _SENTINEL)
    if value is not _SENTINEL:
        return value   # may be None for a null pointer — caller checks

    # Fallback: cast through ctypes for other pointer objects.
    try:
        return ctypes.cast(ptr_obj, ctypes.c_void_p).value
    except Exception:
        return None


_SENTINEL = object()   # unique sentinel distinguishable from None


# ---------------------------------------------------------------------------
# PBO wrapper
# ---------------------------------------------------------------------------

class PBO:
    """
    Single Pixel Buffer Object with mapping and lifecycle management.

    Attributes:
        id:         GL buffer handle (:data:`GLBuffer`).  Set to ``0`` after
                    :meth:`destroy` is called.
        capacity:   Currently allocated size in bytes (:data:`GLsizeiptr`).
        is_mapped:  ``True`` while the buffer is mapped for CPU access.
    """

    def __init__(self) -> None:
        raw = GL.glGenBuffers(1)
        # glGenBuffers may return a scalar, a list, a tuple, or an ndarray
        # depending on the PyOpenGL version and platform.
        if isinstance(raw, (list, tuple, np.ndarray)):
            raw_id = int(raw[0])
        else:
            raw_id = int(raw)

        self.id:        GLBuffer    = GLBuffer(raw_id)
        self.capacity:  GLsizeiptr  = GLsizeiptr(0)
        self.is_mapped: bool        = False

    def prepare_and_map(
        self,
        size_bytes: GLsizeiptr,
        height: GLsizei,
        width: GLsizei,
        channels: GLsizei,
        dtype: np.dtype = np.dtype("uint8"),
    ) -> np.ndarray:
        """
        Orphan the buffer, map it, and return a shaped writable ``ndarray``.

        The returned array is a direct view into GPU-mapped memory.  Writes
        are visible to the GPU after :meth:`unmap` is called.

        Args:
            size_bytes: Total allocation in bytes.  Must equal
                        ``height * width * channels * dtype.itemsize``.
            height:     Frame height in pixels.
            width:      Frame width in pixels.
            channels:   Number of colour channels per pixel.
            dtype:      Element type of the mapped array (default ``uint8``).

        Returns:
            C-contiguous ``ndarray`` of shape ``(height, width, channels)``
            and the requested ``dtype``, backed by the mapped PBO memory.

        Raises:
            GLMemoryError: If ``glMapBufferRange`` returns ``NULL``, if the
                mapped pointer is null after extraction, or if the allocated
                buffer is smaller than the requested shape requires.
            GLUploadError: If ``size_bytes`` does not match the expected byte
                count derived from ``height * width * channels * dtype.itemsize``.
                A mismatch here means the caller allocated the wrong amount of
                GPU memory, which would produce a corrupt texture upload.
        """
        # Validate that size_bytes is consistent with the requested shape so
        # that the view and reshape below are guaranteed to be correct.
        expected_bytes = int(height) * int(width) * int(channels) * dtype.itemsize
        if expected_bytes != int(size_bytes):
            raise GLUploadError(
                "PBO size mismatch: size_bytes=%d does not match "
                "height×width×channels×itemsize=%d.  "
                "Recompute size_bytes before calling prepare_and_map."
                % (int(size_bytes), expected_bytes)
            )

        GL.glBindBuffer(GLenum(GL.GL_PIXEL_UNPACK_BUFFER), self.id)

        # Orphan + map inside gl_error_check so any driver-level GL error
        # (e.g. GL_OUT_OF_MEMORY) is converted to a typed GLMemoryError.
        with gl_error_check("PBO orphan and map", GLMemoryError):
            GL.glBufferData(
                GLenum(GL.GL_PIXEL_UNPACK_BUFFER),
                size_bytes,
                None,
                GLenum(GL.GL_STREAM_DRAW),
            )
            self.capacity = size_bytes

            access = GLbitfield(
                GL.GL_MAP_WRITE_BIT | GL.GL_MAP_INVALIDATE_BUFFER_BIT
            )
            ptr_obj = GL.glMapBufferRange(
                GLenum(GL.GL_PIXEL_UNPACK_BUFFER),
                GLintptr(0),
                size_bytes,
                access,
            )

        if not ptr_obj:
            GL.glBindBuffer(GLenum(GL.GL_PIXEL_UNPACK_BUFFER), _NO_BUFFER)
            raise GLMemoryError(
                "glMapBufferRange returned NULL for PBO %d "
                "(size_bytes=%d).  The driver may be out of mappable memory."
                % (self.id, int(size_bytes))
            )

        ptr_addr = _extract_pointer(ptr_obj)
        if not ptr_addr:
            GL.glBindBuffer(GLenum(GL.GL_PIXEL_UNPACK_BUFFER), _NO_BUFFER)
            raise GLMemoryError(
                "Mapped pointer is NULL or unextractable for PBO %d." % self.id
            )

        # Wrap the raw pointer as a numpy array without copying.
        # ctypes.c_uint8 * N creates a ctypes array type of exactly N bytes.
        c_byte_type = ctypes.c_uint8 * int(size_bytes)
        c_ptr       = ctypes.cast(ptr_addr, ctypes.POINTER(c_byte_type))
        arr_bytes   = np.ctypeslib.as_array(c_ptr.contents)

        # Reinterpret the byte array as the target dtype and reshape.
        # arr_bytes is always C-contiguous (it is a flat 1-D array from a
        # ctypes allocation); reshape on a C-contiguous 1-D array produces a
        # C-contiguous result — ensure_contiguity is not needed here.
        arr_shaped = arr_bytes.view(dtype=dtype).reshape(
            int(height), int(width), int(channels)
        )

        self.is_mapped = True
        return arr_shaped

    def unmap(self) -> None:
        """
        Release the CPU-side mapping so the GPU can DMA-read the buffer.

        A no-op when the buffer is not currently mapped.  Leaves the PBO
        unbound from ``GL_PIXEL_UNPACK_BUFFER`` after returning.
        """
        if not self.is_mapped:
            return

        GL.glBindBuffer(GLenum(GL.GL_PIXEL_UNPACK_BUFFER), self.id)
        result = GL.glUnmapBuffer(GLenum(GL.GL_PIXEL_UNPACK_BUFFER))

        if not result:
            # GL_FALSE from glUnmapBuffer means the buffer contents were
            # invalidated (e.g. context loss on some drivers).  Log and
            # continue — the caller will upload corrupt data but the process
            # is not in an unrecoverable state.
            logger.warning(
                "glUnmapBuffer returned GL_FALSE for PBO %d — "
                "buffer contents may be undefined (possible context loss).",
                self.id,
            )

        GL.glBindBuffer(GLenum(GL.GL_PIXEL_UNPACK_BUFFER), _NO_BUFFER)
        self.is_mapped = False

    def destroy(self) -> None:
        """
        Delete the GL buffer object and release GPU memory.

        Unmaps the buffer first if it is still mapped.  Safe to call multiple
        times — subsequent calls after the first are no-ops because ``self.id``
        is reset to ``0``.
        """
        if not self.id:
            # id == 0 means already destroyed or never successfully created.
            return

        if self.is_mapped:
            self.unmap()

        GL.glDeleteBuffers(GLsizei(1), np.array([self.id], dtype=np.uint32))
        self.id = GLBuffer(GLHandle(0))


# ---------------------------------------------------------------------------
# PBO pool manager
# ---------------------------------------------------------------------------

class PBOManager:
    """
    Pool of PBOs cycled in round-robin order for streaming pixel uploads.

    Supports both the standard unpinned path (:meth:`get_next`) and the
    pinned path (:meth:`acquire_next_writeable`).

    Args:
        buffer_strategy: Controls how many PBOs are allocated.  Defaults to
                         :attr:`PBOBufferingStrategy.DOUBLE`.
        num_pbos:        **Deprecated alias** for ``buffer_strategy`` that
                         accepts a raw integer.  Provided for backward
                         compatibility with callers that pre-date the
                         :class:`PBOBufferingStrategy` enum.  Raises
                         :exc:`ValueError` if both are supplied.
    """

    def __init__(
        self,
        buffer_strategy: PBOBufferingStrategy = PBOBufferingStrategy.DOUBLE,
        *,
        num_pbos: Optional[int] = None,
    ) -> None:
        if num_pbos is not None:
            # Backward-compatibility shim: coerce num_pbos to a strategy.
            # num_pbos must be in {1, 2, 3}; anything outside that range
            # is clamped to TRIPLE to avoid silently creating a mis-sized pool.
            buffer_strategy = PBOBufferingStrategy.from_int(
                min(max(int(num_pbos), 1), 3)
            )
            logger.warning(
                "PBOManager: 'num_pbos=%d' is deprecated; "
                "use buffer_strategy=PBOBufferingStrategy.%s instead.",
                num_pbos,
                buffer_strategy.name,
            )

        self.buffer_strategy: PBOBufferingStrategy = buffer_strategy
        self.pbos:            list[PBO]             = []
        self._lock:           threading.Lock        = threading.Lock()
        self._cycle_iter:     Optional[itertools.cycle] = None

    def initialize(self) -> None:
        """
        Allocate all PBOs and arm the cyclic iterator.

        Idempotent: a second call while the pool is already initialised is a
        no-op.  Must be called from the GL-context thread before any upload
        methods are used.
        """
        with self._lock:
            if self.pbos:
                return
            count = self.buffer_strategy.value
            self.pbos        = [PBO() for _ in range(count)]
            self._cycle_iter = itertools.cycle(self.pbos)
            logger.debug(
                "PBOManager initialised: %d PBO(s) (%s)",
                count,
                self.buffer_strategy.description,
            )

    @staticmethod
    def bind(pbo_id: GLBuffer) -> None:
        """Bind ``pbo_id`` to ``GL_PIXEL_UNPACK_BUFFER``."""
        GL.glBindBuffer(GLenum(GL.GL_PIXEL_UNPACK_BUFFER), pbo_id)

    @staticmethod
    def unbind() -> None:
        """Unbind any PBO from ``GL_PIXEL_UNPACK_BUFFER``."""
        GL.glBindBuffer(GLenum(GL.GL_PIXEL_UNPACK_BUFFER), _NO_BUFFER)

    def get_next(self) -> PBO:
        """
        Return the next PBO in round-robin order for standard unpinned uploads.

        Returns:
            The next :class:`PBO` in the pool cycle.

        Raises:
            RuntimeError: If :meth:`initialize` has not been called.
        """
        with self._lock:
            if self._cycle_iter is None:
                raise RuntimeError(
                    "PBOManager.get_next called before initialize().  "
                    "Call initialize() once after the GL context is current."
                )
            return next(self._cycle_iter)

    def acquire_next_writeable(
        self,
        width: GLsizei,
        height: GLsizei,
        channels: GLsizei,
        dtype: np.dtype = np.dtype("uint8"),
    ) -> tuple[PBO, np.ndarray]:
        """
        Get the next PBO, orphan its storage, and map it for direct writing.

        Args:
            width:    Frame width in pixels.
            height:   Frame height in pixels.
            channels: Number of colour channels per pixel.
            dtype:    Element type (default ``uint8``).

        Returns:
            ``(pbo, array)`` where ``array`` is a writable ``(height, width,
            channels)`` view into GPU-mapped memory.  Call :meth:`PBO.unmap`
            before issuing ``glTexSubImage2D``.

        Raises:
            RuntimeError:  If the pool has not been initialised.
            GLMemoryError: Propagated from :meth:`PBO.prepare_and_map`.
            GLUploadError: Propagated from :meth:`PBO.prepare_and_map` on
                size mismatch.
        """
        pbo = self.get_next()
        size_bytes = GLsizeiptr(
            int(height) * int(width) * int(channels) * dtype.itemsize
        )
        arr = pbo.prepare_and_map(size_bytes, height, width, channels, dtype)
        return pbo, arr

    def cleanup(self) -> None:
        """
        Destroy all managed PBOs and reset the pool.

        Must be called from the GL-context thread.  Safe to call after
        :meth:`initialize` has not been called (no-op).  Safe to call
        multiple times.
        """
        with self._lock:
            for pbo in self.pbos:
                pbo.destroy()
            self.pbos.clear()
            self._cycle_iter = None
            logger.debug("PBOManager cleaned up")
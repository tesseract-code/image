from __future__ import annotations

import ctypes
import itertools
import logging
import threading
from typing import Optional

import numpy as np

from image.gl.backend import GL
from image.gl.errors import GLUploadError, gl_error_check, GLMemoryError
from image.gl.pbo import PBO
from image.gl.pbo.constants import _NO_BUFFER
from image.gl.pbo.strategy import PBOBufferingStrategy
from image.gl.pbo.utils import extract_pointer
from image.gl.types import GLintptr, GLsizeiptr, GLbitfield
from pycore.log.ctx import ContextAdapter

logger = ContextAdapter(logging.getLogger(__name__), {})

class UnpackPBO(PBO):
    """
    PBO specialised for pixel uploads (GL_PIXEL_UNPACK_BUFFER).

    Provides orphan + map for writing (prepare_and_map) and unmap.
    Used exclusively by PBOUploadManager.
    """

    _target: int = GL.GL_PIXEL_UNPACK_BUFFER

    def prepare_and_map(
            self,
            size_bytes: int,
            height: int,
            width: int,
            channels: int,
            dtype: np.dtype = np.dtype("uint8"),
    ) -> np.ndarray:
        """
        Orphan the buffer, map it for writing, and return a shaped ndarray.

        The returned array is a direct view into GPU-mapped memory.  Writes
        are visible to the GPU after unmap() is called.

        Args:
            size_bytes : Total allocation in bytes.
                         Must equal height * width * channels * dtype.itemsize.
            height, width, channels : Frame dimensions and channel count.
            dtype      : Element type of the mapped array (default uint8).

        Returns:
            C-contiguous ndarray of shape (height, width, channels).

        Raises:
            GLUploadError : If size_bytes does not match the expected byte count.
            GLMemoryError : If glMapBufferRange returns NULL.
        """
        expected = height * width * channels * dtype.itemsize
        if expected != size_bytes:
            raise GLUploadError(
                "PBO size mismatch: size_bytes=%d does not match "
                "height×width×channels×itemsize=%d." % (size_bytes, expected)
            )

        GL.glBindBuffer(self._target, self.id)

        with gl_error_check("PBO orphan and map (write)", GLMemoryError):
            GL.glBufferData(self._target, size_bytes, None, GL.GL_STREAM_DRAW)
            self.capacity = size_bytes
            ptr_obj = GL.glMapBufferRange(
                self._target,
                GLintptr(0),
                GLsizeiptr(size_bytes),
                GLbitfield(
                    GL.GL_MAP_WRITE_BIT | GL.GL_MAP_INVALIDATE_BUFFER_BIT),
            )

        if not ptr_obj:
            GL.glBindBuffer(self._target, _NO_BUFFER)
            raise GLMemoryError(
                "glMapBufferRange returned NULL for PBO %d (size=%d)."
                % (self.id, size_bytes)
            )

        ptr_addr = extract_pointer(ptr_obj)
        if not ptr_addr:
            GL.glBindBuffer(self._target, _NO_BUFFER)
            raise GLMemoryError(
                "Mapped pointer is NULL for PBO %d." % self.id
            )

        c_byte_type = ctypes.c_uint8 * size_bytes
        c_ptr = ctypes.cast(ptr_addr, ctypes.POINTER(c_byte_type))
        arr_bytes = np.ctypeslib.as_array(c_ptr.contents)
        arr_shaped = arr_bytes.view(dtype=dtype).reshape(height, width,
                                                         channels)

        self.is_mapped = True
        return arr_shaped

    def unmap(self) -> None:
        """Release the CPU-side mapping so the GPU can DMA-read the buffer."""
        if not self.is_mapped:
            return
        GL.glBindBuffer(self._target, self.id)
        result = GL.glUnmapBuffer(self._target)
        if not result:
            logger.warning(
                "glUnmapBuffer returned GL_FALSE for PBO %d "
                "(possible context loss).", self.id
            )
        GL.glBindBuffer(self._target, _NO_BUFFER)
        self.is_mapped = False

    def destroy(self) -> None:
        if self.id and self.is_mapped:
            self.unmap()
        super().destroy()


class PBOUploadManager:
    """
    Pool of UnpackPBOs cycled in round-robin order for streaming texture uploads.

    Supports both the standard unpinned path (get_next + memmove_pbo) and the
    pinned path (acquire_next_writeable).

    Args:
        buffer_strategy : Controls how many PBOs are allocated.
                          Defaults to PBOBufferingStrategy.DOUBLE.
        num_pbos        : Deprecated alias for buffer_strategy (accepts a raw
                          int).  Raises ValueError if both are supplied.

    Thread-safety
    -------------
    get_next and acquire_next_writeable are safe to call from worker threads.
    All GL calls must be issued from the GL-context thread.
    """

    def __init__(
            self,
            buffer_strategy: PBOBufferingStrategy = PBOBufferingStrategy.DOUBLE,
            *,
            num_pbos: Optional[int] = None,
    ) -> None:
        if num_pbos is not None:
            buffer_strategy = PBOBufferingStrategy.from_int(
                min(max(int(num_pbos), 1), 3)
            )
            logger.warning(
                "PBOUploadManager: 'num_pbos=%d' is deprecated; "
                "use buffer_strategy=PBOBufferingStrategy.%s instead.",
                num_pbos, buffer_strategy.name,
            )

        self.buffer_strategy: PBOBufferingStrategy = buffer_strategy
        self.pbos: list[UnpackPBO] = []
        self._lock: threading.Lock = threading.Lock()
        self._cycle_iter: Optional[itertools.cycle] = None

    # -- lifecycle ----------------------------------------------------------

    def initialize(self) -> None:
        """
        Allocate all PBOs and arm the cyclic iterator.

        Idempotent — a second call while already initialised is a no-op.
        Must be called from the GL-context thread before any upload methods.
        """
        with self._lock:
            if self.pbos:
                return
            count = self.buffer_strategy.value
            self.pbos = [UnpackPBO() for _ in range(count)]
            self._cycle_iter = itertools.cycle(self.pbos)
            logger.debug(
                "PBOUploadManager initialised: %d PBO(s) (%s)",
                count, self.buffer_strategy.description,
            )

    def cleanup(self) -> None:
        """
        Destroy all managed PBOs and reset the pool.

        Must be called from the GL-context thread.  Safe to call before
        initialize or multiple times.
        """
        with self._lock:
            for pbo in self.pbos:
                pbo.destroy()
            self.pbos.clear()
            self._cycle_iter = None
            logger.debug("PBOUploadManager cleaned up")

    # -- static helpers -----------------------------------------------------

    @staticmethod
    def bind(pbo_id: int) -> None:
        """Bind pbo_id to GL_PIXEL_UNPACK_BUFFER."""
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo_id)

    @staticmethod
    def unbind() -> None:
        """Unbind any PBO from GL_PIXEL_UNPACK_BUFFER."""
        GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, _NO_BUFFER)

    # -- upload interface ---------------------------------------------------

    def get_next(self) -> UnpackPBO:
        """
        Return the next PBO in round-robin order for standard unpinned uploads.

        Raises:
            RuntimeError : If initialize() has not been called.
        """
        with self._lock:
            if self._cycle_iter is None:
                raise RuntimeError(
                    "PBOUploadManager.get_next called before initialize()."
                )
            return next(self._cycle_iter)

    def acquire_next_writeable(
            self,
            width: int,
            height: int,
            channels: int,
            dtype: np.dtype = np.dtype("uint8"),
    ) -> tuple[UnpackPBO, np.ndarray]:
        """
        Get the next PBO, orphan its storage, and map it for direct writing.

        Args:
            width, height, channels : Frame dimensions and channel count.
            dtype                   : Element type (default uint8).

        Returns:
            (pbo, array) where array is a writable (height, width, channels)
            view into GPU-mapped memory.  Call pbo.unmap() before
            glTexSubImage2D.

        Raises:
            RuntimeError  : If the pool has not been initialised.
            GLMemoryError : Propagated from UnpackPBO.prepare_and_map.
            GLUploadError : Propagated from UnpackPBO.prepare_and_map.
        """
        pbo = self.get_next()
        size_bytes = height * width * channels * dtype.itemsize
        arr = pbo.prepare_and_map(size_bytes, height, width, channels, dtype)
        return pbo, arr

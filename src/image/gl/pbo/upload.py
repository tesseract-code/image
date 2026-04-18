from __future__ import annotations

import itertools
import logging
import threading
from typing import Optional

import numpy as np

from image.gl.backend import GL
from image.gl.pbo import UnpackPBO
from image.gl.pbo.constants import _NO_BUFFER
from image.gl.pbo.strategy import PBOBufferingStrategy
from pycore.log.ctx import ContextAdapter

logger = ContextAdapter(logging.getLogger(__name__), {})


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

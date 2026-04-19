"""

"""
import ctypes
import logging
import multiprocessing
from typing import Tuple, Any, Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot

from image.pipeline.frame import FrameHeader, FrameSettings
from image.utils.data import ensure_contiguity
from pycore.mtcopy import tuned_parallel_copy
from pycore.shm import SharedMemoryRingBuffer

logger = logging.getLogger(__name__)


class FrameSubmitterCore:
    def __init__(self, input_queue: multiprocessing.Queue, buffer_count: int):
        self.input_queue = input_queue
        # Default safety count
        _count = max(1, buffer_count)
        self._shm_ring = SharedMemoryRingBuffer(buffer_count=_count)
        self._submitted_frame_count: multiprocessing.Value = multiprocessing.Value(
            'i', 0)
        self._active = True

    @property
    def submitted_frame_count(self) -> int:
        with self._submitted_frame_count.get_lock():
            return self._submitted_frame_count.value

    @staticmethod
    def _write_header(shm_view: np.ndarray, header: FrameHeader) -> None:
        header_bytes = header.pack()
        # Direct memory write to offset 0
        src_ptr = ctypes.c_char_p(header_bytes)
        dst_ptr = shm_view.ctypes.data
        ctypes.memmove(dst_ptr, src_ptr, len(header_bytes))

    def submit_image(self,
                     image: np.ndarray,
                     settings: FrameSettings,
                     timestamp: float = 0.0) -> bool:
        """
        Allocates SHM and copies image data.
        Safe against Jitter Padding because we use ctypes.memmove with known input size.
        """
        if not self._active: return False

        try:
            if self.input_queue.full(): return False

            _raw = ensure_contiguity(image)
            height, width = _raw.shape[:2]
            channels = _raw.shape[2] if _raw.ndim > 2 else 1

            header = FrameHeader(
                frame_id=0.0,
                timestamp=timestamp,
                width=width,
                height=height,
                channels=channels
            )

            total_req_size = FrameHeader.SIZE_BYTES + _raw.nbytes

            # Alloc returns a buffer that might be 1.5x larger than total_req_size
            name, full_buffer_view = self._shm_ring.alloc_buffer(total_req_size)

            # Write Header (Offset 0)
            self._write_header(full_buffer_view, header)

            # Write Image (Offset 32)
            # We use the SOURCE image size for memmove, so trailing padding in dst is ignored
            dst_ptr = full_buffer_view.ctypes.data + FrameHeader.SIZE_BYTES
            tuned_parallel_copy(dst_ptr=dst_ptr, src_data=_raw)

            # ctypes.memmove(dst_ptr, src_ptr, _raw.nbytes)

            self.input_queue.put((name, image.shape, image.dtype, settings),
                                 block=True)
            with self._submitted_frame_count.get_lock():
                self._submitted_frame_count.value += 1
            return True

        except Exception as e:
            logger.exception(f"Core Submission Error: {e}")
            return False

    def alloc_shm_write_buffer(self, size_bytes: int) -> Tuple[str, np.ndarray]:
        """
        Zero-Copy Prep.
        CRITICAL FIX: Strictly slices the return view to exclude Jitter Padding.
        """
        req_size = size_bytes + FrameHeader.SIZE_BYTES

        # Ring might return a buffer of size (req_size * 1.5)
        name, full_view = self._shm_ring.alloc_buffer(req_size)

        # We must return a view that is EXACTLY size_bytes long.
        # Start at 32 (skip header)
        start_idx = FrameHeader.SIZE_BYTES
        # End at 32 + requested_size (ignore the 1.5x tail)
        end_idx = start_idx + size_bytes

        # Strict slicing ensures user can .reshape() this view successfully
        data_view = full_view[start_idx:end_idx]

        return name, data_view

    def submit_shm_buffer(self,
                          shm_name: str,
                          shape: Tuple[int, ...],
                          dtype: np.dtype,
                          config: Any,
                          timestamp: float = 0.0) -> bool:
        # (Logic identical to previous, just ensures header is written)
        try:
            if self.input_queue.full(): return False

            full_view = self._shm_ring.get_view(shm_name)
            if full_view is None: return False

            h, w = shape[:2]
            c = shape[2] if len(shape) > 2 else 1

            header = FrameHeader(
                frame_id=0.0, timestamp=timestamp, width=w, height=h, channels=c
            )
            self._write_header(full_view, header)
            self.input_queue.put((shm_name, shape, dtype, config))
            with self._submitted_frame_count.get_lock():
                self._submitted_frame_count.value += 1
            return True
        except Exception as e:
            logger.error(f"Core Prefill Error: {e}")
            return False

    def cleanup(self):
        self._active = False
        if self._shm_ring:
            self._shm_ring.cleanup()
            self._shm_ring = None


class FrameSubmitter(QObject):
    _cleaup_req = pyqtSignal()

    def __init__(self, input_queue: multiprocessing.Queue, buffer_count: int,
                 parent: Optional[QObject] = None):
        super().__init__(parent)
        self._core = FrameSubmitterCore(input_queue, buffer_count)
        self._cleaup_req.connect(self._on_cleanup)

    @property
    def submitted_frame_count(self) -> int:
        return self._core.submitted_frame_count

    @pyqtSlot(np.ndarray, dict)
    def submit_image(self, image: np.ndarray, settings: FrameSettings) -> bool:
        ts = settings.get('timestamp', 0.0) if isinstance(settings,
                                                          dict) else 0.0
        return self._core.submit_image(image, settings, timestamp=ts)

    def alloc_shm_write_buffer(self, size_bytes: int) -> Tuple[str, np.ndarray]:
        return self._core.alloc_shm_write_buffer(size_bytes)

    def submit_shm_buffer(self, shm_name: str, shape: Tuple[int, ...],
                          dtype: np.dtype, config: Any) -> bool:
        ts = config.get('timestamp', 0.0) if isinstance(config, dict) else 0.0
        return self._core.submit_shm_buffer(shm_name, shape, dtype, config,
                                            timestamp=ts)

    @pyqtSlot()
    def _on_cleanup(self):
        self._core.cleanup()

    def cleanup(self):
        self._cleaup_req.emit()

    def __del__(self):
        if hasattr(self, '_core'): self._on_cleanup()

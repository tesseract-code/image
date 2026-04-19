import logging
import multiprocessing
import queue
from multiprocessing import shared_memory
from typing import Dict, Optional, Callable, Any

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QSignalBlocker

from pycore.shm import cleanup_shm_cache
from qtcore.monitor import (PerformanceMonitor,PerfStats)
from qtcore.reference import has_qt_cpp_binding
from image.pipeline.frame import (FrameHeader,
                                                           RenderFrame)
from image.pipeline.stats import FrameStats
from image.pipeline.mailbox import FrameMailbox

logger = logging.getLogger(__name__)


class FrameReceiverCore:
    """
    Pure Python Receiver Logic.
    Handles Queue -> SHM Resolution -> Header Unpacking -> Mailbox.
    """

    def __init__(self,
                 output_queue: multiprocessing.Queue,
                 stop_event: multiprocessing.Event,
                 mailbox: Optional[FrameMailbox] = None,
                 track_metrics: bool = False,
                 # Callbacks for events (instead of Signals)
                 on_frame_ready: Optional[
                     Callable[[np.ndarray, Any], None]] = None,
                 on_perf_stats: Optional[Callable[[PerfStats], None]] = None):

        self.output_queue = output_queue
        self._stop_event = stop_event
        self.mailbox = mailbox

        self._is_tracking_perf = track_metrics
        self._monitor = PerformanceMonitor() if track_metrics else None

        # Callbacks
        self._on_frame_ready = on_frame_ready
        self._on_perf_stats = on_perf_stats

        self._shm_cache: Dict[str, shared_memory.SharedMemory] = {}

    def _get_shm_obj(self, shm_name: str) -> shared_memory.SharedMemory:
        if shm_name not in self._shm_cache:
            try:
                # Open existing SHM
                shm = shared_memory.SharedMemory(name=shm_name)
                self._shm_cache[shm_name] = shm
            except FileNotFoundError:
                logger.error(f"SHM Block {shm_name} not found.")
                raise
        return self._shm_cache[shm_name]

    def _resolve_frame_from_shm(self, shm_name: str,
                                metadata: FrameStats) -> np.ndarray:
        """
        Reads SHM, unpacks Header, returns View of Pixel Data.
        """
        shm_obj = self._get_shm_obj(shm_name)

        # 1. Read Header (First 32 bytes)
        # We trust the SHM header for dimensions, as it travels with the pixels.
        header_bytes = shm_obj.buf[:FrameHeader.SIZE_BYTES]
        header = FrameHeader.unpack(header_bytes)

        # Update metadata timestamps from header if needed
        # metadata.timestamp = header.timestamp 

        # 2. Determine Dtype
        # Header doesn't store dtype (complex mapping), we rely on metadata/queue
        dtype = np.dtype(getattr(metadata, 'dtype_str', 'uint8'))

        # 3. Create View (Offset by 32 bytes)
        # This is the "Zero Copy" view into the SHM
        img_view = np.ndarray(
            (header.height, header.width, header.channels),
            dtype=dtype,
            buffer=shm_obj.buf,
            offset=FrameHeader.SIZE_BYTES
        )

        # Handle squeezing 1-channel if necessary based on your pipeline conventions
        if header.channels == 1 and len(img_view.shape) == 3:
            img_view = img_view.squeeze(axis=2)

        return img_view

    def run_loop(self):
        """Blocking loop intended to be run in a worker thread."""
        try:
            while not self._stop_event.is_set():
                try:
                    task = self.output_queue.get(timeout=0.05)
                except queue.Empty:
                    continue

                if not isinstance(task, tuple) or len(task) != 2:
                    continue

                shm_name, metadata = task

                try:
                    img_view = self._resolve_frame_from_shm(shm_name, metadata)

                    # Path A: Mailbox (Double Buffer)
                    if self.mailbox is not None:
                        self.mailbox.write(RenderFrame(img_view, metadata))
                        # Notify that mailbox has data (optional signal logic handled by wrapper)

                    # Path B: Direct Callback (if configured)
                    elif self._on_frame_ready:
                        # Note: This view is volatile! Receiver must copy if async.
                        self._on_frame_ready(img_view, metadata)

                    # Metrics
                    if self._is_tracking_perf and self._monitor:
                        stats = self._monitor.update(
                            metadata.processing_time_ms)
                        if self._on_perf_stats:
                            self._on_perf_stats(stats)

                except Exception as inner_e:
                    logger.error(f"Frame processing error: {inner_e}")

        except Exception as e:
            logger.error(f"Receiver Core Critical: {e}")

    def cleanup(self):
        cleanup_shm_cache(self._shm_cache, unlink=False)
        self._shm_cache.clear()
        if self._monitor:
            self._monitor = None


# -----------------------------------------------------------------------------
# 3. Qt Interface Wrapper
# -----------------------------------------------------------------------------

class FrameReceiver(QObject):
    """
    Qt Wrapper for FrameReceiverCore.
    Maintains the existing API surface (Signals/Slots).
    """
    # Signals required by existing interface
    processed_img = pyqtSignal(np.ndarray, object)
    have_mail = pyqtSignal()  # Optional: signal when mailbox updates
    perf_stats = pyqtSignal(PerfStats)
    _cleanup_req = pyqtSignal()

    def __init__(self,
                 output_queue: multiprocessing.Queue,
                 stop_event: multiprocessing.Event,
                 mailbox: Optional[FrameMailbox] = None,
                 job_id: str = "worker_1",
                 track_metrics: bool = False):
        super().__init__()
        self._job_id: str = job_id
        self.mailbox = mailbox

        self._core = FrameReceiverCore(
            output_queue=output_queue,
            stop_event=stop_event,
            mailbox=mailbox,
            track_metrics=track_metrics,
            on_frame_ready=self._on_core_frame_ready,
            on_perf_stats=self._on_core_perf_stats
        )

        self._cleanup_req.connect(self._on_cleanup)

    def _on_core_frame_ready(self, img: np.ndarray, meta: Any):
        """Callback from Core -> Emit Qt Signal."""
        pass

    def _on_core_perf_stats(self, stats: PerfStats):
        self.perf_stats.emit(stats)

    @pyqtSlot()
    def run(self):
        """
        Entry point for the QThread.
        Blocking call that runs the Core loop.
        """
        if self.mailbox is None:
            self._core._on_frame_ready = self.processed_img.emit

        self._core.run_loop()

    @pyqtSlot()
    def _on_cleanup(self):
        """Internal slot for cleanup."""
        with QSignalBlocker(self):
            logger.debug(f"Cleaning up {self.__class__.__name__}")
            self._core.cleanup()

    def cleanup(self):
        self._cleanup_req.emit()

    def __del__(self):
        if has_qt_cpp_binding(self) and hasattr(self, "_core"):
            self._core.cleanup()

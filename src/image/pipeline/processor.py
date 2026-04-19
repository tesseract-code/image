import logging
import multiprocessing
import timeit
import uuid
from multiprocessing import Event, Process
from typing import Dict, Any, Optional, Callable, Union

import numpy as np
from PyQt6.QtCore import (QObject, pyqtSignal, QThread, pyqtSlot,
                          QSignalBlocker)

from image.pipeline.mailbox import FrameMailbox
from image.pipeline.operations.process import (
    noop_pipeline
)
from image.pipeline.receive import (
    FrameReceiver
)
from image.pipeline.submit import FrameSubmitter
from image.pipeline.worker import image_worker_entry
from image.settings.base import ImageSettingsSnapshot
from pycore.settings.provider import SettingsProvider
# Internal imports (Refactored Backend)
from qtcore.reference import has_qt_cpp_binding

logger = logging.getLogger(__name__)

# Type Alias for Settings inputs
type SettingsInput = Union[
    SettingsProvider, Dict[str, Any], ImageSettingsSnapshot]


class ImageProcessor(QObject):
    """
    Coordinator for the High-Performance Image Processing Pipeline.
    
    Architecture:
    [Main Thread] -> [Submitter Thread (SHM Write)] -> [Worker Process (Processing)] 
                                                                    |
    [Main Thread] <- [Receiver Thread (SHM Read)]  <-----------------
    """

    # Signals
    image_ready = pyqtSignal(np.ndarray,
                             object)  # Emitted if no mailbox is used
    have_mail = pyqtSignal()  # Emitted if mailbox is used
    dropped_frame = pyqtSignal()  # Emitted on queue overflow

    def __init__(self,
                 max_queue_size: int = 3,
                 track_metrics: bool = False,
                 daemon: bool = False,
                 mailbox: Optional[FrameMailbox] = None,
                 pipeline_func: Callable = noop_pipeline,
                 parent: Optional[QObject] = None):
        """
        Args:
            max_queue_size: Max frames pending in queue before dropping.
            track_metrics: Enable performance monitoring in worker.
            daemon: Run worker as daemon process.
            mailbox: Optional zero-copy mailbox for result consumption.
            pipeline_func: The processing logic (passthrough vs full pipeline).
            parent: Qt parent.
        """
        super().__init__(parent=parent)

        self.mailbox = mailbox
        self._track_metrics = track_metrics
        self._is_shutting_down = False
        self._in_flight_count: int = 0
        self._last_push_time = 0.0

        # 1. Queue Setup
        # We need buffer_count > max_queue_size to prevent "Tearing" wrap-around race conditions
        # buffer_count = queue_size + overhead (producer hold + consumer hold + safety)
        self._buffer_count = max_queue_size + 3

        self._mp_input_queue = multiprocessing.Queue(maxsize=max_queue_size)
        self._mp_output_queue = multiprocessing.Queue(maxsize=max_queue_size)
        self._stop_event = Event()

        # 2. Worker Process Setup (Heavy Lifting)
        self._worker_process = Process(
            target=image_worker_entry,
            args=(
                self._mp_input_queue,
                self._mp_output_queue,
                self._buffer_count,
                self._stop_event,
                True,  # High Priority
                pipeline_func  # Injectable Logic
            ),
            daemon=daemon,
            name=f"ImageWorker-{uuid.uuid4().hex[:6]}"
        )
        self._worker_process.start()

        # 3. Submitter Thread Setup (Write to SHM)
        self._submitter_thread = QThread()
        self._submitter_thread.setObjectName("ImageSubmitterThread")
        self.send_frame_worker = FrameSubmitter(
            self._mp_input_queue,
            self._buffer_count
        )
        self.send_frame_worker.moveToThread(self._submitter_thread)
        self._submitter_thread.start(QThread.Priority.TimeCriticalPriority)

        # 4. Receiver Thread Setup (Read from SHM)
        self._receive_thread = QThread()
        self._receive_thread.setObjectName("ImageReceiverThread")

        self._result_worker_id = uuid.uuid4()
        self.receive_image_worker = FrameReceiver(
            output_queue=self._mp_output_queue,
            stop_event=self._stop_event,
            mailbox=self.mailbox,
            job_id=f"worker_{self._result_worker_id}",
            track_metrics=track_metrics
        )
        self.receive_image_worker.moveToThread(self._receive_thread)

        # 5. Signal Wiring
        self._connect_receiver_signals()

        # Start Receiver Loop
        self._receive_thread.started.connect(self.receive_image_worker.run)
        self._receive_thread.start(QThread.Priority.TimeCriticalPriority)

        logger.debug(f"ImageProcessor started with queue size {max_queue_size}")

    def _connect_receiver_signals(self):
        """Wires up the receiver signals based on Mailbox configuration."""

        # In-flight tracking is common to both modes
        # We use a lambda wrapper or direct slot to ensure thread affinity is handled correctly
        # though decrement is simple arithmetic.

        if self.mailbox is None:
            # Direct Signal Mode (Standard PyQt)
            self.receive_image_worker.processed_img.connect(
                self.image_ready.emit)
            self.receive_image_worker.processed_img.connect(
                self._on_frame_finished)
        else:
            # Mailbox Mode (Zero-Copy Read)
            self.receive_image_worker.have_mail.connect(self.have_mail.emit)
            self.receive_image_worker.have_mail.connect(self._on_frame_finished)

    @property
    def is_running(self):
        return not self._is_shutting_down

    @property
    def in_flight_count(self) -> int:
        """Returns the number of frames currently being processed."""
        return self._in_flight_count

    @pyqtSlot()
    def _on_frame_finished(self):
        """Slot called when a frame returns from the worker."""
        # Atomic decrement in the Main Thread (Slot context)
        if self._in_flight_count > 0:
            self._in_flight_count -= 1

    @pyqtSlot(np.ndarray, object)
    def queue_image(self,
                    image: np.ndarray,
                    settings: SettingsInput) -> bool:
        """
        Submits an image to the processing pipeline.
        
        Returns:
            True if submitted successfully.
            False if dropped (Queue Full or Shutdown).
        """
        if self._is_shutting_down:
            return False

        # Basic Validation
        if image is None or image.size < 1:
            return False

        # Settings Normalization (Handle Provider vs Dict vs Snapshot)
        if not isinstance(settings, (dict, tuple)):
            # Assuming SettingsProvider or similar object with get_copy()
            try:
                # Prefer snapshotting to avoid threading race conditions on mutable settings
                payload = settings.get_copy()
            except AttributeError:
                payload = {}
        else:
            payload = settings

        # Attempt Submission
        # Note: submit_image is checking mp_queue.full() internally.
        success = self.send_frame_worker.submit_image(image, payload)

        if success:
            self._in_flight_count += 1
            self._last_push_time = timeit.default_timer()
            return True
        else:
            # Overflow handling
            self.dropped_frame.emit()
            # If tracking metrics, we might want to log this
            if self._track_metrics:
                logger.warning("ImagePipeline: Frame dropped (Queue Full)")
            return False

    def shutdown(self):
        """
        Gracefully shuts down the pipeline.
        Stops acceptance of new frames, waits for workers to finish, then cleans resources.
        """
        if self._is_shutting_down:
            return

        logger.debug("ImageProcessor shutting down...")
        self._is_shutting_down = True

        # 1. Stop Signals to prevent UI updates during teardown
        with QSignalBlocker(self):

            # 2. Signal Workers to Stop
            self._stop_event.set()

            # 3. Cleanup Submitter (Stops allocating new SHM)
            # We invoke cleanup via metaobject to ensure it runs in the submitter thread context
            # if specific thread-local cleanup was required (less critical for pure SHM but good practice)
            if hasattr(self.send_frame_worker, 'cleanup'):
                self.send_frame_worker.cleanup()

            if self._submitter_thread.isRunning():
                self._submitter_thread.quit()
                self._submitter_thread.wait(500)  # Wait 500ms
                if self._submitter_thread.isRunning():
                    self._submitter_thread.terminate()

            # 4. Cleanup Worker Process
            if self._worker_process.is_alive():
                # Give it time to finish the current frame
                self._worker_process.join(timeout=0.5)

                if self._worker_process.is_alive():
                    logger.warning(
                        "ImageWorker did not exit cleanly, forcing termination.")
                    self._worker_process.terminate()
                    self._worker_process.join(timeout=0.1)

            # 5. Cleanup Receiver (Stops reading SHM)
            if hasattr(self.receive_image_worker, 'cleanup'):
                self.receive_image_worker.cleanup()

            if self._receive_thread.isRunning():
                self._receive_thread.quit()
                self._receive_thread.wait(500)
                if self._receive_thread.isRunning():
                    self._receive_thread.terminate()

            # 6. Final Reset
            self._in_flight_count = 0
        logger.debug("ImageProcessor shutdown complete.")

    def __del__(self):
        """Destructor safeguard."""
        if has_qt_cpp_binding(self) and not self._is_shutting_down:
            self.shutdown()

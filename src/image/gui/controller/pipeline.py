import logging
import time
from enum import Enum, auto
from typing import Optional, Any, Tuple

import numpy as np
from PyQt6.QtCore import (
    QObject, pyqtSignal, pyqtSlot
)
from PyQt6.QtStateMachine import QStateMachine, QState

from image.gl.view import GLFrameViewer
from image.pipeline.frame import RenderFrame
from image.pipeline.mailbox import FrameMailbox
from image.pipeline.processor import ImageProcessor
from image.pipeline.stats import get_frame_stats
from image.settings.base import ImageSettings
from image.settings.mngr import ImageSettingsManager
from image.utils.types import (ImageLike, is_image,
                               IntegerArray)
from pycore.jobs import ExecutionMode
from pycore.shm import SharedMemoryRingBuffer
from qtcore.worker import WorkerSignals

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Interfaces & Enums
# -----------------------------------------------------------------------------

class PipelineMode(Enum):
    """Immutable operation modes for the controller."""
    SYNCHRONOUS = auto()
    ASYNCHRONOUS = auto()


# class GLPipelineViewer(Protocol):
#     """Protocol defining the required interface for the View component."""
#
#     def upload_image(self, image: np.ndarray,
#                      metadata: ImageMetadata) -> None: ...
#

# -----------------------------------------------------------------------------
# Controller Implementation
# -----------------------------------------------------------------------------

class PipelineState(Enum):
    """
    Internal states for the QStateMachine.
    These are mapped to public JobStatus values.
    """
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    ERROR = auto()


# -----------------------------------------------------------------------------
# Main Controller
# -----------------------------------------------------------------------------
class PipelineController(QObject):
    """
    Manages the GL Viewer pipeline lifecycle.

    Integrates QStateMachine for strict lifecycle control with
    WorkerSignals for standardized status reporting.
    """

    # Internal Signals for State Machine Transitions
    _start_requested = pyqtSignal()
    _stop_requested = pyqtSignal()
    _setup_finished = pyqtSignal()
    _teardown_finished = pyqtSignal()
    _error_trigger = pyqtSignal(Exception)

    def __init__(self,
                 viewer: GLFrameViewer,
                 settings: ImageSettings,
                 mode: ExecutionMode = ExecutionMode.AUTO,
                 controller_id: str = "gl_pipeline",
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        self._id = controller_id
        self._viewer = viewer
        self._settings = settings
        self.shm_ring: Optional[SharedMemoryRingBuffer] = None

        # signals object for external consumers
        self.signals = WorkerSignals()

        # Resolve Auto mode -> Default to Async for heavy GL pipelines
        if mode == ExecutionMode.AUTO:
            self._mode = ExecutionMode.ASYNC
        else:
            self._mode = mode

        # Track current state
        self._current_state = PipelineState.STOPPED

        # Timing and error tracking
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._last_exception: Optional[Exception] = None
        self._last_result: Any = None

        # Component placeholders
        self.mailbox = FrameMailbox()
        self._settings_manager: ImageSettingsManager | None = None
        self._processor: ImageProcessor | None = None

        # Optimization flag for hot-path
        self._is_active_flag = False

        # Build State Machine BEFORE starting it
        self._machine = QStateMachine(self)
        self._build_state_machine()

        logger.info(
            f"[{self._id}] GLPipeline initialized: Mode={self._mode.name}")

        # Start the state machine event loop - THIS IS CRITICAL
        logger.debug(f"[{self._id}] Starting state machine...")
        self._machine.start()
        logger.debug(
            f"[{self._id}] State machine started, isRunning={self._machine.isRunning()}")
        logger.debug(
            f"[{self._id}] Initial state entered, current_state={self._current_state.name}")

    # =========================================================================
    # Public API
    # =========================================================================

    @property
    def current_state(self) -> PipelineState:
        """Return current pipeline state."""
        return self._current_state

    @property
    def is_running(self) -> bool:
        """Check if pipeline is actively processing frames."""
        return self._is_active_flag

    @property
    def image_settings(self) -> ImageSettings:
        return self._settings

    @pyqtSlot()
    def start_pipeline(self):
        """Request pipeline startup."""
        # logger.info(f"[{self._id}] ========================================")
        logger.info(f"[{self._id}] start_pipeline() called")
        logger.debug(f"[{self._id}] Current state: {self._current_state.name}")
        logger.debug(f"[{self._id}] Active flag: {self._is_active_flag}")
        logger.debug(
            f"[{self._id}] State machine running: {self._machine.isRunning()}")

        # Check what state the machine thinks it's in
        current_states = self._machine.configuration()
        logger.debug(
            f"[{self._id}] State machine configuration: {[s.objectName() for s in current_states]}")

        # Simply emit the signal - state machine handles valid transitions
        logger.debug(f"[{self._id}] Emitting _start_requested signal...")
        self._machine.setInitialState(self._s_starting)
        self._start_requested.emit()
        logger.debug(f"[{self._id}] Signal emitted")
        # logger.info(f"[{self._id}] ========================================")

    @pyqtSlot()
    def stop_pipeline(self):
        """Request pipeline shutdown."""
        logger.info(f"[{self._id}] stop_pipeline() called")
        logger.debug(f"[{self._id}] Current state: {self._current_state}")
        logger.debug(f"[{self._id}] Emitting _stop_requested signal...")
        self._stop_requested.emit()

    @pyqtSlot(np.ndarray)
    def ingest_frame(self, image: ImageLike):
        """
        Hot-path for frame ingestion.
        """
        if not self._is_active_flag:
            logger.debug(f"[{self._id}] Frame dropped - pipeline not active")
            return

        logger.debug(f"[{self._id}] Ingesting frame in {self._mode.name} mode")

        # Route based on immutable mode
        if self._mode == ExecutionMode.SYNC:
            self._handle_sync_frame(image)
        elif self._mode == ExecutionMode.ASYNC:
            self._handle_async_frame(image)

    def alloc_shm_write_buffer(self, required_size: int
                               ) -> Tuple[str, IntegerArray] | None:
        if self._mode == ExecutionMode.ASYNC:
            logger.debug(
                f"[{self._id}] Allocating SHM buffer, size={required_size}")
            if required_size > 0:
                name, view = self._processor.send_frame_worker.alloc_buffer(
                    required_size=required_size)
                logger.debug(
                    f"[{self._id}] Buffer allocated: name={name}, shape={view.shape}")
                return name, view

    def submit_shm_buffer(self,
                          shm_name: str,
                          shape: Tuple[int, ...],
                          dtype: np.dtype,
                          settings: Any):
        if self._mode == ExecutionMode.ASYNC:
            logger.debug(f"[{self._id}] Submitting SHM buffer: {shm_name}")
            return self._processor.send_frame_worker.submit_shm_buffer(
                shm_name=shm_name,
                shape=shape, dtype=dtype, config=settings)

    # =========================================================================
    # State Machine Configuration
    # =========================================================================

    def _build_state_machine(self):
        logger.debug(f"[{self._id}] Building state machine...")

        # 1. Create States with names for debugging
        self._s_stopped = QState()
        self._s_stopped.setObjectName("STOPPED")

        self._s_starting = QState()
        self._s_starting.setObjectName("STARTING")

        self._s_running = QState()
        self._s_running.setObjectName("RUNNING")

        self._s_stopping = QState()
        self._s_stopping.setObjectName("STOPPING")

        self._s_error = QState()
        self._s_error.setObjectName("ERROR")

        # 2. Add States
        self._machine.addState(self._s_stopped)
        self._machine.addState(self._s_starting)
        self._machine.addState(self._s_running)
        self._machine.addState(self._s_stopping)
        self._machine.addState(self._s_error)

        # 3. Transitions with logging
        logger.debug(f"[{self._id}] Setting up transitions...")

        # Stopped -> Starting
        self._s_stopped.addTransition(self._start_requested,
                                      self._s_starting)
        logger.debug(
            f"[{self._id}] Transition added: STOPPED -> STARTING on _start_requested (signal={self._start_requested})")

        # Starting -> Running (Success)
        self._s_starting.addTransition(self._setup_finished,
                                       self._s_running)
        logger.debug(
            f"[{self._id}] Transition added: STARTING -> RUNNING on _setup_finished (signal={self._setup_finished})")

        # Starting -> Error (Fail)
        self._s_starting.addTransition(self._error_trigger, self._s_error)
        logger.debug(
            f"[{self._id}] Transition added: STARTING -> ERROR on _error_trigger")

        # Running -> Stopping
        self._s_running.addTransition(self._stop_requested, self._s_stopping)
        logger.debug(
            f"[{self._id}] Transition added: RUNNING -> STOPPING on _stop_requested")

        # Running -> Error
        self._s_running.addTransition(self._error_trigger, self._s_error)
        logger.debug(
            f"[{self._id}] Transition added: RUNNING -> ERROR on _error_trigger")

        # Stopping -> Stopped
        self._s_stopping.addTransition(self._teardown_finished, self._s_stopped)
        logger.debug(
            f"[{self._id}] Transition added: STOPPING -> STOPPED on _teardown_finished")

        # Error -> Stopping (Cleanup attempt)
        self._s_error.addTransition(self._stop_requested, self._s_stopping)
        logger.debug(
            f"[{self._id}] Transition added: ERROR -> STOPPING on _stop_requested")

        # 4. Connect Hooks
        logger.debug(f"[{self._id}] Connecting state entry/exit handlers...")
        self._s_stopped.entered.connect(self._on_enter_stopped)
        self._s_starting.entered.connect(self._on_enter_starting)
        self._s_running.entered.connect(self._on_enter_running)
        self._s_running.exited.connect(self._on_exit_running)
        self._s_stopping.entered.connect(self._on_enter_stopping)
        self._s_error.entered.connect(self._on_enter_error)

        # Debug: Connect to signals to see if they fire
        self._start_requested.connect(lambda: logger.debug(
            f"[{self._id}] SIGNAL FIRED: _start_requested"))
        self._setup_finished.connect(
            lambda: logger.debug(f"[{self._id}] SIGNAL FIRED: _setup_finished"))
        self._stop_requested.connect(
            lambda: logger.debug(f"[{self._id}] SIGNAL FIRED: _stop_requested"))
        self._teardown_finished.connect(lambda: logger.debug(
            f"[{self._id}] SIGNAL FIRED: _teardown_finished"))
        self._error_trigger.connect(lambda e: logger.debug(
            f"[{self._id}] SIGNAL FIRED: _error_trigger({e})"))

        # 5. Set Initial State
        self._machine.setInitialState(self._s_stopped)
        logger.debug(f"[{self._id}] Initial state set to: STOPPED")
        logger.debug(f"[{self._id}] State machine build complete")

    # =========================================================================
    # State Logic
    # =========================================================================

    def _update_state(self, new_state: PipelineState):
        """Update internal state and emit signals."""
        if self._current_state != new_state:
            old_state = self._current_state
            self._current_state = new_state
            logger.info(
                f"[{self._id}] State changed: {old_state.name} -> {new_state.name}")
            self.signals.status_changed.emit(self._id, self._current_state)

    def _on_enter_stopped(self):
        """Pipeline idle."""
        logger.info(f"[{self._id}] >>> ENTERED STATE: STOPPED")
        self._update_state(PipelineState.STOPPED)

        self._end_time = time.time()

        # Determine if this was a clean stop or error
        if self._last_exception:
            logger.warning(
                f"[{self._id}] Stopped with error: {self._last_exception}")
            self.signals.error.emit(self._id, self._last_exception)
        else:
            logger.info(f"[{self._id}] Pipeline stopped cleanly")
            self.signals.finished.emit(self._id, self._last_result)

    def _on_enter_starting(self):
        """Transitioning from STOPPED to RUNNING."""
        logger.info(f"[{self._id}] >>> ENTERED STATE: STARTING")
        self._update_state(PipelineState.STARTING)

        self._start_time = time.time()
        self._end_time = None
        self._last_result = None
        self._last_exception = None

        self.signals.started.emit(self._id)

        logger.debug(
            f"[{self._id}] Configuring pipeline components for {self._mode.name} mode...")

        logger.debug(f"[{self._id}] Initializing ImageSettingsManager...")

        self._settings_manager = ImageSettingsManager(
            settings=self._settings,
            auto_start_server=True,
            parent=self
        )
        # self._settings_manager.start()
        logger.debug(f"[{self._id}] ImageSettingsManager started")

        try:
            if self._mode == ExecutionMode.ASYNC:
                logger.debug(f"[{self._id}] Setting up ASYNC components...")
                self._setup_async_components()
            elif self._mode == ExecutionMode.SYNC:
                logger.debug(f"[{self._id}] Setting up SYNC components...")
                self._setup_sync_components()

            logger.info(
                f"[{self._id}] Setup complete, emitting _setup_finished signal")
            self._setup_finished.emit()

        except Exception as e:
            logger.error(f"[{self._id}] Startup failed: {e}", exc_info=True)
            self._last_exception = e
            logger.debug(f"[{self._id}] Emitting _error_trigger signal")
            self._error_trigger.emit(e)

    def _on_enter_running(self):
        """Pipeline is fully operational."""
        logger.info(f"[{self._id}] >>> ENTERED STATE: RUNNING")
        self._update_state(PipelineState.RUNNING)

        self._is_active_flag = True
        logger.info(
            f"[{self._id}] Pipeline is ACTIVE and ready for frames (mode={self._mode.name})")

    def _on_exit_running(self):
        logger.info(f"[{self._id}] <<< EXITING STATE: RUNNING")
        self._is_active_flag = False
        logger.debug(f"[{self._id}] Active flag set to False")

    def _on_enter_stopping(self):
        """Cleaning up."""
        logger.info(f"[{self._id}] >>> ENTERED STATE: STOPPING")
        self._update_state(PipelineState.STOPPING)

        logger.debug(f"[{self._id}] Teardown initiated...")

        try:
            if self._mode == ExecutionMode.ASYNC:
                logger.debug(f"[{self._id}] Tearing down ASYNC components...")
                self._teardown_async_components()
            else:
                logger.debug(f"[{self._id}] SYNC mode - no teardown needed")

        except Exception as e:
            logger.error(f"[{self._id}] Teardown error: {e}", exc_info=True)
            # Record error but don't prevent transition to stopped
            if not self._last_exception:
                self._last_exception = e
        finally:
            logger.debug(f"[{self._id}] Emitting _teardown_finished signal")
            self._teardown_finished.emit()

    def _on_enter_error(self):
        """Critical failure state."""
        logger.error(f"[{self._id}] >>> ENTERED STATE: ERROR")
        self._update_state(PipelineState.ERROR)

        ex = self._last_exception or Exception("Unknown Error")
        self._last_exception = ex
        self._end_time = time.time()

        logger.error(f"[{self._id}] Pipeline error: {ex}", exc_info=True)

        # Auto-shutdown to cleanup
        logger.debug(f"[{self._id}] Requesting stop for cleanup...")
        self._stop_requested.emit()

    # =========================================================================
    # Component Implementation
    # =========================================================================

    def _setup_sync_components(self):
        """Sync mode logic is inline, no extra objects needed."""
        logger.debug(
            f"[{self._id}] Creating SharedMemoryRingBuffer for SYNC mode")
        self.shm_ring = SharedMemoryRingBuffer()
        logger.debug(f"[{self._id}] SYNC components ready")

    def _setup_async_components(self):
        """Initialize settings server and processing worker."""
        logger.debug(f"[{self._id}] Initializing ImageProcessor...")
        self._processor = ImageProcessor(
            track_metrics=True,
            daemon=True,
            mailbox=self.mailbox,
            parent=self
        )
        if self.mailbox is None:
            self._processor.image_ready.connect(self._on_async_image_ready)
        logger.debug(f"[{self._id}] ImageProcessor initialized and connected")
        logger.info(f"[{self._id}] ASYNC components ready")

    def _teardown_async_components(self):
        """Cleanup async components."""
        if self._settings_manager:
            logger.debug(f"[{self._id}] Stopping ImageSettingsManager...")
            self._settings_manager.stop()
            self._settings_manager = None
            logger.debug(f"[{self._id}] ImageSettingsManager stopped")

        if self._processor:
            logger.debug(f"[{self._id}] Shutting down ImageProcessor...")
            self._processor.shutdown()
            self._processor = None
            logger.debug(f"[{self._id}] ImageProcessor shut down")

    # =========================================================================
    # Frame Handling Logic
    # =========================================================================

    def _handle_sync_frame(self, image: np.ndarray):
        """
        Synchronous Mode:
        Directly pass image to viewer using current local settings.
        Main Thread -> GL Context.
        """
        if not is_image(image):
            logger.warning(f"[{self._id}] Invalid image received in SYNC mode")
            return

        try:
            metadata = get_frame_stats(image=image)
            # print(metadata)
            logger.debug(f"[{self._id}] Uploading image to viewer (SYNC)")
            if self.mailbox is not None:
                self.mailbox.write(RenderFrame(image_view=image,
                                               metadata=metadata,
                                               # Questionable if the user
                                               # wants a specific output
                                               # format, but GL converts
                                               # regardless
                                               format=None))
        except Exception as e:
            logger.error(f"[{self._id}] SYNC frame handling error: {e}",
                         exc_info=True)
            self._last_exception = e
            self._error_trigger.emit(e)

    def _handle_async_frame(self, image: np.ndarray):
        """
        Asynchronous Mode:
        Main Thread -> SHM Buffer -> Multiprocessing Worker.
        """
        if self._processor and self._processor.is_running:
            try:
                logger.debug(
                    f"[{self._id}] Queueing image to async processor")
                self._processor.queue_image(image=image,
                                            settings=self._settings.get_copy())
            except Exception as e:
                logger.error(f"[{self._id}] Failed to queue image: {e}",
                             exc_info=True)

    @pyqtSlot(np.ndarray, object)
    def _on_async_image_ready(self, image: np.ndarray, metadata: Any):
        """
        Asynchronous Return:
        Worker Thread -> Signal -> Main Thread -> GL Context.
        """
        logger.info(
            f"[{self._id}] Processed image ready from worker, metadata={metadata}")
        try:
            self._viewer.upload_image(image, metadata)
        except Exception as e:
            logger.error(f"[{self._id}] ASYNC image upload error: {e}",
                         exc_info=True)
            self._last_exception = e
            self._error_trigger.emit(e)

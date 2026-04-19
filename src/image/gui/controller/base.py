import logging
import time
from abc import abstractmethod
from enum import Enum, auto
from typing import Any, Optional

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot, QTimer, QCoreApplication
from PyQt6.QtStateMachine import QStateMachine, QState

from image.pipeline.mailbox import FrameMailbox
from qtcore.meta import QABCMeta
from qtcore.worker import WorkerSignals

logger = logging.getLogger(__name__)


class PipelineState(Enum):
    STOPPED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    ERROR = auto()


class BasePipelineController(QObject, metaclass=QABCMeta):
    """
    Abstract base controller for managing pipeline lifecycle.
    """

    # Signals must be class attributes
    _start_requested = pyqtSignal()
    _stop_requested = pyqtSignal()
    _setup_finished = pyqtSignal()
    _teardown_finished = pyqtSignal()
    _error_trigger = pyqtSignal(Exception)

    def __init__(self,
                 viewer: 'GLFrameViewer',
                 settings: 'ImageSettings',
                 controller_id: str = "pipeline",
                 parent: Optional[QObject] = None):
        super().__init__(parent)

        self._id = controller_id
        self._viewer = viewer
        self._settings = settings

        # Shared components
        self.mailbox = FrameMailbox()
        self.signals = WorkerSignals()

        # Track current state (Python side)
        self._current_state = PipelineState.STOPPED

        # Internal flags
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        self._last_exception: Optional[Exception] = None
        self._last_result: Any = None
        self._is_active_flag = False

        # 1. Initialize Machine with SELF as parent
        # This prevents Python Garbage Collection from killing states
        self._machine = QStateMachine(QState.ChildMode.ExclusiveStates, self)

        # 2. Build Graph
        self._build_state_machine()

        logger.info(f"[{self._id}] Pipeline initialized")

        # 3. Start Machine immediately
        # Note: It enters the event loop queue, it is not "running" yet
        self._start_state_machine()

    def _start_state_machine(self):
        """Start the state machine."""
        self._machine.start()
        # Note: isRunning() might still be False here until the event loop spins once
        logger.debug(f"[{self._id}] State machine start() called")
        # FORCE the event loop to process the 'start' event immediately.
        # Without this, the machine is not actually in the 'STOPPED' state
        # when start_pipeline() is called.
        QCoreApplication.processEvents()

    @property
    def current_state(self) -> PipelineState:
        return self._current_state

    @property
    def is_running(self) -> bool:
        return self._is_active_flag

    @pyqtSlot()
    def start_pipeline(self):
        """Request start and FORCE the state machine to update immediately."""
        logger.info(f"[{self._id}] start_pipeline() called")

        # 1. Emit the signal (normally async)
        self._start_requested.emit()

        # 2. Force the Event Loop to process the signal NOW.
        # This makes the State Machine transition to STARTING (and RUNNING if setup is fast)
        # before this function returns.
        QCoreApplication.processEvents()

        logger.debug(
            f"[{self._id}] Pipeline started (Sync). State is now: {self._current_state}")

    @pyqtSlot()
    def stop_pipeline(self):
        logger.info(f"[{self._id}] stop_pipeline() called")
        self._stop_requested.emit()

    @pyqtSlot(np.ndarray)
    def ingest_frame(self, image: np.ndarray):
        if not self._is_active_flag:
            return
        self._handle_frame(image)

    # =========================================================================
    # State Machine Configuration
    # =========================================================================

    def _build_state_machine(self):
        logger.debug(f"[{self._id}] Building state machine...")

        # 1. Create States
        # CRITICAL: Pass self._machine as parent to QState
        self._s_stopped = QState(self._machine)
        self._s_stopped.setObjectName("STOPPED")

        self._s_starting = QState(self._machine)
        self._s_starting.setObjectName("STARTING")

        self._s_running = QState(self._machine)
        self._s_running.setObjectName("RUNNING")

        self._s_stopping = QState(self._machine)
        self._s_stopping.setObjectName("STOPPING")

        self._s_error = QState(self._machine)
        self._s_error.setObjectName("ERROR")

        # 2. Transitions
        # Stopped -> Starting
        self._s_stopped.addTransition(self._start_requested, self._s_starting)

        # Starting -> Running
        self._s_starting.addTransition(self._setup_finished, self._s_running)
        self._s_starting.addTransition(self._error_trigger, self._s_error)

        # Running -> Stopping
        self._s_running.addTransition(self._stop_requested, self._s_stopping)
        self._s_running.addTransition(self._error_trigger, self._s_error)

        # Stopping -> Stopped
        self._s_stopping.addTransition(self._teardown_finished, self._s_stopped)

        # Error -> Stopping
        self._s_error.addTransition(self._stop_requested, self._s_stopping)

        # 3. Signals (Hooks)
        self._s_stopped.entered.connect(self._on_enter_stopped)
        self._s_starting.entered.connect(self._on_enter_starting)
        self._s_running.entered.connect(self._on_enter_running)
        self._s_running.exited.connect(self._on_exit_running)
        self._s_stopping.entered.connect(self._on_enter_stopping)
        self._s_error.entered.connect(self._on_enter_error)

        # Debug logging for signals
        self._start_requested.connect(
            lambda: logger.debug(f"[{self._id}] SIGNAL: _start_requested"))

        # 4. Initial State
        self._machine.setInitialState(self._s_stopped)

    # =========================================================================
    # State Logic
    # =========================================================================

    def _update_state(self, new_state: PipelineState):
        if self._current_state != new_state:
            old_state = self._current_state
            self._current_state = new_state
            logger.info(
                f"[{self._id}] State: {old_state.name} -> {new_state.name}")
            self.signals.status_changed.emit(self._id, self._current_state)

    def _on_enter_stopped(self):
        logger.info(f"[{self._id}] >>> STOPPED")
        self._update_state(PipelineState.STOPPED)

        if self._last_exception:
            self.signals.error.emit(self._id, self._last_exception)
            self._last_exception = None  # Clear after reporting
        elif self._start_time:
            self.signals.finished.emit(self._id, self._last_result)

    def _on_enter_starting(self):
        logger.info(f"[{self._id}] >>> STARTING")
        self._update_state(PipelineState.STARTING)
        self._start_time = time.time()
        self.signals.started.emit(self._id)

        # Defer execution to let state settle
        QTimer.singleShot(0, self._perform_setup)

    def _perform_setup(self):
        try:
            self._setup_components()
            self._setup_finished.emit()
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            self._last_exception = e
            self._error_trigger.emit(e)

    def _on_enter_running(self):
        logger.info(f"[{self._id}] >>> RUNNING")
        self._update_state(PipelineState.RUNNING)
        self._is_active_flag = True

    def _on_exit_running(self):
        self._is_active_flag = False

    def _on_enter_stopping(self):
        logger.info(f"[{self._id}] >>> STOPPING")
        self._update_state(PipelineState.STOPPING)
        QTimer.singleShot(0, self._perform_teardown)

    def _perform_teardown(self):
        try:
            self._teardown_components()
        except Exception as e:
            logger.error(f"Teardown failed: {e}")
            if not self._last_exception:
                self._last_exception = e
        finally:
            self._teardown_finished.emit()

    def _on_enter_error(self):
        logger.error(f"[{self._id}] >>> ERROR")
        self._update_state(PipelineState.ERROR)
        # Auto-recover/cleanup
        QTimer.singleShot(100, self._stop_requested.emit)

    # =========================================================================
    # Abstracts
    # =========================================================================
    @abstractmethod
    def _setup_components(self):
        pass

    @abstractmethod
    def _teardown_components(self):
        pass

    @abstractmethod
    def _handle_frame(self, image: np.ndarray):
        pass
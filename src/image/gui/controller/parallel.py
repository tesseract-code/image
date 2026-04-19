import logging
from typing import Optional, Tuple, Any

import numpy as np
from PyQt6.QtCore import QObject, pyqtSlot

from image.gui.controller.base import BasePipelineController
from image.pipeline.processor import ImageProcessor
from image.settings.mngr import ImageSettingsManager

logger = logging.getLogger(__name__)


class AsyncPipelineController(BasePipelineController):
    """
    Asynchronous pipeline controller.

    Processes frames in a separate worker process via shared memory:
    Main Thread -> SHM Buffer -> Worker Process -> Signal -> Main Thread -> GL Context

    More complex but handles heavy processing without blocking the UI.
    Includes network-based settings server for remote configuration.
    """

    def __init__(self,
                 viewer: 'GLFrameViewer',
                 settings: 'ImageSettings',
                 controller_id: str = "async_pipeline",
                 parent: Optional[QObject] = None):

        logger.info(f"[{controller_id}] Initializing AsyncPipelineController")

        # Async-specific components (initialized during setup)
        self._settings_manager: Optional['ImageSettingsManager'] = None
        self._processor: Optional['ImageProcessor'] = None

        super().__init__(
            viewer=viewer,
            settings=settings,
            controller_id=controller_id,
            parent=parent
        )

    # =========================================================================
    # Component Management (Asynchronous)
    # =========================================================================

    def _setup_components(self):
        """
        Initialize async-specific components:
        - ImageSettingsManager: Network server for remote settings updates
        - ImageProcessor: Worker process for frame processing
        """
        logger.debug(f"[{self._id}] Setting up ASYNC components...")

        # 1. Initialize Settings Manager (network server)
        logger.debug(f"[{self._id}] Initializing ImageSettingsManager...")
        self._settings_manager = ImageSettingsManager(
            settings=self._settings,
            auto_start_server=True,
            parent=self
        )
        logger.debug(f"[{self._id}] ImageSettingsManager started")

        # 2. Initialize Image Processor (worker process)
        logger.debug(f"[{self._id}] Initializing ImageProcessor...")
        self._processor = ImageProcessor(
            track_metrics=True,
            daemon=True,
            mailbox=self.mailbox,
            parent=self
        )

        # Connect processor output if not using mailbox directly
        if self.mailbox is None:
            self._processor.image_ready.connect(self._on_async_image_ready)
            logger.debug(
                f"[{self._id}] ImageProcessor connected to image_ready signal")

        logger.debug(f"[{self._id}] ImageProcessor initialized")
        logger.info(f"[{self._id}] ASYNC components ready")

    def _teardown_components(self):
        """
        Clean up async components in reverse order of initialization.
        """
        logger.debug(f"[{self._id}] Tearing down ASYNC components...")

        # 1. Stop Settings Manager
        if self._settings_manager:
            logger.debug(f"[{self._id}] Stopping ImageSettingsManager...")
            self._settings_manager.stop()
            self._settings_manager = None
            logger.debug(f"[{self._id}] ImageSettingsManager stopped")

        # 2. Shutdown Processor
        if self._processor:
            logger.debug(f"[{self._id}] Shutting down ImageProcessor...")
            self._processor.shutdown()
            self._processor = None
            logger.debug(f"[{self._id}] ImageProcessor shut down")

        logger.info(f"[{self._id}] ASYNC teardown complete")

    # =========================================================================
    # Frame Processing (Asynchronous)
    # =========================================================================

    def _handle_frame(self, image: np.ndarray):
        """
        Asynchronous Mode:
        Queue image to worker process via shared memory.
        Main Thread -> SHM Buffer -> Multiprocessing Worker.

        Args:
            image: Input frame to process
        """
        if self._processor and self._processor.is_running:
            try:
                logger.debug(f"[{self._id}] Queueing image to async processor")
                self._processor.queue_image(
                    image=image,
                    settings=self._settings.get_copy()
                )
            except Exception as e:
                logger.error(f"[{self._id}] Failed to queue image: {e}",
                             exc_info=True)
                self._last_exception = e
                self._error_trigger.emit(e)
        else:
            logger.warning(f"[{self._id}] Processor not running, frame dropped")

    @pyqtSlot(np.ndarray, object)
    def _on_async_image_ready(self, image: np.ndarray, metadata: Any):
        """
        Handle processed image from worker.
        Asynchronous Return:
        Worker Thread -> Signal -> Main Thread -> GL Context.

        Args:
            image: Processed image from worker
            metadata: Associated metadata
        """
        logger.info(
            f"[{self._id}] Processed image ready from worker, metadata={metadata}")
        try:
            # Upload to viewer
            self._viewer.upload_image(image, metadata)
        except Exception as e:
            logger.error(f"[{self._id}] ASYNC image upload error: {e}",
                         exc_info=True)
            self._last_exception = e
            self._error_trigger.emit(e)

    # =========================================================================
    # Async-Specific Public API
    # =========================================================================

    def alloc_shm_write_buffer(self, required_size: int) -> Optional[
        Tuple[str, 'IntegerArray']]:
        """
        Allocate a shared memory buffer for writing frame data.

        Args:
            required_size: Size of buffer needed in bytes

        Returns:
            Tuple of (buffer_name, memory_view) or None if allocation fails
        """
        if self._processor is None:
            logger.warning(
                f"[{self._id}] Cannot allocate buffer: processor not initialized")
            return None

        logger.debug(
            f"[{self._id}] Allocating SHM buffer, size={required_size}")

        if required_size > 0:
            name, view = self._processor.send_frame_worker.alloc_buffer(
                required_size=required_size
            )
            logger.debug(
                f"[{self._id}] Buffer allocated: name={name}, shape={view.shape}")
            return name, view

        return None

    def submit_shm_buffer(self,
                          shm_name: str,
                          shape: Tuple[int, ...],
                          dtype: np.dtype,
                          settings: Any):
        """
        Submit a populated shared memory buffer for processing.

        Args:
            shm_name: Name of the shared memory buffer
            shape: Shape of the data in the buffer
            dtype: Data type of the buffer contents
            settings: Processing settings to use
        """
        if self._processor is None:
            logger.warning(
                f"[{self._id}] Cannot submit buffer: processor not initialized")
            return

        logger.debug(f"[{self._id}] Submitting SHM buffer: {shm_name}")
        return self._processor.send_frame_worker.submit_shm_buffer(
            shm_name=shm_name,
            shape=shape,
            dtype=dtype,
            config=settings
        )

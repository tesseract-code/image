import logging
from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject

from image.gl.view import GLFrameViewer
from image.gui.controller.base import (
    BasePipelineController)
from image.pipeline.frame import RenderFrame
from image.pipeline.stats import get_frame_stats
from image.utils.types import is_image

logger = logging.getLogger(__name__)


class SyncPipelineController(BasePipelineController):
    """
    Synchronous pipeline controller.

    Processes frames directly on the main thread:
    Main Thread -> Process -> GL Context

    Simple, lightweight implementation with minimal overhead.
    Suitable for lighter workloads or when latency is critical.
    """

    def __init__(self,
                 viewer: 'GLFrameViewer',
                 settings: 'ImageSettings',
                 controller_id: str = "sync_pipeline",
                 parent: Optional[QObject] = None):

        logger.info(f"[{controller_id}] Initializing SyncPipelineController")
        super().__init__(
            viewer=viewer,
            settings=settings,
            controller_id=controller_id,
            parent=parent
        )

    def _setup_components(self):
        """
        Sync mode requires no external components.
        Processing happens inline on the main thread.
        """
        logger.debug(f"[{self._id}] SYNC mode - no components to initialize")
        # Mailbox is already initialized in base class
        logger.debug(f"[{self._id}] SYNC components ready")

    def _teardown_components(self):
        """
        Sync mode has no components to clean up.
        """
        logger.debug(f"[{self._id}] SYNC mode - no teardown needed")

    def _handle_frame(self, image: np.ndarray):
        """
        Synchronous Mode:
        Directly process and pass image to viewer using current local settings.
        Main Thread -> GL Context.

        Args:
            image: Input frame to process
        """
        if not is_image(image):
            logger.warning(f"[{self._id}] Invalid image received in SYNC mode")
            return

        try:
            # Get frame statistics
            metadata = get_frame_stats(image=image)

            logger.debug(f"[{self._id}] Processing frame synchronously")

            if self.mailbox is not None:
                self.mailbox.write(RenderFrame(
                    image_view=image,
                    metadata=metadata,
                    format=None  # GL converts regardless
                ))
                logger.debug(f"[{self._id}] Frame written to mailbox")
            else:
                logger.warning(f"[{self._id}] Mailbox is None, frame dropped")

        except Exception as e:
            logger.error(f"[{self._id}] SYNC frame handling error: {e}",
                         exc_info=True)
            self._last_exception = e
            self._error_trigger.emit(e)

import threading
from typing import Optional, Tuple

import logging
import numpy as np

from image.pipeline.frame import RenderFrame
from image.pipeline.stats import FrameStats

logger = logging.getLogger(__name__)


class FrameMailbox:
    """
    Frame mailbox client.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._buffer: Optional[np.ndarray] = None
        self._latest_metadata: Optional[FrameStats] = None
        self._new_data_available = False

    def write(self, payload: RenderFrame):
        """Called by Receiver Thread (Core). Stabilizes SHM view."""
        incoming_view = payload.image_view
        meta = payload.metadata

        with self._lock:
            if (self._buffer is None or
                    self._buffer.shape != incoming_view.shape or
                    self._buffer.dtype != incoming_view.dtype):
                logger.debug("Mailbox: Allocating new stable buffer.")
                self._buffer = np.empty_like(incoming_view, order='C')

            np.copyto(self._buffer, incoming_view)

            self._latest_metadata = meta
            self._new_data_available = True

    def read(self) -> Optional[Tuple[np.ndarray, FrameStats]]:
        """Return a copy of the current buffer and metadata."""
        with self._lock:
            if not self._new_data_available or self._buffer is None:
                return None

            self._new_data_available = False
            # Return copy to avoid holding lock during rendering/GPU upload
            return self._buffer.copy(), self._latest_metadata

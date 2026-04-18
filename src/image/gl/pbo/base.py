from __future__ import annotations

import logging

import numpy as np
from OpenGL import GL as GL

from pycore.log.ctx import ContextAdapter

logger = ContextAdapter(logging.getLogger(__name__), {})


class PBO:
    """
    RAII wrapper around a single OpenGL buffer object.

    Subclasses set ``_target`` to specialise for upload (GL_PIXEL_UNPACK_BUFFER)
    or download (GL_PIXEL_PACK_BUFFER).
    """

    _target: int = GL.GL_PIXEL_UNPACK_BUFFER  # overridden by subclasses

    def __init__(self) -> None:
        raw = GL.glGenBuffers(1)
        self.id: int = int(raw[0]) if isinstance(raw, (
            list, tuple, np.ndarray)) else int(raw)
        self.capacity: int = 0
        self.is_mapped: bool = False

    def destroy(self) -> None:
        """Delete the GL buffer and release GPU memory. Safe to call multiple times."""
        if not self.id:
            return
        if self.is_mapped:
            logger.warning("Destroying PBO %d while still mapped.", self.id)
        # numpy array, not a list: PyOpenGL with ERROR_ON_COPY rejects plain lists.
        GL.glDeleteBuffers(1, np.array([self.id], dtype=np.uint32))
        self.id = 0
        self.capacity = 0
        self.is_mapped = False

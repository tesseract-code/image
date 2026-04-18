"""
pbo_core.py
===========
Pixel Buffer Object (PBO) allocation, mapping, and pipeline management.

Provides three public layers, all framework-agnostic (no Qt imports):

PBOUploadManager
    Pool of UnpackPBOs cycled in round-robin order for streaming texture
    uploads.  Supports both the standard map/copy/unmap path and a
    pinned zero-copy path.

PBODownloadManager
    Async double-buffered pixel readback using PackPBOs.  All coupling to
    the host widget goes through the ``WidgetBridge`` duck-typed protocol;
    results are delivered via a plain ``FrameCallback`` callable.

Shared infrastructure
    ``PBOBufferingStrategy``, ``PBO`` base, ``UnpackPBO``, ``PackPBO``,
    ``gl_error_check``, ``calculate_pixel_alignment``,
    ``configure_pixel_storage``, ``memmove_pbo``, ``write_pbo_buffer``.

Upload pipelines
----------------
Standard (unpinned)
    ``CPU array → memmove_pbo (map/copy/unmap) → glTexSubImage2D``

Pinned (zero-copy)
    ``acquire_next_writeable → write directly → unmap → glTexSubImage2D``

Download pipeline  (one frame of latency)
------------------------------------------
paintGL call A  (frame_count == 0)
    glReadPixels / glGetTexImage → current_pbo      # async DMA starts
    glFenceSync inserted                            # marks when DMA is done
    swap A/B; frame_count = 1
    bridge.schedule_update() forces call B

paintGL call B  (frame_count >= 1)
    glReadPixels / glGetTexImage → current_pbo      # queue next transfer
    glFenceSync inserted
    glClientWaitSync on previous_pbo.fence          # block only if not done
    glGetBufferSubData → numpy array                # copy out of PBO
    on_frame_ready(arr, w, h) callback              # hand data to caller

Thread-safety
-------------
``PBOUploadManager`` guards its pool and iterator with a lock so that
``get_next`` and ``acquire_next_writeable`` are safe to call from worker
threads.  All ``GL.*`` calls must still be issued from the GL-context thread.
"""

from __future__ import annotations

import logging

from image.gl.types import GLHandle
from pycore.log.ctx import ContextAdapter

# Sentinels / constants
_NO_BUFFER = GLHandle(0)
_NO_SYNC = None
_SENTINEL = object()  # unique sentinel distinguishable from None

# glClientWaitSync timeout: 100 ms in nanoseconds (~6× a 60 fps frame).
_SYNC_TIMEOUT_NS = 100_000_000

logger = ContextAdapter(logging.getLogger(__name__), {})

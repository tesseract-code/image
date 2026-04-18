from __future__ import annotations

import ctypes
import logging
from typing import Callable

import numpy as np
from OpenGL import GL as GL

from image.gl.errors import gl_error_check, GLSyncTimeout, GLMemoryError
from image.gl.pbo import WidgetBridge, PackPBO
from image.gl.pbo.constants import _NO_SYNC, _NO_BUFFER
from pycore.log.ctx import ContextAdapter

logger = ContextAdapter(logging.getLogger(__name__), {})

FrameCallback = Callable[[np.ndarray, int, int], None]


class PBODownloadManager:
    """
    Async double-buffered PBO pixel readback.

    Framework-agnostic: all widget coupling goes through WidgetBridge.
    Results are delivered via on_frame_ready, a plain FrameCallback callable
    that receives a (H×W×4) uint8 RGBA numpy array plus pixel dimensions.

    Supports two transfer sources
    ------------------------------
    Framebuffer   request_capture()                 → glReadPixels
    Texture       request_texture_capture(id, w, h) → glGetTexImage

    Integration contract
    --------------------
    initializeGL  → manager.initialize()
    resizeGL      → manager.on_resize(w, h)   (args are ignored; bridge queried)
    paintGL       → manager.capture_now()     (must be the very last call)
    """

    def __init__(
            self,
            bridge: WidgetBridge,
            on_frame_ready: FrameCallback,
    ) -> None:
        """
        bridge         : WidgetBridge implementation for the host widget.
        on_frame_ready : Called with (arr, width, height) when a frame is ready.
                         arr is a (H, W, 4) uint8 RGBA ndarray, top-left origin,
                         fully independent of GPU memory.
        """
        self._bridge = bridge
        self._on_frame_ready = on_frame_ready

        self.pbo_a: PackPBO | None = None
        self.pbo_b: PackPBO | None = None
        self._current_pbo: PackPBO | None = None  # receives this frame's transfer
        self._previous_pbo: PackPBO | None = None  # holds last frame's data

        self.width: int = 0
        self.height: int = 0
        self._frame_count: int = 0

        self._capture_requested: bool = False
        self._pending_texture: tuple[int, int, int] | None = None
        self._initialized: bool = False

    def initialize(self) -> None:
        """
        Call once inside initializeGL().
        The GL context is already current; do not call make_current here.
        """
        self.pbo_a = PackPBO()
        self.pbo_b = PackPBO()
        self._current_pbo = self.pbo_a
        self._previous_pbo = self.pbo_b
        self._allocate_buffers(
            self._bridge.physical_width(),
            self._bridge.physical_height(),
        )
        self._initialized = True

    def on_resize(self, _w: int = 0, _h: int = 0) -> None:
        """
        Call from resizeGL(w, h).

        The arguments are intentionally ignored — bridge.physical_width/height
        are queried directly so the result is always correct regardless of Qt
        version or HiDPI scaling behaviour.  Context is current.
        """
        if not self._initialized:
            return
        w = self._bridge.physical_width()
        h = self._bridge.physical_height()
        if w != self.width or h != self.height:
            self._allocate_buffers(w, h)

    def request_capture(self) -> None:
        """
        Request a framebuffer screenshot (glReadPixels path).
        on_frame_ready fires after the next completed two-frame cycle.
        """
        if not self._initialized:
            raise RuntimeError(
                "PBODownloadManager.initialize() must be called "
                "before request_capture()."
            )
        self._pending_texture = None
        self._capture_requested = True
        self._bridge.schedule_update()

    def request_texture_capture(
            self, texture_id: int, width: int, height: int
    ) -> None:
        """
        Request a texture download (glGetTexImage path).

        texture_id   : GL texture name (from glGenTextures)
        width/height : pixel dimensions of mip level 0

        on_frame_ready fires after the next completed two-frame cycle.
        Must be called while the GL context is current.
        """
        if not self._initialized:
            raise RuntimeError(
                "PBODownloadManager.initialize() must be called "
                "before request_texture_capture()."
            )
        if width != self.width or height != self.height:
            self._allocate_buffers(width, height)
        self._pending_texture = (texture_id, width, height)
        self._capture_requested = True
        self._bridge.schedule_update()

    def capture_now(self) -> None:
        """
        Call from paintGL() AFTER all scene rendering commands.

        Two-frame pipeline
        ------------------
        Call A (frame_count == 0)
            Issue transfer into _current_pbo; insert fence.
            Nothing to read yet (_previous_pbo uninitialised).
            Swap; frame_count = 1; schedule_update() → call B.

        Call B (frame_count >= 1)
            Issue transfer into _current_pbo; insert fence.
            Wait on _previous_pbo fence (almost always instant).
            glGetBufferSubData → numpy → on_frame_ready callback.
            Swap; frame_count = 2; clear _capture_requested.
        """
        if not self._capture_requested:
            return
        if not self._initialized:
            logger.error("capture_now() called before initialize().")
            return
        if self.width == 0 or self.height == 0:
            return  # dimensions not yet known; retry next paintGL

        size_bytes = self.width * self.height * 4  # GL_RGBA

        # Step 1 — issue async transfer and fence
        if self._pending_texture is not None:
            self._issue_texture_transfer(self._current_pbo,
                                         self._pending_texture)
        else:
            self._issue_readpixels(self._current_pbo)
        self._current_pbo.insert_fence()

        # Step 2 — read previous frame (uninitialised on first call)
        if self._frame_count > 0:
            self._wait_and_deliver(self._previous_pbo, size_bytes)

        # Step 3 — swap roles
        self._current_pbo, self._previous_pbo = (
            self._previous_pbo, self._current_pbo
        )
        self._frame_count += 1

        # Step 4 — loop control
        if self._frame_count == 1:
            # First transfer queued; no data emitted yet. Schedule call B.
            self._bridge.schedule_update()
        else:
            # Delivery happened in step 2; pipeline complete.
            self._capture_requested = False
            self._pending_texture = None

    def cleanup(self) -> None:
        """
        Release all GPU resources.  Safe to call multiple times.

        Uses bridge.is_context_valid / make_current / done_current; if the
        context is already gone handles are zeroed without GL calls.
        """
        if not self._initialized:
            return
        self._initialized = False

        if self._bridge.is_context_valid():
            self._bridge.make_current()
            for pbo in (self.pbo_a, self.pbo_b):
                if pbo is not None:
                    pbo.destroy()
            self._bridge.done_current()
        else:
            for pbo in (self.pbo_a, self.pbo_b):
                if pbo is not None:
                    pbo.id = 0
                    pbo.is_mapped = False
                    pbo.fence = _NO_SYNC

        self.pbo_a = self.pbo_b = None
        self._current_pbo = self._previous_pbo = None

    def _allocate_buffers(self, width: int, height: int) -> None:
        """
        (Re)create both PackPBOs for width × height × 4 bytes (GL_RGBA).
        Caller must ensure the GL context is current.
        Resets _frame_count so uninitialised PBO data is never read.
        """
        if width <= 0 or height <= 0:
            return

        for pbo in (self.pbo_a, self.pbo_b):
            if pbo is not None:
                pbo.destroy()

        self.pbo_a = PackPBO()
        self.pbo_b = PackPBO()
        self._current_pbo = self.pbo_a
        self._previous_pbo = self.pbo_b

        self.width = width
        self.height = height
        buf_size = width * height * 4  # GL_RGBA

        for pbo in (self.pbo_a, self.pbo_b):
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pbo.id)
            GL.glBufferData(GL.GL_PIXEL_PACK_BUFFER, buf_size, None,
                            GL.GL_STREAM_READ)
            pbo.capacity = buf_size

        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, _NO_BUFFER)
        self._frame_count = 0

    # ------------------------------------------------------------------
    # Private: transfer issuing
    # ------------------------------------------------------------------

    def _issue_readpixels(self, pbo: PackPBO) -> None:
        """
        Bind *pbo* and issue async glReadPixels into it.

        With a PACK PBO bound, glReadPixels returns immediately; the DMA runs
        in the background.  ctypes.c_void_p(0) is a byte offset of 0 into the
        bound buffer — it is NOT a client-memory pointer.

        GL_RGBA is required on macOS: GL_RGB on the default framebuffer
        generates GL_INVALID_OPERATION on Apple's driver.
        """
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pbo.id)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 4)
        with gl_error_check("glReadPixels into PBO", RuntimeError):
            GL.glReadPixels(
                0, 0, self.width, self.height,
                GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
                ctypes.c_void_p(0),  # offset 0 into the bound PBO
            )
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, _NO_BUFFER)

    def _issue_texture_transfer(
            self, pbo: PackPBO, pending: tuple[int, int, int]
    ) -> None:
        """
        Bind *pbo* and issue async glGetTexImage into it.

        With a PACK PBO bound, the last argument is a byte offset into the
        buffer — NOT a client-memory pointer.  ctypes.c_void_p(0) = offset 0.
        """
        tex_id, _w, _h = pending
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pbo.id)
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 4)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex_id)
        with gl_error_check("glGetTexImage into PBO", RuntimeError):
            GL.glGetTexImage(
                GL.GL_TEXTURE_2D,
                0,  # mip level 0
                GL.GL_RGBA,
                GL.GL_UNSIGNED_BYTE,
                ctypes.c_void_p(0),  # offset 0 into the bound PBO
            )
        GL.glBindTexture(GL.GL_TEXTURE_2D, _NO_BUFFER)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, _NO_BUFFER)

    # ------------------------------------------------------------------
    # Private: readback and delivery
    # ------------------------------------------------------------------

    def _wait_and_deliver(self, pbo: PackPBO, size_bytes: int) -> None:
        """
        1. Wait for pbo's fence (DMA complete).
        2. Copy pixels via glGetBufferSubData into a Python-owned array.
        3. Flip vertically (OpenGL bottom-left → top-left origin).
        4. Call on_frame_ready(arr, width, height).

        glGetBufferSubData writes into a numpy array we own — no mapping,
        no ctypes pointer casting, no map/unmap lifecycle.
        arr is fully independent of GPU memory before the callback fires.
        """
        # 1. Fence wait
        try:
            pbo.wait_for_fence()
        except GLSyncTimeout:
            logger.warning("PBO %d: fence timed out — skipping capture.",
                           pbo.id)
            return
        except GLMemoryError as exc:
            logger.error("PBO %d: fence error: %s", pbo.id, exc)
            return

        # 2. Copy out
        dest = np.empty(size_bytes, dtype=np.uint8)
        GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, pbo.id)
        try:
            with gl_error_check("glGetBufferSubData", GLMemoryError):
                GL.glGetBufferSubData(GL.GL_PIXEL_PACK_BUFFER, 0, size_bytes,
                                      dest)
        except Exception as exc:
            logger.error("PBO %d: glGetBufferSubData failed: %s", pbo.id, exc)
            return
        finally:
            GL.glBindBuffer(GL.GL_PIXEL_PACK_BUFFER, _NO_BUFFER)

        # 3. Flip — np.flipud on a reshape view has negative strides;
        #    ascontiguousarray gives a safe C-contiguous independent copy.
        arr = np.ascontiguousarray(
            np.flipud(dest.reshape(self.height, self.width, 4))
        )

        # 4. Deliver — arr is fully owned by the caller from this point.
        self._on_frame_ready(arr, self.width, self.height)

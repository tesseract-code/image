"""
gl_frame_viewer.py
==================
Hardware-accelerated OpenGL image viewer widget for PyQt6.

Provides real-time shader effects (color correction, normalization, colourmap
application) via a PBO double-buffering pipeline.  Two upload paths are
supported:

* **Standard** – CPU ndarray → PBO (map/copy/unmap) → Texture
* **Pinned**   – Pre-filled mapped PBO → Texture (zero-copy)

Thread-safety:
    All public GL-touching methods must be called from the Qt main thread
    (the thread that owns the QOpenGLWidget's context).
"""

from __future__ import annotations

import ctypes
import logging
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QImage
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtWidgets import QWidget

from image.gl.backend import GL, initialize_context
from image.gl.errors import (
    GLError,
    GLInitializationError,
    GLMemoryError,
    GLTextureError,
    GLUploadError, clear_gl_errors,
)
from image.gl.format import get_gl_texture_spec
from image.gl.pbo import (PBOBufferingStrategy, PBOUploadManager,
                          configure_pixel_storage, memmove_pbo,
                          write_pbo_buffer)
from image.gl.pbo.bridge import QtPBOBridge
from image.gl.program import ShaderProgramManager
from image.gl.quad import GeometryManager
from image.gl.shaders.paths import (
    IMAGE_SHADERS,
    validate_shader_paths,
)
from image.gl.texture import (
    TextureManager,
    TextureState,
    TextureUploadPayload,
    alloc_texture_storage,
)
from image.gl.types import (GLenum, GLbitfield,
                            GLfloat,
                            GLint, GLuint,
                            GLsizei, GLBuffer,
                            GLTexture, GLHandle)
from image.gl.uniform import (
    UniformManager,
    UniformType,
    FragmentShaderUniforms, VertexShaderUniforms,
)
from image.gl.utils import gl_context
from image.gl.viewport import ViewManager
from image.model.cmap import ColormapModel
from image.pipeline.stats import FrameStats
from image.settings.base import (
    ImageSettings,
    ImageSettingsSnapshot,
)
from image.settings.pixels import PixelFormat
from image.utils.types import is_standard_image
from pycore.log.ctx import ContextAdapter
from pycore.mtcopy import get_global_executor
from qtcore.monitor import PerfStats, PerformanceMonitor
from qtcore.reference import has_qt_cpp_binding

logger = ContextAdapter(logging.getLogger(__name__), {})


@dataclass(slots=True, frozen=False)
class GLState:
    """
    Tracks mutable OpenGL resource-binding state between calls.

    Used to avoid redundant bind/unbind round-trips and to detect
    inconsistent state early.  All fields reflect the driver-side state
    at the time of the last recorded operation; they are *not* authoritative
    in the presence of external GL calls.
    """
    texture_bound: Optional[GLTexture] = None
    pbo_bound: Optional[GLBuffer] = None
    program_active: Optional[GLuint] = None
    current_texture_state: Optional[TextureState] = None
    current_filter_mode: Optional[bool] = None
    active_texture_unit: GLenum = GLenum(GL.GL_TEXTURE0)
    uniforms_dirty: bool = True
    initialized: bool = False
    paint_event_count: int = 0

    def reset(self) -> None:
        """Reset all tracked state to uninitialised defaults."""
        self.texture_bound = None
        self.pbo_bound = None
        self.program_active = None
        self.uniforms_dirty = True
        self.current_texture_state = None
        self.current_filter_mode = None
        self.paint_event_count = 0
        self.active_texture_unit = GLenum(GL.GL_TEXTURE0)
        self.initialized = False


class GLFrameViewer(QOpenGLWidget):
    """
    OpenGL image viewer widget.

    Provides hardware-accelerated display with real-time shader effects
    (color correction, normalization, colormap application).  Supports both
    standard CPU→PBO→Texture uploads and zero-copy pinned-buffer paths.

    Upload pipelines
    ----------------
    Unpinned: ``CPU Array → PBO (map/copy/unmap) → Texture (glTexSubImage2D)``
    Pinned:   ``CPU Array → PBO (direct write)   → Texture (glTexSubImage2D)``

    Signals:
        frame_changed(object): Emitted with frame metadata after each
            successful upload.
        gl_error(str): Emitted with a human-readable message when a
            non-fatal GL error is caught and logged.

    Attributes:
        settings (ImageSettings): Live display and shader configuration.
    """

    @staticmethod
    def _as_uint(handle) -> int:
        """Return the underlying integer for any GL NewType handle."""
        return int(handle)

    frameChanged = pyqtSignal(object)  # payload.meta (FrameStats or similar)
    image_ready = pyqtSignal(QImage)
    glError = pyqtSignal(str)

    def __init__(
            self,
            settings: ImageSettings,
            buffer_strategy: PBOBufferingStrategy = PBOBufferingStrategy.DOUBLE,
            monitor_performance: bool = True,
            parent: Optional[QWidget] = None,
    ) -> None:
        """
        Initialize the viewer.

        Args:
            settings:           Image display settings and shader parameters.
            buffer_strategy:    Number of Pixel Buffer Objects for async
                                uploads (minimum 2 for double-buffering).
            monitor_performance: When ``True``, collect FPS and frame-timing
                                statistics via :class:`PerformanceMonitor`.
            parent:             Parent Qt widget.
        """
        super().__init__(parent=parent)

        # Warm up the global thread-pool used by async PBO transfers.

        get_global_executor(max_workers=4)

        self.settings: ImageSettings = settings
        self._settings_snapshot: Optional[ImageSettingsSnapshot] = None

        # Image texture (unit 0)
        self._image_texture_id: Optional[GLTexture] = None
        self._image_texture_unit: GLint = GLint(0)

        # Colormap texture (unit 1)
        self._cmap_texture_id: Optional[GLTexture] = None
        self._cmap_texture_unit: GLint = GLint(1)
        self._cmap_cache = ColormapModel()

        # Resource managers
        self._program_manager: ShaderProgramManager = ShaderProgramManager()
        self._texture_manager: TextureManager = TextureManager()
        self._view_manager: ViewManager = ViewManager()
        self._pbo_upload_mngr: PBOUploadManager = PBOUploadManager(
            buffer_strategy=buffer_strategy)
        self._pbo_download_bridge: QtPBOBridge | None = None
        self._geo_manager: GeometryManager = GeometryManager()

        # GL state shadow
        self._gl_state = GLState()
        self._cleaned_up = False

        # Pending upload bookkeeping
        self._frame_data: Optional[np.ndarray] = None
        self._latest_payload: Optional[TextureUploadPayload] = None
        self._pending_frame_update: bool = False

        # Input handling
        self._last_mouse_pos: Optional[QPointF] = None
        self._is_panning = False

        # Performance monitoring
        self._perf_monitor: Optional[PerformanceMonitor] = None
        self._current_stats: Optional[PerfStats] = None
        if monitor_performance:
            self._perf_monitor = PerformanceMonitor(window_size=30)

        logger.info(
            "GLFrameViewer created",
            buffer_strategy=buffer_strategy,
            monitoring=monitor_performance,
        )

    @property
    def data(self) -> Optional[np.ndarray]:
        """Current frame data (read-only reference)."""
        return self._frame_data

    @property
    def has_data(self) -> bool:
        """``True`` when at least one frame has been uploaded."""
        return self._frame_data is not None

    @property
    def dtype(self) -> Optional[np.dtype]:
        """Data type of the current frame, or ``None`` if no frame is loaded."""
        return self.data.dtype if self.has_data else None

    @property
    def aspect_ratio(self) -> float:
        """Image aspect ratio (width / height); ``1.0`` when height is zero."""
        if self._view_manager.image_h == 0:
            return 1.0
        return self._view_manager.image_w / self._view_manager.image_h

    @property
    def is_initialized(self) -> bool:
        """``True`` after :meth:`initializeGL` has completed successfully."""
        return self._gl_state.initialized

    def initializeGL(self) -> None:
        """
        Set up all OpenGL resources for this widget.

        Called once by the Qt framework after the GL context becomes current.
        Creates the shader program, PBOs, geometry buffers, and the colormap
        texture, then connects the settings-change signal.

        Raises:
            GLInitializationError: On any failure during setup.  The
                ``gl_error`` signal is emitted before raising so the UI can
                react without catching the exception itself.
        """
        try:
            initialize_context()

            GL.glClearColor(GLfloat(0.1), GLfloat(0.1), GLfloat(0.1),
                            GLfloat(1.0))

            GL.glEnable(GLenum(GL.GL_BLEND))
            GL.glBlendFunc(GLenum(GL.GL_SRC_ALPHA),
                           GLenum(GL.GL_ONE_MINUS_SRC_ALPHA))
            GL.glEnable(GLenum(GL.GL_FRAMEBUFFER_SRGB))

            self._pbo_upload_mngr.initialize()
            self._pbo_download_bridge = QtPBOBridge(self)
            self._pbo_download_bridge.initialize()

            validate_shader_paths(shaders=IMAGE_SHADERS)
            self._program_manager.initialize(
                vertex_path=IMAGE_SHADERS["image_vertex"],
                fragment_path=IMAGE_SHADERS["image_fragment"],
            )

            if not self._program_manager.is_valid:
                raise GLInitializationError("Failed to create shader program")

            self._program_manager.uniform_manager.register_members(
                VertexShaderUniforms
            )
            self._program_manager.uniform_manager.register_members(
                FragmentShaderUniforms
            )

            if not self._geo_manager.initialize():
                raise GLInitializationError("Failed to initialize geometry")

            ctx = self.context()
            if not ctx or not ctx.isValid():
                raise GLInitializationError("OpenGL context is invalid")

            self._cmap_texture_id = GLTexture(
                self._texture_manager.create_texture("cmap"))
            self._gl_state.initialized = True

            self._sync_settings_to_view()

            self._view_manager.handle_resize(self.width(), self.height())

            self.settings.changed.connect(self._on_settings_changed)
            self._pbo_download_bridge.image_ready.connect(self.image_ready)

            logger.info(
                "OpenGL initialisation complete",
                program_id=self._program_manager.handle,
                cmap_texture_id=self._cmap_texture_id,
            )

        except GLInitializationError:
            # Already the right type — emit signal then re-raise unchanged.
            raise
        except GLError as e:
            # A specific GL subclass from a subsystem (shader, texture, etc.)
            # — wrap it so callers only need to catch one init-phase type.
            msg = f"GL initialization failed: {e}"
            logger.error(msg)
            self.glError.emit(msg)
            raise GLInitializationError(msg) from e
        except Exception as e:
            msg = f"Unexpected error during GL initialization: {e}"
            logger.error(msg, exception_type=type(e).__name__)
            self.glError.emit(msg)
            raise GLInitializationError(msg) from e

    def resizeGL(self, width: GLsizei, height: GLsizei) -> None:
        """
        Handle viewport resize.

        Args:
            width:  New viewport width in pixels.
            height: New viewport height in pixels.
        """
        logger.debug("Resizing viewport", width=width, height=height)
        GL.glViewport(GLint(0), GLint(0), width, height)
        self._view_manager.handle_resize(width, height)
        self._pbo_download_bridge.on_resize(width, height)
        self._gl_state.uniforms_dirty = True
        if self._can_render():
            self.update()

    def paintGL(self) -> None:
        """
        Render one frame with the active shader effects.

        Called automatically by Qt on ``update()`` or when the widget needs
        repainting.  All GL errors are caught, logged, and re-emitted via
        ``gl_error`` so the caller is never left with an unhandled exception
        on the render thread.
        """
        logger.debug(
            "Paint event started",
            event_count=self._gl_state.paint_event_count,
            has_data=self.has_data,
            initialized=self._gl_state.initialized,
        )

        if not self._can_render():
            logger.debug(
                "Skipping paint — resources not ready",
                gl_init=self._gl_state.initialized,
                has_texture=bool(self._image_texture_id),
                has_program=self._program_manager.is_valid,
            )
            return

        try:
            clear_mask: GLbitfield = GLbitfield(GL.GL_COLOR_BUFFER_BIT)
            GL.glClear(clear_mask)
            self._geo_manager.bind()

            if self._gl_state.uniforms_dirty:
                logger.debug("Uploading uniforms (dirty flag set)")
                self._upload_uniforms()
                self._gl_state.uniforms_dirty = False

            with self._program_manager as prog_id:
                self._gl_state.program_active = GLuint(prog_id)
                try:
                    self._bind_image_texture()
                    self._bind_colormap_texture()
                    self._geo_manager.draw()
                    self._gl_state.paint_event_count += 1
                    logger.debug(
                        "Frame rendered successfully",
                        draw_count=self._gl_state.paint_event_count,
                    )
                finally:
                    self._geo_manager.unbind()
                    self._cleanup_texture_bindings()
                    self._pbo_download_bridge.capture_now()

            self._gl_state.program_active = None

        except GLTextureError as e:
            # Texture errors during paint are non-fatal but worth flagging.
            msg = f"Texture error during render: {e}"
            logger.error(msg)
            self.glError.emit(msg)
        except GLError as e:
            msg = f"Render error: {e}"
            logger.error(msg)
            self.glError.emit(msg)
        except Exception as e:
            msg = f"Unexpected render error: {e}"
            logger.error(msg, exception_type=type(e).__name__)
            self.glError.emit(msg)

    def _can_render(self) -> bool:
        """
        Return ``True`` when all resources required for a draw call are ready.

        Checks initialization, texture, shader program, and VAO availability.
        """
        return (
                self._gl_state.initialized
                and self._image_texture_id is not None
                and self._program_manager.is_valid
                and self._geo_manager.vao is not None
        )

    def _can_upload(self) -> Tuple[bool, Optional[str]]:
        """
        Validate that the widget is in a state that allows a texture upload.

        Returns:
            ``(True, None)`` when an upload may proceed, or
            ``(False, reason_string)`` when a precondition is unmet.
        """
        if not self._gl_state.initialized:
            return False, "OpenGL not initialised"
        if not self.isValid():
            return False, "Widget is not valid"
        if not self.context() or not self.context().isValid():
            return False, "OpenGL context is invalid"
        if not self.isVisible():
            return False, "Widget is not visible"
        return True, None

    def _bind_image_texture(self) -> None:
        """
        Activate texture unit 0 and bind the main image texture to it.

        Raises:
            GLTextureError: If the texture state is inconsistent (renderer_id
                is ``None`` despite the texture being registered).
        """
        state = self._texture_manager.get_state("main")
        if state is None:
            # No texture allocated yet — silently skip; _can_render guards this.
            return
        if state.renderer_id is None:
            raise GLTextureError(
                "Main texture is registered but has no renderer_id; "
                "the texture was likely deleted without being recreated."
            )
        self._texture_manager.activate(self._image_texture_unit)
        self._gl_state.active_texture_unit = GLenum(self._image_texture_unit)
        self._texture_manager.bind("main", self._image_texture_unit)
        self._gl_state.texture_bound = self._image_texture_id

    def _bind_colormap_texture(self) -> None:
        """
        Activate texture unit 1 and bind either the colormap or a fallback.

        When the colormap is enabled and a colormap texture exists, that
        texture is bound.  Otherwise, the main image texture is rebound to
        unit 1 to prevent the sampler from reading an unbound texture unit.

        Raises:
            GLTextureError: If the fallback image texture is also unavailable.
        """
        self._texture_manager.activate(self._cmap_texture_unit)
        self._gl_state.active_texture_unit = GLenum(self._cmap_texture_unit)

        use_cmap = (
                self.settings.colormap_enabled
                and self._cmap_texture_id is not None
        )

        if use_cmap:
            logger.debug(
                "Binding colormap texture", texture_id=self._cmap_texture_id
            )
            self._texture_manager.bind("cmap", self._cmap_texture_unit)
            self._gl_state.texture_bound = self._cmap_texture_id
            self._texture_manager.set_sampling_mode(
                min_filter=GLenum(GL.GL_LINEAR),
                mag_filter=GLenum(GL.GL_LINEAR),
                generate_mipmaps=False,
            )
        else:
            # Fallback: rebind the main image to unit 1 to avoid sampling
            # an unbound texture unit, which produces GL_INVALID_OPERATION.
            if self._image_texture_id is None:
                raise GLTextureError(
                    "Colormap fallback failed: main image texture is not "
                    "allocated.  Cannot bind a valid texture to unit 1."
                )
            logger.debug("Using main texture as colormap fallback")
            GL.glBindTexture(GLenum(GL.GL_TEXTURE_2D), self._image_texture_id)
            self._gl_state.texture_bound = self._image_texture_id

    def _set_cmap_texture_params(self) -> None:
        """
        Set wrap / filter parameters on the currently bound colormap texture.

        Must be called while texture unit ``_cmap_texture_unit`` is active
        and the colormap texture is bound.  Disables mip-mapping explicitly
        (the LUT is a flat 256×1 strip).
        """
        if self._gl_state.active_texture_unit != self._cmap_texture_unit:
            return
        GL.glTexParameteri(GLenum(GL.GL_TEXTURE_2D),
                           GLenum(GL.GL_TEXTURE_WRAP_S),
                           GLint(GL.GL_CLAMP_TO_EDGE))
        GL.glTexParameteri(GLenum(GL.GL_TEXTURE_2D),
                           GLenum(GL.GL_TEXTURE_WRAP_T),
                           GLint(GL.GL_CLAMP_TO_EDGE))
        GL.glTexParameteri(GLenum(GL.GL_TEXTURE_2D),
                           GLenum(GL.GL_TEXTURE_MIN_FILTER),
                           GLint(GL.GL_LINEAR))
        GL.glTexParameteri(GLenum(GL.GL_TEXTURE_2D),
                           GLenum(GL.GL_TEXTURE_MAG_FILTER),
                           GLint(GL.GL_LINEAR))
        GL.glTexParameteri(GLenum(GL.GL_TEXTURE_2D),
                           GLenum(GL.GL_TEXTURE_BASE_LEVEL),
                           GLint(0))
        GL.glTexParameteri(GLenum(GL.GL_TEXTURE_2D),
                           GLenum(GL.GL_TEXTURE_MAX_LEVEL),
                           GLint(0))

    def _cleanup_texture_bindings(self) -> None:
        """
        Unbind all texture units used during rendering.

        Leaves the driver in a clean state after each paint.  Errors here are
        logged as warnings rather than raised: a failed unbind does not corrupt
        the rendered output and should not abort the render loop.
        """
        try:
            self._texture_manager.unbind(self._cmap_texture_unit)
            self._texture_manager.unbind(self._image_texture_unit)
            self._gl_state.texture_bound = None
            self._gl_state.active_texture_unit = GLenum(GL.GL_TEXTURE0)
        except Exception as e:
            logger.warning("Failed to clean up texture bindings", error=str(e))

    def _set_sampling_mode(self) -> None:
        """
        Apply the current interpolation setting to the main image texture.

        Sets ``GL_NEAREST`` or ``GL_LINEAR`` filter modes and disables
        mip-mapping via ``BASE_LEVEL`` / ``MAX_LEVEL`` to prevent the
        "incomplete texture" driver warning on non-power-of-two images.

        Raises:
            GLTextureError: If the texture manager fails to apply parameters.
        """
        settings = self.settings.get_copy()
        mode = GLenum(GL.GL_LINEAR) if settings.interpolation else GLenum(
            GL.GL_NEAREST)

        # Re-activate the image unit: a previous update_swizzle call may have
        # left a different unit active.
        self._texture_manager.activate(self._image_texture_unit)
        self._texture_manager.set_sampling_mode(
            min_filter=mode,
            mag_filter=mode,
            wrap_s=GLenum(GL.GL_CLAMP_TO_EDGE),
            wrap_t=GLenum(GL.GL_CLAMP_TO_EDGE),
        )
        # Disable mipmaps explicitly to fix the "Incomplete Texture" warning
        # that the driver raises for non-power-of-two textures when MAX_LEVEL > 0.
        GL.glTexParameteri(GLenum(GL.GL_TEXTURE_2D),
                           GLenum(GL.GL_TEXTURE_BASE_LEVEL), GLint(0))
        GL.glTexParameteri(GLenum(GL.GL_TEXTURE_2D),
                           GLenum(GL.GL_TEXTURE_MAX_LEVEL), GLint(0))
        self._gl_state.current_filter_mode = settings.interpolation

    def _alloc_image_texture(
            self,
            width: GLsizei,
            height: GLsizei,
            gl_format: GLenum,
            internal_format: GLenum,
            gl_type: GLenum
    ) -> None:
        """
        Allocate (or reallocate) the main image texture.

        Skips reallocation when the existing texture state is unchanged.
        When a resize or format change is detected, the old texture is deleted
        before the new one is created so GPU memory is not double-allocated.

        Args:
            width:           Texture width in pixels.
            height:          Texture height in pixels.
            gl_format:       GL base format token (e.g. ``GL_RGB``).
            internal_format: GL internal format token (e.g. ``GL_RGB8``).
            gl_type:         GL element type token (e.g. ``GL_UNSIGNED_BYTE``).

        Raises:
            GLTextureError: If texture creation or storage allocation fails.
            GLMemoryError:  If the driver reports ``GL_OUT_OF_MEMORY`` during
                            storage allocation.
        """
        state: TextureState | None = self._texture_manager.get_state("main")

        if state and state.renderer_id is not None:
            if self._gl_state.current_texture_state == state:
                logger.debug("Texture state unchanged, skipping recreation")
                return
            self._texture_manager.delete_texture("main")
            self._image_texture_id = None

        try:
            self._image_texture_id = GLTexture(
                self._texture_manager.create_texture("main"))
            self._bind_image_texture()
            self._set_sampling_mode()
            self._texture_manager.allocate(
                key="main",
                width=width,
                height=height,
                gl_int_fmt=internal_format,
                gl_fmt=gl_format,
                gl_type=gl_type,
                data=ctypes.c_void_p(0),
            )
        except MemoryError as e:
            raise GLMemoryError(
                f"GPU out of memory allocating {width}×{height} texture"
            ) from e
        except GLError:
            raise
        except Exception as e:
            raise GLTextureError(
                f"Failed to allocate image texture ({width}×{height}): {e}"
            ) from e
        finally:
            # Always unbind regardless of success/failure to leave clean state.
            self._texture_manager.unbind(self._image_texture_unit)

        self._gl_state.current_texture_state = state

    def _realloc_image_texture(self, payload: TextureUploadPayload) -> bool:
        """
        Reallocate the image texture if its dimensions have changed.

        Args:
            payload: Upload payload carrying the new image dimensions and
                     format tokens.

        Returns:
            ``True`` if a reallocation occurred, ``False`` if the existing
            texture was reused unchanged.

        Raises:
            GLTextureError: Propagated from :meth:`_alloc_image_texture`.
            GLMemoryError:  Propagated from :meth:`_alloc_image_texture`.
        """
        state = self._texture_manager.get_state("main")
        should_realloc = state is None or (
                payload.width != state.width or payload.height != state.height
        )

        if should_realloc:
            # Unbind PBO before glTexImage2D so the driver allocates from
            # client memory rather than treating the PBO offset as a pointer.
            self._pbo_upload_mngr.unbind()
            self._alloc_image_texture(
                payload.width,
                payload.height,
                payload.gl_format,
                payload.gl_internal_format,
                payload.gl_type,
            )

        return should_realloc

    def _upload_image_texture(self, payload: TextureUploadPayload) -> None:
        """
        Transfer pixel data from the bound PBO into the image texture.

        This is the final stage of both upload paths (standard and pinned).
        The caller must have a PBO already bound before invoking this method;
        passing offset ``0`` to ``glTexSubImage2D`` instructs the driver to
        read from the bound PBO rather than a CPU pointer.

        Args:
            payload: Describes the image dimensions, format, and type tokens.

        Raises:
            GLUploadError: If the texture sub-image transfer fails.
        """
        try:
            self._texture_manager.activate(self._image_texture_unit)
            self._texture_manager.bind("main", self._image_texture_unit)
            self._set_sampling_mode()
            configure_pixel_storage(payload.gl_type, payload.gl_format)
            self._texture_manager.update_swizzle("main", payload.gl_format)

            GL.glTexSubImage2D(
                GLenum(GL.GL_TEXTURE_2D), GLint(0), GLint(0), GLint(0),
                payload.width, payload.height,
                payload.gl_format, payload.gl_type,
                ctypes.c_void_p(0),  # offset 0 → read from bound PBO
            )
        except GLError:
            raise
        except Exception as e:
            raise GLUploadError(
                f"glTexSubImage2D failed for {payload.width}×{payload.height} "
                f"image: {e}"
            ) from e

    def _upload_cmap_texture(self, data: np.ndarray) -> None:
        """
        Upload a colormap LUT as a 256×1 RGB texture on unit 1.

        Must be called with an active GL context.

        Args:
            data: ``(256, 3)`` uint8 array of RGB color entries.

        Raises:
            GLTextureError: If the colormap texture upload fails.
        """
        try:
            self._bind_colormap_texture()
            self._set_cmap_texture_params()
            GL.glPixelStorei(GLenum(GL.GL_UNPACK_ALIGNMENT), GLint(4))
            alloc_texture_storage(
                target=GLenum(GL.GL_TEXTURE_2D),
                width=GLsizei(self._cmap_cache.resolution),
                height=GLsizei(1),
                gl_int_fmt=GLenum(GL.GL_RGB8),
                gl_fmt=GLenum(GL.GL_RGB),
                gl_type=GLenum(GL.GL_UNSIGNED_BYTE),
                data=data,
            )
        except GLError:
            raise
        except Exception as e:
            raise GLTextureError(f"Colormap texture upload failed: {e}") from e
        finally:
            # Restore image texture binding so subsequent code sees unit 0.
            self._bind_image_texture()

    @pyqtSlot(object)  # TextureUploadPayload
    def _upload_frame(
            self,
            payload: TextureUploadPayload,
            repaint: bool = True,
    ) -> None:
        """
        Orchestrate a complete CPU→PBO→Texture upload.

        Args:
            payload: Prepared upload descriptor (from :meth:`present` or
                     :meth:`present_pinned`).
            repaint: When ``True``, calls ``update()`` after the upload to
                     trigger a repaint.

        Raises:
            GLUploadError: If any stage of the upload pipeline fails.
            GLTextureError: Propagated if texture (re)allocation fails.
            GLMemoryError: Propagated if the GPU runs out of memory.
        """
        with gl_context(self, "upload_frame"):
            try:
                self._realloc_image_texture(payload)

                if not payload.is_pinned and payload.data is not None:
                    memmove_pbo(payload.pbo_id, payload.data)
                else:
                    self._pbo_upload_mngr.bind(payload.pbo_id)

                self._view_manager.set_image_size(payload.width, payload.height)
                self._upload_image_texture(payload)
                self._pbo_upload_mngr.unbind()
                self._texture_manager.unbind(self._image_texture_unit)

                if not payload.is_pinned:
                    self._frame_data = payload.data

            except (GLUploadError, GLTextureError, GLMemoryError):
                raise
            except GLError as e:
                raise GLUploadError(
                    f"Upload pipeline failed: {e}"
                ) from e
            except Exception as e:
                raise GLUploadError(
                    f"Unexpected error in upload pipeline: {e}"
                ) from e

        if self.settings.colormap_enabled:
            self._set_colormap()

        if repaint:
            self.update()

    @pyqtSlot()
    def _process_frame(self, repaint: bool = True) -> None:
        """
        Flush the pending frame payload through the upload pipeline.

        A no-op when no payload is queued.  Emits ``frame_changed`` on success
        and clears the pending-update flag regardless of outcome.
        """
        payload = self._latest_payload
        if payload is None:
            return

        try:
            self._upload_frame(payload, repaint=repaint)
            self.frameChanged.emit(payload.meta)
        except (GLUploadError, GLTextureError, GLMemoryError) as e:
            msg = str(e)
            logger.error("Frame processing failed", error=msg)
            self.glError.emit(msg)
        finally:
            self._latest_payload = None
            self._pending_frame_update = False

    def _set_colormap(self) -> None:
        """
        Fetch the active colormap LUT and upload it to the GPU.

        Safe to call at runtime whenever the colormap name or reverse flag
        changes.  Failures are caught and emitted via ``gl_error`` rather than
        propagated, because a colormap update failure should degrade display
        quality gracefully rather than crash rendering.
        """
        if self._cmap_texture_id is None:
            return

        try:
            lut_data = self._cmap_cache.get_lut(
                cmap_name=self.settings.colormap_name,
                reverse=self.settings.colormap_reverse,
            )
            with gl_context(self, "set_colormap"):
                logger.debug("Updating colormap texture")
                self._upload_cmap_texture(lut_data)
                self._gl_state.uniforms_dirty = True

        except GLTextureError as e:
            msg = f"Colormap update failed: {e}"
            logger.error(msg)
            self.glError.emit(msg)
        except Exception as e:
            msg = f"Unexpected error updating colormap: {e}"
            logger.error(msg, exception_type=type(e).__name__)
            self.glError.emit(msg)

    def _set_norm_uniforms(
            self,
            mngr: UniformManager,
            vmin: Optional[float] = None,
            vmax: Optional[float] = None,
    ) -> None:
        """
        Write normalization range uniforms into the active shader batch.

        When both ``vmin`` and ``vmax`` are ``None`` the range is
        auto-detected: RGB/RGBA images get ``[0, 1]``; grayscale arrays get
        the actual ``[min, max]`` of the current frame for contrast
        stretching.

        Args:
            mngr:  The uniform manager obtained from
                   :meth:`ShaderProgramManager.batch_update_uniforms`.
            vmin:  Lower normalization bound, or ``None`` to auto-detect.
            vmax:  Upper normalization bound, or ``None`` to auto-detect.
        """
        # Reference — not a copy — to avoid a potentially large array copy
        # on every uniform upload.
        data = self._frame_data

        if vmin is None and vmax is None:
            if data is not None and data.ndim == 3 and data.shape[2] in (3, 4):
                render_vmin, render_vmax = GLfloat(0.0), GLfloat(1.0)
            else:
                render_vmin = GLfloat(
                    float(np.min(data))) if data is not None else GLfloat(0.0)
                render_vmax = GLfloat(
                    float(np.max(data))) if data is not None else GLfloat(1.0)
        else:
            render_vmin = GLfloat(vmin) if vmin is not None else GLfloat(0.0)
            render_vmax = GLfloat(vmax) if vmax is not None else GLfloat(1.0)

        mngr.set_fast(
            location=getattr(mngr.locs, FragmentShaderUniforms.NORM_VMIN),
            value=render_vmin,
            uniform_type=UniformType.FLOAT,
        )
        mngr.set_fast(
            location=getattr(mngr.locs, FragmentShaderUniforms.NORM_VMAX),
            value=render_vmax,
            uniform_type=UniformType.FLOAT,
        )

    def _upload_uniforms(self) -> None:
        """
        Flush all shader uniforms from current settings and view state.

        Uses the program manager's batch context to minimize individual
        ``glUniform*`` call overhead.
        """
        snapshot = self.settings.get_copy()
        img_unit = self._image_texture_unit
        cmap_unit = self._cmap_texture_unit
        u_transform = self._view_manager.get_transform_data()
        u_projection = self._view_manager.get_projection_data()

        with self._program_manager.batch_update_uniforms() as um:
            locs = um.locs

            um.set_fast(
                location=getattr(locs, VertexShaderUniforms.TRANSFORM_MATRIX),
                # ← vertex
                value=u_transform,
                uniform_type=UniformType.MAT4,
            )
            um.set_fast(
                location=getattr(locs, VertexShaderUniforms.PROJECTION_MATRIX),
                # ← vertex
                value=u_projection,
                uniform_type=UniformType.MAT4,
            )
            um.set_fast(
                location=getattr(locs, FragmentShaderUniforms.IMAGE_TEXTURE),
                value=img_unit,
                uniform_type=UniformType.INT,
            )
            um.set_fast(
                location=getattr(locs,
                                 FragmentShaderUniforms.COLORMAP_TEXTURE),
                value=cmap_unit,
                uniform_type=UniformType.INT,
            )
            um.set_fast(
                location=getattr(locs, FragmentShaderUniforms.USE_CMAP),
                value=snapshot.colormap_enabled,
                uniform_type=UniformType.BOOL,
            )

            self._set_norm_uniforms(um, snapshot.norm_vmin, snapshot.norm_vmax)

            um.set_fast(
                location=getattr(locs, FragmentShaderUniforms.BRIGHTNESS),
                value=GLfloat(snapshot.brightness),
                uniform_type=UniformType.FLOAT,
            )
            um.set_fast(
                location=getattr(locs, FragmentShaderUniforms.CONTRAST),
                value=GLfloat(snapshot.contrast),
                uniform_type=UniformType.FLOAT,
            )

            gamma_val = GLfloat(max(snapshot.gamma, 0.001))
            um.set_fast(
                location=getattr(locs, FragmentShaderUniforms.INV_GAMMA),
                value=GLfloat(1.0) / gamma_val,
                uniform_type=UniformType.FLOAT,
            )
            um.set_fast(
                location=getattr(locs, FragmentShaderUniforms.COLOR_BALANCE),
                value=[
                    GLfloat(snapshot.color_balance_r),
                    GLfloat(snapshot.color_balance_g),
                    GLfloat(snapshot.color_balance_b),
                ],
                uniform_type=UniformType.VEC3,
            )
            um.set_fast(
                location=getattr(locs, FragmentShaderUniforms.INVERT),
                value=GLint(snapshot.invert),
                uniform_type=UniformType.BOOL,
            )
            um.set_fast(
                location=getattr(locs, FragmentShaderUniforms.LUT_ENABLED),
                value=GLint(snapshot.lut_enabled),
                uniform_type=UniformType.BOOL,
            )

            if snapshot.lut_enabled:
                um.set_fast(
                    location=getattr(locs, FragmentShaderUniforms.LUT_MIN),
                    value=GLfloat(snapshot.lut_min),
                    uniform_type=UniformType.FLOAT,
                )
                lut_range = GLfloat(
                    max(snapshot.lut_max - snapshot.lut_min, 0.001))
                um.set_fast(
                    location=getattr(locs,
                                     FragmentShaderUniforms.LUT_NORM_FACTOR),
                    value=GLfloat(1.0) / lut_range,
                    uniform_type=UniformType.FLOAT,
                )
                um.set_fast(
                    location=getattr(locs, FragmentShaderUniforms.LUT_TYPE),
                    value=GLint(snapshot.lut_type.value),
                    uniform_type=UniformType.INT,
                )

    def _sync_view_to_settings(self) -> None:
        """Persist the current ViewManager state back into :attr:`settings`."""
        self.settings.update_setting("zoom", self._view_manager.zoom_level)
        self.settings.update_setting("pan_x", self._view_manager.pan_x)
        self.settings.update_setting("pan_y", self._view_manager.pan_y)
        self.settings.update_setting(
            "rotation", np.radians(self._view_manager.rotation)
        )

    def _sync_settings_to_view(self) -> None:
        """Load persisted view state from :attr:`settings` into the ViewManager."""
        self._view_manager.zoom_level = self.settings.zoom
        self._view_manager.pan_x = self.settings.pan_x
        self._view_manager.pan_y = self.settings.pan_y
        self._view_manager.handle_rotation(np.degrees(self.settings.rotation))
        self._view_manager.update_transform()

    @pyqtSlot()
    def _on_settings_changed(self) -> None:
        """React to a settings change: mark uniforms dirty and repaint."""
        self._gl_state.uniforms_dirty = True
        if self._gl_state.initialized:
            self._sync_settings_to_view()
            if self.settings.colormap_enabled:
                self._set_colormap()
            self.update()

    def fit_to_viewport(self) -> None:
        """Scale and center the image to fill the current viewport."""
        self._view_manager.fit_to_viewport()
        self._sync_view_to_settings()
        self.update()

    def reset_view(self) -> None:
        """Reset zoom, pan, and rotation to their default values."""
        self._view_manager.reset_view()
        self._sync_view_to_settings()
        self.update()

    # ------------------------------------------------------------------
    # Public upload API
    # ------------------------------------------------------------------

    @pyqtSlot(np.ndarray, object)
    def present(
            self,
            image: np.ndarray,
            metadata: FrameStats,
            pixel_fmt: PixelFormat,
    ) -> bool:
        """
        Upload and display an image via the standard CPU→PBO→Texture path.

        Args:
            image:      HxW or HxWxC array to display.
            metadata:   Caller-defined frame metadata emitted with
                        ``frame_changed``.
            pixel_fmt:  Pixel format descriptor; falls back to
                        ``settings.format`` when ``None``.

        Returns:
            ``True`` on success, ``False`` when a precondition is unmet or
            an upload error occurs.  In both failure cases the ``gl_error``
            signal is emitted with a human-readable reason.

        Example::

            success = viewer.present(
                image=np.random.rand(480, 640, 3).astype(np.float32),
                metadata=FrameStats(...),
                pixel_fmt=PixelFormat.RGB_FLOAT32
            )
        """
        valid, error_msg = self._can_upload()
        if not valid or not is_standard_image(image):
            logger.warning("Upload validation failed", reason=error_msg)
            return False

        start_time = time.perf_counter_ns() if self._perf_monitor else None

        try:
            h, w = image.shape[:2]
            logger.debug(
                "Starting present upload",
                width=w, height=h, dtype=image.dtype, shape=image.shape,
            )

            gl_format, gl_internal_format, gl_type = get_gl_texture_spec(
                pixel_fmt or self.settings.format, image.dtype.name
            )
            logger.debug(
                "GL formats resolved",
                gl_format=gl_format,
                gl_internal_format=gl_internal_format,
                gl_type=gl_type,
            )

            pbo = self._pbo_upload_mngr.get_next()
            logger.debug("Acquired PBO", pbo_id=pbo.id)

            self._latest_payload = TextureUploadPayload(
                pbo_id=GLBuffer(GLHandle(pbo.id)),
                data=image,
                width=GLsizei(w),
                height=GLsizei(h),
                gl_format=GLenum(gl_format),
                gl_internal_format=GLenum(gl_internal_format),
                gl_type=GLenum(gl_type),
                meta=metadata,
                is_pinned=False,
            )

            if not self._pending_frame_update:
                self._pending_frame_update = True
                self._process_frame(repaint=True)

            if start_time is not None:
                self._update_perf_stats(time.perf_counter_ns() - start_time)

            return True

        except GLUploadError as e:
            msg = str(e)
            logger.error("Upload failed", error=msg)
            self.glError.emit(msg)
            return False
        except GLError as e:
            msg = f"GL error during upload: {e}"
            logger.error(msg)
            self.glError.emit(msg)
            return False
        except Exception as e:
            msg = f"Unexpected upload error: {e}"
            logger.error(msg, exception_type=type(e).__name__)
            self.glError.emit(msg)
            return False

    def request_pinned_buffer(
            self,
            width: int,
            height: int,
            pixel_fmt: PixelFormat,
            dtype: str = "uint8",
    ) -> Tuple[object, np.ndarray]:
        """
        Map and return a pinned PBO buffer for direct zero-copy writing.

        The caller writes image data directly into the returned array, then
        passes the PBO object to :meth:`present_pinned`.

        Args:
            width:     Buffer width in pixels.
            height:    Buffer height in pixels.
            pixel_fmt: Pixel format (channel count and layout).
            dtype:     NumPy dtype string (default ``"uint8"``).

        Returns:
            ``(pbo_object, ndarray)`` — write directly into the array, then
            call :meth:`present_pinned` with the PBO object.
        """
        with gl_context(self, "request_pinned_buffer"):
            fmt = pixel_fmt or self.settings.format
            pbo, arr_shaped = self._pbo_upload_mngr.acquire_next_writeable(
                width=GLsizei(width),
                height=GLsizei(height),
                channels=GLsizei(fmt.channels),
                dtype=np.dtype(dtype),
            )
            return pbo, arr_shaped

    def write_to_pinned_buffer(
            self,
            pbo_array: np.ndarray,
            image: np.ndarray,
            pixel_fmt: PixelFormat,
    ) -> None:
        """
        Copy (or broadcast) image data into a mapped PBO buffer.

        Handles grayscale→RGB broadcasting automatically.

        Args:
            pbo_array:  The mapped buffer array from :meth:`request_pinned_buffer`.
            image:      Source data (HxW or HxWxC).
            pixel_fmt:  Pixel format of the target PBO buffer.

        Raises:
            GLUploadError: If the data cannot be written due to a shape or
                           dtype mismatch.
        """
        try:
            write_pbo_buffer(pbo_array=pbo_array, image=image,
                             pixel_fmt=pixel_fmt)
            self._frame_data = image
        except ValueError as e:
            raise GLUploadError(
                f"Cannot write image into PBO buffer — shape/dtype mismatch: {e}"
            ) from e

    @pyqtSlot(object, object, int, int)
    def present_pinned(
            self,
            pbo_object,
            metadata: FrameStats,
            width: int,
            height: int,
            img_fmt: PixelFormat,
            dtype=np.float32,
    ) -> bool:
        """
        Display a frame from a pre-filled, mapped PBO (zero-copy path).

        The PBO must have been obtained from :meth:`request_pinned_buffer` and
        written via :meth:`write_to_pinned_buffer`.  This method unmaps the PBO
        before handing it to the GPU.

        Pipeline: ``Pre-filled PBO → Texture``

        Args:
            pbo_object: PBO handle returned by :meth:`request_pinned_buffer`.
            metadata:   Frame metadata emitted with ``frame_changed``.
            width:      Frame width in pixels.
            height:     Frame height in pixels.
            img_fmt:    Pixel format of the data in the PBO.
            dtype:      Element type of the data in the PBO.

        Returns:
            ``True`` on success, ``False`` on any failure (``gl_error`` is
            emitted in all failure cases).
        """
        valid, error_msg = self._can_upload()
        if not valid:
            logger.warning("Upload validation failed", reason=error_msg)
            return False

        start_time = time.perf_counter_ns() if self._perf_monitor else None

        try:
            # Unmap the PBO so the GPU can safely DMA from it.
            if pbo_object.is_mapped:
                with gl_context(self, "present_pinned"):
                    pbo_object.unmap()
            else:
                logger.warning(
                    "PBO is already unmapped before present_pinned",
                    pbo=str(pbo_object),
                )

            fmt = img_fmt or self.settings.format
            gl_format, gl_internal_format, gl_type = get_gl_texture_spec(fmt,
                                                                         dtype)

            self._latest_payload = TextureUploadPayload(
                pbo_id=GLBuffer(pbo_object.id),
                width=GLsizei(width),
                height=GLsizei(height),
                gl_format=GLenum(gl_format),
                gl_internal_format=GLenum(gl_internal_format),
                gl_type=GLenum(gl_type),
                meta=metadata,
                is_pinned=True,
                data=None,
            )

            if not self._pending_frame_update:
                self._pending_frame_update = True
                self._process_frame(repaint=True)

            if start_time is not None:
                self._update_perf_stats(time.perf_counter_ns() - start_time)

            return True

        except GLUploadError as e:
            msg = str(e)
            logger.error("present_pinned failed", error=msg)
            self.glError.emit(msg)
            return False
        except GLError as e:
            msg = f"GL error in present_pinned: {e}"
            logger.error(msg)
            self.glError.emit(msg)
            return False
        except Exception as e:
            msg = f"Unexpected error in present_pinned: {e}"
            logger.error(msg, exception_type=type(e).__name__)
            self.glError.emit(msg)
            return False

    def set_range(self, vmin: float, vmax: float) -> None:
        """
        Set the normalization range mapped to the low and high ends of the gradient.

        Delegates to :attr:`settings` so the change propagates through the
        normal settings-changed signal path and marks uniforms dirty automatically.
        """
        self.settings.update_setting("norm_vmin", vmin)
        self.settings.update_setting("norm_vmax", vmax)
        self.update()

    @pyqtSlot()
    def request_capture(self):
        self._pbo_download_bridge.request_capture()

    # ------------------------------------------------------------------
    # Input handling
    # ------------------------------------------------------------------

    def mousePressEvent(self, event) -> None:
        """Begin a pan gesture on left-button press."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = True
            self._last_mouse_pos = event.position()

    def mouseMoveEvent(self, event) -> None:
        """Apply pan delta while left button is held."""
        if not self._is_panning or self._last_mouse_pos is None:
            return
        if not (event.buttons() & Qt.MouseButton.LeftButton):
            self._is_panning = False
            return
        if not self.settings.panning_enabled:
            return

        current_pos = event.position()
        dx = current_pos.x() - self._last_mouse_pos.x()
        # Flip Y: Qt is top-down, OpenGL NDC is bottom-up.
        dy = -(current_pos.y() - self._last_mouse_pos.y())

        self._view_manager.handle_pan(dx, dy)
        self.settings.update_setting("pan_x", self._view_manager.pan_x)
        self.settings.update_setting("pan_y", self._view_manager.pan_y)

        self._last_mouse_pos = current_pos
        self.update()

    def mouseReleaseEvent(self, event) -> None:
        """End a pan gesture on left-button release."""
        if event.button() == Qt.MouseButton.LeftButton:
            self._is_panning = False
            self._last_mouse_pos = None

    def wheelEvent(self, event) -> None:
        """Zoom towards the cursor position on wheel scroll."""
        if not self.settings.zoom_enabled:
            return

        delta = event.angleDelta().y()
        factor = GLfloat(1.1) if delta > 0 else GLfloat(1.0 / 1.1)
        new_zoom = np.clip(
            self._view_manager.zoom_level * factor, GLfloat(0.01),
            GLfloat(100.0)
        )
        actual_factor = new_zoom / self._view_manager.zoom_level

        cursor_pos = event.position()
        self._view_manager.handle_zoom(
            actual_factor,
            cursor_pos.x(),
            self.height() - cursor_pos.y(),  # flip Y
        )
        self.settings.update_setting("zoom",
                                     self._view_manager.zoom_level)
        self.update()

    def keyPressEvent(self, event) -> None:
        """
        Handle keyboard shortcuts.

        ======= =============================
        Key     Action
        ======= =============================
        ``R``   Reset view
        ``F``   Fit image to viewport
        ``←``   Rotate counter-clockwise 5°
        ``→``   Rotate clockwise 5°
        ======= =============================
        """
        key = event.key()
        if key == Qt.Key.Key_R:
            self._view_manager.reset_view()
        elif key == Qt.Key.Key_F:
            self._view_manager.fit_to_viewport()
        elif key == Qt.Key.Key_Left:
            self._view_manager.handle_rotation(self._view_manager.rotation + 5)
        elif key == Qt.Key.Key_Right:
            self._view_manager.handle_rotation(self._view_manager.rotation - 5)
        self.update()

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def get_image_region(
            self, x: int, y: int, w: int, h: int
    ) -> Optional[np.ndarray]:
        """
        Extract a pixel region from the current frame (CPU-side).

        Coordinates are in image space (top-left origin).  Out-of-bounds
        coordinates are clamped to the image extent.

        Args:
            x: Left edge of the region.
            y: Top edge of the region.
            w: Region width in pixels.
            h: Region height in pixels.

        Returns:
            A copy of the requested region as a ndarray, or ``None`` if no
            frame has been loaded or the clamped region is empty.
        """
        if self._frame_data is None:
            return None

        # Flip vertically: OpenGL framebuffer origin is bottom-left.
        img = np.flip(self._frame_data, 0)
        img_h, img_w = img.shape[:2]

        x = max(0, int(x))
        y = max(0, int(y))
        x2 = min(x + w, img_w)
        y2 = min(y + h, img_h)

        if x2 > x and y2 > y:
            return img[y:y2, x:x2].copy()
        return None

    def _update_perf_stats(self, frame_time_ns: float) -> None:
        """
        Feed a frame duration measurement into the performance monitor.

        Args:
            frame_time_ns: Elapsed time in nanoseconds for the upload or
                           render operation being timed.
        """
        dur_ms = frame_time_ns / 1_000_000
        self._current_stats = self._perf_monitor.update(dur_ms)
        logger.debug("GL performance", stats=str(self._current_stats))

    def get_performance_stats(self) -> dict:
        """
        Return the latest rendering performance metrics.

        Returns:
            ``{"fps": float, "avg_ms": float}`` — zeroed when no stats have
            been collected yet.
        """
        if not self._current_stats:
            return {"fps": GLfloat(0.0), "avg_ms": GLfloat(0.0)}
        return {
            "fps": GLfloat(self._current_stats.rate),
            "avg_ms": GLfloat(self._current_stats.avg_processing_ms),
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """
        Release all GPU resources owned by this widget.

        Must be called before widget destruction to avoid GPU memory leaks.
        Safe to call multiple times (idempotent).  Invoked automatically by
        :meth:`__del__` when the Qt binding is still valid.

        Errors during individual deletion steps are caught and logged so that
        a single bad resource does not prevent the remaining ones from being
        freed.
        """
        if not self._gl_state.initialized:
            logger.debug("Cleanup called but GL not initialised — skipping")
            return

        if self._cleaned_up:
            return

        self._cleaned_up = True

        logger.debug("Starting GL resource cleanup")

        try:
            self._frame_data = None

            if self._image_texture_id:
                logger.debug(
                    "Deleting image texture",
                    texture_id=self._image_texture_id,
                )
                GL.glDeleteTextures(
                    GLsizei(1),
                    np.array([self._image_texture_id], dtype=np.uint32)
                )
                self._image_texture_id = None

            if self._cmap_texture_id:
                logger.debug(
                    "Deleting colormap texture",
                    texture_id=self._cmap_texture_id,
                )
                GL.glDeleteTextures(
                    GLsizei(1),
                    np.array([self._cmap_texture_id], dtype=np.uint32)
                )
                self._cmap_texture_id = None

            logger.debug("Cleaning up PBO manager(s)")
            self._pbo_upload_mngr.cleanup()
            self._pbo_download_bridge.cleanup()

            logger.debug("Cleaning up geometry manager")
            self._geo_manager.cleanup()

            self._gl_state.reset()
            logger.debug("GL resource cleanup complete")

        except GLError as e:
            logger.error("GL error during cleanup", error=str(e))
        except Exception as e:
            logger.error(
                "Unexpected error during cleanup",
                error=str(e),
                exception_type=type(e).__name__,
            )

    def __del__(self) -> None:
        """Destructor — runs :meth:`cleanup` if the Qt binding is still live."""
        if has_qt_cpp_binding(self) and self._gl_state.initialized:
            logger.debug("Destructor invoked, running cleanup")
            self.cleanup()

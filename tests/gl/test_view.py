"""
test_view.py
============
Comprehensive pytest test suite for :class:`GLFrameViewer`.

Mocking strategy
----------------
1. **Qt layer** — a minimal ``QOpenGLWidget`` stub is injected into
   ``sys.modules`` before the module under test is imported so that
   ``super().__init__()`` is a no-op and widget methods (``isValid``,
   ``update``, etc.) are replaced with ``MagicMock`` stubs per-test.

2. **GL driver layer** — the ``GL`` object inside ``view`` is replaced
   with a ``MagicMock`` whose GL constant attributes return real enum
   integers so comparison logic inside the widget still works.

3. **Sub-system layer** — ``PBOManager``, ``ShaderProgramManager``,
   ``TextureManager``, ``GeometryManager``, ``ViewManager``, and
   ``ColormapModel`` are each replaced with ``MagicMock(spec=…)`` so
   attribute access on unknown names raises ``AttributeError`` rather than
   silently returning a new mock.

4. **Helper utilities** — ``gl_context``, ``initialize_context``,
   ``validate_shader_paths``, and ``get_gl_texture_spec`` are patched to
   no-ops / fixed return values.

Naming convention
-----------------
The ``mock_subsystems`` fixture stores the ``ViewManager`` mock under the
key ``"view_mgr"`` (not ``"view"``) to avoid shadowing the ``view`` module
imported at the top of this file.
"""

from __future__ import annotations

import types
from contextlib import contextmanager
from dataclasses import replace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PyQt6.QtCore import Qt

# ---------------------------------------------------------------------------
# Inject a fake QOpenGLWidget before the module under test is imported so
# the constructor never touches a display server.
# ---------------------------------------------------------------------------
_FakeQOpenGLWidget = type(
    "QOpenGLWidget",
    (),
    {
        "__init__": lambda self, parent=None: None,
        "update":   lambda self: None,
        "isValid":  lambda self: True,
    },
)

with patch("PyQt6.QtOpenGLWidgets.QOpenGLWidget", _FakeQOpenGLWidget, create=True):
    import sys
    _fake_ogl_mod = types.ModuleType("PyQt6.QtOpenGLWidgets")
    _fake_ogl_mod.QOpenGLWidget = _FakeQOpenGLWidget
    sys.modules["PyQt6.QtOpenGLWidgets"] = _fake_ogl_mod

from cross_platform.qt6_utils.image.gl import view                        # noqa: E402
from cross_platform.qt6_utils.image.gl.view import GLFrameViewer, GLState  # noqa: E402
from cross_platform.qt6_utils.image.gl.error import (                     # noqa: E402
    GLError,
    GLInitializationError,
    GLMemoryError,
    GLTextureError,
    GLUploadError,
)
from cross_platform.qt6_utils.image.gl.texture import (                    # noqa: E402
    FrameStats,
    TextureUploadPayload,
)
from cross_platform.qt6_utils.image.gl.types import (  # noqa: E402
    GLenum,
    GLfloat,
    GLint,
    GLsizei,
    GLTexture,
    GLBuffer, GLHandle, GLuint,
)
from cross_platform.qt6_utils.image.settings.pixels import PixelFormat     # noqa: E402

# ---------------------------------------------------------------------------
# Module path constant — used by every monkeypatch.setattr call so the
# target string never drifts out of sync with the actual module.
# ---------------------------------------------------------------------------
_MOD = "cross_platform.qt6_utils.image.gl.view"


# ===========================================================================
# Helpers
# ===========================================================================

@contextmanager
def _noop_gl_context(*_args, **_kwargs):
    """Drop-in replacement for ``gl_context`` that never touches the driver."""
    yield


def _make_frame_stats(**overrides) -> FrameStats:
    """Return a minimal ``FrameStats`` suitable for use as upload metadata."""
    defaults = dict(
        shape=(64, 64, 3),
        dtype_str="uint8",
        vmin=0.0,
        vmax=255.0,
        mean=127.0,
        std=50.0,
        dmin=0.0,
        dmax=255.0,
        processing_time_ms=1.0,
    )
    return FrameStats(**{**defaults, **overrides})


def _make_payload(
    w: int = 64,
    h: int = 64,
    pinned: bool = False,
    data: np.ndarray | None = None,
    meta: FrameStats | None = None,
) -> TextureUploadPayload:
    """
    Build a minimal :class:`TextureUploadPayload`.

    Accepts keyword overrides for every field so individual tests vary only
    the field they care about without a separate ``dataclasses.replace`` call.
    """
    return TextureUploadPayload(
        pbo_id=GLBuffer(GLHandle(1)),
        data=(
            data
            if data is not None
            else (None if pinned else np.zeros((h, w, 3), dtype=np.uint8))
        ),
        width=GLsizei(w),
        height=GLsizei(h),
        gl_format=GLenum(0x1907),           # GL_RGB
        gl_internal_format=GLenum(0x8051),  # GL_RGB8
        gl_type=GLenum(0x1401),             # GL_UNSIGNED_BYTE
        meta=meta or _make_frame_stats(),
        is_pinned=pinned,
    )


def _rgb_frame(h: int = 64, w: int = 64) -> np.ndarray:
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


# ===========================================================================
# Fixtures
# ===========================================================================

@pytest.fixture()
def mock_gl(monkeypatch):
    """Replace the GL driver object with a ``MagicMock`` with real enum values."""
    gl = MagicMock(name="GL")
    gl.GL_TEXTURE0              = 0x84C0
    gl.GL_TEXTURE_2D            = 0x0DE1
    gl.GL_COLOR_BUFFER_BIT      = 0x4000
    gl.GL_BLEND                 = 0x0BE2
    gl.GL_DEPTH_TEST            = 0x0B71
    gl.GL_CULL_FACE             = 0x0B44
    gl.GL_SRC_ALPHA             = 0x0302
    gl.GL_ONE_MINUS_SRC_ALPHA   = 0x0303
    gl.GL_FRAMEBUFFER_SRGB      = 0x8DB9
    gl.GL_LINEAR                = 0x2601
    gl.GL_NEAREST               = 0x2600
    gl.GL_CLAMP_TO_EDGE         = 0x812F
    gl.GL_UNPACK_ALIGNMENT      = 0x0CF5
    gl.GL_TEXTURE_MIN_FILTER    = 0x2800
    gl.GL_TEXTURE_MAG_FILTER    = 0x2801
    gl.GL_TEXTURE_WRAP_S        = 0x2802
    gl.GL_TEXTURE_WRAP_T        = 0x2803
    gl.GL_TEXTURE_BASE_LEVEL    = 0x813C
    gl.GL_TEXTURE_MAX_LEVEL     = 0x813D
    gl.GL_NO_ERROR              = 0
    gl.glGetError.return_value  = gl.GL_NO_ERROR
    monkeypatch.setattr(f"{_MOD}.GL", gl)
    return gl


@pytest.fixture()
def mock_initialize_context(monkeypatch):
    monkeypatch.setattr(f"{_MOD}.initialize_context", MagicMock())


@pytest.fixture()
def mock_validate_shader_paths(monkeypatch):
    monkeypatch.setattr(f"{_MOD}.validate_shader_paths", MagicMock())


@pytest.fixture()
def mock_get_gl_texture_spec(monkeypatch):
    """Return fixed (GL_RGB, GL_RGB8, GL_UNSIGNED_BYTE) for every call."""
    spec_mock = MagicMock(return_value=(0x1907, 0x8051, 0x1401))
    monkeypatch.setattr(f"{_MOD}.get_gl_texture_spec", spec_mock)
    return spec_mock


@pytest.fixture()
def mock_gl_context(monkeypatch):
    monkeypatch.setattr(f"{_MOD}.gl_context", _noop_gl_context)


@pytest.fixture()
def mock_subsystems(monkeypatch):
    """
    Replace all heavy sub-system managers with spec-constrained mocks.

    The ``ViewManager`` mock is stored under the key ``"view_mgr"`` to avoid
    shadowing the ``view`` module imported at module scope.
    """
    from cross_platform.qt6_utils.image.gl.pbo import PBOManager
    from cross_platform.qt6_utils.image.gl.program import ShaderProgramManager
    from cross_platform.qt6_utils.image.gl.texture import TextureManager
    from cross_platform.qt6_utils.image.gl.quad import GeometryManager
    from cross_platform.qt6_utils.image.gl.viewport import ViewManager
    from cross_platform.qt6_utils.image.model.cmap import ColormapModel

    pbo     = MagicMock(spec=PBOManager)
    prog    = MagicMock(spec=ShaderProgramManager)
    tex     = MagicMock(spec=TextureManager)
    geo     = MagicMock(spec=GeometryManager)
    view_mgr = MagicMock(spec=ViewManager)
    cmap    = MagicMock(spec=ColormapModel)

    # Defaults expected by initializeGL / paintGL.
    prog.is_valid            = True
    prog.handle              = 42
    geo.vao                  = object()     # truthy sentinel
    geo.initialize.return_value = True
    view_mgr.image_w         = 640
    view_mgr.image_h         = 480
    view_mgr.zoom_level      = 1.0
    view_mgr.pan_x           = 0.0
    view_mgr.pan_y           = 0.0
    view_mgr.rotation        = 0.0
    cmap.resolution          = 256

    # Uniform manager — needs context-manager and locs.
    um = MagicMock()
    um.__enter__ = MagicMock(return_value=um)
    um.__exit__  = MagicMock(return_value=False)
    um.locs      = MagicMock()
    prog.batch_update_uniforms.return_value = um
    prog.uniform_manager = MagicMock()

    # ShaderProgramManager context manager.
    prog.__enter__ = MagicMock(return_value=42)
    prog.__exit__  = MagicMock(return_value=False)

    monkeypatch.setattr(f"{_MOD}.PBOManager",           lambda **kw: pbo)
    monkeypatch.setattr(f"{_MOD}.ShaderProgramManager", lambda: prog)
    monkeypatch.setattr(f"{_MOD}.TextureManager",       lambda: tex)
    monkeypatch.setattr(f"{_MOD}.GeometryManager",      lambda: geo)
    monkeypatch.setattr(f"{_MOD}.ViewManager",          lambda: view_mgr)
    monkeypatch.setattr(f"{_MOD}.ColormapModel",        lambda: cmap)

    return {
        "pbo": pbo, "prog": prog, "tex": tex,
        "geo": geo, "view_mgr": view_mgr, "cmap": cmap,
        "um": um,
    }


@pytest.fixture()
def frame_stats() -> FrameStats:
    """Canonical ``FrameStats`` instance shared across tests."""
    return _make_frame_stats()


@pytest.fixture()
def settings():
    """Spec-constrained mock for ``ImageSettings`` with sensible defaults."""
    from cross_platform.qt6_utils.image.settings.image import ImageSettings
    s = MagicMock(spec=ImageSettings)

    # Live attributes read directly by the widget.
    s.format            = PixelFormat.RGB
    s.colormap_enabled  = False
    s.colormap_name     = "viridis"
    s.colormap_reverse  = False
    s.interpolation     = True
    s.norm_vmin         = None
    s.norm_vmax         = None
    s.brightness        = 1.0
    s.contrast          = 1.0
    s.gamma             = 1.0
    s.color_balance_r   = 1.0
    s.color_balance_g   = 1.0
    s.color_balance_b   = 1.0
    s.invert            = False
    s.lut_enabled       = False
    s.lut_min           = 0.0
    s.lut_max           = 1.0
    s.zoom              = 1.0
    s.pan_x             = 0.0
    s.pan_y             = 0.0
    s.rotation          = 0.0
    s.panning_enabled   = True
    s.zoom_enabled      = True

    # Snapshot returned by get_copy() — used by _upload_uniforms.
    snap = MagicMock()
    snap.format            = PixelFormat.RGB
    snap.colormap_enabled  = False
    snap.colormap_name     = "viridis"
    snap.interpolation     = True
    snap.norm_vmin         = None
    snap.norm_vmax         = None
    snap.brightness        = 1.0
    snap.contrast          = 1.0
    snap.gamma             = 1.0
    snap.color_balance_r   = 1.0
    snap.color_balance_g   = 1.0
    snap.color_balance_b   = 1.0
    snap.invert            = False
    snap.lut_enabled       = False
    snap.lut_min           = 0.0
    snap.lut_max           = 1.0
    s.get_copy.return_value = snap

    s.changed         = MagicMock()
    s.changed.connect = MagicMock()
    return s


@pytest.fixture()
def viewer(
    mock_gl,
    mock_initialize_context,
    mock_validate_shader_paths,
    mock_get_gl_texture_spec,
    mock_gl_context,
    mock_subsystems,
    settings,
):
    """
    Fully-constructed, GL-initialised ``GLFrameViewer``.

    The widget is never rendered; Qt's paint loop is never entered.
    ``_image_texture_id`` is pre-set to a non-zero value so ``_can_render()``
    passes in every test that does not explicitly test the guard conditions.
    """
    from cross_platform.qt6_utils.image.gl.pbo import PBOBufferingStrategy

    v = GLFrameViewer.__new__(GLFrameViewer)
    GLFrameViewer.__init__(
        v,
        settings=settings,
        buffer_strategy=PBOBufferingStrategy.DOUBLE,
        monitor_performance=False,
    )

    # Stub Qt widget methods that would normally require a live display.
    v.isValid          = MagicMock(return_value=True)
    v.isVisible        = MagicMock(return_value=True)
    v.context          = MagicMock(
        return_value=MagicMock(isValid=MagicMock(return_value=True))
    )
    v.width            = MagicMock(return_value=640)
    v.height           = MagicMock(return_value=480)
    v.devicePixelRatio = MagicMock(return_value=1.0)
    v.update           = MagicMock()
    v.makeCurrent      = MagicMock()
    v.doneCurrent      = MagicMock()

    mock_subsystems["tex"].create_texture.return_value = 99  # cmap texture id
    v.initializeGL()

    # Provide a valid image texture so _can_render() returns True by default.
    v._image_texture_id = GLTexture(GLHandle(1))

    return v


# ===========================================================================
# GLState
# ===========================================================================

class TestGLState:
    def test_defaults(self):
        state = GLState()
        assert state.texture_bound    is None
        assert state.pbo_bound        is None
        assert state.program_active   is None
        assert state.uniforms_dirty   is True
        assert state.initialized      is False
        assert state.paint_event_count == 0

    def test_reset_clears_every_field(self):
        state = GLState()
        state.texture_bound     = GLTexture(GLHandle(5))
        state.pbo_bound         = GLBuffer(GLHandle(2))
        state.program_active    = GLuint(7)
        state.uniforms_dirty    = False
        state.initialized       = True
        state.paint_event_count = 42

        state.reset()

        assert state.texture_bound    is None
        assert state.pbo_bound        is None
        assert state.program_active   is None
        assert state.uniforms_dirty   is True
        assert state.initialized      is False
        assert state.paint_event_count == 0

    def test_reset_is_idempotent(self):
        state = GLState()
        state.reset()
        state.reset()
        assert not state.initialized


# ===========================================================================
# Construction
# ===========================================================================

class TestConstruction:
    def _build(self, mock_subsystems, settings):
        from cross_platform.qt6_utils.image.gl.pbo import PBOBufferingStrategy
        v = GLFrameViewer.__new__(GLFrameViewer)
        GLFrameViewer.__init__(
            v,
            settings=settings,
            buffer_strategy=PBOBufferingStrategy.DOUBLE,
            monitor_performance=False,
        )
        return v

    def test_not_initialized_before_initializeGL(self, mock_gl, mock_subsystems, settings):
        v = self._build(mock_subsystems, settings)
        assert not v.is_initialized

    def test_no_frame_data_on_creation(self, mock_gl, mock_subsystems, settings):
        v = self._build(mock_subsystems, settings)
        assert v.data  is None
        assert not v.has_data
        assert v.dtype is None

    def test_performance_monitor_created_when_requested(
        self, mock_gl, mock_subsystems, settings
    ):
        from cross_platform.qt6_utils.image.gl.pbo import PBOBufferingStrategy
        v = GLFrameViewer.__new__(GLFrameViewer)
        GLFrameViewer.__init__(
            v, settings=settings,
            buffer_strategy=PBOBufferingStrategy.DOUBLE,
            monitor_performance=True,
        )
        assert v._perf_monitor is not None

    def test_performance_monitor_absent_when_disabled(
        self, mock_gl, mock_subsystems, settings
    ):
        v = self._build(mock_subsystems, settings)
        assert v._perf_monitor is None


# ===========================================================================
# Properties
# ===========================================================================

class TestProperties:
    def test_has_data_false_initially(self, viewer):
        viewer._frame_data = None
        assert not viewer.has_data

    def test_has_data_true_after_frame_assigned(self, viewer):
        viewer._frame_data = np.zeros((10, 10), dtype=np.uint8)
        assert viewer.has_data

    def test_dtype_none_when_no_data(self, viewer):
        viewer._frame_data = None
        assert viewer.dtype is None

    def test_dtype_reflects_array_dtype(self, viewer):
        viewer._frame_data = np.zeros((10, 10), dtype=np.float32)
        assert viewer.dtype == np.float32

    def test_aspect_ratio_correct(self, viewer, mock_subsystems):
        mock_subsystems["view_mgr"].image_w = 640
        mock_subsystems["view_mgr"].image_h = 480
        assert abs(viewer.aspect_ratio - 640 / 480) < 1e-6

    def test_aspect_ratio_defaults_to_one_on_zero_height(self, viewer, mock_subsystems):
        mock_subsystems["view_mgr"].image_h = 0
        assert viewer.aspect_ratio == 1.0

    def test_is_initialized_reflects_gl_state(self, viewer):
        assert viewer.is_initialized
        viewer._gl_state.initialized = False
        assert not viewer.is_initialized

    def test_data_property_returns_same_reference(self, viewer):
        arr = np.zeros((5, 5), dtype=np.uint8)
        viewer._frame_data = arr
        assert viewer.data is arr


# ===========================================================================
# initializeGL
# ===========================================================================

class TestInitializeGL:
    def test_sets_initialized_flag(self, viewer):
        assert viewer._gl_state.initialized is True

    def test_glclearcolor_called_once(self, mock_gl, viewer):
        mock_gl.glClearColor.assert_called_once()

    def test_blend_is_enabled(self, mock_gl, viewer):
        enabled_args = [c.args[0] for c in mock_gl.glEnable.call_args_list]
        assert mock_gl.GL_BLEND in enabled_args

    def test_cmap_texture_id_set(self, viewer):
        assert viewer._cmap_texture_id is not None

    def test_settings_changed_signal_connected(self, viewer, settings):
        settings.changed.connect.assert_called()

    def test_invalid_shader_program_raises_init_error(
        self,
        mock_gl, mock_initialize_context, mock_validate_shader_paths,
        mock_gl_context, mock_subsystems, settings,
    ):
        from cross_platform.qt6_utils.image.gl.pbo import PBOBufferingStrategy
        mock_subsystems["prog"].is_valid = False

        v = GLFrameViewer.__new__(GLFrameViewer)
        GLFrameViewer.__init__(
            v, settings=settings,
            buffer_strategy=PBOBufferingStrategy.DOUBLE,
            monitor_performance=False,
        )
        v.isValid  = MagicMock(return_value=True)
        v.context  = MagicMock(return_value=MagicMock(isValid=MagicMock(return_value=True)))
        v.gl_error = MagicMock(); v.gl_error.emit = MagicMock()

        with pytest.raises(GLInitializationError):
            v.initializeGL()

    def test_geometry_init_failure_raises_init_error(
        self,
        mock_gl, mock_initialize_context, mock_validate_shader_paths,
        mock_gl_context, mock_subsystems, settings,
    ):
        from cross_platform.qt6_utils.image.gl.pbo import PBOBufferingStrategy
        mock_subsystems["geo"].initialize.return_value = False

        v = GLFrameViewer.__new__(GLFrameViewer)
        GLFrameViewer.__init__(
            v, settings=settings,
            buffer_strategy=PBOBufferingStrategy.DOUBLE,
            monitor_performance=False,
        )
        v.isValid  = MagicMock(return_value=True)
        v.context  = MagicMock(return_value=MagicMock(isValid=MagicMock(return_value=True)))
        v.gl_error = MagicMock(); v.gl_error.emit = MagicMock()

        with pytest.raises(GLInitializationError):
            v.initializeGL()

    def test_gl_error_during_init_is_wrapped_as_init_error(
        self,
        mock_gl, mock_initialize_context, mock_validate_shader_paths,
        mock_gl_context, mock_subsystems, settings,
    ):
        from cross_platform.qt6_utils.image.gl.pbo import PBOBufferingStrategy
        mock_gl.glClearColor.side_effect = GLError("bang")

        v = GLFrameViewer.__new__(GLFrameViewer)
        GLFrameViewer.__init__(
            v, settings=settings,
            buffer_strategy=PBOBufferingStrategy.DOUBLE,
            monitor_performance=False,
        )
        v.isValid  = MagicMock(return_value=True)
        v.context  = MagicMock(return_value=MagicMock(isValid=MagicMock(return_value=True)))
        v.gl_error = MagicMock(); v.gl_error.emit = MagicMock()

        with pytest.raises(GLInitializationError):
            v.initializeGL()


# ===========================================================================
# Render / upload guards
# ===========================================================================

class TestRenderUploadGuards:
    # --- _can_render ---

    def test_can_render_all_conditions_met(self, viewer):
        assert viewer._can_render()

    def test_can_render_false_when_not_initialized(self, viewer):
        viewer._gl_state.initialized = False
        assert not viewer._can_render()

    def test_can_render_false_without_image_texture(self, viewer):
        viewer._image_texture_id = None
        assert not viewer._can_render()

    def test_can_render_false_without_valid_program(self, viewer, mock_subsystems):
        mock_subsystems["prog"].is_valid = False
        assert not viewer._can_render()

    def test_can_render_false_without_vao(self, viewer, mock_subsystems):
        mock_subsystems["geo"].vao = None
        assert not viewer._can_render()

    # --- _can_upload ---

    def test_can_upload_all_conditions_met(self, viewer):
        ok, reason = viewer._can_upload()
        assert ok
        assert reason is None

    def test_can_upload_false_when_not_initialized(self, viewer):
        viewer._gl_state.initialized = False
        ok, reason = viewer._can_upload()
        assert not ok
        assert "not initialised" in reason.lower()

    def test_can_upload_false_when_widget_invalid(self, viewer):
        viewer.isValid.return_value = False
        ok, _ = viewer._can_upload()
        assert not ok

    def test_can_upload_false_when_not_visible(self, viewer):
        viewer.isVisible.return_value = False
        ok, _ = viewer._can_upload()
        assert not ok

    def test_can_upload_false_when_context_invalid(self, viewer):
        viewer.context.return_value.isValid.return_value = False
        ok, _ = viewer._can_upload()
        assert not ok


# ===========================================================================
# paintGL
# ===========================================================================

class TestPaintGL:
    def test_skip_when_not_ready(self, viewer, mock_gl):
        viewer._gl_state.initialized = False
        viewer.paintGL()
        mock_gl.glClear.assert_not_called()

    def test_glclear_called_when_ready(self, viewer, mock_gl):
        viewer._gl_state.uniforms_dirty = False
        viewer.paintGL()
        mock_gl.glClear.assert_called_once()

    def test_paint_event_count_incremented(self, viewer):
        viewer._gl_state.uniforms_dirty = False
        initial = viewer._gl_state.paint_event_count
        viewer.paintGL()
        assert viewer._gl_state.paint_event_count == initial + 1

    def test_uniforms_dirty_flag_cleared_after_upload(self, viewer):
        viewer._gl_state.uniforms_dirty = True
        viewer.paintGL()
        assert not viewer._gl_state.uniforms_dirty

    def test_program_active_reset_after_paint(self, viewer):
        viewer._gl_state.uniforms_dirty = False
        viewer.paintGL()
        assert viewer._gl_state.program_active is None

    def test_gl_texture_error_caught_and_emitted(self, viewer, mock_subsystems):
        viewer.gl_error = MagicMock(); viewer.gl_error.emit = MagicMock()
        mock_subsystems["tex"].get_state.side_effect = GLTextureError("tex boom")
        viewer.paintGL()
        viewer.gl_error.emit.assert_called_once()
        assert "Texture error" in viewer.gl_error.emit.call_args[0][0]

    def test_gl_error_caught_and_emitted(self, viewer, mock_subsystems):
        viewer.gl_error = MagicMock(); viewer.gl_error.emit = MagicMock()
        mock_subsystems["geo"].bind.side_effect = GLError("generic boom")
        viewer.paintGL()
        viewer.gl_error.emit.assert_called_once()

    def test_unexpected_exception_caught_and_emitted(self, viewer, mock_subsystems):
        viewer.gl_error = MagicMock(); viewer.gl_error.emit = MagicMock()
        mock_subsystems["geo"].bind.side_effect = RuntimeError("unexpected")
        viewer.paintGL()
        viewer.gl_error.emit.assert_called_once()


# ===========================================================================
# Texture binding helpers
# ===========================================================================

class TestTextureBindingHelpers:
    def test_bind_image_texture_no_op_when_no_state(self, viewer, mock_subsystems):
        mock_subsystems["tex"].get_state.return_value = None
        viewer._bind_image_texture()   # must not raise

    def test_bind_image_texture_raises_when_renderer_id_is_none(
        self, viewer, mock_subsystems
    ):
        state = MagicMock(); state.renderer_id = None
        mock_subsystems["tex"].get_state.return_value = state
        with pytest.raises(GLTextureError, match="no renderer_id"):
            viewer._bind_image_texture()

    def test_bind_image_texture_updates_bound_texture_state(
        self, viewer, mock_subsystems
    ):
        state = MagicMock(); state.renderer_id = 7
        mock_subsystems["tex"].get_state.return_value = state
        viewer._bind_image_texture()
        assert viewer._gl_state.texture_bound == viewer._image_texture_id

    def test_bind_colormap_uses_cmap_texture_when_enabled(
        self, viewer, mock_subsystems
    ):
        viewer.settings.colormap_enabled = True
        viewer._cmap_texture_id = GLTexture(GLHandle(99))
        viewer._bind_colormap_texture()
        mock_subsystems["tex"].bind.assert_called_with(
            "cmap", viewer._cmap_texture_unit
        )

    def test_bind_colormap_falls_back_to_main_when_disabled(
        self, viewer, mock_gl
    ):
        viewer.settings.colormap_enabled = False
        viewer._image_texture_id = GLTexture(GLHandle(1))
        viewer._bind_colormap_texture()
        mock_gl.glBindTexture.assert_called()

    def test_bind_colormap_raises_when_both_unavailable(self, viewer):
        viewer.settings.colormap_enabled = False
        viewer._image_texture_id = None
        with pytest.raises(GLTextureError, match="Colormap fallback failed"):
            viewer._bind_colormap_texture()

    def test_cleanup_texture_bindings_resets_bound_state(
        self, viewer, mock_subsystems
    ):
        viewer._gl_state.texture_bound = GLTexture(GLHandle(5))
        viewer._cleanup_texture_bindings()
        assert viewer._gl_state.texture_bound is None

    def test_cleanup_texture_bindings_swallows_exceptions(
        self, viewer, mock_subsystems
    ):
        mock_subsystems["tex"].unbind.side_effect = RuntimeError("unbind fail")
        viewer._cleanup_texture_bindings()   # must not propagate


# ===========================================================================
# Texture allocation
# ===========================================================================

class TestTextureAllocation:
    def test_realloc_returns_true_when_no_existing_state(
        self, viewer, mock_subsystems
    ):
        mock_subsystems["tex"].get_state.return_value = None
        assert viewer._realloc_image_texture(_make_payload(w=320, h=240)) is True

    def test_realloc_returns_false_when_dimensions_unchanged(
        self, viewer, mock_subsystems
    ):
        existing = MagicMock()
        existing.renderer_id = 1
        existing.width  = GLsizei(320)
        existing.height = GLsizei(240)
        mock_subsystems["tex"].get_state.return_value = existing
        viewer._gl_state.current_texture_state = existing

        assert viewer._realloc_image_texture(_make_payload(w=320, h=240)) is False

    def test_realloc_returns_true_when_dimensions_change(
        self, viewer, mock_subsystems
    ):
        existing = MagicMock()
        existing.renderer_id = 1
        existing.width  = GLsizei(100)
        existing.height = GLsizei(100)
        mock_subsystems["tex"].get_state.return_value = existing

        assert viewer._realloc_image_texture(_make_payload(w=320, h=240)) is True

    def test_alloc_raises_gl_memory_error_on_oom(
        self, viewer, mock_subsystems
    ):
        mock_subsystems["tex"].get_state.return_value = None
        mock_subsystems["tex"].create_texture.return_value = 5
        mock_subsystems["tex"].allocate.side_effect = MemoryError("OOM")

        with pytest.raises(GLMemoryError, match="out of memory"):
            viewer._alloc_image_texture(
                GLsizei(64), GLsizei(64),
                GLenum(0x1907), GLenum(0x8051), GLenum(0x1401),
            )


# ===========================================================================
# Upload pipeline
# ===========================================================================

class TestUploadPipeline:
    def test_process_frame_is_noop_with_no_payload(self, viewer):
        viewer._latest_payload = None
        viewer._process_frame()   # must not raise

    def test_process_frame_emits_frame_changed_on_success(self, viewer, frame_stats):
        viewer.frame_changed = MagicMock(); viewer.frame_changed.emit = MagicMock()
        payload = _make_payload(meta=frame_stats)
        viewer._latest_payload = payload

        with patch.object(viewer, "_upload_frame"):
            viewer._process_frame(repaint=False)

        viewer.frame_changed.emit.assert_called_once_with(frame_stats)

    def test_process_frame_emits_gl_error_on_upload_failure(self, viewer):
        viewer.gl_error = MagicMock(); viewer.gl_error.emit = MagicMock()
        viewer._latest_payload = _make_payload()

        with patch.object(viewer, "_upload_frame", side_effect=GLUploadError("fail")):
            viewer._process_frame(repaint=False)

        viewer.gl_error.emit.assert_called_once()

    def test_process_frame_always_clears_payload_and_flag(self, viewer):
        viewer.gl_error = MagicMock(); viewer.gl_error.emit = MagicMock()
        viewer._latest_payload = _make_payload()
        viewer._pending_frame_update = True

        with patch.object(viewer, "_upload_frame", side_effect=GLUploadError("fail")):
            viewer._process_frame(repaint=False)

        assert viewer._latest_payload       is None
        assert viewer._pending_frame_update is False

    def test_upload_cmap_texture_propagates_gl_texture_error(
        self, viewer, mock_gl
    ):
        lut = np.zeros((256, 3), dtype=np.uint8)
        with patch(
            f"{_MOD}.alloc_texture_storage",
            side_effect=GLTextureError("alloc fail"),
        ), pytest.raises(GLTextureError):
            viewer._upload_cmap_texture(lut)

    def test_upload_frame_stores_frame_data_on_success(self, viewer):
        img     = np.zeros((64, 64, 3), dtype=np.uint8)
        payload = _make_payload(data=img)

        with patch.object(viewer, "_realloc_image_texture", return_value=False), \
             patch(f"{_MOD}.memmove_pbo"), \
             patch.object(viewer, "_upload_image_texture"):
            viewer._upload_frame(payload, repaint=False)

        assert viewer._frame_data is img


# ===========================================================================
# Colormap
# ===========================================================================

class TestColormap:
    def test_set_colormap_skips_when_cmap_texture_is_none(self, viewer, mock_gl):
        viewer._cmap_texture_id = None
        viewer._set_colormap()
        mock_gl.glTexImage2D.assert_not_called()

    def test_set_colormap_fetches_lut_and_calls_upload(
        self, viewer, mock_subsystems
    ):
        lut = np.zeros((256, 3), dtype=np.uint8)
        mock_subsystems["cmap"].get_lut.return_value = lut

        with patch.object(viewer, "_upload_cmap_texture") as mock_upload:
            viewer._set_colormap()

        mock_upload.assert_called_once_with(lut)

    def test_set_colormap_marks_uniforms_dirty(self, viewer, mock_subsystems):
        mock_subsystems["cmap"].get_lut.return_value = np.zeros((256, 3), dtype=np.uint8)
        viewer._gl_state.uniforms_dirty = False

        with patch.object(viewer, "_upload_cmap_texture"):
            viewer._set_colormap()

        assert viewer._gl_state.uniforms_dirty

    def test_set_colormap_emits_gl_error_on_failure(self, viewer, mock_subsystems):
        viewer.gl_error = MagicMock(); viewer.gl_error.emit = MagicMock()
        mock_subsystems["cmap"].get_lut.side_effect = GLTextureError("lut fail")

        viewer._set_colormap()

        viewer.gl_error.emit.assert_called_once()


# ===========================================================================
# Uniforms
# ===========================================================================

class TestUniforms:
    def test_norm_uniforms_use_zero_one_range_for_rgb(self, viewer, mock_subsystems):
        viewer._frame_data = np.zeros((10, 10, 3), dtype=np.uint8)
        um = mock_subsystems["um"]

        viewer._set_norm_uniforms(um)

        values = [c.kwargs["value"] for c in um.set_fast.call_args_list]
        assert GLfloat(0.0) in values
        assert GLfloat(1.0) in values

    def test_norm_uniforms_use_data_range_for_grayscale(self, viewer, mock_subsystems):
        viewer._frame_data = np.array([[10, 200]], dtype=np.uint8)
        um = mock_subsystems["um"]

        viewer._set_norm_uniforms(um)

        values = [c.kwargs["value"] for c in um.set_fast.call_args_list]
        assert GLfloat(10.0)  in values
        assert GLfloat(200.0) in values

    def test_norm_uniforms_honour_explicit_range(self, viewer, mock_subsystems):
        um = mock_subsystems["um"]
        viewer._set_norm_uniforms(um, vmin=0.2, vmax=0.9)
        values = [c.kwargs["value"] for c in um.set_fast.call_args_list]
        assert GLfloat(0.2) in values
        assert GLfloat(0.9) in values

    def test_norm_uniforms_safe_when_no_frame_data(self, viewer, mock_subsystems):
        viewer._frame_data = None
        viewer._set_norm_uniforms(mock_subsystems["um"])   # must not raise

    def test_upload_uniforms_enters_batch_context(self, viewer, mock_subsystems):
        viewer._upload_uniforms()
        mock_subsystems["prog"].batch_update_uniforms.assert_called_once()

    def test_gamma_clamped_to_prevent_zero_division(self, viewer, settings):
        snap = settings.get_copy.return_value
        snap.gamma       = 0.0
        snap.lut_enabled = False
        viewer._upload_uniforms()   # must not raise ZeroDivisionError


# ===========================================================================
# View sync
# ===========================================================================

class TestViewSync:
    def test_sync_view_to_settings_writes_correct_keys(
        self, viewer, mock_subsystems
    ):
        mock_subsystems["view_mgr"].zoom_level = 2.5
        mock_subsystems["view_mgr"].pan_x      = 10.0
        mock_subsystems["view_mgr"].pan_y      = -5.0
        mock_subsystems["view_mgr"].rotation   = 0.0

        viewer._sync_view_to_settings()

        viewer.settings.update_setting.assert_any_call("zoom",  2.5)
        viewer.settings.update_setting.assert_any_call("pan_x", 10.0)

    def test_on_settings_changed_marks_uniforms_dirty(self, viewer):
        viewer._gl_state.uniforms_dirty = False
        with patch.object(viewer, "_sync_settings_to_view"):
            viewer._on_settings_changed()
        assert viewer._gl_state.uniforms_dirty

    def test_on_settings_changed_triggers_repaint(self, viewer):
        with patch.object(viewer, "_sync_settings_to_view"):
            viewer._on_settings_changed()
        viewer.update.assert_called()


# ===========================================================================
# Public API — colormap & range
# ===========================================================================

class TestPublicColormapRangeAPI:
    def test_set_colormap_fetches_lut_uploads_and_updates(
        self, viewer, mock_subsystems
    ):
        lut = np.zeros((256, 3), dtype=np.uint8)
        mock_subsystems["cmap"].get_lut.return_value = lut
        viewer.settings.colormap_enabled = True

        with patch.object(viewer, "_upload_cmap_texture") as mock_upload:
            viewer._set_colormap()

        mock_upload.assert_called_once_with(lut)
        # _set_colormap marks uniforms dirty; _on_settings_changed triggers update.
        assert viewer._gl_state.uniforms_dirty

    def test_set_colormap_data_delegates_and_updates(self, viewer, mock_subsystems):
        arr = np.zeros((256, 3), dtype=np.uint8)
        with patch.object(
            viewer._gradient_renderer if hasattr(viewer, "_gradient_renderer") else viewer,
            "set_colormap_data",
            create=True,
        ):
            viewer.set_colormap_data(arr)
        viewer.update.assert_called()

    def test_set_range_calls_update_setting_for_both_bounds(self, viewer):
        viewer.set_range(-1.0, 2.0)
        viewer.settings.update_setting.assert_any_call("norm_vmin", -1.0)
        viewer.settings.update_setting.assert_any_call("norm_vmax",  2.0)

    def test_set_range_triggers_repaint(self, viewer):
        viewer.update.reset_mock()
        viewer.set_range(0.0, 1.0)
        viewer.update.assert_called_once()


# ===========================================================================
# Public upload API — present (standard path)
# ===========================================================================

class TestPresent:
    def test_returns_false_when_not_initialized(self, viewer, frame_stats):
        viewer._gl_state.initialized = False
        assert not viewer.present(_rgb_frame(), frame_stats, PixelFormat.RGB)

    def test_returns_false_for_invalid_image(self, viewer, frame_stats):
        assert not viewer.present(np.array([]), frame_stats, PixelFormat.RGB)

    def test_returns_true_on_success(
        self, viewer, mock_subsystems, mock_get_gl_texture_spec, frame_stats
    ):
        mock_subsystems["pbo"].get_next.return_value = MagicMock(id=1)

        with patch.object(viewer, "_process_frame"):
            ok = viewer.present(_rgb_frame(), frame_stats, PixelFormat.RGB)

        assert ok

    def test_sets_pending_update_flag_before_process_frame(
        self, viewer, mock_subsystems, mock_get_gl_texture_spec, frame_stats
    ):
        mock_subsystems["pbo"].get_next.return_value = MagicMock(id=1)

        with patch.object(viewer, "_process_frame"):
            viewer.present(_rgb_frame(), frame_stats, PixelFormat.RGB)

        # Flag is set to True before _process_frame is called; because
        # _process_frame is patched (no-op) it is never cleared back to False.
        assert viewer._pending_frame_update

    def test_emits_gl_error_and_returns_false_on_upload_error(
        self, viewer, mock_subsystems, mock_get_gl_texture_spec, frame_stats
    ):
        viewer.gl_error = MagicMock(); viewer.gl_error.emit = MagicMock()
        mock_subsystems["pbo"].get_next.return_value = MagicMock(id=1)

        with patch.object(viewer, "_process_frame", side_effect=GLUploadError("fail")):
            ok = viewer.present(_rgb_frame(), frame_stats, PixelFormat.RGB)

        assert not ok
        viewer.gl_error.emit.assert_called_once()

    def test_emits_gl_error_and_returns_false_on_unexpected_exception(
        self, viewer, mock_subsystems, mock_get_gl_texture_spec, frame_stats
    ):
        viewer.gl_error = MagicMock(); viewer.gl_error.emit = MagicMock()
        mock_subsystems["pbo"].get_next.side_effect = RuntimeError("kaboom")

        ok = viewer.present(_rgb_frame(), frame_stats, PixelFormat.RGB)

        assert not ok
        viewer.gl_error.emit.assert_called_once()


# ===========================================================================
# Public upload API — pinned (zero-copy) path
# ===========================================================================

class TestPresentPinned:
    def _pbo(self, mapped: bool = True) -> MagicMock:
        pbo = MagicMock(); pbo.id = 2; pbo.is_mapped = mapped
        return pbo

    def test_returns_false_when_not_initialized(self, viewer, frame_stats):
        viewer._gl_state.initialized = False
        assert not viewer.present_pinned(
            self._pbo(), frame_stats, 64, 64, PixelFormat.RGB
        )

    def test_returns_true_on_success(
        self, viewer, mock_get_gl_texture_spec, frame_stats
    ):
        with patch.object(viewer, "_process_frame"):
            ok = viewer.present_pinned(
                self._pbo(), frame_stats, 64, 64, PixelFormat.RGB, dtype=np.uint8
            )
        assert ok

    def test_unmaps_pbo_before_gpu_transfer(
        self, viewer, mock_get_gl_texture_spec, frame_stats
    ):
        pbo = self._pbo(mapped=True)
        with patch.object(viewer, "_process_frame"):
            viewer.present_pinned(pbo, frame_stats, 64, 64, PixelFormat.RGB, dtype=np.uint8)
        pbo.unmap.assert_called_once()

    def test_safe_when_pbo_already_unmapped(
        self, viewer, mock_get_gl_texture_spec, frame_stats
    ):
        with patch.object(viewer, "_process_frame"):
            ok = viewer.present_pinned(
                self._pbo(mapped=False), frame_stats, 64, 64, PixelFormat.RGB, dtype=np.uint8
            )
        assert ok

    def test_write_to_pinned_buffer_raises_gl_upload_error_on_mismatch(self, viewer):
        with patch(
            f"{_MOD}.write_pbo_buffer",
            side_effect=ValueError("shape mismatch"),
        ), pytest.raises(GLUploadError, match="shape/dtype mismatch"):
            viewer.write_to_pinned_buffer(
                np.zeros((64, 64, 3), dtype=np.uint8),
                np.zeros((32,), dtype=np.float32),
                PixelFormat.RGB,
            )


# ===========================================================================
# Input handling
# ===========================================================================

class TestInputHandling:
    def _event(
        self,
        button: Qt.MouseButton = Qt.MouseButton.LeftButton,
        pos: tuple[float, float] = (100.0, 200.0),
    ) -> MagicMock:
        ev = MagicMock()
        ev.button.return_value   = button
        ev.buttons.return_value  = button
        ev.position.return_value = MagicMock(
            x=MagicMock(return_value=pos[0]),
            y=MagicMock(return_value=pos[1]),
        )
        return ev

    def test_left_press_starts_pan(self, viewer):
        viewer.mousePressEvent(self._event())
        assert viewer._is_panning
        assert viewer._last_mouse_pos is not None

    def test_left_release_ends_pan(self, viewer):
        viewer.mousePressEvent(self._event())
        viewer.mouseReleaseEvent(self._event())
        assert not viewer._is_panning
        assert viewer._last_mouse_pos is None

    def test_mouse_move_ignored_when_not_panning(self, viewer, mock_subsystems):
        viewer._is_panning = False
        viewer.mouseMoveEvent(self._event())
        mock_subsystems["view_mgr"].handle_pan.assert_not_called()

    def test_mouse_move_calls_handle_pan(self, viewer, mock_subsystems):
        viewer.settings.panning_enabled = True
        viewer._is_panning    = True
        viewer._last_mouse_pos = MagicMock(
            x=MagicMock(return_value=50.0),
            y=MagicMock(return_value=50.0),
        )
        viewer.mouseMoveEvent(self._event(pos=(60.0, 40.0)))
        mock_subsystems["view_mgr"].handle_pan.assert_called_once_with(10.0, 10.0)

    def test_mouse_move_skipped_when_panning_disabled(self, viewer, mock_subsystems):
        viewer.settings.panning_enabled = False
        viewer._is_panning    = True
        viewer._last_mouse_pos = MagicMock(
            x=MagicMock(return_value=0.0),
            y=MagicMock(return_value=0.0),
        )
        viewer.mouseMoveEvent(self._event(pos=(10.0, 10.0)))
        mock_subsystems["view_mgr"].handle_pan.assert_not_called()

    def test_scroll_up_zooms_in(self, viewer, mock_subsystems):
        viewer.settings.zoom_enabled = True
        mock_subsystems["view_mgr"].zoom_level = 1.0
        ev = MagicMock()
        ev.angleDelta.return_value.y.return_value = 120
        ev.position.return_value.x.return_value   = 320.0
        ev.position.return_value.y.return_value   = 240.0
        viewer.wheelEvent(ev)
        mock_subsystems["view_mgr"].handle_zoom.assert_called_once()

    def test_scroll_no_op_when_zoom_disabled(self, viewer, mock_subsystems):
        viewer.settings.zoom_enabled = False
        viewer.wheelEvent(MagicMock())
        mock_subsystems["view_mgr"].handle_zoom.assert_not_called()

    def test_key_r_resets_view(self, viewer, mock_subsystems):
        ev = MagicMock(); ev.key.return_value = Qt.Key.Key_R
        viewer.keyPressEvent(ev)
        mock_subsystems["view_mgr"].reset_view.assert_called_once()

    def test_key_f_fits_to_viewport(self, viewer, mock_subsystems):
        ev = MagicMock(); ev.key.return_value = Qt.Key.Key_F
        viewer.keyPressEvent(ev)
        mock_subsystems["view_mgr"].fit_to_viewport.assert_called_once()

    def test_key_left_rotates_counter_clockwise(self, viewer, mock_subsystems):
        mock_subsystems["view_mgr"].rotation = 10
        ev = MagicMock(); ev.key.return_value = Qt.Key.Key_Left
        viewer.keyPressEvent(ev)
        mock_subsystems["view_mgr"].handle_rotation.assert_called_with(15)

    def test_key_right_rotates_clockwise(self, viewer, mock_subsystems):
        mock_subsystems["view_mgr"].rotation = 10
        ev = MagicMock(); ev.key.return_value = Qt.Key.Key_Right
        viewer.keyPressEvent(ev)
        mock_subsystems["view_mgr"].handle_rotation.assert_called_with(5)


# ===========================================================================
# Utilities
# ===========================================================================

class TestUtilities:
    def test_get_image_region_none_when_no_frame_data(self, viewer):
        viewer._frame_data = None
        assert viewer.get_image_region(0, 0, 10, 10) is None

    def test_get_image_region_correct_shape(self, viewer):
        viewer._frame_data = np.zeros((100, 200, 3), dtype=np.uint8)
        region = viewer.get_image_region(10, 10, 20, 30)
        assert region is not None
        assert region.shape == (30, 20, 3)

    def test_get_image_region_clamps_to_image_bounds(self, viewer):
        viewer._frame_data = np.zeros((50, 50, 3), dtype=np.uint8)
        region = viewer.get_image_region(40, 40, 1000, 1000)
        assert region is not None
        assert region.shape[0] <= 50
        assert region.shape[1] <= 50

    def test_get_image_region_none_for_degenerate_rect(self, viewer):
        viewer._frame_data = np.zeros((50, 50, 3), dtype=np.uint8)
        assert viewer.get_image_region(60, 60, 0, 0) is None

    def test_performance_stats_zeroed_with_no_monitor_data(self, viewer):
        viewer._current_stats = None
        stats = viewer.get_performance_stats()
        assert stats["fps"]    == GLfloat(0.0)
        assert stats["avg_ms"] == GLfloat(0.0)

    def test_performance_stats_reflect_monitor(self, viewer):
        viewer._current_stats = MagicMock(fps=60.0, avg_processing_ms=16.7)
        stats = viewer.get_performance_stats()
        assert stats["fps"]    == GLfloat(60.0)
        assert stats["avg_ms"] == GLfloat(16.7)

    def test_fit_to_viewport_delegates_and_repaints(self, viewer, mock_subsystems):
        viewer.fit_to_viewport()
        mock_subsystems["view_mgr"].fit_to_viewport.assert_called_once()
        viewer.update.assert_called()

    def test_reset_view_delegates_and_repaints(self, viewer, mock_subsystems):
        viewer.reset_view()
        mock_subsystems["view_mgr"].reset_view.assert_called_once()
        viewer.update.assert_called()


# ===========================================================================
# Cleanup / lifecycle
# ===========================================================================

class TestCleanup:
    def test_noop_when_not_initialized(self, viewer, mock_gl):
        viewer._gl_state.initialized = False
        viewer.cleanup()
        mock_gl.glDeleteTextures.assert_not_called()

    def test_deletes_image_texture(self, viewer, mock_gl):
        viewer._image_texture_id = GLTexture(1)
        viewer.cleanup()
        mock_gl.glDeleteTextures.assert_any_call(
            GLsizei(1), np.array([1], dtype=np.uint32)
        )

    def test_deletes_cmap_texture(self, viewer, mock_gl):
        viewer._cmap_texture_id = GLTexture(99)
        viewer.cleanup()
        mock_gl.glDeleteTextures.assert_any_call(
            GLsizei(1), np.array([99], dtype=np.uint32)
        )

    def test_resets_gl_state(self, viewer):
        viewer.cleanup()
        assert not viewer._gl_state.initialized

    def test_is_idempotent(self, viewer, mock_gl):
        viewer.cleanup()
        mock_gl.glDeleteTextures.reset_mock()
        viewer.cleanup()
        mock_gl.glDeleteTextures.assert_not_called()

    def test_clears_frame_data(self, viewer):
        viewer._frame_data = np.zeros((10, 10), dtype=np.uint8)
        viewer.cleanup()
        assert viewer._frame_data is None

    def test_calls_pbo_cleanup(self, viewer, mock_subsystems):
        viewer.cleanup()
        mock_subsystems["pbo"].cleanup.assert_called_once()

    def test_calls_geo_cleanup(self, viewer, mock_subsystems):
        viewer.cleanup()
        mock_subsystems["geo"].cleanup.assert_called_once()

    def test_swallows_gl_error_during_deletion(self, viewer, mock_gl):
        mock_gl.glDeleteTextures.side_effect = GLError("del fail")
        viewer.cleanup()   # must not propagate

    def test_del_never_raises(self, viewer):
        viewer.__del__()   # must not raise regardless of state


# ===========================================================================
# resizeGL
# ===========================================================================

class TestResizeGL:
    def test_delegates_to_view_manager(self, viewer, mock_subsystems):
        viewer.resizeGL(800, 600)
        mock_subsystems["view_mgr"].handle_resize.assert_called_with(800, 600)

    def test_marks_uniforms_dirty(self, viewer):
        viewer._gl_state.uniforms_dirty = False
        viewer.resizeGL(800, 600)
        assert viewer._gl_state.uniforms_dirty

    def test_triggers_repaint_when_renderable(self, viewer):
        viewer.update.reset_mock()
        viewer.resizeGL(800, 600)
        viewer.update.assert_called()

    def test_no_repaint_when_not_renderable(self, viewer):
        viewer._image_texture_id = None   # _can_render() → False
        viewer.update.reset_mock()
        viewer.resizeGL(800, 600)
        viewer.update.assert_not_called()
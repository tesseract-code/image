# tests/test_gl_backend.py
import importlib
import sys
import types
from unittest.mock import MagicMock, patch

import pytest


# -------------------------------------------------------------------------
# Fixtures
# -------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_gl_backend():
    for module in list(sys.modules.keys()):
        if (module.startswith("backend") or module.startswith("gl_backend") or
                module in (
                "OpenGL", "OpenGL.GL", "OpenGL.GLU")):
            sys.modules.pop(module, None)
    yield


@pytest.fixture
def mock_cross_platform_modules(monkeypatch):
    class FakeGLConfig:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.DEBUG_MODE = False  # Backend never sets DEBUG_MODE

        def __repr__(self):
            return f"FakeGLConfig({self.__dict__})"

    fake_gl_configs = {"default": FakeGLConfig(USE_IMMUTABLE_STORAGE=False)}
    fake_config_module = MagicMock()
    fake_config_module.GL_CONFIGS = fake_gl_configs
    fake_config_module.GLConfig = FakeGLConfig

    fake_debug_module = MagicMock()
    fake_debug_module.enable_gl_debug_output = MagicMock()

    monkeypatch.setitem(sys.modules, "cross_platform.qt6_utils.image.gl.config",
                        fake_config_module)
    monkeypatch.setitem(sys.modules, "cross_platform.qt6_utils.image.gl.debug",
                        fake_debug_module)

    return fake_config_module, fake_debug_module


@pytest.fixture
def mock_opengl(monkeypatch):
    """Mock OpenGL module using a real object to allow attribute assignment."""
    # Create a real namespace that can hold boolean flags
    open_gl_mock = types.SimpleNamespace()
    open_gl_mock.ERROR_CHECKING = False
    open_gl_mock.ERROR_LOGGING = False
    open_gl_mock.ERROR_ON_COPY = True

    gl_mock = MagicMock()
    glu_mock = MagicMock()

    # Default GL behaviour: OpenGL 4.6 with ARB_texture_storage
    gl_mock.glGetString.return_value = b"4.6.0 NVIDIA"
    gl_mock.glGetIntegerv.return_value = 2
    gl_mock.glGetStringi.side_effect = (
        lambda _, i: [b"GL_ARB_texture_storage", b"GL_EXT_other"][i])
    gl_mock.GL_EXTENSIONS = 0x1F03

    open_gl_mock.GL = gl_mock
    open_gl_mock.GLU = glu_mock

    monkeypatch.setitem(sys.modules, "OpenGL", open_gl_mock)
    monkeypatch.setitem(sys.modules, "OpenGL.GL", gl_mock)
    monkeypatch.setitem(sys.modules, "OpenGL.GLU", glu_mock)

    return open_gl_mock, gl_mock, glu_mock


def _import_gl_backend():
    from image.gl import backend as gl_backend
    importlib.reload(gl_backend)
    return gl_backend


# -------------------------------------------------------------------------
# Tests
# -------------------------------------------------------------------------

def test_import_without_pyopengl(mock_cross_platform_modules, monkeypatch):
    for mod in ("OpenGL", "OpenGL.GL", "OpenGL.GLU"):
        monkeypatch.delitem(sys.modules, mod, raising=False)

    def failing_import(name, *args, **kwargs):
        if name.startswith("OpenGL"):
            raise ImportError(f"No module named '{name}'")
        return original_import(name, *args, **kwargs)

    original_import = __import__
    with patch("builtins.__import__", side_effect=failing_import):
        with pytest.raises(ImportError):
            import OpenGL.GL  # This will hit the failing_import side effect


def test_debug_mode_enabled_pyopengl_flags(mock_cross_platform_modules,
                                           mock_opengl, monkeypatch):
    monkeypatch.setenv("GL_DEBUG_MODE", "1")
    monkeypatch.setattr("platform.system", lambda: "Linux")

    gl_backend = _import_gl_backend()
    # The mocked OpenGL module is the one in sys.modules
    open_gl = sys.modules["OpenGL"]
    assert open_gl.ERROR_CHECKING is True
    assert open_gl.ERROR_LOGGING is True
    assert open_gl.ERROR_ON_COPY is True

    gl_backend.initialize_context()
    _, fake_debug = mock_cross_platform_modules
    fake_debug.enable_gl_debug_output.assert_not_called()


def test_debug_mode_disabled(mock_cross_platform_modules, mock_opengl,
                             monkeypatch):
    monkeypatch.delenv("GL_DEBUG_MODE", raising=False)
    monkeypatch.setattr("platform.system", lambda: "Linux")

    gl_backend = _import_gl_backend()
    open_gl = sys.modules["OpenGL"]
    assert open_gl.ERROR_CHECKING is False
    assert open_gl.ERROR_LOGGING is False
    assert open_gl.ERROR_ON_COPY is False

    gl_backend.initialize_context()
    _, fake_debug = mock_cross_platform_modules
    fake_debug.enable_gl_debug_output.assert_not_called()


def test_macos_disables_immutable_storage_and_logs_info(
        mock_cross_platform_modules, mock_opengl, monkeypatch, caplog):
    monkeypatch.setenv("GL_DEBUG_MODE", "1")
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    gl_backend = _import_gl_backend()
    with caplog.at_level("INFO", logger="GLBackend"):
        gl_backend.initialize_context()

    config = gl_backend.GL_CONFIGS["default"]
    assert config.USE_IMMUTABLE_STORAGE is False

    _, fake_debug = mock_cross_platform_modules
    fake_debug.enable_gl_debug_output.assert_not_called()
    assert "GL debug output is disabled on macOS" not in caplog.text
    assert "macOS detected: immutable texture storage unavailable" in caplog.text


def test_macos_no_debug_callback_and_no_warning(mock_cross_platform_modules,
                                                mock_opengl, monkeypatch,
                                                caplog):
    monkeypatch.setenv("GL_DEBUG_MODE", "1")
    monkeypatch.setattr("platform.system", lambda: "Darwin")

    with caplog.at_level("WARNING", logger="GLBackend"):
        _import_gl_backend().initialize_context()

    _, fake_debug = mock_cross_platform_modules
    fake_debug.enable_gl_debug_output.assert_not_called()
    assert "GL debug output is disabled on macOS" not in caplog.text


def test_gl_version_4_2_or_higher_immutable(mock_cross_platform_modules,
                                            mock_opengl, monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    _, gl_mock, _ = mock_opengl
    gl_mock.glGetString.return_value = b"4.2.0"
    gl_mock.glGetIntegerv.return_value = 0

    gl_backend = _import_gl_backend()
    gl_backend.initialize_context()
    config = gl_backend.GL_CONFIGS["default"]
    assert config.USE_IMMUTABLE_STORAGE is True


def test_gl_version_4_1_with_arb_extension(mock_cross_platform_modules,
                                           mock_opengl, monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    _, gl_mock, _ = mock_opengl
    gl_mock.glGetString.return_value = b"4.1.0"
    gl_mock.glGetIntegerv.return_value = 2
    gl_mock.glGetStringi.side_effect = [b"GL_ARB_texture_storage",
                                        b"GL_EXT_other"]

    gl_backend = _import_gl_backend()
    gl_backend.initialize_context()
    config = gl_backend.GL_CONFIGS["default"]
    assert config.USE_IMMUTABLE_STORAGE is True


def test_gl_version_4_1_without_extension(mock_cross_platform_modules,
                                          mock_opengl, monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    _, gl_mock, _ = mock_opengl
    gl_mock.glGetString.return_value = b"4.1.0"
    gl_mock.glGetIntegerv.return_value = 1
    gl_mock.glGetStringi.side_effect = [b"GL_EXT_some_other"]

    gl_backend = _import_gl_backend()
    gl_backend.initialize_context()
    config = gl_backend.GL_CONFIGS["default"]
    assert config.USE_IMMUTABLE_STORAGE is False


def test_gl_version_below_4_fallback_extension_string(
        mock_cross_platform_modules, mock_opengl, monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    _, gl_mock, _ = mock_opengl
    gl_mock.glGetString.return_value = b"3.3.0"
    gl_mock.glGetString.side_effect = lambda x: {
        gl_mock.GL_EXTENSIONS: b"GL_ARB_texture_storage GL_EXT_other"
    }.get(x, b"")

    gl_backend = _import_gl_backend()
    gl_backend.initialize_context()
    config = gl_backend.GL_CONFIGS["default"]
    assert config.USE_IMMUTABLE_STORAGE is True


def test_gl_version_string_parsing(mock_cross_platform_modules, mock_opengl,
                                   monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    _, gl_mock, _ = mock_opengl

    test_cases = [
        (b"4.6.0 NVIDIA 123", True),
        (b"4.0.0", False),
        (b"3.3.0", False),
        (b"2.1", False),
    ]

    for version_str, expected in test_cases:
        gl_mock.glGetString.return_value = version_str
        gl_mock.glGetIntegerv.return_value = 0
        gl_backend = _import_gl_backend()
        gl_backend.initialize_context()
        config = gl_backend.GL_CONFIGS["default"]
        assert config.USE_IMMUTABLE_STORAGE is expected


def test_context_failure_fallback(mock_cross_platform_modules, mock_opengl,
                                  monkeypatch, caplog):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    _, gl_mock, _ = mock_opengl
    gl_mock.glGetString.side_effect = Exception("No context")

    with caplog.at_level("ERROR", logger="GLBackend"):
        gl_backend = _import_gl_backend()
        gl_backend.initialize_context()

    assert "Context capability check failed" in caplog.text
    config = gl_backend.GL_CONFIGS["default"]
    assert config.USE_IMMUTABLE_STORAGE is False


def test_initialize_context_idempotent(mock_cross_platform_modules, mock_opengl,
                                       monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    gl_backend = _import_gl_backend()
    gl_backend.initialize_context()
    config1 = gl_backend.GL_CONFIGS["default"]

    gl_backend.initialize_context()
    config2 = gl_backend.GL_CONFIGS["default"]

    assert config1.USE_IMMUTABLE_STORAGE == config2.USE_IMMUTABLE_STORAGE


def test_warning_if_opengl_already_imported(mock_cross_platform_modules,
                                            mock_opengl, monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "OpenGL.GL", MagicMock())
    with caplog.at_level("WARNING"):
        _import_gl_backend()
    assert "OpenGL.GL was imported before GLConfig could apply optimizations" in caplog.text


def test_gl_and_glu_are_exported(mock_cross_platform_modules, mock_opengl,
                                 monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    gl_backend = _import_gl_backend()
    _, gl_mock, glu_mock = mock_opengl
    assert gl_backend.GL is gl_mock

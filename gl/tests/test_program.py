"""
tests/test_gl_program.py
========================
Unit tests for cross_platform.qt6_utils.image.gl.program.

All OpenGL driver calls are intercepted by patching the ``GL`` object in
the program module's namespace.  No real OpenGL context is required.

Coverage
--------
compile_shader       — success, warning log, create failure, compile failure,
                       shader deletion on failure, path in error message.
link_program         — success, warning log, create failure, link failure,
                       detach sequencing, program deletion on failure.
create_program       — file-not-found, read error, compile failure leak guard,
                       link failure leak guard, shader deletion on success.
validate_program     — None/0 guard, success, failure with and without log.
delete_program       — valid handle, None no-op, zero no-op.
set_uniform          — location=-1 guard, scalar/vector/matrix type inference,
                       explicit type override, program bind/unbind sequencing,
                       unbind on exception, unknown type log.
ShaderProgramManager — pre-init state, initialize, idempotent init, error
                       propagation, cleanup, context manager bind/unbind,
                       unbind on block exception, enter guard, batch uniform
                       context, batch guard.
"""

from __future__ import annotations

import logging
import unittest
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from cross_platform.qt6_utils.image.gl.error import GLShaderError
from cross_platform.qt6_utils.image.gl.program import (
    ShaderProgramManager,
    compile_shader,
    create_program,
    delete_program,
    link_program,
    set_uniform,
    validate_program,
)
from cross_platform.qt6_utils.image.gl.types import GLenum, GLint, GLuint

# Patch target — the GL name as it exists in the program module's namespace.
_MOD = "cross_platform.qt6_utils.image.gl.program"

# Stable fake handles used across tests.
_SHADER_ID = 42
_PROGRAM_ID = 99


# =============================================================================
# Shared fixtures
# =============================================================================

@pytest.fixture()
def gl():
    """
    Fully configured mock GL object injected into the program module.

    Default return values reflect the happy path (everything succeeds, no
    driver logs).  Individual tests override specific attributes as needed.
    """
    with patch(f"{_MOD}.GL") as mock_gl:
        # Integer constants the module compares against at runtime.
        mock_gl.GL_VERTEX_SHADER = 35633
        mock_gl.GL_FRAGMENT_SHADER = 35632
        mock_gl.GL_COMPILE_STATUS = 35713
        mock_gl.GL_LINK_STATUS = 35714
        mock_gl.GL_VALIDATE_STATUS = 35715
        mock_gl.GL_INFO_LOG_LENGTH = 35716
        mock_gl.GL_TRUE = 1
        mock_gl.GL_FALSE = 0

        # Default: object creation succeeds.
        mock_gl.glCreateShader.return_value = _SHADER_ID
        mock_gl.glCreateProgram.return_value = _PROGRAM_ID

        # Default: compilation and linking succeed with empty logs.
        mock_gl.glGetShaderiv.return_value = 1       # GL_TRUE
        mock_gl.glGetShaderInfoLog.return_value = b""
        mock_gl.glGetProgramiv.return_value = 1      # GL_TRUE
        mock_gl.glGetProgramInfoLog.return_value = b""

        yield mock_gl


@pytest.fixture()
def shader_files(tmp_path: Path):
    """
    Write minimal GLSL stubs to a temporary directory.

    Returns ``(vertex_path, fragment_path)`` as resolved ``Path`` objects.
    The source is syntactically valid GLSL 4.10 so the content can be read
    by ``create_program`` without triggering an I/O error.
    """
    vert = tmp_path / "image.vert"
    frag = tmp_path / "image.frag"
    vert.write_text(
        "#version 410 core\nvoid main() { gl_Position = vec4(0.0); }\n",
        encoding="utf-8",
    )
    frag.write_text(
        "#version 410 core\nout vec4 color;\nvoid main() { color = vec4(1.0); }\n",
        encoding="utf-8",
    )
    return vert, frag


# Helper: configure glGetShaderiv to return (compile_ok, log_len) sequentially.
def _shader_iv(compile_ok: int, log_len: int = 0):
    return [compile_ok, log_len]


# Helper: configure glGetProgramiv to return (link_ok, log_len) sequentially.
def _program_iv(link_ok: int, log_len: int = 0):
    return [link_ok, log_len]


# =============================================================================
# compile_shader
# =============================================================================

class TestCompileShader:

    def test_vertex_success_returns_gluint(self, gl):
        gl.glGetShaderiv.side_effect = _shader_iv(1)
        result = compile_shader("void main(){}", GLenum(gl.GL_VERTEX_SHADER))
        assert result == GLuint(_SHADER_ID)

    def test_fragment_success_returns_gluint(self, gl):
        gl.glGetShaderiv.side_effect = _shader_iv(1)
        result = compile_shader("void main(){}", GLenum(gl.GL_FRAGMENT_SHADER))
        assert result == GLuint(_SHADER_ID)

    def test_source_and_compile_called_in_order(self, gl):
        gl.glGetShaderiv.side_effect = _shader_iv(1)
        compile_shader("src", GLenum(gl.GL_VERTEX_SHADER))

        gl.glShaderSource.assert_called_once_with(_SHADER_ID, "src")
        gl.glCompileShader.assert_called_once_with(_SHADER_ID)
        # glShaderSource must precede glCompileShader
        source_idx = gl.mock_calls.index(call.glShaderSource(_SHADER_ID, "src"))
        compile_idx = gl.mock_calls.index(call.glCompileShader(_SHADER_ID))
        assert source_idx < compile_idx

    def test_raises_when_create_shader_returns_zero(self, gl):
        gl.glCreateShader.return_value = 0
        with pytest.raises(GLShaderError, match="glCreateShader returned 0"):
            compile_shader("void main(){}", GLenum(gl.GL_VERTEX_SHADER))

    def test_raises_on_compile_failure_with_log(self, gl):
        gl.glGetShaderiv.side_effect = _shader_iv(0, log_len=60)
        gl.glGetShaderInfoLog.return_value = b"ERROR: 0:1: undeclared identifier 'x'"

        with pytest.raises(GLShaderError, match="undeclared identifier"):
            compile_shader("bad src", GLenum(gl.GL_VERTEX_SHADER), path="image.vert")

    def test_raises_on_compile_failure_no_log(self, gl):
        gl.glGetShaderiv.side_effect = _shader_iv(0, log_len=0)

        with pytest.raises(GLShaderError, match="no compiler log"):
            compile_shader("bad src", GLenum(gl.GL_VERTEX_SHADER))

    def test_shader_deleted_on_compile_failure(self, gl):
        gl.glGetShaderiv.side_effect = _shader_iv(0, log_len=0)

        with pytest.raises(GLShaderError):
            compile_shader("bad src", GLenum(gl.GL_VERTEX_SHADER))

        gl.glDeleteShader.assert_called_once_with(_SHADER_ID)

    def test_shader_not_deleted_on_success(self, gl):
        gl.glGetShaderiv.side_effect = _shader_iv(1)
        compile_shader("void main(){}", GLenum(gl.GL_VERTEX_SHADER))
        gl.glDeleteShader.assert_not_called()

    def test_path_included_in_error_message(self, gl):
        gl.glGetShaderiv.side_effect = _shader_iv(0, log_len=0)
        with pytest.raises(GLShaderError, match="shaders/image.vert"):
            compile_shader("bad", GLenum(gl.GL_VERTEX_SHADER), path="shaders/image.vert")

    def test_warning_logged_for_nonempty_success_log(self, gl, caplog):
        gl.glGetShaderiv.side_effect = _shader_iv(1, log_len=30)
        gl.glGetShaderInfoLog.return_value = b"warning: unused variable 'uv'"

        with caplog.at_level(logging.WARNING, logger=_MOD):
            compile_shader("void main(){}", GLenum(gl.GL_VERTEX_SHADER))

        assert caplog.records, "Expected at least one WARNING log record"
        assert "unused variable" in caplog.text

    def test_no_warning_for_null_terminator_only_log(self, gl, caplog):
        # log_length == 1 means only the null byte — must not produce a warning.
        gl.glGetShaderiv.side_effect = _shader_iv(1, log_len=1)
        gl.glGetShaderInfoLog.return_value = b"\x00"

        with caplog.at_level(logging.WARNING, logger=_MOD):
            compile_shader("void main(){}", GLenum(gl.GL_VERTEX_SHADER))

        assert not caplog.records


# =============================================================================
# link_program
# =============================================================================

class TestLinkProgram:

    def test_success_returns_gluint(self, gl):
        gl.glGetProgramiv.side_effect = _program_iv(1)
        result = link_program(GLuint(10), GLuint(20))
        assert result == GLuint(_PROGRAM_ID)

    def test_both_shaders_attached(self, gl):
        gl.glGetProgramiv.side_effect = _program_iv(1)
        link_program(GLuint(10), GLuint(20))
        gl.glAttachShader.assert_any_call(_PROGRAM_ID, 10)
        gl.glAttachShader.assert_any_call(_PROGRAM_ID, 20)

    def test_both_shaders_detached_after_success(self, gl):
        gl.glGetProgramiv.side_effect = _program_iv(1)
        link_program(GLuint(10), GLuint(20))
        gl.glDetachShader.assert_any_call(_PROGRAM_ID, 10)
        gl.glDetachShader.assert_any_call(_PROGRAM_ID, 20)

    def test_shaders_detached_before_status_check(self, gl):
        """Detach must occur regardless of link outcome."""
        gl.glGetProgramiv.side_effect = _program_iv(0, log_len=10)
        gl.glGetProgramInfoLog.return_value = b"link error"

        with pytest.raises(GLShaderError):
            link_program(GLuint(10), GLuint(20))

        gl.glDetachShader.assert_any_call(_PROGRAM_ID, 10)
        gl.glDetachShader.assert_any_call(_PROGRAM_ID, 20)

    def test_raises_when_create_program_returns_zero(self, gl):
        gl.glCreateProgram.return_value = 0
        with pytest.raises(GLShaderError, match="glCreateProgram returned 0"):
            link_program(GLuint(10), GLuint(20))

    def test_raises_on_link_failure_with_log(self, gl):
        gl.glGetProgramiv.side_effect = _program_iv(0, log_len=40)
        gl.glGetProgramInfoLog.return_value = b"undefined symbol: main"

        with pytest.raises(GLShaderError, match="undefined symbol"):
            link_program(GLuint(10), GLuint(20))

    def test_raises_on_link_failure_no_log(self, gl):
        gl.glGetProgramiv.side_effect = _program_iv(0, log_len=0)
        with pytest.raises(GLShaderError, match="no linker log"):
            link_program(GLuint(10), GLuint(20))

    def test_program_deleted_on_link_failure(self, gl):
        gl.glGetProgramiv.side_effect = _program_iv(0, log_len=0)

        with pytest.raises(GLShaderError):
            link_program(GLuint(10), GLuint(20))

        gl.glDeleteProgram.assert_called_once_with(_PROGRAM_ID)

    def test_warning_logged_for_nonempty_link_log(self, gl, caplog):
        gl.glGetProgramiv.side_effect = _program_iv(1, log_len=25)
        gl.glGetProgramInfoLog.return_value = b"performance hint: ..."

        with caplog.at_level(logging.WARNING, logger=_MOD):
            link_program(GLuint(10), GLuint(20))

        assert caplog.records
        assert "warnings" in caplog.text.lower()


# =============================================================================
# create_program
# =============================================================================

class TestCreateProgram:

    def _success_iv(self, gl):
        # Both shaders compile (2 glGetShaderiv calls each) then link succeeds.
        gl.glGetShaderiv.side_effect = [1, 0, 1, 0]
        gl.glGetProgramiv.side_effect = _program_iv(1)

    def test_success_returns_gluint(self, gl, shader_files):
        self._success_iv(gl)
        vert, frag = shader_files
        result = create_program(vert, frag)
        assert result == GLuint(_PROGRAM_ID)

    def test_both_shaders_deleted_on_success(self, gl, shader_files):
        self._success_iv(gl)
        vert, frag = shader_files
        create_program(vert, frag)
        assert gl.glDeleteShader.call_count == 2

    def test_raises_when_vertex_file_missing(self, gl, tmp_path, shader_files):
        _, frag = shader_files
        with pytest.raises(GLShaderError, match="not found"):
            create_program(tmp_path / "nonexistent.vert", frag)

    def test_raises_when_fragment_file_missing(self, gl, tmp_path, shader_files):
        vert, _ = shader_files
        with pytest.raises(GLShaderError, match="not found"):
            create_program(vert, tmp_path / "nonexistent.frag")

    def test_raises_on_read_error(self, gl, shader_files):
        vert, frag = shader_files
        with patch.object(Path, "read_text", side_effect=OSError("permission denied")):
            with pytest.raises(GLShaderError, match="Failed to read"):
                create_program(vert, frag)

    import pytest
    from unittest.mock import Mock

    def test_vertex_shader_deleted_when_fragment_compile_fails(self, gl,
                                                               shader_files):
        # Make glCreateShader return distinct handles
        shader_handles = [100, 200]  # vertex=100, fragment=200
        gl.glCreateShader.side_effect = shader_handles

        # Compile status: vertex ok, fragment fail
        gl.glGetShaderiv.side_effect = [
            1, 0,  # vertex: GL_TRUE, no log
            0, 0,  # fragment: GL_FALSE, no log
        ]

        vert, frag = shader_files
        with pytest.raises(GLShaderError):
            create_program(vert, frag)

        # Both shaders should be deleted exactly once each
        vertex_handle, fragment_handle = shader_handles
        expected_calls = [
            unittest.mock.call(vertex_handle),
            unittest.mock.call(fragment_handle),
        ]
        gl.glDeleteShader.assert_has_calls(expected_calls, any_order=True)
        assert gl.glDeleteShader.call_count == 2

    def test_both_shaders_deleted_when_link_fails(self, gl, shader_files):
        gl.glGetShaderiv.side_effect = [1, 0, 1, 0]
        gl.glGetProgramiv.side_effect = _program_iv(0, log_len=0)

        vert, frag = shader_files
        with pytest.raises(GLShaderError):
            create_program(vert, frag)

        assert gl.glDeleteShader.call_count == 2


# =============================================================================
# validate_program
# =============================================================================

class TestValidateProgram:

    def test_returns_false_tuple_for_none(self, gl):
        ok, msg = validate_program(None)
        assert ok is False
        assert msg is not None
        assert "None or 0" in msg
        gl.glValidateProgram.assert_not_called()

    def test_returns_false_tuple_for_zero(self, gl):
        ok, msg = validate_program(GLuint(0))
        assert ok is False
        gl.glValidateProgram.assert_not_called()

    def test_returns_true_none_on_success(self, gl):
        # glGetProgramiv must return GL_TRUE (1) for equality check.
        gl.glGetProgramiv.return_value = 1
        ok, msg = validate_program(GLuint(_PROGRAM_ID))
        assert ok is True
        assert msg is None

    def test_returns_false_with_log_on_failure(self, gl):
        gl.glGetProgramiv.return_value = 0
        gl.glGetProgramInfoLog.return_value = b"draw framebuffer invalid"
        ok, msg = validate_program(GLuint(_PROGRAM_ID))
        assert ok is False
        assert "framebuffer" in msg

    def test_returns_false_with_fallback_message_when_log_empty(self, gl):
        gl.glGetProgramiv.return_value = 0
        gl.glGetProgramInfoLog.return_value = b""
        ok, msg = validate_program(GLuint(_PROGRAM_ID))
        assert ok is False
        assert msg  # Must be a non-empty string even with no driver log.


# =============================================================================
# delete_program
# =============================================================================

class TestDeleteProgram:

    def test_deletes_valid_handle(self, gl):
        delete_program(GLuint(_PROGRAM_ID))
        gl.glDeleteProgram.assert_called_once_with(_PROGRAM_ID)

    def test_noop_for_none(self, gl):
        delete_program(None)
        gl.glDeleteProgram.assert_not_called()

    def test_noop_for_zero(self, gl):
        delete_program(GLuint(0))
        gl.glDeleteProgram.assert_not_called()


# =============================================================================
# set_uniform
# =============================================================================

class TestSetUniform:

    def test_skips_entirely_when_location_is_minus_one(self, gl):
        set_uniform(GLuint(1), "inactive", 42, GLint(-1))
        gl.glUseProgram.assert_not_called()

    @pytest.mark.parametrize("value, gl_call", [
        (7,           "glUniform1i"),
        (True,        "glUniform1i"),
        (np.int32(3), "glUniform1i"),
        (1.5,         "glUniform1f"),
        (np.float32(0.5), "glUniform1f"),
    ])
    def test_auto_detects_scalar_type(self, gl, value, gl_call):
        set_uniform(GLuint(1), "u_val", value, GLint(0))
        getattr(gl, gl_call).assert_called_once()

    @pytest.mark.parametrize("value, gl_call", [
        ([0.1, 0.2],           "glUniform2fv"),
        ([0.1, 0.2, 0.3],      "glUniform3fv"),
        ([0.1, 0.2, 0.3, 0.4], "glUniform4fv"),
    ])
    def test_auto_detects_vector_type_from_list(self, gl, value, gl_call):
        set_uniform(GLuint(1), "u_vec", value, GLint(0))
        getattr(gl, gl_call).assert_called_once()

    def test_auto_detects_mat4_from_ndarray(self, gl):
        mat = np.eye(4, dtype=np.float32)
        set_uniform(GLuint(1), "u_mvp", mat, GLint(0))
        gl.glUniformMatrix4fv.assert_called_once()

    def test_auto_detects_mat3_from_ndarray(self, gl):
        mat = np.eye(3, dtype=np.float32)
        set_uniform(GLuint(1), "u_normal", mat, GLint(0))
        gl.glUniformMatrix3fv.assert_called_once()

    def test_explicit_type_overrides_auto_inference(self, gl):
        # Pass an int but force float upload.
        set_uniform(GLuint(1), "u_val", 1, GLint(0), uniform_type="float")
        gl.glUniform1f.assert_called_once_with(0, 1.0)
        gl.glUniform1i.assert_not_called()

    def test_program_bound_before_uniform_call(self, gl):
        set_uniform(GLuint(1), "u_int", 5, GLint(0))
        # glUseProgram(program) must be called before any glUniform*.
        use_idx   = gl.mock_calls.index(call.glUseProgram(1))
        uniform_idx = gl.mock_calls.index(call.glUniform1i(0, 5))
        assert use_idx < uniform_idx

    def test_program_unbound_after_call(self, gl):
        set_uniform(GLuint(1), "u_int", 5, GLint(0))
        last_use = gl.glUseProgram.call_args_list[-1]
        assert last_use == call(0)

    def test_program_unbound_even_when_uniform_raises(self, gl):
        gl.glUniform1i.side_effect = RuntimeError("driver fault")
        with pytest.raises(RuntimeError):
            set_uniform(GLuint(1), "u_int", 5, GLint(0))
        last_use = gl.glUseProgram.call_args_list[-1]
        assert last_use == call(0)

    def test_logs_error_for_unresolvable_type(self, gl, caplog):
        with caplog.at_level(logging.ERROR, logger=_MOD):
            # An object() cannot be inferred and has no matching explicit tag.
            set_uniform(GLuint(1), "u_wat", object(), GLint(0), uniform_type="tensor")
        assert caplog.records, "Expected an ERROR log for unknown uniform type"


# =============================================================================
# ShaderProgramManager
# =============================================================================

@pytest.fixture()
def patched_create():
    """Patch create_program to return a stable fake handle without touching files."""
    with patch(f"{_MOD}.create_program", return_value=GLuint(_PROGRAM_ID)) as m:
        yield m


@pytest.fixture()
def patched_uniform_manager():
    """Patch UniformManager so its constructor never touches a real GL context."""
    with patch(f"{_MOD}.UniformManager") as m:
        m.return_value = MagicMock(name="UniformManagerInstance")
        yield m


class TestShaderProgramManager:

    # --- Pre-initialisation state -----------------------------------------

    def test_not_valid_before_initialize(self):
        assert not ShaderProgramManager().is_valid

    def test_handle_is_zero_before_initialize(self):
        assert ShaderProgramManager().handle == 0

    def test_uniform_manager_is_none_before_initialize(self):
        assert ShaderProgramManager().uniform_manager is None

    # --- initialize() -------------------------------------------------------

    def test_initialize_sets_valid_state(self, gl, patched_create, patched_uniform_manager):
        mgr = ShaderProgramManager()
        mgr.initialize("v.glsl", "f.glsl")
        assert mgr.is_valid
        assert mgr.handle == GLuint(_PROGRAM_ID)

    def test_initialize_calls_create_program_with_paths(self, gl, patched_create, patched_uniform_manager):
        mgr = ShaderProgramManager()
        mgr.initialize("v.glsl", "f.glsl")
        patched_create.assert_called_once_with(vertex_path="v.glsl", fragment_path="f.glsl")

    def test_initialize_creates_uniform_manager_with_program_id(self, gl, patched_create, patched_uniform_manager):
        mgr = ShaderProgramManager()
        mgr.initialize("v.glsl", "f.glsl")
        patched_uniform_manager.assert_called_once_with(GLuint(_PROGRAM_ID))
        assert mgr.uniform_manager is patched_uniform_manager.return_value

    def test_initialize_is_idempotent(self, gl, patched_create, patched_uniform_manager):
        mgr = ShaderProgramManager()
        mgr.initialize("v.glsl", "f.glsl")
        mgr.initialize("v.glsl", "f.glsl")
        patched_create.assert_called_once()   # second call must be a no-op

    def test_initialize_propagates_shader_error(self, gl):
        with patch(f"{_MOD}.create_program", side_effect=GLShaderError("bad source")):
            mgr = ShaderProgramManager()
            with pytest.raises(GLShaderError, match="bad source"):
                mgr.initialize("bad.vert", "bad.frag")
        assert not mgr.is_valid

    # --- cleanup() ----------------------------------------------------------

    def test_cleanup_deletes_program(self, gl, patched_create, patched_uniform_manager):
        mgr = ShaderProgramManager()
        mgr.initialize("v.glsl", "f.glsl")
        mgr.cleanup()
        gl.glDeleteProgram.assert_called_once_with(_PROGRAM_ID)

    def test_cleanup_resets_state(self, gl, patched_create, patched_uniform_manager):
        mgr = ShaderProgramManager()
        mgr.initialize("v.glsl", "f.glsl")
        mgr.cleanup()
        assert not mgr.is_valid
        assert mgr.uniform_manager is None

    def test_cleanup_is_idempotent(self, gl, patched_create, patched_uniform_manager):
        mgr = ShaderProgramManager()
        mgr.initialize("v.glsl", "f.glsl")
        mgr.cleanup()
        mgr.cleanup()
        gl.glDeleteProgram.assert_called_once()  # must not double-delete

    # --- Context manager (__enter__ / __exit__) ----------------------------

    def test_context_manager_binds_program(self, gl, patched_create, patched_uniform_manager):
        mgr = ShaderProgramManager()
        mgr.initialize("v.glsl", "f.glsl")
        with mgr as prog_id:
            gl.glUseProgram.assert_called_with(_PROGRAM_ID)
            assert prog_id == GLuint(_PROGRAM_ID)

    def test_context_manager_unbinds_on_clean_exit(self, gl, patched_create, patched_uniform_manager):
        mgr = ShaderProgramManager()
        mgr.initialize("v.glsl", "f.glsl")
        with mgr:
            pass
        gl.glUseProgram.assert_called_with(0)

    def test_context_manager_unbinds_on_exception(self, gl, patched_create, patched_uniform_manager):
        mgr = ShaderProgramManager()
        mgr.initialize("v.glsl", "f.glsl")
        with pytest.raises(RuntimeError):
            with mgr:
                raise RuntimeError("render error")
        # Final glUseProgram call must still be the unbind.
        assert gl.glUseProgram.call_args_list[-1] == call(0)

    def test_enter_raises_when_not_initialized(self, gl):
        mgr = ShaderProgramManager()
        with pytest.raises(GLShaderError, match="not initialised"):
            with mgr:
                pass  # pragma: no cover

    # --- batch_update_uniforms() -------------------------------------------

    def test_batch_update_yields_uniform_manager(self, gl, patched_create, patched_uniform_manager):
        mgr = ShaderProgramManager()
        mgr.initialize("v.glsl", "f.glsl")
        with mgr.batch_update_uniforms() as um:
            assert um is mgr.uniform_manager

    def test_batch_update_keeps_program_bound_during_block(self, gl, patched_create, patched_uniform_manager):
        mgr = ShaderProgramManager()
        mgr.initialize("v.glsl", "f.glsl")
        with mgr.batch_update_uniforms():
            gl.glUseProgram.assert_called_with(_PROGRAM_ID)

    def test_batch_update_unbinds_after_block(self, gl, patched_create, patched_uniform_manager):
        mgr = ShaderProgramManager()
        mgr.initialize("v.glsl", "f.glsl")
        with mgr.batch_update_uniforms():
            pass
        assert gl.glUseProgram.call_args_list[-1] == call(0)

    def test_batch_update_raises_when_not_initialized(self, gl):
        mgr = ShaderProgramManager()
        with pytest.raises(GLShaderError, match="not available"):
            with mgr.batch_update_uniforms():
                pass  # pragma: no cover
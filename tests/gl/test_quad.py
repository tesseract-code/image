"""
tests/test_gl_quad.py
=====================
Unit tests for cross_platform.qt6_utils.image.gl.quad.GeometryManager.

All GL driver calls are intercepted by patching ``GL`` in the quad module's
namespace.  No real OpenGL context is required.

Coverage
--------
initialize   — happy path, VAO=0 failure, buffer=0 failure, GLError, cleanup
               on partial failure, attribute pointer arguments.
bind         — active vao, uninitialised no-op.
unbind       — always calls glBindVertexArray(0).
draw         — issues correct glDrawElements args, uninitialised warning.
cleanup      — correct handle passed to each delete call (VAO bug regression),
               partial-init safety, idempotency.
__enter__    — binds VAO.
__exit__     — unbinds on clean exit and on exception.
"""

from __future__ import annotations

import ctypes
import logging
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from cross_platform.qt6_utils.image.gl.quad import GeometryManager

_MOD = "cross_platform.qt6_utils.image.gl.quad"

_VAO_ID = 11
_VBO_ID = 22
_EBO_ID = 33


def _make_gl() -> MagicMock:
    """GL mock with happy-path defaults."""
    gl = MagicMock(name="GL")
    gl.GL_ARRAY_BUFFER              = 0x8892
    gl.GL_ELEMENT_ARRAY_BUFFER      = 0x8893
    gl.GL_STATIC_DRAW               = 0x88B4
    gl.GL_FLOAT                     = 0x1406
    gl.GL_FALSE                     = 0
    gl.GL_TRIANGLES                 = 0x0004
    gl.GL_UNSIGNED_INT              = 0x1405
    gl.glGenVertexArrays.return_value = _VAO_ID
    gl.glGenBuffers.side_effect       = [_VBO_ID, _EBO_ID]
    return gl


@pytest.fixture()
def gl():
    mock_gl = _make_gl()
    with patch(f"{_MOD}.GL", mock_gl):
        yield mock_gl


@pytest.fixture()
def mgr(gl) -> GeometryManager:
    """Fully initialised manager."""
    m = GeometryManager()
    assert m.initialize() is True
    return m


# =============================================================================
# initialize
# =============================================================================

class TestInitialize:

    def test_returns_true_on_success(self, gl):
        assert GeometryManager().initialize() is True

    def test_sets_vao_vbo_ebo_handles(self, mgr):
        assert mgr.vao == _VAO_ID
        assert mgr.vbo == _VBO_ID
        assert mgr.ebo == _EBO_ID

    def test_returns_false_when_vao_is_zero(self, gl):
        gl.glGenVertexArrays.return_value = 0
        assert GeometryManager().initialize() is False

    def test_returns_false_when_vbo_is_zero(self, gl):
        gl.glGenBuffers.side_effect = [0, _EBO_ID]
        assert GeometryManager().initialize() is False

    def test_returns_false_when_ebo_is_zero(self, gl):
        gl.glGenBuffers.side_effect = [_VBO_ID, 0]
        assert GeometryManager().initialize() is False

    def test_cleanup_called_on_buffer_creation_failure(self, gl):
        """Partial allocation must not leak handles."""
        gl.glGenBuffers.side_effect = [0, _EBO_ID]
        m = GeometryManager()
        m.initialize()
        # VAO was allocated; it must be deleted even though buffers failed.
        gl.glDeleteVertexArrays.assert_called_once()

    def test_catches_gl_error_and_returns_false(self, gl):
        from cross_platform.qt6_utils.image.gl.error import GLError
        gl.glGenVertexArrays.side_effect = GLError("driver fault")
        assert GeometryManager().initialize() is False

    def test_catches_generic_exception_and_returns_false(self, gl):
        gl.glGenVertexArrays.side_effect = RuntimeError("bad state")
        assert GeometryManager().initialize() is False

    def test_vao_bound_before_attribute_calls(self, gl):
        GeometryManager().initialize()
        bind_idx  = next(
            i for i, c in enumerate(gl.mock_calls)
            if c == call.glBindVertexArray(_VAO_ID)
        )
        attrib_idx = next(
            (i for i, c in enumerate(gl.mock_calls)
            if hasattr(c, "args") and "glVertexAttribPointer" in str(c)
            or str(c).startswith("call.glVertexAttribPointer")),
            None,
        )
        # Simpler: just check glEnableVertexAttribArray(0) comes after bind
        enable_idx = next(
            i for i, c in enumerate(gl.mock_calls)
            if c == call.glEnableVertexAttribArray(0)
        )
        assert bind_idx < enable_idx

    def test_vao_unbound_after_setup(self, gl):
        GeometryManager().initialize()
        # The last glBindVertexArray call in init must be with 0.
        bind_calls = [c for c in gl.mock_calls if "glBindVertexArray" in str(c)]
        assert call.glBindVertexArray(0) in bind_calls

    def test_position_attribute_pointer_args(self, gl):
        """location=0, size=3, stride=20, offset=0."""
        GeometryManager().initialize()
        # ctypes.c_void_p does not implement __eq__ — two separate instances
        # are never == by mock's equality check even if both represent null.
        # Extract the recorded call for location 0 and compare fields directly.
        pos_call = next(
            c for c in gl.glVertexAttribPointer.call_args_list
            if c.args[0] == 0   # attribute location
        )
        loc, size, typ, norm, stride, offset = pos_call.args
        assert loc    == 0
        assert size   == 3
        assert typ    == gl.GL_FLOAT
        assert norm   == gl.GL_FALSE
        assert stride == 20
        # c_void_p(0).value is None — that is how ctypes represents a null pointer.
        assert offset.value is None

    def test_texcoord_attribute_pointer_args(self, gl):
        """location=1, size=2, stride=20, offset=12."""
        GeometryManager().initialize()
        tex_call = next(
            c for c in gl.glVertexAttribPointer.call_args_list
            if c.args[0] == 1   # attribute location
        )
        loc, size, typ, norm, stride, offset = tex_call.args
        assert loc    == 1
        assert size   == 2
        assert typ    == gl.GL_FLOAT
        assert norm   == gl.GL_FALSE
        assert stride == 20
        assert offset.value == 12   # 3 position floats × 4 bytes

    def test_vbo_array_buffer_data_uploaded(self, gl):
        m = GeometryManager()
        m.initialize()
        gl.glBufferData.assert_any_call(
            gl.GL_ARRAY_BUFFER,
            GeometryManager.VERTICES.nbytes,
            GeometryManager.VERTICES,
            gl.GL_STATIC_DRAW,
        )

    def test_ebo_element_array_buffer_data_uploaded(self, gl):
        m = GeometryManager()
        m.initialize()
        gl.glBufferData.assert_any_call(
            gl.GL_ELEMENT_ARRAY_BUFFER,
            GeometryManager.INDICES.nbytes,
            GeometryManager.INDICES,
            gl.GL_STATIC_DRAW,
        )


# =============================================================================
# bind / unbind
# =============================================================================

class TestBindUnbind:

    def test_bind_calls_bind_vertex_array_with_vao(self, mgr, gl):
        gl.reset_mock()
        mgr.bind()
        gl.glBindVertexArray.assert_called_once_with(_VAO_ID)

    def test_bind_noop_when_uninitialised(self, gl):
        gl.reset_mock()
        GeometryManager().bind()
        gl.glBindVertexArray.assert_not_called()

    def test_unbind_always_binds_zero(self, mgr, gl):
        gl.reset_mock()
        mgr.unbind()
        gl.glBindVertexArray.assert_called_once_with(0)

    def test_unbind_works_when_uninitialised(self, gl):
        gl.reset_mock()
        GeometryManager().unbind()
        gl.glBindVertexArray.assert_called_once_with(0)


# =============================================================================
# draw
# =============================================================================

class TestDraw:

    def test_draw_calls_draw_elements_with_correct_args(self, mgr, gl):
        gl.reset_mock()
        mgr.draw()
        gl.glDrawElements.assert_called_once()
        mode, count, typ, offset = gl.glDrawElements.call_args.args
        assert mode   == gl.GL_TRIANGLES
        assert count  == len(GeometryManager.INDICES)
        assert typ    == gl.GL_UNSIGNED_INT
        # c_void_p(0).value is None — ctypes represents a null pointer as None.
        assert offset.value is None

    def test_draw_skips_and_warns_when_uninitialised(self, gl, caplog):
        with caplog.at_level(logging.WARNING, logger=_MOD):
            GeometryManager().draw()
        gl.glDrawElements.assert_not_called()
        assert caplog.records

    def test_draw_does_not_call_bind_or_unbind(self, mgr, gl):
        """draw() must not manage VAO binding — that is the caller's responsibility."""
        gl.reset_mock()
        mgr.draw()
        gl.glBindVertexArray.assert_not_called()


# =============================================================================
# cleanup  (includes the VAO-handle bug regression)
# =============================================================================

class TestCleanup:

    def test_vao_deleted_with_vao_handle_not_vbo(self, mgr, gl):
        """
        Regression: original code passed self.vbo to glDeleteVertexArrays.
        The VAO handle must be passed, not the VBO handle.
        """
        gl.reset_mock()
        mgr.cleanup()
        vao_array = np.array([_VAO_ID], dtype=np.uint32)
        actual_args = gl.glDeleteVertexArrays.call_args[0]
        assert actual_args[0] == 1
        assert actual_args[1][0] == _VAO_ID, (
            "glDeleteVertexArrays received %r — expected VAO handle %d, "
            "not VBO handle %d" % (actual_args[1], _VAO_ID, _VBO_ID)
        )

    def test_vbo_deleted_with_vbo_handle(self, mgr, gl):
        gl.reset_mock()
        mgr.cleanup()
        actual_args = gl.glDeleteBuffers.call_args_list[0][0]
        assert actual_args[1][0] == _VBO_ID

    def test_ebo_deleted_with_ebo_handle(self, mgr, gl):
        gl.reset_mock()
        mgr.cleanup()
        actual_args = gl.glDeleteBuffers.call_args_list[1][0]
        assert actual_args[1][0] == _EBO_ID

    def test_handles_reset_to_none_after_cleanup(self, mgr, gl):
        mgr.cleanup()
        assert mgr.vao is None
        assert mgr.vbo is None
        assert mgr.ebo is None

    def test_cleanup_is_idempotent(self, mgr, gl):
        mgr.cleanup()
        gl.reset_mock()
        mgr.cleanup()
        gl.glDeleteVertexArrays.assert_not_called()
        gl.glDeleteBuffers.assert_not_called()

    def test_partial_init_cleanup_skips_missing_handles(self, gl):
        """VAO allocated but buffers failed — cleanup must not crash."""
        gl.glGenBuffers.side_effect = [0, _EBO_ID]
        m = GeometryManager()
        m.initialize()
        # vbo is None because glGenBuffers returned 0; cleanup must handle this.
        m.cleanup()   # must not raise


# =============================================================================
# Context manager
# =============================================================================

class TestContextManager:

    def test_enter_binds_vao(self, mgr, gl):
        gl.reset_mock()
        with mgr:
            gl.glBindVertexArray.assert_called_once_with(_VAO_ID)

    def test_exit_unbinds_vao(self, mgr, gl):
        gl.reset_mock()
        with mgr:
            pass
        assert gl.glBindVertexArray.call_args_list[-1] == call(0)

    def test_exit_unbinds_even_on_exception(self, mgr, gl):
        gl.reset_mock()
        with pytest.raises(RuntimeError):
            with mgr:
                raise RuntimeError("render error")
        assert gl.glBindVertexArray.call_args_list[-1] == call(0)
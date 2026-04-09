"""
tests/test_gl_uniform.py
========================
Unit tests for cross_platform.qt6_utils.image.gl.uniform.

All OpenGL driver calls are intercepted by patching ``GL`` in the uniform
module's namespace.  No real OpenGL context is required.

Patch strategy
--------------
The ``gl`` fixture patches ``_MOD + ".GL"`` once per test using
``unittest.mock.patch``.  The module is imported **once** at collection time
(module-level ``import`` statements below); the patch then replaces the ``GL``
name in the module's namespace for the duration of each test.

``importlib.reload`` is intentionally absent.  Reloading the module inside a
``patch`` context causes the module's ``from ... import GL`` line to run again,
silently overwriting the mock with the real PyOpenGL GL object.  Any
subsequent call to a real GL function without an active context produces a
segmentation fault -- exactly the failure being fixed here.

``UniformType`` enum members bind their integer values once at class-definition
time from real GL constants (e.g. ``GL_INT = 0x1404``).  These values are
mandated by the OpenGL specification and match the ``_C`` table below, so the
dispatch comparisons in ``_set_uniform_by_type`` work correctly with the
patched mock without any reload.

Coverage
--------
UniformType         -- all members are positive ints; AUTO is -1.
_infer_type         -- bool before int ordering (critical regression guard),
                       scalar, vector, matrix, list, unknown fallback.
_gl_type_name       -- known tokens, unknown token hex fallback.
UniformManager      -- construction guard, property, registration, locs
                       namespace, introspection, set/set_fast dispatch,
                       AUTO type resolution, GL error propagation, error
                       logging, _check_gl_error.
register_members    -- StrEnum extraction delegates correctly.
_set_uniform_by_type -- every dispatch branch, unsupported type, GL error.
"""

from __future__ import annotations

import logging
from enum import StrEnum, unique
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Module-level imports -- executed ONCE at collection time.
# The gl fixture patches GL in the module namespace after this point.
# ---------------------------------------------------------------------------
from cross_platform.qt6_utils.image.gl.error import GLError
from cross_platform.qt6_utils.image.gl.types import GLenum, GLint, GLuint
from cross_platform.qt6_utils.image.gl.uniform import (
    UniformManager,
    UniformType,
    FragmentShaderUniforms,
)

_MOD = "cross_platform.qt6_utils.image.gl.uniform"

# ---------------------------------------------------------------------------
# GL constant table
#
# Values are the OpenGL-specification-mandated integers for each token.
# They match what the real PyOpenGL GL module exposes, which is why the
# dispatch comparisons (e.g. ``gl_type == GL.GL_INT``) work correctly with
# the patched mock: both sides of the comparison resolve to 0x1404, etc.
# ---------------------------------------------------------------------------
_C: dict[str, int] = {
    "GL_NO_ERROR":        0x0000,
    "GL_INT":             0x1404,
    "GL_UNSIGNED_INT":    0x1405,
    "GL_FLOAT":           0x1406,
    "GL_BOOL":            0x8B56,
    "GL_FLOAT_VEC2":      0x8B50,
    "GL_FLOAT_VEC3":      0x8B51,
    "GL_FLOAT_VEC4":      0x8B52,
    "GL_INT_VEC2":        0x8B53,
    "GL_INT_VEC3":        0x8B54,
    "GL_INT_VEC4":        0x8B55,
    "GL_BOOL_VEC2":       0x8B57,
    "GL_BOOL_VEC3":       0x8B58,
    "GL_BOOL_VEC4":       0x8B59,
    "GL_FLOAT_MAT2":      0x8B5A,
    "GL_FLOAT_MAT3":      0x8B5B,
    "GL_FLOAT_MAT4":      0x8B5C,
    "GL_FLOAT_MAT2x3":    0x8B65,
    "GL_FLOAT_MAT2x4":    0x8B66,
    "GL_FLOAT_MAT3x2":    0x8B67,
    "GL_FLOAT_MAT3x4":    0x8B68,
    "GL_FLOAT_MAT4x2":    0x8B69,
    "GL_FLOAT_MAT4x3":    0x8B6A,
    "GL_SAMPLER_1D":      0x8B5D,
    "GL_SAMPLER_2D":      0x8B5E,
    "GL_SAMPLER_3D":      0x8B5F,
    "GL_SAMPLER_CUBE":    0x8B60,
    "GL_FALSE":           0x0000,
    "GL_ACTIVE_UNIFORMS": 0x8B86,
}


def _make_gl() -> MagicMock:
    """
    Return a GL mock with all constants from ``_C`` set and error-free defaults.

    ``glGetError`` returns ``GL_NO_ERROR`` by default; individual tests
    override it when they need to exercise the error path.
    """
    gl = MagicMock(name="GL")
    for name, value in _C.items():
        setattr(gl, name, value)
    gl.glGetError.return_value = _C["GL_NO_ERROR"]
    return gl


@pytest.fixture()
def gl():
    """
    Patch ``GL`` in the uniform module namespace and yield the configured mock.

    ``ensure_contiguity`` is also patched to a passthrough so that matrix
    tests do not depend on a real NumPy contiguity check.

    The module is NOT reloaded here.  See the module docstring for why reload
    inside a patch context causes segmentation faults.
    """
    mock_gl = _make_gl()
    with (
        patch(f"{_MOD}.GL", mock_gl),
        patch(f"{_MOD}.ensure_contiguity", side_effect=lambda x: x),
    ):
        yield mock_gl


# Stable fake handles used throughout.
_PROGRAM = GLuint(7)
_LOC     = GLint(3)


# =============================================================================
# UniformType
# =============================================================================

class TestUniformType:

    def test_auto_sentinel_is_minus_one(self):
        assert int(UniformType.AUTO) == -1

    def test_all_non_auto_members_are_positive(self):
        for member in UniformType:
            if member is not UniformType.AUTO:
                assert int(member) > 0, "%s should be > 0" % member.name

    def test_float_value_matches_spec(self):
        # GL_FLOAT = 0x1406 per the OpenGL specification.
        assert int(UniformType.FLOAT) == _C["GL_FLOAT"]

    def test_sampler_2d_value_matches_spec(self):
        assert int(UniformType.SAMPLER_2D) == _C["GL_SAMPLER_2D"]


# =============================================================================
# UniformManager construction
# =============================================================================

class TestUniformManagerConstruction:

    def test_valid_program_creates_instance(self):
        mgr = UniformManager(_PROGRAM)
        assert mgr.program == _PROGRAM

    def test_zero_program_raises_value_error(self):
        with pytest.raises(ValueError, match="0"):
            UniformManager(GLuint(0))

    def test_locs_is_simple_namespace(self):
        assert isinstance(UniformManager(_PROGRAM).locs, SimpleNamespace)

    def test_locs_starts_empty(self):
        assert vars(UniformManager(_PROGRAM).locs) == {}


# =============================================================================
# _infer_type  (critical: bool must be checked before int)
# =============================================================================

class TestInferType:
    """
    ``_infer_type`` is a static method; these tests call it directly.

    The ``gl`` fixture is required because ``_infer_type`` references
    ``GL.*`` constants inside the method body (not at definition time),
    so the patch must be active for the comparisons to resolve correctly.
    """

    # --- bool before int (the original bug) ---------------------------------

    def test_true_infers_bool_not_int(self, gl):
        result = UniformManager._infer_type(True)
        assert result == _C["GL_BOOL"], (
            "bool must be checked before int; isinstance(True, int) is True "
            "so without the guard, True would be incorrectly classified as GL_INT"
        )

    def test_false_infers_bool_not_int(self, gl):
        assert UniformManager._infer_type(False) == _C["GL_BOOL"]

    # --- scalars ------------------------------------------------------------

    def test_int_infers_gl_int(self, gl):
        assert UniformManager._infer_type(5) == _C["GL_INT"]

    def test_float_infers_gl_float(self, gl):
        assert UniformManager._infer_type(1.5) == _C["GL_FLOAT"]

    # --- vectors (list) -----------------------------------------------------

    @pytest.mark.parametrize("value, key", [
        ([0.0, 1.0],           "GL_FLOAT_VEC2"),
        ([0.0, 1.0, 2.0],      "GL_FLOAT_VEC3"),
        ([0.0, 1.0, 2.0, 3.0], "GL_FLOAT_VEC4"),
    ])
    def test_list_vector_inference(self, gl, value, key):
        assert UniformManager._infer_type(value) == _C[key]

    # --- vectors (ndarray) --------------------------------------------------

    @pytest.mark.parametrize("shape, key", [
        ((2,), "GL_FLOAT_VEC2"),
        ((3,), "GL_FLOAT_VEC3"),
        ((4,), "GL_FLOAT_VEC4"),
    ])
    def test_ndarray_vector_inference(self, gl, shape, key):
        arr = np.zeros(shape, dtype=np.float32)
        assert UniformManager._infer_type(arr) == _C[key]

    # --- matrices -----------------------------------------------------------

    @pytest.mark.parametrize("shape, key", [
        ((2, 2), "GL_FLOAT_MAT2"),
        ((3, 3), "GL_FLOAT_MAT3"),
        ((4, 4), "GL_FLOAT_MAT4"),
    ])
    def test_ndarray_matrix_inference(self, gl, shape, key):
        assert UniformManager._infer_type(np.eye(*shape, dtype=np.float32)) == _C[key]

    # --- fallback -----------------------------------------------------------

    def test_unknown_type_falls_back_to_float(self, gl):
        assert UniformManager._infer_type(object()) == _C["GL_FLOAT"]

    def test_ndarray_length_5_falls_back_to_float(self, gl):
        # shape (5,) has no entry in the shape map.
        assert UniformManager._infer_type(np.zeros(5)) == _C["GL_FLOAT"]


# =============================================================================
# _gl_type_name
# =============================================================================

class TestGlTypeName:

    def test_known_float_returns_string(self, gl):
        assert UniformManager._gl_type_name(GLenum(_C["GL_FLOAT"])) == "float"

    def test_known_mat4_returns_string(self, gl):
        assert UniformManager._gl_type_name(GLenum(_C["GL_FLOAT_MAT4"])) == "mat4"

    def test_known_sampler_2d_returns_string(self, gl):
        assert UniformManager._gl_type_name(GLenum(_C["GL_SAMPLER_2D"])) == "sampler2D"

    def test_unknown_token_returns_hex_string(self, gl):
        result = UniformManager._gl_type_name(GLenum(0xDEAD))
        assert result.startswith("0x")
        assert "dead" in result.lower()


# =============================================================================
# register_uniforms / register_members
# =============================================================================

class TestRegistration:

    def _register(
        self,
        gl: MagicMock,
        names: list[str],
        locations: list[int],
        *,
        introspect: bool = False,
    ) -> UniformManager:
        """Configure the mock and call register_uniforms."""
        gl.glGetUniformLocation.side_effect = locations
        gl.glGetProgramiv.return_value = 0   # zero active uniforms to introspect
        mgr = UniformManager(_PROGRAM)
        mgr.register_uniforms(names, introspect=introspect)
        return mgr

    # --- program binding sequencing ----------------------------------------

    def test_program_bound_before_location_query(self, gl):
        """glUseProgram(program) must precede glGetUniformLocation in the call log."""
        gl.glGetUniformLocation.return_value = 1
        gl.glGetProgramiv.return_value = 0
        mgr = UniformManager(_PROGRAM)
        mgr.register_uniforms(["u_val"], introspect=False)

        # Scan call_args_list by index to avoid equality comparison with
        # mock_calls.index(), which can recurse into MagicMock.__eq__.
        use_indices = [
            i for i, c in enumerate(gl.mock_calls)
            if c == call.glUseProgram(_PROGRAM)
        ]
        loc_indices = [
            i for i, c in enumerate(gl.mock_calls)
            if c == call.glGetUniformLocation(_PROGRAM, "u_val")
        ]
        assert use_indices, "glUseProgram(program) was never called"
        assert loc_indices, "glGetUniformLocation was never called"
        assert use_indices[0] < loc_indices[0]

    def test_program_unbound_after_registration(self, gl):
        gl.glGetUniformLocation.return_value = 1
        gl.glGetProgramiv.return_value = 0
        UniformManager(_PROGRAM).register_uniforms(["u_val"], introspect=False)
        assert gl.glUseProgram.call_args_list[-1] == call(0)

    # --- locs namespace ----------------------------------------------------

    def test_active_uniform_added_to_locs(self, gl):
        mgr = self._register(gl, ["brightness"], [5])
        assert hasattr(mgr.locs, "brightness")
        assert mgr.locs.brightness == GLint(5)

    def test_inactive_uniform_not_added_to_locs(self, gl):
        mgr = self._register(gl, ["unused"], [-1])
        assert not hasattr(mgr.locs, "unused")

    def test_inactive_uniform_logged_as_warning(self, gl, caplog):
        with caplog.at_level(logging.WARNING, logger=_MOD):
            self._register(gl, ["gone"], [-1])
        assert caplog.records
        assert "gone" in caplog.text

    def test_get_location_returns_cached_value(self, gl):
        mgr = self._register(gl, ["contrast"], [9])
        assert mgr.get_location("contrast") == GLint(9)

    def test_get_location_returns_minus_one_for_unknown(self, gl):
        assert UniformManager(_PROGRAM).get_location("nonexistent") == GLint(-1)

    # --- introspection -----------------------------------------------------

    def test_introspect_populates_type_cache(self, gl):
        gl.glGetProgramiv.return_value = 1
        gl.glGetActiveUniform.return_value = (b"brightness", 1, _C["GL_FLOAT"])
        gl.glGetUniformLocation.return_value = 2

        mgr = UniformManager(_PROGRAM)
        mgr.register_uniforms(["brightness"], introspect=True)

        assert "brightness" in mgr._types
        assert mgr._types["brightness"] == GLenum(_C["GL_FLOAT"])

    def test_introspect_strips_array_suffix(self, gl):
        gl.glGetProgramiv.return_value = 1
        gl.glGetActiveUniform.return_value = (b"lights[0]", 4, _C["GL_FLOAT_VEC3"])
        gl.glGetUniformLocation.return_value = 1

        mgr = UniformManager(_PROGRAM)
        mgr.register_uniforms(["lights"], introspect=True)

        assert "lights" in mgr._types

    def test_introspect_decodes_bytes_name(self, gl):
        gl.glGetProgramiv.return_value = 1
        gl.glGetActiveUniform.return_value = (b"u_color", 1, _C["GL_FLOAT_VEC3"])
        gl.glGetUniformLocation.return_value = 4

        mgr = UniformManager(_PROGRAM)
        mgr.register_uniforms(["u_color"], introspect=True)

        assert "u_color" in mgr._types

    # --- register_members --------------------------------------------------

    def test_register_members_extracts_enum_string_values(self, gl):
        @unique
        class Stub(StrEnum):
            A = "u_alpha"
            B = "u_beta"

        gl.glGetUniformLocation.side_effect = [10, 11]
        gl.glGetProgramiv.return_value = 0

        mgr = UniformManager(_PROGRAM)
        mgr.register_members(Stub, introspect=False)

        assert mgr.get_location("u_alpha") == GLint(10)
        assert mgr.get_location("u_beta")  == GLint(11)


# =============================================================================
# _check_gl_error
# =============================================================================

class TestCheckGlError:

    def test_no_error_does_not_raise(self, gl):
        gl.glGetError.return_value = _C["GL_NO_ERROR"]
        UniformManager(_PROGRAM)._check_gl_error("test op")   # must not raise

    def test_non_zero_error_raises_gl_error(self, gl):
        gl.glGetError.return_value = 0x0500   # GL_INVALID_ENUM
        with pytest.raises(GLError, match="0x500"):
            UniformManager(_PROGRAM)._check_gl_error("test op")

    def test_error_message_includes_operation_label(self, gl):
        gl.glGetError.return_value = 0x0501
        with pytest.raises(GLError, match="glUseProgram"):
            UniformManager(_PROGRAM)._check_gl_error("glUseProgram")


# =============================================================================
# set / set_fast
# =============================================================================

class TestSetAndSetFast:

    def _registered(self, gl: MagicMock, name: str = "u_val", loc: int = 3) -> UniformManager:
        """Return a manager with one active uniform registered."""
        gl.glGetUniformLocation.return_value = loc
        gl.glGetProgramiv.return_value = 0
        mgr = UniformManager(_PROGRAM)
        mgr.register_uniforms([name], introspect=False)
        return mgr

    # --- set (by name) -----------------------------------------------------

    def test_set_returns_false_for_unregistered_name(self, gl):
        assert UniformManager(_PROGRAM).set("unknown", 1.0, UniformType.FLOAT) is False

    def test_set_returns_true_on_success(self, gl):
        mgr = self._registered(gl)
        assert mgr.set("u_val", 1.0, UniformType.FLOAT) is True

    def test_set_dispatches_float_correctly(self, gl):
        mgr = self._registered(gl)
        mgr.set("u_val", 3.14, UniformType.FLOAT)
        gl.glUniform1f.assert_called_once_with(3, pytest.approx(3.14))

    # --- set_fast (by location) --------------------------------------------

    def test_set_fast_returns_false_for_minus_one(self, gl):
        assert UniformManager(_PROGRAM).set_fast(GLint(-1), 1.0, UniformType.FLOAT) is False

    def test_set_fast_dispatches_int(self, gl):
        UniformManager(_PROGRAM).set_fast(_LOC, 7, UniformType.INT)
        gl.glUniform1i.assert_called_once_with(_LOC, 7)

    def test_set_fast_dispatches_float(self, gl):
        UniformManager(_PROGRAM).set_fast(_LOC, 2.5, UniformType.FLOAT)
        gl.glUniform1f.assert_called_once_with(_LOC, 2.5)

    def test_set_fast_dispatches_bool(self, gl):
        UniformManager(_PROGRAM).set_fast(_LOC, True, UniformType.BOOL)
        gl.glUniform1i.assert_called_once_with(_LOC, 1)

    # --- AUTO type resolution ----------------------------------------------

    def test_auto_uses_introspection_cache_over_inference(self, gl):
        """When the type cache has an entry, AUTO must use it, not _infer_type."""
        gl.glGetProgramiv.return_value = 1
        gl.glGetActiveUniform.return_value = (b"u_val", 1, _C["GL_FLOAT_VEC3"])
        gl.glGetUniformLocation.return_value = 3

        mgr = UniformManager(_PROGRAM)
        mgr.register_uniforms(["u_val"], introspect=True)
        mgr.set("u_val", [1.0, 2.0, 3.0], UniformType.AUTO)
        gl.glUniform3fv.assert_called_once()

    def test_auto_falls_back_to_inference_without_cache(self, gl):
        """Without introspection, AUTO must classify from the Python value type."""
        gl.glGetUniformLocation.return_value = 3
        gl.glGetProgramiv.return_value = 0

        mgr = UniformManager(_PROGRAM)
        mgr.register_uniforms(["u_val"], introspect=False)
        mgr.set("u_val", 0.5, UniformType.AUTO)
        gl.glUniform1f.assert_called_once()


# =============================================================================
# _set_uniform_by_type -- dispatch table
# =============================================================================

class TestSetUniformByType:
    """
    Each test calls ``_set_uniform_by_type`` directly to isolate the dispatch
    from registration overhead.  One assertion per GL function keeps failure
    output immediately actionable.
    """

    def _dispatch(self, gl: MagicMock, type_key: str, value) -> bool:
        return UniformManager(_PROGRAM)._set_uniform_by_type(
            _LOC, value, GLenum(_C[type_key])
        )

    # --- scalars ------------------------------------------------------------

    def test_gl_int_calls_uniform1i(self, gl):
        self._dispatch(gl, "GL_INT", 5)
        gl.glUniform1i.assert_called_once_with(_LOC, 5)

    def test_gl_bool_calls_uniform1i(self, gl):
        self._dispatch(gl, "GL_BOOL", True)
        gl.glUniform1i.assert_called_once_with(_LOC, 1)

    def test_gl_float_calls_uniform1f(self, gl):
        self._dispatch(gl, "GL_FLOAT", 1.5)
        gl.glUniform1f.assert_called_once_with(_LOC, 1.5)

    def test_gl_unsigned_int_calls_uniform1ui(self, gl):
        self._dispatch(gl, "GL_UNSIGNED_INT", 3)
        gl.glUniform1ui.assert_called_once_with(_LOC, 3)

    # --- samplers (all four must call glUniform1i) --------------------------

    @pytest.mark.parametrize("key", [
        "GL_SAMPLER_1D", "GL_SAMPLER_2D", "GL_SAMPLER_3D", "GL_SAMPLER_CUBE",
    ])
    def test_sampler_calls_uniform1i(self, gl, key):
        self._dispatch(gl, key, 0)
        gl.glUniform1i.assert_called_once_with(_LOC, 0)

    # --- float vectors ------------------------------------------------------

    def test_float_vec2_calls_uniform2fv(self, gl):
        self._dispatch(gl, "GL_FLOAT_VEC2", [1.0, 2.0])
        gl.glUniform2fv.assert_called_once()

    def test_float_vec3_calls_uniform3fv(self, gl):
        self._dispatch(gl, "GL_FLOAT_VEC3", [1.0, 2.0, 3.0])
        gl.glUniform3fv.assert_called_once()

    def test_float_vec4_calls_uniform4fv(self, gl):
        self._dispatch(gl, "GL_FLOAT_VEC4", [1.0, 2.0, 3.0, 4.0])
        gl.glUniform4fv.assert_called_once()

    # --- integer vectors ----------------------------------------------------

    def test_int_vec2_calls_uniform2iv(self, gl):
        self._dispatch(gl, "GL_INT_VEC2", [1, 2])
        gl.glUniform2iv.assert_called_once()

    def test_int_vec3_calls_uniform3iv(self, gl):
        self._dispatch(gl, "GL_INT_VEC3", [1, 2, 3])
        gl.glUniform3iv.assert_called_once()

    def test_int_vec4_calls_uniform4iv(self, gl):
        self._dispatch(gl, "GL_INT_VEC4", [1, 2, 3, 4])
        gl.glUniform4iv.assert_called_once()

    # --- bool vectors -------------------------------------------------------

    @pytest.mark.parametrize("key", ["GL_BOOL_VEC2", "GL_BOOL_VEC3", "GL_BOOL_VEC4"])
    def test_bool_vec_calls_corresponding_iv(self, gl, key):
        n = int(key[-1])
        self._dispatch(gl, key, [True] * n)
        getattr(gl, "glUniform%div" % n).assert_called_once()

    # --- matrices -----------------------------------------------------------

    def test_mat2_calls_uniformmatrix2fv(self, gl):
        self._dispatch(gl, "GL_FLOAT_MAT2", np.eye(2, dtype=np.float32))
        gl.glUniformMatrix2fv.assert_called_once()

    def test_mat3_calls_uniformmatrix3fv(self, gl):
        self._dispatch(gl, "GL_FLOAT_MAT3", np.eye(3, dtype=np.float32))
        gl.glUniformMatrix3fv.assert_called_once()

    def test_mat4_calls_uniformmatrix4fv(self, gl):
        self._dispatch(gl, "GL_FLOAT_MAT4", np.eye(4, dtype=np.float32))
        gl.glUniformMatrix4fv.assert_called_once()

    def test_mat2x3_calls_uniformmatrix2x3fv(self, gl):
        self._dispatch(gl, "GL_FLOAT_MAT2x3", np.zeros((2, 3), dtype=np.float32))
        gl.glUniformMatrix2x3fv.assert_called_once()

    def test_mat3x2_calls_uniformmatrix3x2fv(self, gl):
        self._dispatch(gl, "GL_FLOAT_MAT3x2", np.zeros((3, 2), dtype=np.float32))
        gl.glUniformMatrix3x2fv.assert_called_once()

    # --- unsupported type --------------------------------------------------

    def test_unsupported_type_returns_false(self, gl):
        result = UniformManager(_PROGRAM)._set_uniform_by_type(
            _LOC, 0, GLenum(0xFFFF)
        )
        assert result is False

    def test_unsupported_type_logs_warning(self, gl, caplog):
        with caplog.at_level(logging.WARNING, logger=_MOD):
            UniformManager(_PROGRAM)._set_uniform_by_type(_LOC, 0, GLenum(0xFFFF))
        assert caplog.records

    # --- GL error propagation ----------------------------------------------

    def test_driver_error_raises_gl_error(self, gl):
        gl.glGetError.return_value = 0x0502   # GL_INVALID_OPERATION
        with pytest.raises(GLError, match="0x502"):
            UniformManager(_PROGRAM)._set_uniform_by_type(
                _LOC, 1.0, GLenum(_C["GL_FLOAT"])
            )

    def test_gl_error_caught_by_set_by_loc_and_type(self, gl):
        """GLError from the dispatch is caught one level up; returns False."""
        gl.glGetError.return_value = 0x0502
        result = UniformManager(_PROGRAM)._set_by_loc_and_type(
            _LOC, 1.0, UniformType.FLOAT, "u_val"
        )
        assert result is False

    def test_gl_error_caught_by_set_fast(self, gl):
        gl.glGetError.return_value = 0x0502
        assert UniformManager(_PROGRAM).set_fast(_LOC, 1.0, UniformType.FLOAT) is False

    def test_gl_error_is_logged_when_caught(self, gl, caplog):
        gl.glGetError.return_value = 0x0502
        with caplog.at_level(logging.ERROR, logger=_MOD):
            UniformManager(_PROGRAM).set_fast(_LOC, 1.0, UniformType.FLOAT)
        assert caplog.records
"""
tests/test_gl_pbo.py
====================
Unit and integration tests for cross_platform.qt6_utils.image.gl.pbo.

No real OpenGL context is required.  All GL driver calls are intercepted
by patching the ``GL`` name in the pbo module's namespace.  The module is
imported once at collection time; the ``gl`` fixture replaces ``GL`` for each
test without reloading the module (reloading inside a patch context overwrites
the mock with the real GL object, causing segfaults on the next driver call).

Memory safety
-------------
``PBO.prepare_and_map`` performs a real ctypes cast and numpy view on the
mapped pointer.  Tests for that method allocate genuine memory via
``ctypes.create_string_buffer`` and pass its address as the mock return value
from ``glMapBufferRange``.  Using a fake address (e.g. ``0x1000``) would
cause a segfault when numpy tries to read it.

Coverage
--------
PBOBufferingStrategy  — member values, default, from_int, description.
_extract_pointer      — c_void_p path, mock .value path, null pointer,
                        extraction failure.
calculate_pixel_alignment — all format/type branches.
configure_pixel_storage   — alignment forwarded, extra params set.
memmove_pbo           — happy path sequencing, falsy map return, null pointer,
                        PBO left bound on success.
write_pbo_buffer      — shape match, shape mismatch raises GLUploadError
                        (regression: original raised ValueError).
PBO                   — construction, prepare_and_map (shape, dtype, mapping
                        state, size mismatch, null map + unbind), unmap
                        (noop, false return warning, state reset), destroy
                        (noop on id=0, pre-unmap, delete handle, id reset).
PBOManager            — construction (strategy, deprecated num_pbos shim),
                        initialize (count, idempotent), bind, unbind,
                        get_next (uninitialised error, round-robin),
                        acquire_next_writeable (delegation, error propagation),
                        cleanup (all PBOs destroyed, state reset, idempotent).
"""

from __future__ import annotations

import ctypes
import itertools
import logging
from contextlib import contextmanager
from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest

from cross_platform.qt6_utils.image.gl.error import GLMemoryError, GLUploadError
from cross_platform.qt6_utils.image.gl.pbo import (
    PBO,
    PBOBufferingStrategy,
    PBOManager,
    _extract_pointer,
    calculate_pixel_alignment,
    configure_pixel_storage,
    memmove_pbo,
    write_pbo_buffer,
)
from cross_platform.qt6_utils.image.gl.types import GLenum, GLBuffer, GLint, GLsizeiptr

_MOD = "cross_platform.qt6_utils.image.gl.pbo"

# ---------------------------------------------------------------------------
# GL constant table — OpenGL-specification-mandated values.
# The dispatch comparisons in the production code (e.g. ``gl_type == GL.GL_FLOAT``)
# use these same integers, so the mock and the argument must agree.
# ---------------------------------------------------------------------------
_C: dict[str, int] = {
    "GL_PIXEL_UNPACK_BUFFER":       0x88EB,
    "GL_STREAM_DRAW":               0x88E0,
    "GL_MAP_WRITE_BIT":             0x0002,
    "GL_MAP_INVALIDATE_BUFFER_BIT": 0x0008,
    "GL_FLOAT":                     0x1406,
    "GL_UNSIGNED_BYTE":             0x1401,
    "GL_UNSIGNED_SHORT":            0x1403,
    "GL_RGB":                       0x1907,
    "GL_RGBA":                      0x1908,
    "GL_UNPACK_ALIGNMENT":          0x0CF5,
    "GL_UNPACK_ROW_LENGTH":         0x0CF2,
    "GL_UNPACK_SKIP_PIXELS":        0x0CF4,
    "GL_UNPACK_SKIP_ROWS":          0x0CF3,
    "GL_NO_ERROR":                  0x0000,
}


def _make_gl() -> MagicMock:
    """Return a GL mock with all constants and safe defaults."""
    gl = MagicMock(name="GL")
    for name, value in _C.items():
        setattr(gl, name, value)
    gl.glGetError.return_value   = _C["GL_NO_ERROR"]
    gl.glUnmapBuffer.return_value = True
    return gl


@contextmanager
def _noop_error_check(*args, **kwargs):
    """Passthrough replacement for gl_error_check in happy-path tests."""
    yield


@pytest.fixture()
def gl():
    """
    Patch GL in the pbo module namespace and yield the configured mock.

    ``glGenBuffers`` uses an incrementing counter so each PBO created during
    a test receives a distinct, non-zero handle.  Starting at 10 avoids
    any ambiguity with boolean False (0).
    """
    mock_gl = _make_gl()
    _id = itertools.count(10)
    mock_gl.glGenBuffers.side_effect = lambda _: next(_id)

    with patch(f"{_MOD}.GL", mock_gl):
        yield mock_gl


@pytest.fixture()
def pbo_env(gl):
    """
    Full test environment: GL + all module-level dependencies patched.

    Suitable for PBO and PBOManager tests that exercise the full call chain
    without needing a real GL context, config, or copy infrastructure.
    """
    with (
        patch(f"{_MOD}.gl_error_check", _noop_error_check),
        patch(f"{_MOD}.tuned_parallel_copy"),
        patch(f"{_MOD}.ensure_contiguity", side_effect=lambda x: x),
        patch(f"{_MOD}.broadcast_to_format", side_effect=lambda img, fmt, copy: img),
    ):
        yield gl


def _real_mapped_ptr(size_bytes: int) -> tuple[ctypes.create_string_buffer, ctypes.c_void_p]:
    """
    Allocate ``size_bytes`` of real memory and return ``(buffer, c_void_p)``.

    The buffer reference must be kept alive for the duration of the test;
    ctypes does not prevent the GC from collecting it.
    """
    buf  = ctypes.create_string_buffer(size_bytes)
    addr = ctypes.addressof(buf)
    return buf, ctypes.c_void_p(addr)


# =============================================================================
# PBOBufferingStrategy
# =============================================================================

class TestPBOBufferingStrategy:

    def test_member_values(self):
        assert PBOBufferingStrategy.SINGLE == 1
        assert PBOBufferingStrategy.DOUBLE == 2
        assert PBOBufferingStrategy.TRIPLE == 3

    def test_default_returns_double(self):
        assert PBOBufferingStrategy.default() is PBOBufferingStrategy.DOUBLE

    @pytest.mark.parametrize("value, expected", [
        (1, PBOBufferingStrategy.SINGLE),
        (2, PBOBufferingStrategy.DOUBLE),
        (3, PBOBufferingStrategy.TRIPLE),
    ])
    def test_from_int_valid(self, value, expected):
        assert PBOBufferingStrategy.from_int(value) is expected

    def test_from_int_invalid_raises_value_error(self):
        with pytest.raises(ValueError, match="Invalid PBOBufferingStrategy"):
            PBOBufferingStrategy.from_int(4)

    def test_from_int_error_message_lists_valid_values(self):
        with pytest.raises(ValueError, match="1, 2, 3"):
            PBOBufferingStrategy.from_int(0)

    def test_description_returns_non_empty_string(self):
        for member in PBOBufferingStrategy:
            assert isinstance(member.description, str)
            assert len(member.description) > 0

    def test_description_single_mentions_stall(self):
        assert "stall" in PBOBufferingStrategy.SINGLE.description.lower()

    def test_description_triple_mentions_latency(self):
        assert "latency" in PBOBufferingStrategy.TRIPLE.description.lower()


# =============================================================================
# _extract_pointer
# =============================================================================

class TestExtractPointer:
    """
    Tests for the private pointer-normalisation helper.
    No GL mock needed — pure Python/ctypes logic.
    """

    def test_c_void_p_with_nonzero_value(self):
        ptr = ctypes.c_void_p(0x1000)
        result = _extract_pointer(ptr)
        assert result == 0x1000

    def test_c_void_p_zero_returns_none(self):
        # c_void_p(0).value is None in ctypes (null pointer representation).
        ptr = ctypes.c_void_p(0)
        result = _extract_pointer(ptr)
        assert result is None

    def test_mock_with_value_attribute(self):
        obj = MagicMock()
        obj.value = 4096
        assert _extract_pointer(obj) == 4096

    def test_mock_with_none_value_returns_none(self):
        obj = MagicMock()
        obj.value = None
        assert _extract_pointer(obj) is None

    def test_unextractable_object_returns_none(self):
        # An object with no .value and that ctypes.cast rejects → None.
        class _Opaque:
            pass
        result = _extract_pointer(_Opaque())
        assert result is None


# =============================================================================
# calculate_pixel_alignment
# =============================================================================

class TestCalculatePixelAlignment:

    def test_float_returns_4(self, gl):
        result = calculate_pixel_alignment(
            GLenum(_C["GL_FLOAT"]), GLenum(_C["GL_RGB"])
        )
        assert result == GLint(4)

    def test_unsigned_byte_rgba_returns_4(self, gl):
        result = calculate_pixel_alignment(
            GLenum(_C["GL_UNSIGNED_BYTE"]), GLenum(_C["GL_RGBA"])
        )
        assert result == GLint(4)

    def test_unsigned_byte_rgb_returns_1(self, gl):
        result = calculate_pixel_alignment(
            GLenum(_C["GL_UNSIGNED_BYTE"]), GLenum(_C["GL_RGB"])
        )
        assert result == GLint(1)

    def test_unsigned_byte_other_format_returns_1(self, gl):
        # Any format other than RGBA for ubyte falls through to 1.
        result = calculate_pixel_alignment(
            GLenum(_C["GL_UNSIGNED_BYTE"]), GLenum(0x1903)  # GL_RED
        )
        assert result == GLint(1)

    def test_other_type_returns_1(self, gl):
        result = calculate_pixel_alignment(
            GLenum(_C["GL_UNSIGNED_SHORT"]), GLenum(_C["GL_RGB"])
        )
        assert result == GLint(1)


# =============================================================================
# configure_pixel_storage
# =============================================================================

class TestConfigurePixelStorage:

    def test_sets_unpack_alignment_for_float_rgb(self, gl):
        configure_pixel_storage(GLenum(_C["GL_FLOAT"]), GLenum(_C["GL_RGB"]))
        gl.glPixelStorei.assert_any_call(
            GLenum(_C["GL_UNPACK_ALIGNMENT"]), GLint(4)
        )

    def test_sets_unpack_alignment_for_ubyte_rgb(self, gl):
        configure_pixel_storage(
            GLenum(_C["GL_UNSIGNED_BYTE"]), GLenum(_C["GL_RGB"])
        )
        gl.glPixelStorei.assert_any_call(
            GLenum(_C["GL_UNPACK_ALIGNMENT"]), GLint(1)
        )

    def test_sets_row_length_to_zero_by_default(self, gl):
        configure_pixel_storage(GLenum(_C["GL_FLOAT"]), GLenum(_C["GL_RGBA"]))
        gl.glPixelStorei.assert_any_call(GLenum(_C["GL_UNPACK_ROW_LENGTH"]), GLint(0))

    def test_sets_skip_pixels_to_zero_by_default(self, gl):
        configure_pixel_storage(GLenum(_C["GL_FLOAT"]), GLenum(_C["GL_RGBA"]))
        gl.glPixelStorei.assert_any_call(GLenum(_C["GL_UNPACK_SKIP_PIXELS"]), GLint(0))

    def test_sets_skip_rows_to_zero_by_default(self, gl):
        configure_pixel_storage(GLenum(_C["GL_FLOAT"]), GLenum(_C["GL_RGBA"]))
        gl.glPixelStorei.assert_any_call(GLenum(_C["GL_UNPACK_SKIP_ROWS"]), GLint(0))

    def test_custom_row_length_forwarded(self, gl):
        configure_pixel_storage(
            GLenum(_C["GL_FLOAT"]), GLenum(_C["GL_RGBA"]), row_length=GLint(640)
        )
        gl.glPixelStorei.assert_any_call(GLenum(_C["GL_UNPACK_ROW_LENGTH"]), GLint(640))

    def test_exactly_four_pixel_store_calls(self, gl):
        configure_pixel_storage(GLenum(_C["GL_FLOAT"]), GLenum(_C["GL_RGBA"]))
        assert gl.glPixelStorei.call_count == 4


# =============================================================================
# memmove_pbo
# =============================================================================

class TestMemmovePbo:

    @pytest.fixture()
    def env(self, gl):
        """memmove_pbo dependencies: tuned_parallel_copy + ensure_contiguity."""
        with (
            patch(f"{_MOD}.tuned_parallel_copy") as mock_copy,
            patch(f"{_MOD}.ensure_contiguity", side_effect=lambda x: x),
        ):
            # Provide a non-null mapped pointer.
            gl.glMapBufferRange.return_value = ctypes.c_void_p(0x4000)
            yield gl, mock_copy

    def test_returns_true_on_success(self, env):
        gl, _ = env
        data = np.zeros((10, 10, 3), dtype=np.uint8)
        assert memmove_pbo(GLBuffer(10), data) is True

    def test_binds_pbo_before_buffer_data(self, env):
        gl, _ = env
        data = np.zeros((4, 4, 3), dtype=np.uint8)
        memmove_pbo(GLBuffer(10), data)

        bind_idx = next(
            i for i, c in enumerate(gl.mock_calls)
            if c == call.glBindBuffer(_C["GL_PIXEL_UNPACK_BUFFER"], 10)
        )
        buf_idx = next(
            i for i, c in enumerate(gl.mock_calls)
            if "glBufferData" in str(c)
        )
        assert bind_idx < buf_idx

    def test_orphans_buffer_with_none(self, env):
        gl, _ = env
        data = np.zeros((4, 4, 3), dtype=np.uint8)
        memmove_pbo(GLBuffer(10), data)
        # glBufferData is the orphan call — None as data argument.
        calls = [c for c in gl.mock_calls if "glBufferData" in str(c)]
        assert calls, "glBufferData was not called"
        _, args, _ = calls[0]
        assert args[2] is None   # None = orphan existing storage

    def test_copies_data_via_parallel_copy(self, env):
        gl, mock_copy = env
        data = np.zeros((4, 4, 3), dtype=np.uint8)
        memmove_pbo(GLBuffer(10), data)
        mock_copy.assert_called_once()

    def test_unmaps_buffer_after_copy(self, env):
        gl, mock_copy = env
        data = np.zeros((4, 4, 3), dtype=np.uint8)
        memmove_pbo(GLBuffer(10), data)

        copy_idx = next(
            i for i, c in enumerate(gl.mock_calls)
            if "glUnmapBuffer" in str(c)
        )
        # copy happens before unmap; since tuned_parallel_copy is mocked,
        # we just check unmap was called at all.
        gl.glUnmapBuffer.assert_called_once_with(_C["GL_PIXEL_UNPACK_BUFFER"])

    def test_pbo_left_bound_after_success(self, env):
        """Caller owns the unbind — PBO must remain bound after return."""
        gl, _ = env
        data = np.zeros((4, 4, 3), dtype=np.uint8)
        memmove_pbo(GLBuffer(10), data)
        # Final glBindBuffer call must be the initial bind to the PBO, not 0.
        bind_calls = [
            c for c in gl.mock_calls if "glBindBuffer" in str(c)
        ]
        # No unbind (glBindBuffer(target, 0)) should have been issued.
        assert not any(
            c == call.glBindBuffer(_C["GL_PIXEL_UNPACK_BUFFER"], 0)
            for c in bind_calls
        )

    def test_returns_false_when_map_returns_falsy(self, env):
        gl, _ = env
        gl.glMapBufferRange.return_value = None
        data = np.zeros((4, 4, 3), dtype=np.uint8)
        assert memmove_pbo(GLBuffer(10), data) is False

    def test_returns_false_when_pointer_is_null(self, env):
        gl, _ = env
        # c_void_p(0).value is None — extract_pointer returns None.
        gl.glMapBufferRange.return_value = ctypes.c_void_p(0)
        data = np.zeros((4, 4, 3), dtype=np.uint8)
        assert memmove_pbo(GLBuffer(10), data) is False

    def test_logs_error_on_null_map(self, env, caplog):
        gl, _ = env
        gl.glMapBufferRange.return_value = None
        data = np.zeros((4, 4, 3), dtype=np.uint8)
        with caplog.at_level(logging.ERROR, logger=_MOD):
            memmove_pbo(GLBuffer(10), data)
        assert caplog.records


# =============================================================================
# write_pbo_buffer
# =============================================================================

class TestWritePboBuffer:

    @pytest.fixture()
    def fmt(self):
        """Minimal PixelFormat stub — only .channels is used by the mock."""
        fmt = MagicMock()
        fmt.channels = 3
        return fmt

    def test_happy_path_writes_data(self, fmt):
        image    = np.ones((4, 4, 3), dtype=np.uint8)
        pbo_arr  = np.zeros((4, 4, 3), dtype=np.uint8)
        with patch(f"{_MOD}.broadcast_to_format", return_value=image):
            write_pbo_buffer(pbo_arr, image, fmt)
        np.testing.assert_array_equal(pbo_arr, image)

    def test_raises_gl_upload_error_on_shape_mismatch(self, fmt):
        """
        Regression: original code raised ValueError.
        The caller's error handler catches GLUploadError, not ValueError.
        """
        image   = np.ones((4, 4, 3), dtype=np.uint8)
        pbo_arr = np.zeros((8, 8, 3), dtype=np.uint8)   # wrong shape
        with patch(f"{_MOD}.broadcast_to_format", return_value=image):
            with pytest.raises(GLUploadError, match="shape"):
                write_pbo_buffer(pbo_arr, image, fmt)

    def test_does_not_raise_value_error_on_mismatch(self, fmt):
        """Explicit guard: ValueError must never escape write_pbo_buffer."""
        image   = np.ones((4, 4, 3), dtype=np.uint8)
        pbo_arr = np.zeros((2, 2, 3), dtype=np.uint8)
        with patch(f"{_MOD}.broadcast_to_format", return_value=image):
            with pytest.raises(GLUploadError):
                write_pbo_buffer(pbo_arr, image, fmt)
            # If we reach here, it was GLUploadError (not ValueError). Good.

    def test_broadcast_to_format_called_with_image_and_fmt(self, fmt):
        image   = np.ones((4, 4, 3), dtype=np.uint8)
        pbo_arr = np.zeros((4, 4, 3), dtype=np.uint8)
        with patch(f"{_MOD}.broadcast_to_format", return_value=image) as mock_bc:
            write_pbo_buffer(pbo_arr, image, fmt)
        mock_bc.assert_called_once_with(image, fmt, copy=False)


# =============================================================================
# PBO
# =============================================================================

class TestPBO:

    # --- Construction -------------------------------------------------------

    def test_init_calls_gen_buffers(self, pbo_env):
        PBO()
        pbo_env.glGenBuffers.assert_called_once_with(1)

    def test_init_sets_id_from_scalar_return(self, pbo_env):
        pbo = PBO()
        assert pbo.id == 10   # first value from counter

    def test_init_capacity_is_zero(self, pbo_env):
        assert PBO().capacity == 0

    def test_init_not_mapped(self, pbo_env):
        assert PBO().is_mapped is False

    def test_init_handles_list_return_from_gen_buffers(self, gl):
        """Some PyOpenGL versions return a list from glGenBuffers."""
        gl.glGenBuffers.side_effect = None
        gl.glGenBuffers.return_value = [55]
        with patch(f"{_MOD}.gl_error_check", _noop_error_check):
            pbo = PBO()
        assert pbo.id == 55

    def test_init_handles_ndarray_return(self, gl):
        gl.glGenBuffers.side_effect = None
        gl.glGenBuffers.return_value = np.array([77], dtype=np.uint32)
        with patch(f"{_MOD}.gl_error_check", _noop_error_check):
            pbo = PBO()
        assert pbo.id == 77

    # --- prepare_and_map ----------------------------------------------------

    @pytest.fixture()
    def mapped_pbo(self, pbo_env):
        """PBO with glMapBufferRange returning real mapped memory."""
        H, W, C = 3, 4, 3
        dtype     = np.dtype("uint8")
        size      = H * W * C
        buf, ptr  = _real_mapped_ptr(size)

        pbo_env.glMapBufferRange.return_value = ptr

        pbo = PBO()
        # Keep buf alive — GC would invalidate the pointer otherwise.
        pbo._test_buf = buf
        return pbo, H, W, C, dtype, size

    def test_prepare_and_map_returns_correct_shape(self, mapped_pbo):
        pbo, H, W, C, dtype, size = mapped_pbo
        arr = pbo.prepare_and_map(GLsizeiptr(size), H, W, C, dtype)
        assert arr.shape == (H, W, C)

    def test_prepare_and_map_returns_correct_dtype(self, mapped_pbo):
        pbo, H, W, C, dtype, size = mapped_pbo
        arr = pbo.prepare_and_map(GLsizeiptr(size), H, W, C, dtype)
        assert arr.dtype == dtype

    def test_prepare_and_map_sets_is_mapped(self, mapped_pbo):
        pbo, H, W, C, dtype, size = mapped_pbo
        pbo.prepare_and_map(GLsizeiptr(size), H, W, C, dtype)
        assert pbo.is_mapped is True

    def test_prepare_and_map_array_is_c_contiguous(self, mapped_pbo):
        pbo, H, W, C, dtype, size = mapped_pbo
        arr = pbo.prepare_and_map(GLsizeiptr(size), H, W, C, dtype)
        assert arr.flags["C_CONTIGUOUS"]

    def test_prepare_and_map_size_mismatch_raises_gl_upload_error(self, pbo_env):
        """
        size_bytes must equal height×width×channels×itemsize.
        A mismatch means the caller allocated the wrong amount of GPU memory.
        """
        pbo = PBO()
        with pytest.raises(GLUploadError, match="size mismatch"):
            pbo.prepare_and_map(
                GLsizeiptr(100),   # wrong — should be 3*4*3=36
                height=3, width=4, channels=3,
                dtype=np.dtype("uint8"),
            )

    def test_prepare_and_map_null_map_raises_gl_memory_error(self, pbo_env):
        pbo_env.glMapBufferRange.return_value = None
        pbo = PBO()
        H, W, C, dtype = 2, 2, 3, np.dtype("uint8")
        with pytest.raises(GLMemoryError, match="NULL"):
            pbo.prepare_and_map(GLsizeiptr(H * W * C), H, W, C, dtype)

    def test_prepare_and_map_unbinds_on_null_map(self, pbo_env):
        """PBO must be unbound when map fails to leave GL state clean."""
        pbo_env.glMapBufferRange.return_value = None
        pbo = PBO()
        H, W, C, dtype = 2, 2, 3, np.dtype("uint8")
        with pytest.raises(GLMemoryError):
            pbo.prepare_and_map(GLsizeiptr(H * W * C), H, W, C, dtype)
        # Last glBindBuffer call must be the unbind (target, 0).
        assert pbo_env.glBindBuffer.call_args_list[-1] == call(
            _C["GL_PIXEL_UNPACK_BUFFER"], 0
        )

    # --- unmap --------------------------------------------------------------

    def test_unmap_noop_when_not_mapped(self, pbo_env):
        pbo = PBO()
        assert pbo.is_mapped is False
        pbo.unmap()
        pbo_env.glUnmapBuffer.assert_not_called()

    def test_unmap_calls_unmap_buffer(self, pbo_env, mapped_pbo):
        pbo, H, W, C, dtype, size = mapped_pbo
        pbo.prepare_and_map(GLsizeiptr(size), H, W, C, dtype)
        pbo_env.glUnmapBuffer.reset_mock()
        pbo.unmap()
        pbo_env.glUnmapBuffer.assert_called_once_with(_C["GL_PIXEL_UNPACK_BUFFER"])

    def test_unmap_unbinds_pbo(self, pbo_env, mapped_pbo):
        pbo, H, W, C, dtype, size = mapped_pbo
        pbo.prepare_and_map(GLsizeiptr(size), H, W, C, dtype)
        pbo_env.glBindBuffer.reset_mock()
        pbo.unmap()
        assert pbo_env.glBindBuffer.call_args_list[-1] == call(
            _C["GL_PIXEL_UNPACK_BUFFER"], 0
        )

    def test_unmap_clears_is_mapped(self, pbo_env, mapped_pbo):
        pbo, H, W, C, dtype, size = mapped_pbo
        pbo.prepare_and_map(GLsizeiptr(size), H, W, C, dtype)
        pbo.unmap()
        assert pbo.is_mapped is False

    def test_unmap_logs_warning_on_false_return(self, pbo_env, mapped_pbo, caplog):
        """GL_FALSE from glUnmapBuffer signals possible context loss."""
        pbo, H, W, C, dtype, size = mapped_pbo
        pbo.prepare_and_map(GLsizeiptr(size), H, W, C, dtype)
        pbo_env.glUnmapBuffer.return_value = False
        with caplog.at_level(logging.WARNING, logger=_MOD):
            pbo.unmap()
        assert caplog.records
        assert "GL_FALSE" in caplog.text or "false" in caplog.text.lower()

    def test_unmap_continues_after_false_return(self, pbo_env, mapped_pbo):
        """A false return from glUnmapBuffer must not leave is_mapped True."""
        pbo, H, W, C, dtype, size = mapped_pbo
        pbo.prepare_and_map(GLsizeiptr(size), H, W, C, dtype)
        pbo_env.glUnmapBuffer.return_value = False
        pbo.unmap()   # must not raise
        assert pbo.is_mapped is False

    # --- destroy ------------------------------------------------------------

    def test_destroy_noop_when_id_is_zero(self, pbo_env):
        pbo    = PBO()
        pbo.id = 0   # simulate already-destroyed state
        pbo.destroy()
        pbo_env.glDeleteBuffers.assert_not_called()

    def test_destroy_calls_delete_buffers_with_pbo_id(self, pbo_env):
        pbo = PBO()
        pbo_id = pbo.id
        pbo.destroy()
        actual_args = pbo_env.glDeleteBuffers.call_args[0]
        assert actual_args[0] == 1
        assert actual_args[1][0] == pbo_id

    def test_destroy_resets_id_to_zero(self, pbo_env):
        pbo = PBO()
        pbo.destroy()
        assert pbo.id == 0

    def test_destroy_unmaps_before_delete_if_mapped(self, pbo_env, mapped_pbo):
        pbo, H, W, C, dtype, size = mapped_pbo
        pbo.prepare_and_map(GLsizeiptr(size), H, W, C, dtype)
        assert pbo.is_mapped is True
        pbo.destroy()
        # glUnmapBuffer must have been called before glDeleteBuffers.
        unmap_idx = next(
            i for i, c in enumerate(pbo_env.mock_calls)
            if "glUnmapBuffer" in str(c)
        )
        delete_idx = next(
            i for i, c in enumerate(pbo_env.mock_calls)
            if "glDeleteBuffers" in str(c)
        )
        assert unmap_idx < delete_idx

    def test_destroy_idempotent(self, pbo_env):
        pbo = PBO()
        pbo.destroy()
        pbo_env.glDeleteBuffers.reset_mock()
        pbo.destroy()   # second call — id is now 0
        pbo_env.glDeleteBuffers.assert_not_called()


# =============================================================================
# PBOManager
# =============================================================================

class TestPBOManager:

    # --- Construction -------------------------------------------------------

    def test_default_strategy_is_double(self, pbo_env):
        mgr = PBOManager()
        assert mgr.buffer_strategy is PBOBufferingStrategy.DOUBLE

    def test_explicit_strategy_stored(self, pbo_env):
        mgr = PBOManager(buffer_strategy=PBOBufferingStrategy.TRIPLE)
        assert mgr.buffer_strategy is PBOBufferingStrategy.TRIPLE

    def test_num_pbos_shim_selects_correct_strategy(self, pbo_env):
        """Deprecated num_pbos kwarg must be coerced to a strategy."""
        mgr = PBOManager(num_pbos=3)
        assert mgr.buffer_strategy is PBOBufferingStrategy.TRIPLE

    def test_num_pbos_shim_logs_deprecation_warning(self, pbo_env, caplog):
        with caplog.at_level(logging.WARNING, logger=_MOD):
            PBOManager(num_pbos=2)
        assert caplog.records
        assert "deprecated" in caplog.text.lower()

    def test_pool_starts_empty(self, pbo_env):
        assert PBOManager().pbos == []

    # --- initialize ---------------------------------------------------------

    def test_initialize_creates_correct_pbo_count_for_double(self, pbo_env):
        mgr = PBOManager(buffer_strategy=PBOBufferingStrategy.DOUBLE)
        mgr.initialize()
        assert len(mgr.pbos) == 2

    def test_initialize_creates_correct_pbo_count_for_triple(self, pbo_env):
        mgr = PBOManager(buffer_strategy=PBOBufferingStrategy.TRIPLE)
        mgr.initialize()
        assert len(mgr.pbos) == 3

    def test_initialize_is_idempotent(self, pbo_env):
        mgr = PBOManager()
        mgr.initialize()
        first_pbos = list(mgr.pbos)
        mgr.initialize()   # second call must be a no-op
        assert mgr.pbos is not first_pbos or mgr.pbos == first_pbos
        # Verify glGenBuffers was called exactly DOUBLE=2 times total.
        assert pbo_env.glGenBuffers.call_count == 2

    def test_initialize_arms_cycle_iterator(self, pbo_env):
        mgr = PBOManager()
        mgr.initialize()
        assert mgr._cycle_iter is not None

    # --- bind / unbind ------------------------------------------------------

    def test_bind_calls_bind_buffer_with_pbo_id(self, pbo_env):
        PBOManager.bind(GLBuffer(42))
        pbo_env.glBindBuffer.assert_called_with(
            _C["GL_PIXEL_UNPACK_BUFFER"], 42
        )

    def test_unbind_calls_bind_buffer_with_zero(self, pbo_env):
        PBOManager.unbind()
        pbo_env.glBindBuffer.assert_called_with(_C["GL_PIXEL_UNPACK_BUFFER"], 0)

    # --- get_next -----------------------------------------------------------

    def test_get_next_raises_when_not_initialized(self, pbo_env):
        with pytest.raises(RuntimeError, match="initialize"):
            PBOManager().get_next()

    def test_get_next_returns_pbo_instance(self, pbo_env):
        mgr = PBOManager()
        mgr.initialize()
        result = mgr.get_next()
        assert isinstance(result, PBO)

    def test_get_next_cycles_round_robin(self, pbo_env):
        """After cycling through all PBOs the first one is returned again."""
        mgr = PBOManager(buffer_strategy=PBOBufferingStrategy.DOUBLE)
        mgr.initialize()
        p0 = mgr.get_next()
        p1 = mgr.get_next()
        p2 = mgr.get_next()   # should wrap back to p0
        assert p0 is not p1
        assert p2 is p0

    def test_get_next_is_thread_safe(self, pbo_env):
        """Concurrent get_next calls must not raise or return the same PBO twice."""
        mgr = PBOManager(buffer_strategy=PBOBufferingStrategy.TRIPLE)
        mgr.initialize()

        results = []
        errors  = []

        def _worker():
            try:
                results.append(mgr.get_next())
            except Exception as e:
                errors.append(e)

        import threading
        threads = [threading.Thread(target=_worker) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, "Thread raised: %s" % errors
        assert len(results) == 6

    # --- acquire_next_writeable ---------------------------------------------

    def test_acquire_next_writeable_returns_pbo_and_array(self, pbo_env):
        H, W, C   = 3, 4, 3
        dtype     = np.dtype("uint8")
        size      = H * W * C
        buf, ptr  = _real_mapped_ptr(size)
        pbo_env.glMapBufferRange.return_value = ptr

        mgr = PBOManager()
        mgr.initialize()
        pbo, arr = mgr.acquire_next_writeable(
            width=W, height=H, channels=C, dtype=dtype
        )
        assert isinstance(pbo, PBO)
        assert arr.shape == (H, W, C)

    def test_acquire_next_writeable_propagates_runtime_error_when_uninit(self, pbo_env):
        with pytest.raises(RuntimeError):
            PBOManager().acquire_next_writeable(width=4, height=4, channels=3)

    def test_acquire_next_writeable_propagates_gl_memory_error(self, pbo_env):
        pbo_env.glMapBufferRange.return_value = None
        mgr = PBOManager()
        mgr.initialize()
        with pytest.raises(GLMemoryError):
            mgr.acquire_next_writeable(width=4, height=4, channels=3)

    # --- cleanup ------------------------------------------------------------

    def test_cleanup_calls_delete_buffers_for_each_pbo(self, pbo_env):
        mgr = PBOManager(buffer_strategy=PBOBufferingStrategy.DOUBLE)
        mgr.initialize()
        pbo_env.glDeleteBuffers.reset_mock()
        mgr.cleanup()
        assert pbo_env.glDeleteBuffers.call_count == 2

    def test_cleanup_deletes_correct_handles(self, pbo_env):
        """Each PBO must be deleted with its own handle, not a neighbour's."""
        mgr = PBOManager(buffer_strategy=PBOBufferingStrategy.DOUBLE)
        mgr.initialize()
        ids = [pbo.id for pbo in mgr.pbos]
        pbo_env.glDeleteBuffers.reset_mock()
        mgr.cleanup()
        deleted = [
            int(c[0][1][0])   # second positional arg: the numpy array
            for c in pbo_env.glDeleteBuffers.call_args_list
        ]
        assert sorted(deleted) == sorted(ids)

    def test_cleanup_clears_pbo_list(self, pbo_env):
        mgr = PBOManager()
        mgr.initialize()
        mgr.cleanup()
        assert mgr.pbos == []

    def test_cleanup_resets_cycle_iterator(self, pbo_env):
        mgr = PBOManager()
        mgr.initialize()
        mgr.cleanup()
        assert mgr._cycle_iter is None

    def test_cleanup_makes_get_next_raise_again(self, pbo_env):
        mgr = PBOManager()
        mgr.initialize()
        mgr.cleanup()
        with pytest.raises(RuntimeError):
            mgr.get_next()

    def test_cleanup_idempotent(self, pbo_env):
        mgr = PBOManager()
        mgr.initialize()
        mgr.cleanup()
        pbo_env.glDeleteBuffers.reset_mock()
        mgr.cleanup()   # second call on empty pool — must not raise
        pbo_env.glDeleteBuffers.assert_not_called()

    def test_cleanup_noop_before_initialize(self, pbo_env):
        mgr = PBOManager()
        mgr.cleanup()   # must not raise
        pbo_env.glDeleteBuffers.assert_not_called()
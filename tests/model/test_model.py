"""
Comprehensive pytest suite for ImageDataModel.

Tests cover:
- Data lifecycle (set, get, clear)
- Read-only enforcement
- Caching behavior
- Coordinate access
- ROI extraction
- Edge cases and error handling
- Memory management
- Dtype preservation

Run with: pytest test_image_model.py -v
"""

import numpy as np
import pytest

from cross_platform.qt6_utils.image.model.cmap import (normalize_value_for_lut,
                                                       apply_colormap_to_value,
                                                       apply_colormap_to_region)
from cross_platform.qt6_utils.image.model.model import ImageDataModel
from cross_platform.qt6_utils.image.model.utils import (get_value_at_position,
                                                        get_roi)


@pytest.fixture
def empty_model():
    """Empty image model."""
    return ImageDataModel()


@pytest.fixture
def model_uint8():
    """Model with uint8 RGB image."""
    model = ImageDataModel()
    data = np.random.randint(0, 255, (100, 200, 3), dtype=np.uint8)
    model.set_data(data)
    return model


@pytest.fixture
def model_float32():
    """Model with float32 grayscale image."""
    model = ImageDataModel()
    data = np.random.rand(50, 50).astype(np.float32)
    model.set_data(data)
    return model


@pytest.fixture
def model_uint16():
    """Model with uint16 image."""
    model = ImageDataModel()
    data = np.random.randint(0, 65535, (64, 64), dtype=np.uint16)
    model.set_data(data)
    return model


@pytest.fixture
def simple_lut():
    """Simple grayscale LUT for testing (256x3)."""
    lut = np.linspace(0, 255, 256, dtype=np.uint8).reshape(-1, 1)
    return np.repeat(lut, 3, axis=1)


@pytest.fixture
def viridis_like_lut():
    """Viridis-like colormap LUT for testing (256x3)."""
    # Simplified viridis-like colors
    indices = np.linspace(0, 1, 256)
    r = np.clip(255 * (-0.5 + 1.5 * indices), 0, 255).astype(np.uint8)
    g = np.clip(255 * (0.5 * indices), 0, 255).astype(np.uint8)
    b = np.clip(255 * (1.0 - 0.5 * indices), 0, 255).astype(np.uint8)
    return np.stack([r, g, b], axis=1)


# ============================================================================
# Basic Functionality Tests
# ============================================================================

class TestBasicFunctionality:
    """Test core model functionality."""

    def test_empty_model_has_no_data(self, empty_model):
        """Empty model should report no data."""
        assert not empty_model.has_data()
        assert empty_model.get_data() is None
        assert empty_model.get_shape() is None
        assert empty_model.get_resolution() is None

    def test_set_data_stores_data(self, empty_model):
        """set_data should store data correctly."""
        data = np.ones((10, 20, 3), dtype=np.uint8)
        empty_model.set_data(data)

        assert empty_model.has_data()
        assert empty_model.get_shape() == (10, 20, 3)
        assert empty_model.get_resolution() == (20, 10)  # (width, height)

    def test_set_data_copies_by_default(self):
        """set_data should copy data by default."""
        model = ImageDataModel()
        original = np.ones((10, 10), dtype=np.uint8)
        model.set_data(original)

        # Modify original
        original[0, 0] = 255

        # Model data should be unchanged
        model_data = model.get_copy()
        assert model_data[0, 0] == 1

    def test_set_data_no_copy_ownership(self):
        """set_data(copy=False) should take ownership."""
        model = ImageDataModel()
        data = np.ones((10, 10), dtype=np.uint8)
        model.set_data(data, copy=False)

        assert model.has_data()
        # Model should have the data (though caller shouldn't modify original)

    def test_clear_removes_data(self, model_uint8):
        """clear() should remove all data."""
        assert model_uint8.has_data()
        model_uint8.clear()

        assert not model_uint8.has_data()
        assert model_uint8.get_data() is None


# ============================================================================
# Read-Only Enforcement Tests
# ============================================================================

class TestReadOnlyEnforcement:
    """Test that views are properly read-only."""

    def test_get_data_view_is_readonly(self, model_uint8):
        """get_data() should return read-only view."""
        view = model_uint8.get_data()

        assert not view.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            view[0, 0] = [255, 255, 255]

    def test_get_view_is_readonly(self, model_uint8):
        """get_view() should return read-only view."""
        view = model_uint8.get_view()

        assert not view.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            view[0, 0, 0] = 255

    def test_get_copy_is_writeable(self, model_uint8):
        """get_copy() should return writeable copy."""
        copy = model_uint8.get_copy()

        assert copy.flags.writeable
        # Should not raise
        copy[0, 0] = [255, 255, 255]

    def test_get_data_copy_is_writeable(self, model_uint8):
        """get_data(copy=True) should return writeable copy."""
        copy = model_uint8.get_data(copy=True)

        assert copy.flags.writeable
        copy[0, 0] = [255, 255, 255]  # Should not raise

    def test_copy_modification_doesnt_affect_model(self, model_uint8):
        """Modifying a copy should not affect model data."""
        original_value = model_uint8.get_copy()[0, 0].copy()

        copy = model_uint8.get_copy()
        copy[0, 0] = [255, 255, 255]

        # Model data should be unchanged
        model_value = model_uint8.get_copy()[0, 0]
        np.testing.assert_array_equal(model_value, original_value)


# ============================================================================
# Caching Tests
# ============================================================================

class TestCaching:
    """Test that caching works correctly."""

    def test_resolution_cached(self, model_uint8):
        """get_resolution() should be cached."""
        res1 = model_uint8.get_resolution()
        res2 = model_uint8.get_resolution()

        # Should return slice of shape, not the same object
        assert res1 == res2

    def test_shape_cached(self, model_uint8):
        """get_shape() should be cached."""
        shape1 = model_uint8.get_shape()
        shape2 = model_uint8.get_shape()

        assert shape1 is shape2

    def test_readonly_view_cached(self, model_uint8):
        """Read-only view should be cached."""
        view1 = model_uint8.get_data()
        view2 = model_uint8.get_data()

        # Should return same view object
        assert view1 is view2

    def test_cache_invalidated_on_set_data(self, model_uint8):
        """Cache should be invalidated when set_data is called."""
        old_res = model_uint8.get_resolution()

        # Set new data with different size
        new_data = np.ones((50, 100, 3), dtype=np.uint8)
        model_uint8.set_data(new_data)

        new_res = model_uint8.get_resolution()
        assert new_res != old_res
        assert new_res == (100, 50)

    def test_cache_invalidated_on_clear(self, model_uint8):
        """Cache should be invalidated when clear is called."""
        model_uint8.get_resolution()  # Populate cache
        model_uint8.clear()

        assert model_uint8.get_resolution() is None


# ============================================================================
# Dtype Preservation Tests
# ============================================================================

class TestDtypePreservation:
    """Test that dtypes are preserved correctly."""

    def test_uint8_preserved(self, model_uint8):
        """uint8 dtype should be preserved."""
        assert model_uint8.get_dtype() == np.uint8
        data = model_uint8.get_copy()
        assert data.dtype == np.uint8

    def test_float32_preserved(self, model_float32):
        """float32 dtype should be preserved."""
        assert model_float32.get_dtype() == np.float32
        data = model_float32.get_copy()
        assert data.dtype == np.float32

    def test_uint16_preserved(self, model_uint16):
        """uint16 dtype should be preserved."""
        assert model_uint16.get_dtype() == np.uint16
        data = model_uint16.get_copy()
        assert data.dtype == np.uint16

    def test_int32_preserved(self, empty_model):
        """int32 dtype should be preserved."""
        data = np.ones((10, 10), dtype=np.int32)
        empty_model.set_data(data)

        assert empty_model.get_dtype() == np.int32

    def test_float64_preserved(self, empty_model):
        """float64 dtype should be preserved."""
        data = np.ones((10, 10), dtype=np.float64)
        empty_model.set_data(data)

        assert empty_model.get_dtype() == np.float64


# ============================================================================
# Shape and Resolution Tests
# ============================================================================

class TestShapeAndResolution:
    """Test shape and resolution handling."""

    def test_resolution_width_height_order(self, empty_model):
        """get_resolution should return (width, height)."""
        data = np.ones((100, 200, 3), dtype=np.uint8)  # shape is (H, W, C)
        empty_model.set_data(data)

        width, height = empty_model.get_resolution()
        assert width == 200
        assert height == 100

    def test_grayscale_channels(self, model_float32):
        """Grayscale image should report 1 channel."""
        assert model_float32.get_channels() == 1

    def test_rgb_channels(self, model_uint8):
        """RGB image should report 3 channels."""
        assert model_uint8.get_channels() == 3

    def test_rgba_channels(self, empty_model):
        """RGBA image should report 4 channels."""
        data = np.ones((10, 10, 4), dtype=np.uint8)
        empty_model.set_data(data)
        assert empty_model.get_channels() == 4


# ============================================================================
# Coordinate Access Tests
# ============================================================================

class TestCoordinateAccess:
    """Test pixel value access at coordinates."""

    def test_get_value_at_valid_position(self, model_uint8):
        """Should get value at valid position."""
        value = model_uint8.get_value_at(10, 20)
        assert value is not None
        assert isinstance(value, np.ndarray)
        assert value.shape == (3,)  # RGB

    def test_get_value_at_out_of_bounds(self, model_uint8):
        """Should return None for out of bounds."""
        width, height = model_uint8.get_resolution()

        assert model_uint8.get_value_at(-1, 0) is None
        assert model_uint8.get_value_at(0, -1) is None
        assert model_uint8.get_value_at(width, 0) is None
        assert model_uint8.get_value_at(0, height) is None

    def test_get_value_at_flip_y(self, empty_model):
        """flip_y should flip Y coordinate."""
        data = np.arange(100).reshape(10, 10).astype(np.uint8)
        empty_model.set_data(data)

        # Without flip
        val_normal = empty_model.get_value_at(0, 0)

        # With flip (should get bottom-left instead of top-left)
        val_flipped = empty_model.get_value_at(0, 0, flip_y=True)

        assert val_normal != val_flipped

    def test_get_value_at_float_coords(self, model_uint8):
        """Should handle float coordinates."""
        value = model_uint8.get_value_at(10.7, 20.3)
        assert value is not None


# ============================================================================
# ROI Extraction Tests
# ============================================================================

class TestROIExtraction:
    """Test region of interest extraction."""

    def test_get_region_valid(self, model_uint8):
        """Should extract valid region."""
        roi = model_uint8.get_region(10, 20, 30, 40)

        assert roi is not None
        assert roi.shape == (40, 30, 3)  # (height, width, channels)

    def test_get_region_is_copy_by_default(self, model_uint8):
        """get_region should return writeable copy by default."""
        roi = model_uint8.get_region(0, 0, 10, 10)

        assert roi.flags.writeable
        roi[0, 0] = [255, 255, 255]  # Should not raise

    def test_get_region_view_is_readonly(self, model_uint8):
        """get_region(copy=False) should return read-only view."""
        roi = model_uint8.get_region(0, 0, 10, 10, copy=False)

        assert not roi.flags.writeable
        with pytest.raises(ValueError, match="read-only"):
            roi[0, 0] = [255, 255, 255]

    def test_get_region_clamping(self, model_uint8):
        """Should clamp region to valid bounds."""
        width, height = model_uint8.get_resolution()

        # Request region that extends beyond bounds
        roi = model_uint8.get_region(width - 10, height - 10, 50, 50)

        assert roi is not None
        assert roi.shape[0] <= 10  # Height clamped
        assert roi.shape[1] <= 10  # Width clamped

    def test_get_region_completely_out_of_bounds(self, model_uint8):
        """Should return None for completely invalid region."""
        width, height = model_uint8.get_resolution()

        roi = model_uint8.get_region(width + 100, 0, 10, 10)
        assert roi is None

    def test_get_region_zero_size(self, model_uint8):
        """Should return None for zero-size region."""
        roi = model_uint8.get_region(10, 10, 0, 10)
        assert roi is None

        roi = model_uint8.get_region(10, 10, 10, 0)
        assert roi is None


# ============================================================================
# Validation Tests
# ============================================================================

class TestValidation:
    """Test input validation."""

    def test_set_data_rejects_1d_array(self, empty_model):
        """Should reject 1D array."""
        data = np.ones(100, dtype=np.uint8)

        with pytest.raises(ValueError,):
            empty_model.set_data(data)

    def test_set_data_rejects_4d_array(self, empty_model):
        """Should reject 4D array."""
        data = np.ones((10, 10, 3, 2), dtype=np.uint8)

        with pytest.raises(ValueError):
            empty_model.set_data(data)

    def test_set_data_accepts_2d_array(self, empty_model):
        """Should accept 2D array (grayscale)."""
        data = np.ones((10, 10), dtype=np.uint8)
        empty_model.set_data(data)

        assert empty_model.has_data()

    def test_set_data_accepts_3d_array(self, empty_model):
        """Should accept 3D array (color)."""
        data = np.ones((10, 10, 3), dtype=np.uint8)
        empty_model.set_data(data)

        assert empty_model.has_data()


# ============================================================================
# Standalone Function Tests
# ============================================================================

class TestStandaloneFunctions:
    """Test standalone utility functions."""

    def test_get_value_at_position_direct(self):
        """Test get_value_at_position function directly."""
        data = np.arange(100).reshape(10, 10).astype(np.uint8)

        value = get_value_at_position(data, 5, 5)
        assert value == 55  # row 5, col 5

    def test_get_roi_direct(self):
        """Test get_roi function directly."""
        data = np.ones((100, 100, 3), dtype=np.uint8)

        roi = get_roi(data, 10, 20, 30, 40)
        assert roi.shape == (40, 30, 3)

    def test_get_roi_handles_grayscale(self):
        """get_roi should work with grayscale images."""
        data = np.ones((100, 100), dtype=np.uint8)

        roi = get_roi(data, 10, 10, 20, 20)
        assert roi.shape == (20, 20)

    def test_get_value_at_position_none_data(self):
        """Should handle None data gracefully."""
        value = get_value_at_position(None, 0, 0)
        assert value is None


# ============================================================================
# Colormap Utility Tests
# ============================================================================

class TestColormapUtilities:
    """Test colormap application utilities (separate from model)."""

    def test_normalize_uint8(self):
        """Should normalize uint8 to [0, 1]."""
        value = np.uint8(128)
        normalized = normalize_value_for_lut(value, np.dtype(np.uint8))

        assert 0.0 <= normalized <= 1.0
        assert np.isclose(normalized, 128 / 255.0)

    def test_normalize_uint16(self):
        """Should normalize uint16 to [0, 1]."""
        value = np.uint16(32768)
        normalized = normalize_value_for_lut(value, np.dtype(np.uint16))

        assert 0.0 <= normalized <= 1.0
        assert np.isclose(normalized, 32768 / 65535.0)

    def test_normalize_float32(self):
        """Should clamp float32 to [0, 1]."""
        value = np.float32(0.5)
        normalized = normalize_value_for_lut(value, np.dtype(np.float32))

        assert np.isclose(normalized, 0.5)

    def test_normalize_float32_clamps(self):
        """Should clamp out-of-range float values."""
        value = np.float32(1.5)
        normalized = normalize_value_for_lut(value, np.dtype(np.float32))

        assert normalized == 1.0

    def test_apply_colormap_to_value_uint8(self, simple_lut):
        """Should map uint8 value to RGB."""
        value = np.uint8(128)
        rgb = apply_colormap_to_value(value, simple_lut, np.dtype(np.uint8))

        assert rgb is not None
        assert rgb.shape == (3,)
        assert rgb.dtype == np.uint8
        # Middle value should map to middle of LUT
        assert np.allclose(rgb, [128, 128, 128], atol=2)

    def test_apply_colormap_to_value_uint16(self, simple_lut):
        """Should map uint16 value to RGB."""
        value = np.uint16(32768)
        rgb = apply_colormap_to_value(value, simple_lut, np.dtype(np.uint16))

        assert rgb is not None
        assert rgb.shape == (3,)
        # Middle value should map to middle of LUT
        assert np.allclose(rgb, [128, 128, 128], atol=2)

    def test_apply_colormap_to_value_float32(self, simple_lut):
        """Should map float32 value to RGB."""
        value = np.float32(0.5)
        rgb = apply_colormap_to_value(value, simple_lut, np.dtype(np.float32))

        assert rgb is not None
        assert np.allclose(rgb, [128, 128, 128], atol=2)

    def test_apply_colormap_to_value_array(self, simple_lut):
        """Should handle array input (uses first element)."""
        value = np.array([100, 200], dtype=np.uint8)
        rgb = apply_colormap_to_value(value, simple_lut, np.dtype(np.uint8))

        assert rgb is not None
        # Should use first element (100)
        expected_idx = int((100 / 255.0) * 255)
        assert np.allclose(rgb, [expected_idx, expected_idx, expected_idx],
                           atol=2)

    def test_apply_colormap_to_value_none(self, simple_lut):
        """Should handle None input."""
        rgb = apply_colormap_to_value(None, simple_lut, np.dtype(np.uint8))
        assert rgb is None

    def test_apply_colormap_to_region_grayscale(self, simple_lut):
        """Should apply colormap to grayscale region."""
        region = np.array([[0, 127, 255]], dtype=np.uint8)
        rgb_region = apply_colormap_to_region(region, simple_lut)

        assert rgb_region.shape == (1, 3, 3)
        assert rgb_region.dtype == np.uint8

        # Check mapping
        assert np.allclose(rgb_region[0, 0], [0, 0, 0], atol=2)
        assert np.allclose(rgb_region[0, 1], [127, 127, 127], atol=2)
        assert np.allclose(rgb_region[0, 2], [255, 255, 255], atol=2)

    def test_apply_colormap_to_region_multichannel(self, simple_lut):
        """Should use first channel of multi-channel region."""
        region = np.array([[[100, 200, 50]]], dtype=np.uint8)
        rgb_region = apply_colormap_to_region(region, simple_lut)
        assert rgb_region.shape == (1, 1, 3)
        # Should map using first channel (100)
        expected_idx = int(100 * 0.299 + 200 * 0.587 + 50 * 0.114)
        assert np.allclose(rgb_region[0, 0],
                           [expected_idx, expected_idx, expected_idx], atol=2)

    def test_apply_colormap_to_region_uint16(self, simple_lut):
        """Should apply colormap to uint16 region."""
        region = np.array([[0, 32768, 65535]], dtype=np.uint16)
        rgb_region = apply_colormap_to_region(region, simple_lut)

        assert rgb_region.shape == (1, 3, 3)
        assert rgb_region.dtype == np.uint8

    def test_apply_colormap_to_region_float32(self, simple_lut):
        """Should apply colormap to float32 region."""
        region = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)
        rgb_region = apply_colormap_to_region(region, simple_lut)

        assert rgb_region.shape == (1, 3, 3)
        assert np.allclose(rgb_region[0, 1], [128, 128, 128], atol=2)

    def test_apply_colormap_with_custom_lut(self, viridis_like_lut):
        """Should work with custom colormaps."""
        region = np.linspace(0, 255, 10, dtype=np.uint8).reshape(2, 5)
        rgb_region = apply_colormap_to_region(region, viridis_like_lut)

        assert rgb_region.shape == (2, 5, 3)
        assert rgb_region.dtype == np.uint8


class TestColormapIntegration:
    """Test colormap utilities integration with model."""

    def test_model_stays_pure_after_colormap(self, model_uint8, simple_lut):
        """Model data should remain unchanged after colormap operations."""
        original_data = model_uint8.get_copy()

        # Apply colormap to value
        value = model_uint8.get_value_at(10, 10)
        apply_colormap_to_value(value[0], simple_lut, model_uint8.get_dtype())

        # Model data should be unchanged
        current_data = model_uint8.get_copy()
        np.testing.assert_array_equal(original_data, current_data)

    def test_colormap_value_workflow(self, model_uint8, simple_lut):
        """Test typical workflow for colormapped value lookup."""
        # Get raw value from model
        raw_value = model_uint8.get_value_at(50, 50)
        assert raw_value is not None

        # Apply colormap separately
        rgb_value = apply_colormap_to_value(
            raw_value[0],  # Use first channel
            simple_lut,
            model_uint8.get_dtype()
        )

        assert rgb_value.shape == (3,)
        assert rgb_value.dtype == np.uint8

    def test_colormap_region_workflow(self, model_uint8, simple_lut):
        """Test typical workflow for colormapped region extraction."""
        # Get raw region from model
        raw_region = model_uint8.get_region(0, 0, 50, 50)
        assert raw_region is not None

        # Apply colormap separately
        rgb_region = apply_colormap_to_region(raw_region, simple_lut)

        assert rgb_region.shape == (50, 50, 3)
        assert rgb_region.dtype == np.uint8

    def test_colormap_with_different_dtypes(self, simple_lut):
        """Test colormap works with all supported dtypes."""
        dtypes = [np.uint8, np.uint16, np.float32, np.float64]

        for dtype in dtypes:
            model = ImageDataModel()
            if np.issubdtype(dtype, np.floating):
                data = np.random.rand(10, 10).astype(dtype)
            else:
                info = np.iinfo(dtype)
                data = np.random.randint(0, info.max // 2, (10, 10),
                                         dtype=dtype)

            model.set_data(data)

            # Test value
            raw_value = model.get_value_at(5, 5)
            rgb_value = apply_colormap_to_value(
                raw_value, simple_lut, model.get_dtype()
            )
            assert rgb_value is not None

            # Test region
            raw_region = model.get_region(0, 0, 5, 5)
            rgb_region = apply_colormap_to_region(raw_region, simple_lut)
            assert rgb_region.shape == (5, 5, 3)

    def test_colormap_empty_region(self, simple_lut):
        """Test colormap handles empty region."""
        empty_region = np.array([], dtype=np.uint8).reshape(0, 0)
        rgb_region = apply_colormap_to_region(empty_region, simple_lut)

        assert rgb_region.size == 0


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_array(self, empty_model):
        """Should handle empty arrays."""
        data = np.array([], dtype=np.uint8).reshape(0, 0)

        with pytest.raises(ValueError):
            empty_model.set_data(data)

    def test_single_pixel_image(self, empty_model):
        """Should handle single pixel image."""
        data = np.array([[255]], dtype=np.uint8)
        empty_model.set_data(data)

        assert empty_model.get_resolution() == (1, 1)
        value = empty_model.get_value_at(0, 0)
        assert value == 255

    def test_very_large_channels(self, empty_model):
        """Should handle images with many channels."""
        data = np.ones((10, 10, 10), dtype=np.uint8)
        empty_model.set_data(data)

        assert empty_model.get_channels() == 10

    def test_negative_coordinates(self, model_uint8):
        """Should handle negative coordinates."""
        assert model_uint8.get_value_at(-1, -1) is None
        assert model_uint8.get_region(-10, -10, 5, 5) is None


# ============================================================================
# Memory and Performance Tests
# ============================================================================

class TestMemoryAndPerformance:
    """Test memory efficiency and performance characteristics."""

    def test_view_shares_memory(self, model_uint8):
        """View should share memory with original data."""
        view = model_uint8.get_view()
        internal_data = model_uint8.get_copy()

        # View should point to same memory (though it's read-only)
        assert view.base is not None or np.shares_memory(view, internal_data)

    def test_copy_does_not_share_memory(self, model_uint8):
        """Copy should not share memory."""
        copy1 = model_uint8.get_copy()
        copy2 = model_uint8.get_copy()

        # Copies should be independent
        assert not np.shares_memory(copy1, copy2)

    def test_multiple_views_return_same_object(self, model_uint8):
        """Multiple view calls should return cached object."""
        view1 = model_uint8.get_view()
        view2 = model_uint8.get_view()

        assert view1 is view2  # Same object due to caching


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

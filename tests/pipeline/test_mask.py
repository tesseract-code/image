import pytest
import numpy as np

from cross_platform.qt6_utils.image.pipeline.operations.mask import compute_masked_stats, \
    correct_bad_pixels, compute_robust_window_levels, sanitize_float_buffer
from cross_platform.qt6_utils.image.pipeline.operations.transform import \
    calc_linear_transform_coefficients, _apply_colormap_pipeline, \
    _apply_raw_pipeline
from cross_platform.qt6_utils.image.pipeline.operations.stats import \
    ImageStats, compute_image_stats
from cross_platform.qt6_utils.image.pipeline.config import ProcessingConfig


# -----------------------------------------------------------------------------
# IMPORTS (Assuming implementation is in 'pipeline_ops.py')
# -----------------------------------------------------------------------------
# try:
#
# except ImportError:
#     pytest.fail(
#         "Could not import pipeline utilities. Ensure 'pipeline_ops.py' exists.")


# -----------------------------------------------------------------------------
# FIXTURES
# -----------------------------------------------------------------------------

@pytest.fixture
def gray_ramp_100x100():
    """Generates a 100x100 grayscale gradient (0 to 99)."""
    # Create a gradient 0..99
    row = np.linspace(0, 99, 100).astype(np.uint8)
    return np.tile(row, (100, 1))


@pytest.fixture
def hot_pixel_image():
    """Smooth image with one 'Hot' pixel (255) in the center."""
    img = np.full((10, 10), 50, dtype=np.uint8)
    img[5, 5] = 255  # The defect
    return img


@pytest.fixture
def bayer_aliasing_pattern():
    """
    Creates a pattern that exposes bad striding.
    Pattern: Alternating columns of 0 and 255.
    If you stride by an even number (2, 4, 8), you see ONLY 0 or ONLY 255.
    """
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:, ::2] = 255
    return img


@pytest.fixture
def rainbow_lut():
    """Creates a simple 256-entry RGB LUT (Gradient Red to Blue)."""
    lut = np.zeros((1, 256, 3), dtype=np.uint8)
    # 0 -> Red, 255 -> Blue
    for i in range(256):
        lut[0, i, 0] = 255 - i  # R
        lut[0, i, 1] = 0  # G
        lut[0, i, 2] = i  # B
    return lut


# -----------------------------------------------------------------------------
# PART 1: PIPELINE LOGIC TESTS
# -----------------------------------------------------------------------------

def test_stats_aliasing_protection(bayer_aliasing_pattern):
    """
    Scientific Check: Ensure statistical sampling uses an ODD stride
    to prevent locking onto a single phase of a Bayer/Striped pattern.
    """
    # 1. Even stride (Bad) - mimics simplistic implementation
    bad_view = bayer_aliasing_pattern[::2, ::2]
    # This will be all 255 or all 0 depending on start index
    is_biased = np.std(bad_view) == 0

    # 2. Pipeline Implementation (Should be smart/odd stride)
    stats = compute_image_stats(bayer_aliasing_pattern)

    # The actual image has mean 127.5.
    # If aliased, mean would be 0 or 255.
    assert 100 < stats.mean < 155, \
        f"Stats Aliasing Detected! Mean {stats.mean} indicates biased sampling."
    assert stats.std > 10, "Standard Deviation collapsed (aliasing)."


def test_transform_math_normalization():
    """
    Math Check: Test the y = mx + c logic for normalization.
    """
    config = ProcessingConfig(
        normalize=True,
        normalize_min=10.0,
        normalize_max=110.0,  # Range = 100
        gain=2.0,
        offset=0.5
    )
    # We pass dummy stats because explicit min/max is set in config
    dummy_stats = ImageStats(0, 0, 0, 0)

    scale, shift = calc_linear_transform_coefficients(config, dummy_stats)

    # Expected Math:
    # 1. Norm: Scale = 1/100 = 0.01. Shift = -10 * 0.01 = -0.1
    # 2. Gain: Scale = 0.01 * 2.0 = 0.02.
    # 3. Offset: Shift = (-0.1 * 2.0) + 0.5 = 0.3

    assert np.isclose(scale, 0.02)
    assert np.isclose(shift, 0.3)


def test_lut_pipeline_application(gray_ramp_100x100, rainbow_lut):
    """
    Integration: Verifies 1-channel -> 3-channel LUT mapping.
    """
    h, w = gray_ramp_100x100.shape
    output_buffer = np.zeros((h, w, 3), dtype=np.uint8)

    # -------------------------------------------------------------------------
    # CRITICAL FIX: The Pipeline Contract
    # -------------------------------------------------------------------------
    # The pipeline assumes 'scale' normalizes data to the 0.0-1.0 range.
    # Internally, it calculates: Index = (Input * Scale) * 255
    #
    # For Uint8 Input (0-255) to map 1:1 to LUT Indices (0-255):
    # We need: Input * Scale = Input / 255.0
    # Therefore: Scale must be 1.0 / 255.0
    # -------------------------------------------------------------------------
    scale_factor = 1.0 / 255.0

    _apply_colormap_pipeline(
        gray_ramp_100x100,
        output_buffer,
        rainbow_lut,
        scale=scale_factor,
        shift=0.0
    )

    # Check Column 0:
    # Input Value 0 -> Index 0.
    # LUT[0]: Red=255, Blue=0
    assert output_buffer[0, 0, 0] == 255, "Red channel at index 0 failed"
    assert output_buffer[0, 0, 2] == 0, "Blue channel at index 0 failed"

    # Check Column 99:
    # Input Value 99 -> Index 99.
    # LUT[99]: Red=(255-99)=156, Blue=99
    # If this fails with 255, the scale factor was too high (saturation).
    actual_blue = output_buffer[0, 99, 2]
    assert actual_blue == 99, f"Expected Blue=99, got {actual_blue}. Data Saturated?"

def test_raw_pipeline_inplace_math():
    """
    Engineering Check: Verify inplace operations and type safety.
    """
    # Input: Uint16 [100, 200]
    raw_img = np.array([[100, 200]], dtype=np.uint16)

    # Output: Float32 Buffer
    out_buffer = np.zeros((1, 2), dtype=np.float32)

    stats = ImageStats(100, 200, 150, 50)

    # Transform: value * 0.5 + 10.0
    # Expected: 100->60, 200->110
    _apply_raw_pipeline(raw_img, out_buffer, scale=0.5, shift=10.0, stats=stats)

    expected = np.array([[60.0, 110.0]], dtype=np.float32)
    np.testing.assert_allclose(out_buffer, expected)


# -----------------------------------------------------------------------------
# PART 2: MASKED UTILITIES TESTS
# -----------------------------------------------------------------------------

def test_masked_stats_accuracy(hot_pixel_image):
    """
    Check if masking correctly ignores outliers in stats calculation.
    """
    # 1. Unmasked Stats (Should include the 255 hot pixel)
    stats_unmasked = compute_masked_stats(hot_pixel_image, mask=None)
    assert stats_unmasked.max == 255.0

    # 2. Masked Stats (Mask out the hot pixel at 5,5)
    mask = np.full((10, 10), 255, dtype=np.uint8)  # 255 = Valid
    mask[5, 5] = 0  # 0 = Ignore

    stats_masked = compute_masked_stats(hot_pixel_image, mask=mask)

    assert stats_masked.max == 50.0  # Should ignore the 255
    assert stats_masked.valid_count == 99


def test_bad_pixel_correction(hot_pixel_image):
    """
    Check if hot pixel replacement works using local median.
    """
    # Create mask identifying the hot pixel
    bad_pixel_mask = np.zeros((10, 10), dtype=bool)
    bad_pixel_mask[5, 5] = True

    # Apply Correction
    corrected = correct_bad_pixels(hot_pixel_image, bad_pixel_mask, radius=1)

    # The pixel at [5,5] was 255. Neighbors are 50. Median should be 50.
    assert corrected[5, 5] == 50
    # Ensure neighbors weren't touched
    assert corrected[0, 0] == 50


def test_robust_windowing_levels():
    """
    Check histogram-based robust windowing (ignoring outliers).
    """
    # Create image: 1000 pixels of value 100, 5 pixels of value 255 (outliers)
    # Total pixels = 1005. 5 pixels is < 1%.
    data = np.concatenate([
        np.full(1000, 100, dtype=np.uint8),
        np.full(5, 255, dtype=np.uint8)
    ])
    # Reshape only works if size matches, let's just use 1D array (cv2 hist works on 1D too usually,
    # but let's make it 2D for strictness)
    img = np.zeros((1, 1005), dtype=np.uint8)
    img[0, :1000] = 100
    img[0, 1000:] = 255

    # Compute levels with 5% clipping
    # This should clip the top 5% of data. Since outliers are < 1%,
    # the max level should drop to ~100, not 255.
    rmin, rmax = compute_robust_window_levels(img, percentile=0.05)

    # We expect rmax to be close to 100.
    # Allowing slight binning error (256 bins).
    assert abs(
        rmax - 100) < 5, f"Robust max {rmax} did not exclude outliers (expected ~100)."


def test_sanitize_nan_inf():
    """
    Safety Check: Ensure NaNs don't crash the pipeline.
    """
    buffer = np.array([1.0, np.nan, np.inf, -np.inf, 5.0], dtype=np.float32)

    mask = sanitize_float_buffer(buffer, fill_value=0.0)

    # Check Values
    expected = np.array([1.0, 0.0, 0.0, 0.0, 5.0], dtype=np.float32)
    np.testing.assert_array_equal(buffer, expected)

    # Check Mask (True where data was bad)
    expected_mask = np.array([False, True, True, True, False])
    np.testing.assert_array_equal(mask, expected_mask)


# def test_crop_valid_roi():
#     """
#     Test ROI extraction based on mask.
#     """
#     from pipeline_ops import \
#         crop_to_valid_data  # Importing here to keep context clear
#
#     img = np.zeros((100, 100), dtype=np.uint8)
#     mask = np.zeros((100, 100), dtype=np.uint8)
#
#     # Define ROI: x=20, y=20, w=10, h=10
#     mask[20:30, 20:30] = 255
#
#     cropped, (x, y, w, h) = crop_to_valid_data(img, mask)
#
#     assert x == 20
#     assert y == 20
#     assert w == 10
#     assert h == 10
#     assert cropped.shape == (10, 10)
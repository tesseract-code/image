import time
import pytest
import numpy as np
from typing import TypeAlias

# Import the implementation
from cross_platform.qt6_utils.image.pipeline.operations.bayer import demosaic, mosaic

# Python 3.12 Type Aliases
ImageArray: TypeAlias = np.ndarray
BayerPattern: TypeAlias = str


# -----------------------------------------------------------------------------
# FIXTURES & UTILITIES
# -----------------------------------------------------------------------------

def calculate_psnr(img1: ImageArray, img2: ImageArray,
                   max_val: int = 255) -> float:
    """
    Calculates Peak Signal-to-Noise Ratio (PSNR) efficiently.
    Used to mathematically quantify reconstruction quality.
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0
    return 20 * np.log10(max_val / np.sqrt(mse))


@pytest.fixture
def synthetic_zone_plate() -> ImageArray:
    """
    Generates a Zone Plate (concentric circles getting tighter).
    This is the ultimate mathematical test for aliasing and 'zipper' artifacts.
    """
    size = 256
    x = np.linspace(-size / 2, size / 2, size)
    y = np.linspace(-size / 2, size / 2, size)
    X, Y = np.meshgrid(x, y)
    r = np.sqrt(X ** 2 + Y ** 2)

    # Create a frequency sweep (chirp signal)
    # This creates high frequencies at the edges
    raw = 127.5 + 127.5 * np.cos(0.1 * r ** 2)

    # Convert to RGB (Grayscale, so R=G=B)
    img = np.stack([raw, raw, raw], axis=-1).astype(np.uint8)
    return img


@pytest.fixture
def performance_tracker():
    """Context manager for high-precision Python 3.12 performance tracking."""

    class Tracker:
        def __init__(self):
            self.timings = {}

        def record(self, name: str, func, *args):
            # Warmup
            func(*args)

            # Measurement
            start = time.perf_counter_ns()
            # Run 10 times for average stability
            for _ in range(10):
                func(*args)
            end = time.perf_counter_ns()

            avg_ms = ((end - start) / 10) / 1e6
            self.timings[name] = avg_ms
            return avg_ms

    return Tracker()


# -----------------------------------------------------------------------------
# MATHEMATICAL CONSISTENCY TESTS
# -----------------------------------------------------------------------------

def test_conservation_of_energy(synthetic_zone_plate):
    """
    Math Check: The demosaicing process should not arbitrarily add or remove
    significant luminance energy from the scene.
    """
    original = synthetic_zone_plate

    # 1. Mosaic (Create Bayer)
    bayer = mosaic(original, pattern='RGGB')

    # 2. Demosaic (Reconstruct)
    reconstructed = demosaic(bayer, pattern='RGGB', alg_type='QUALITY')

    # 3. Check Mean Luminance
    # Allow small deviation due to interpolation logic at edges
    mean_orig = np.mean(original)
    mean_recon = np.mean(reconstructed)

    assert np.isclose(mean_orig, mean_recon, atol=2.0), \
        f"Luminance energy leaked! Orig: {mean_orig:.2f}, Recon: {mean_recon:.2f}"


def test_spectral_correlation_improvement(synthetic_zone_plate):
    """
    Algorithmic Quality Check:
    Malvar-He-Cutler (QUALITY) must produce a higher PSNR (better quality)
    than standard Bilinear (FAST/OpenCV) on high-frequency data.
    """
    original = synthetic_zone_plate
    bayer = mosaic(original, pattern='RGGB')

    # Run both algorithms
    res_fast = demosaic(bayer, 'RGGB', alg_type='FAST')
    res_qual = demosaic(bayer, 'RGGB', alg_type='QUALITY')

    # Calculate Quality metrics
    psnr_fast = calculate_psnr(original, res_fast)
    psnr_qual = calculate_psnr(original, res_qual)

    print(f"\n[Quality Metric] FAST (Bilinear): {psnr_fast:.2f} dB")
    print(f"[Quality Metric] QUALITY (MHC):     {psnr_qual:.2f} dB")

    # The assertion: The advanced math MUST be better than the simple math
    # Note: On a zone plate, the difference is usually distinct.
    assert psnr_qual > psnr_fast, \
        "Scientific Failure: The expensive algorithm is performing worse than the cheap one."


def test_spatial_padding_invariant():
    """
    Edge Case: The implementation must handle convolution padding correctly.
    If we input a 101x101 image, we must get a 101x101 image out.
    Naive convolutions often shrink images or crash on odd dimensions.
    """
    # Create odd-dimension random noise
    h, w = 101, 103
    raw = np.random.randint(0, 255, (h, w), dtype=np.uint8)

    result = demosaic(raw, pattern='BGGR', alg_type='QUALITY')

    assert result.shape == (h, w, 3), \
        f"Shape Mismatch! Input: {(h, w)}, Output: {result.shape}. Padding logic failed."


def test_bit_depth_consistency():
    """
    Data Type Check:
    - 8-bit input -> 8-bit output
    - 16-bit input -> 16-bit output
    """
    # 16-bit range check
    raw_16 = np.random.randint(0, 65535, (64, 64), dtype=np.uint16)

    res_16 = demosaic(raw_16, 'RGGB', alg_type='QUALITY')

    assert res_16.dtype == np.uint16
    # Ensure values weren't crushed to 0-255 range
    assert np.max(
        res_16) > 255, "16-bit data was incorrectly compressed to 8-bit!"


# -----------------------------------------------------------------------------
# EDGE CASES
# -----------------------------------------------------------------------------

@pytest.mark.parametrize("pattern", ['RGGB', 'BGGR', 'GRBG', 'GBRG'])
def test_pattern_phase_shifts(pattern):
    """
    Logic Check: Ensure the algorithm respects the Bayer phase shift.
    If we have a pure Red image, but tell the algorithm it is 'BGGR',
    it should result in Black (0) or artifacts, but definitely NOT Red.
    """
    # Create a pure Red image (in Bayer domain)
    # In RGGB, Red is at (0,0).
    shape = (10, 10)
    bayer = np.zeros(shape, dtype=np.uint8)
    bayer[0::2, 0::2] = 255  # Only populate the top-left phase

    # 1. Correct Pattern -> Should see Red
    if pattern == 'RGGB':
        res = demosaic(bayer, pattern, alg_type='QUALITY')
        # The Red channel should have high energy
        assert np.mean(
            res[..., 0]) > 50, "Failed to reconstruct Red from correct pattern"

    # 2. Incorrect Interpretation -> Should look wrong
    # If we treat an RGGB array as BGGR, the (0,0) pixel is interpreted as Blue.
    # So the Red channel should be empty/black.
    wrong_pattern_map = {'RGGB': 'BGGR'}  # Just test one case
    if pattern in wrong_pattern_map:
        res_wrong = demosaic(bayer, wrong_pattern_map[pattern],
                             alg_type='QUALITY')
        # Red channel should be near zero (interpolated from 0 neighbors)
        assert np.mean(res_wrong[..., 0]) < np.mean(res_wrong[..., 2]), \
            "Phase shift logic failure: Red pixel interpreted as Red despite wrong pattern mask."


def test_extreme_gradients():
    """
    Edge Case: 'Zipper' Artifact detection on a hard Step Function.
    This creates a sharp white box on a black background.
    """
    shape = (20, 20)
    raw = np.zeros(shape, dtype=np.uint8)
    raw[5:15, 5:15] = 255  # White box

    # We aren't asserting a specific value, but ensuring no NaN/Inf execution
    # and stability of the Laplacian kernel at sharp edges (overshoot check).
    res = demosaic(raw, 'RGGB', alg_type='QUALITY')

    # Malvar-He-Cutler can overshoot (ringing) slightly, but shouldn't explode.
    # Numpy/Cv2 usually clip wrap-around for uint8, or clip float.
    # Our implementation handles clipping.
    assert np.min(res) >= 0
    assert np.max(res) <= 255


# -----------------------------------------------------------------------------
# PERFORMANCE TRACKING
# -----------------------------------------------------------------------------

def test_performance_benchmarks(performance_tracker):
    """
    Performance: Measures the ms/frame for FAST vs QUALITY.
    Expects FAST to be ~50-100x faster.
    """
    # 1080p frame
    h, w = 1080, 1920
    raw = np.random.randint(0, 255, (h, w), dtype=np.uint8)

    t_fast = performance_tracker.record("OpenCV (FAST)", demosaic, raw, 'RGGB',
                                        'FAST')
    t_qual = performance_tracker.record("MHC (QUALITY)", demosaic, raw, 'RGGB',
                                        'QUALITY')

    print(f"\n[Benchmark 1080p]")
    print(f"FAST Path:    {t_fast:.3f} ms")
    print(f"QUALITY Path: {t_qual:.3f} ms")
    print(f"Slowdown Factor: {t_qual / t_fast:.1f}x")

    # Performance assertion: FAST must be suitable for realtime (>30fps => <33ms)
    assert t_fast < 33.0, f"Critical: FAST path is too slow for realtime! ({t_fast}ms)"

    # Just a sanity check that Quality isn't completely broken (e.g. > 2 seconds)
    assert t_qual < 2000.0, "QUALITY path is unreasonably slow."
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Tuple, Optional, Union, TypeAlias

# Type definition for clarity
MaskType: TypeAlias = np.ndarray  # Boolean or Uint8 array
ImageArray: TypeAlias = np.ndarray


@dataclass(slots=True)
class MaskedStats:
    min: float
    max: float
    mean: float
    std: float
    valid_count: int
    total_count: int


# -----------------------------------------------------------------------------
# 1. FAST STATISTICS (Avoiding numpy.ma overhead)
# -----------------------------------------------------------------------------

def compute_masked_stats(image: ImageArray,
                         mask: Optional[MaskType] = None) -> MaskedStats:
    """
    Computes image statistics ignoring masked-out pixels.

    ENGINEERING NOTE:
    Native `numpy.ma.mean()` is slow because it performs Python-level
    checks. This implementation uses `cv2.meanStdDev` and `cv2.minMaxLoc`
    which run in C++ with SIMD/AVX optimizations.

    Parameters
    ----------
    image : ndarray
        Input image (H, W) or (H, W, C).
    mask : ndarray, optional
        Uint8 mask (0=Ignore, 255=Keep).
        If None, computes on whole image.
    """
    if mask is None:
        # Fast path for unmasked
        dmin, dmax, _, _ = cv2.minMaxLoc(image)
        mean, std = cv2.meanStdDev(image)
        count = image.size
        return MaskedStats(dmin, dmax, float(mean[0]), float(std[0]), count,
                           count)

    # Ensure mask is uint8 for OpenCV
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255

    # 1. Mean / Std (O(1) in C++)
    # Returns tuple of scalars
    mean_val, std_val = cv2.meanStdDev(image, mask=mask)

    # 2. Min / Max (O(1) in C++)
    # cv2.minMaxLoc supports masks
    dmin, dmax, _, _ = cv2.minMaxLoc(image, mask=mask)

    # 3. Valid Pixel Count
    # cv2.countNonZero is significantly faster than np.count_nonzero
    valid_count = cv2.countNonZero(mask)

    return MaskedStats(
        min=dmin,
        max=dmax,
        mean=float(mean_val[0]),
        std=float(std_val[0]),
        valid_count=valid_count,
        total_count=image.size
    )


# -----------------------------------------------------------------------------
# 2. DEFECT CORRECTION (Interpolation)
# -----------------------------------------------------------------------------

def correct_bad_pixels(image: ImageArray,
                       bad_pixel_mask: MaskType,
                       radius: int = 1) -> ImageArray:
    """
    Replaces bad pixels (Hot/Dead) with the median of their local neighbors.

    SCIENTIFIC METHOD:
    Instead of using heavy inpainting algorithms (Navier-Stokes),
    we use a masked replacement strategy.
    1. Calculate Median Blur of the entire image.
    2. Where mask indicates 'Bad Pixel', substitute Original with Median.
    3. Keep Original everywhere else.

    This preserves scientific integrity of valid data while visually
    fixing defects for the viewer.
    """
    # Ensure boolean mask for indexing
    if bad_pixel_mask.dtype != bool:
        # Assuming input is Bad=255 or Bad=True.
        # Adjust logic if Mask means "Good Pixels"
        is_bad = bad_pixel_mask.astype(bool)
    else:
        is_bad = bad_pixel_mask

    if not np.any(is_bad):
        return image

    # Calculate kernel size (must be odd, e.g., radius 1 -> 3x3 kernel)
    ksize = (radius * 2) + 1

    # 1. Generate candidate replacements
    # Median is robust against outliers (other nearby bad pixels)
    corrected = cv2.medianBlur(image, ksize)

    # 2. Sparse substitution
    # Out = Image * (1-Mask) + Corrected * Mask
    # Using np.where is faster than arithmetic for sparse updates
    output = image.copy()
    output[is_bad] = corrected[is_bad]

    return output


# -----------------------------------------------------------------------------
# 3. DYNAMIC RANGE OPTIMIZATION (Auto-Windowing)
# -----------------------------------------------------------------------------

def compute_robust_window_levels(image: ImageArray,
                                 mask: Optional[MaskType] = None,
                                 percentile: float = 0.05) -> Tuple[
    float, float]:
    """
    Calculates robust Min/Max levels for display normalization,
    ignoring outliers and masked regions.

    Algorithm:
    1. Compute Histogram (masked).
    2. Find cumulative distribution points (e.g., 5% and 95%).

    This prevents a single hot pixel from ruining the contrast of the
    entire image stream.
    """
    # Parameters for Histogram
    # If integer, use discrete bins. If float, use defined range.
    is_float = image.dtype.kind == 'f'

    if is_float:
        # Float images (0.0-1.0 or unbounded)
        # We need a range. Let's sample min/max quickly first.
        dmin, dmax, _, _ = cv2.minMaxLoc(image,
                                         mask=mask if mask is not None else None)
        rng = dmax - dmin
        if rng < 1e-6:
            return dmin, dmax
        hist_range = [dmin, dmax]
        hist_size = 1000  # Precision binning
    else:
        # Uint8/Uint16
        max_val = 65536 if image.dtype == np.uint16 else 256
        hist_range = [0, max_val]
        hist_size = 256 if max_val == 256 else 4096  # Bin reduction for speed on uint16

    # Convert mask to CV2 format if present
    cv_mask = None
    if mask is not None:
        cv_mask = mask.astype(np.uint8) * 255 if mask.dtype == bool else mask

    # 1. Compute Histogram (CPU optimized in OpenCV)
    hist = cv2.calcHist([image], [0], cv_mask, [hist_size], hist_range)

    # 2. Cumulative Distribution
    cumsum = hist.cumsum()
    total_pixels = cumsum[-1]

    if total_pixels == 0:
        return 0.0, 1.0

    # 3. Find thresholds
    lower_thresh = total_pixels * percentile
    upper_thresh = total_pixels * (1.0 - percentile)

    # np.searchsorted is binary search (O(log N))
    lower_idx = np.searchsorted(cumsum, lower_thresh)
    upper_idx = np.searchsorted(cumsum, upper_thresh)

    # Map index back to value space
    step = (hist_range[1] - hist_range[0]) / hist_size

    robust_min = hist_range[0] + (lower_idx * step)
    robust_max = hist_range[0] + (upper_idx * step)

    return float(robust_min), float(robust_max)


# -----------------------------------------------------------------------------
# 4. NAN/INF SANITIZATION (Engineering Safety)
# -----------------------------------------------------------------------------

def sanitize_float_buffer(buffer: ImageArray,
                          fill_value: float = 0.0) -> MaskType:
    """
    Checks a float buffer for NaN/Inf, replaces them, and returns
    a mask of where the bad values were.

    Essential for pipeline stability before sending data to OpenGL
    (which may render black artifacts or crash on NaNs).
    """
    if buffer.dtype.kind != 'f':
        return np.zeros(buffer.shape, dtype=bool)

    # Fast boolean check
    bad_mask = ~np.isfinite(buffer)

    if np.any(bad_mask):
        buffer[bad_mask] = fill_value
        return bad_mask

    return bad_mask


# -----------------------------------------------------------------------------
# 5. ROI EXTRACTION (Bounding Box)
# -----------------------------------------------------------------------------

def crop_to_valid_data(image: ImageArray, mask: MaskType) -> Tuple[
    ImageArray, Tuple[int, int, int, int]]:
    """
    Crops the image to the bounding box of the valid mask region.
    Returns (CroppedImage, (x, y, w, h)).
    """
    if mask.dtype == bool:
        mask = mask.astype(np.uint8) * 255

    # cv2.boundingRect is O(N) but optimized
    x, y, w, h = cv2.boundingRect(mask)

    if w == 0 or h == 0:
        return image, (0, 0, image.shape[1], image.shape[0])

    crop = image[y:y + h, x:x + w]
    return crop, (x, y, w, h)
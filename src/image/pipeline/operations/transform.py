import logging
import sys
from typing import Tuple, Final, Optional, NamedTuple

import cv2
import numpy as np

from image.pipeline.config import ProcessingConfig
from image.utils.types import is_image

logger = logging.getLogger(__name__)

type ImageBuffer = np.ndarray
type ColorMap = np.ndarray
type Coefficients = Tuple[float, float]

# --- Constants ---
EPSILON: Final[float] = sys.float_info.epsilon
LUT_SHAPE_TARGET: Final[Tuple[int, int, int]] = (1, 256, 3)


class _SampledStats(NamedTuple):
    """
    Subsampled pixel statistics computed during pipeline processing.

    Private to this module. Use FrameStats for anything that crosses a
    module boundary or gets stored / serialised.
    """
    min: float
    max: float
    mean: float
    std: float


def sample_image_stats(image: np.ndarray, step: int = 9) -> _SampledStats:
    """
    Compute statistics on a strided view of the image.

    The default step is 9 (odd) intentionally.  An even stride on a Bayer
    (2×2 RGGB) sensor would repeatedly land on the same colour channel,
    producing aliased statistics.  An odd stride distributes samples across
    R, G, and B pixels.
    """
    view = image[::step, ::step]

    if view.size == 0:
        return _SampledStats(0.0, 0.0, 0.0, 0.0)

    return _SampledStats(
        min=float(np.min(view)),
        max=float(np.max(view)),
        mean=float(np.mean(view, dtype=np.float64)),
        std=float(np.std(view, dtype=np.float64)),
    )


def calc_linear_coeffs(config: ProcessingConfig,
                       stats: _SampledStats) -> Coefficients:
    """
    Pure function: Calculates linear transformation coefficients (y = mx + c).
    """
    scale = 1.0
    shift = 0.0

    if config.normalize:
        # Sanitize stats (NaN protection)
        src_min = np.nan_to_num(stats.min, nan=0.0)
        src_max = np.nan_to_num(stats.max, nan=1.0)

        tgt_min = config.normalize_min if config.normalize_min is not None else src_min
        tgt_max = config.normalize_max if config.normalize_max is not None else src_max

        src_rng = src_max - src_min
        tgt_rng = tgt_max - tgt_min

        if abs(src_rng) > EPSILON:
            factor = tgt_rng / src_rng
            scale *= factor
            shift = tgt_min - (factor * src_min)
        else:
            # Flat image fallback: Center the values
            scale = 1.0
            shift = tgt_min - src_min

    # Apply Gain/Offset
    final_scale = scale * config.gain
    final_shift = (shift * config.gain) + config.offset

    return float(final_scale), float(final_shift)


def transform_to_float(
        image: ImageBuffer,
        output_buffer: ImageBuffer,
        scale: float,
        shift: float
) -> None:
    """
    Analysis Path: Raw -> Float32

    Responsibility:
    Safely casts input to float and applies linear transform in-place.
    Optimized for numerical analysis (stats, histograms, masking).
    """
    # 1. Fault Tolerance: Shape mismatch
    if output_buffer.shape != image.shape:
        output_buffer.resize(image.shape, refcheck=False)

    # 2. Optimized Cast (Raw -> Output Buffer)
    np.copyto(output_buffer, image, casting='unsafe')

    # 3. In-Place Math
    # Skip identity transforms to save cycles
    if abs(scale - 1.0) > EPSILON:
        np.multiply(output_buffer, scale, out=output_buffer)

    if abs(shift) > EPSILON:
        np.add(output_buffer, shift, out=output_buffer)


def transform_to_visual_indices(
        image: ImageBuffer,
        scale: float,
        shift: float
) -> ImageBuffer:
    """
    Visualization Path Part 1: Raw -> Uint8 Indices

    Responsibility:
    Converts raw data into 0-255 indices for LUT consumption.

    Optimization Note:
    We use `cv2.convertScaleAbs` which fuses (Input * Scale + Shift) + Clipping + Casting.
    This is significantly faster than doing float math then casting to int.
    """
    # Ensure we are working with a single channel (Luminance)
    # 1. Prepare Input (Gray)
    if image.ndim == 3 and image.shape[2] == 3:
        src_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        src_gray = image

    # 2. Generate Indices [0-255]
    # alpha matches the pipeline contract: Input * Scale -> 0..1 range -> * 255 -> Indices
    return cv2.convertScaleAbs(src_gray, alpha=scale * 255.0,
                               beta=shift * 255.0)


def apply_lut(
        indices: np.ndarray,
        lut: np.ndarray,
        output_buffer: np.ndarray
) -> None:
    """
    Robustly applies a Color LUT to an input image/index map.

    Logic:
    1. Data -> Normalize to 0-255 Uint8.
    2. Channels -> Reduce to 1-Channel Grayscale (Intensity).
    3. Expansion -> Expand to 3-Channel RGB (Duplicated).
    4. Mapping -> Apply LUT.
    """

    # --- 1. Sanitize LUT (Strict Shape & Type) ---
    # OpenCV LUT requires exact (1, 256, 3) shape and contiguous memory
    target_shape = (1, 256, 3)
    if lut.shape != target_shape:
        if lut.ndim == 2 and lut.shape == (256, 3):
            lut = lut.reshape(target_shape)
        else:
            # Fallback for weird shapes: resize to linear 256
            lut = cv2.resize(lut, (1, 256),
                             interpolation=cv2.INTER_NEAREST).reshape(
                target_shape)

    if not lut.flags['C_CONTIGUOUS']:
        lut = np.ascontiguousarray(lut)
    if lut.dtype != np.uint8:
        lut = lut.astype(np.uint8)

    # --- 2. Sanitize Input Data (Float vs Int) ---
    # We use a temporary variable 'src' to avoid modifying 'indices' in-place if it's not ours

    # CASE A: Float Input (0.0 to 1.0)
    # Common in generated patterns or normalized data.
    if indices.dtype.kind == 'f':
        # Scale 0.0-1.0 -> 0-255.
        # We must cast to uint8 NOW, otherwise cvtColor math will be wrong later.
        src = (indices * 255).astype(np.uint8)

    # CASE B: Integer Input (Already 0-255)
    else:
        if indices.dtype != np.uint8:
            src = cv2.convertScaleAbs(
                indices)  # Handles int32/int64 -> uint8 safely
        else:
            src = indices

    # --- 3. Sanitize Channels (3 vs 1) ---
    # We need a 1-channel index map for the LUT.

    # CASE A: Multichannel (e.g., RGB Image)
    # Standard Behavior: Calculate Perceived Brightness (Luma)
    if src.ndim == 3 and src.shape[2] == 3:
        # Note: If input is a mask where R=G=B (e.g. 100,100,100),
        # RGB2GRAY results in 100. It is safe for both Masks and Images.
        src_gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

    # CASE B: Single Channel (Gray or (H,W,1))
    elif src.ndim == 3 and src.shape[2] == 1:
        src_gray = src[:, :, 0]
    else:
        src_gray = src

    # --- 4. Expand to RGB Destination ---
    # We populate the output buffer with the indices: [idx, idx, idx]
    # This prepares the buffer for the channel-independent lookup.

    # Safety Check: If output_buffer and src_gray share memory (rare but possible),
    # cvtColor handles it, but we strictly ensure dst shape fits.
    cv2.cvtColor(src_gray, cv2.COLOR_GRAY2RGB, dst=output_buffer)

    # --- 5. Apply LUT ---
    # LUT( [idx, idx, idx] ) -> [ LUT(idx)_R, LUT(idx)_G, LUT(idx)_B ]
    cv2.LUT(output_buffer, lut, dst=output_buffer)


def apply_transformations(
        image: ImageBuffer,
        output_buffer: ImageBuffer,
        config: ProcessingConfig,
        stats: _SampledStats,
        lut: Optional[ColorMap] = None
) -> Tuple[float, float] | None:
    """
    The Main Entry Point.

    Renamed from `apply_transforms` to `process_pipeline_stage` to indicate
    it executes a specific stage of the pipeline logic.

    Logic:
    - If `lut` is provided: Execute Visualization Path (Raw -> Index -> RGB).
    - If `lut` is None: Execute Analysis Path (Raw -> Float).
    """
    if not is_image(image):
        logger.debug("Transformations not applied to array")
        return

    try:
        logger.debug("Calculating linear transform coefficients...")
        scale, shift = calc_linear_coeffs(config, stats)
        if lut is not None:
            logger.debug("Applying color map lut...")
            # --- Visualization Path ---
            # 1. Transform (Raw -> 8-bit Index)
            # We allocate a small intermediate 8-bit buffer here.
            # In Python, this is cheaper than reusing a huge Float32 buffer.
            index_map = transform_to_visual_indices(image, scale, shift)

            # 2. Colorize (8-bit Index -> RGB)
            apply_lut(index_map, lut, output_buffer)
            logger.debug("Color map lut applied")
            final_min, final_max = 0.0, 1.0
            return final_min, final_max

        else:
            logger.debug(f"Casting image from {image.dtype.name} to Float32...")
            # Transform (Raw -> Float32)
            transform_to_float(image, output_buffer, scale, shift)
            final_min = (stats.min * scale) + shift
            final_max = (stats.max * scale) + shift
            return final_min, final_max

    except cv2.error as e:
        logger.error(f"CV2 Pipeline Error: {e}")
    except Exception as e:
        logger.exception(f"Pipeline Processing Error: {e}")

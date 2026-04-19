import cv2
import numpy as np

from cross_platform.qt6_utils.image.pipeline.operations.transform import EPSILON
from cross_platform.qt6_utils.image.settings.fmt import PixelLayout


def normalize(X: np.ndarray, fmt: PixelLayout = PixelLayout.RGB,
              vmin: float = 0.0, vmax: float = 1.0) -> np.ndarray:
    """
    Prepare Image for OpenGL with normalized float values.
    Mimics matplotlib.pyplot.imshow behavior (contrast stretching).

    Args:
        X: Input image data (assumed BGR if color, or Grayscale)
        fmt: Target ImageFormat enum
        vmin: Value to map to 0.0
        vmax: Value to map to 1.0

    Returns:
        np.ndarray: Float32 array with values 0.0-1.0
    """
    # 0. Validate Inputs
    if X is None:
        raise ValueError("Input image X is None")

    # Block YUV formats
    if fmt in [PixelLayout.YUV420, PixelLayout.YUV422, PixelLayout.YUV444,
               PixelLayout.NV12, PixelLayout.NV21]:
        raise NotImplementedError(
            f"Format {fmt.name} is not supported in this utility.")

    # Create a local copy to avoid modifying the original array
    display_frame = X.copy()

    # 1. Handle Color Format Conversion
    # Assumption: Input X is standard OpenCV BGR (3 channel) or Grayscale (2 dim)

    # Handle Grayscale input being converted to Color
    if len(display_frame.shape) == 2 and fmt not in [PixelLayout.MONOCHROME]:
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_GRAY2BGR)

    if fmt == PixelLayout.RGB:
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

    elif fmt == PixelLayout.RGBA:
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)

    elif fmt == PixelLayout.BGRA:
        display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2BGRA)

    elif fmt == PixelLayout.MONOCHROME:
        if len(display_frame.shape) == 3:
            display_frame = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)

    elif fmt == PixelLayout.RG:
        # Construct RG by taking the R (index 2 in BGR) and G (index 1 in BGR) channels
        # Note: output is 2-channel
        if len(display_frame.shape) == 3:
            display_frame = display_frame[:, :, [2, 1]]

    # If format == 'BGR', keep as is

    # 2. Convert to float32
    display_frame = display_frame.astype(np.float32)

    # 3. Base Normalization (Integer to Float 0-1)
    # If input was integer type (common for images), scale to 0-1 first
    if X.dtype.kind in 'iu':  # int or uint
        # For 8-bit, div by 255. For 16-bit, div by 65535
        max_val = np.iinfo(X.dtype).max
        display_frame /= max_val

    # 4. Apply Imshow-style Contrast Stretching (vmin/vmax)
    # Formula: result = (pixel - vmin) / (vmax - vmin)
    if vmin != 0.0 or vmax != 1.0:
        denom = vmax - vmin
        # Avoid division by zero
        if abs(denom) < EPSILON:
            denom = EPSILON
        display_frame = (display_frame - vmin) / denom

    # 5. Final Clamp and Layout
    display_frame = np.clip(display_frame, 0.0, 1.0)
    display_frame = np.ascontiguousarray(display_frame)

    return display_frame
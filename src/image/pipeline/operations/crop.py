from typing import Tuple, Optional

import numpy as np


def is_valid_roi(
        roi: Tuple[int, int, int, int],
        image_shape: Tuple[int, ...]) -> bool:
    """
    Fast check if this ROI fits strictly within the given image dimensions.
    Assumes image_shape is (Height, Width, ...) following Numpy convention.
    """
    img_h, img_w = image_shape[:2]

    x, y, width, height = roi

    # 1. Non-zero area check
    if width <= 0 or height <= 0:
        return False

    # 2. Bounds check
    if x < 0 or y < 0:
        return False

    # 3. Overflow check
    if (x + width) > img_w:
        return False
    if (y + height) > img_h:
        return False

    return True


def get_roi_slice(roi: Tuple[int, int, int, int]) -> Tuple[slice, slice]:
        """Returns the numpy slices for y and x."""
        x, y, width, height = roi
        return slice(y, y + height), slice(x, x + width)


def apply_crop(
        image: np.ndarray,
        roi: Optional[Tuple[int, int, int, int]],
        output_buffer: Optional[np.ndarray] = None
) -> Optional[np.ndarray]:
    """
    Crops the image based on the ROI.

    Args:
        image: Source image (H, W, C) or (H, W).
        roi: The ROI definition. If None, returns full image or copy.
        output_buffer: Optional pre-allocated buffer to write into.
                       If provided, data is COPIED.
                       If None, a VIEW (zero-copy) is returned.

    Returns:
        The cropped numpy array (view or filled buffer), or None if ROI is invalid.
    """
    # 1. Handle No ROI (Passthrough)
    if roi is None:
        if output_buffer is not None:
            # Check shape compatibility
            if output_buffer.shape != image.shape:
                return None
            np.copyto(output_buffer, image, casting='unsafe')
            return output_buffer
        return image  # Return View

    # 2. Validate
    if not is_valid_roi(roi, image.shape):
        # ROI is invalid (out of bounds or zero size)
        # You might choose to raise an error or return None depending on strictness
        return None

    # 3. Create View (Instant, Zero-Copy)
    # Using slices is faster than fancy indexing
    y_slice, x_slice = get_roi_slice(roi)
    crop_view = image[y_slice, x_slice]

    # 4. Handle Output Buffer (Copy Mode)
    if output_buffer is not None:
        # Validate output shape matches ROI shape
        # Handle Channel dimension logic
        expected_shape = (roi.height, roi.width) + image.shape[2:]

        if output_buffer.shape != expected_shape:
            # Shape mismatch
            return None

        # Perform C-level copy
        np.copyto(output_buffer, crop_view, casting='unsafe')
        return output_buffer

    # 5. Return View (Zero-Copy Mode)
    return crop_view

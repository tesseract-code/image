from typing import Union, Optional, Any

import numpy as np
from numpy import typing as npt


# ============================================================================
# Standalone Utility Functions (Optimized)
# ============================================================================

def get_value_at_position(
        data: npt.NDArray[Any],
        x: Union[int, float],
        y: Union[int, float],
        flip_x: bool = False,
        flip_y: bool = False
) -> Optional[Union[np.generic, npt.NDArray[Any]]]:
    """
    Get pixel value at specified position with O(1) optimization for flipping.

    Args:
        data: Image data array (H, W) or (H, W, C)
        x: X coordinate (column)
        y: Y coordinate (row)
        flip_x: If True, flip X axis (e.g., mirror mode)
        flip_y: If True, flip Y axis (e.g., scene coordinates origin bottom-left)

    Returns:
        Pixel value (scalar or array) or None if out of bounds.
    """
    if data is None or data.size == 0:
        return None

    # 1. Fast Integer Casting (Truncate towards zero)
    ix, iy = int(x), int(y)

    # 2. Get Dimensions (Height=Rows, Width=Cols)
    height, width = data.shape[:2]

    # 3. Bounds Check (Strictly enforce 0 <= index < size)
    # We check BEFORE flipping to validate the input coordinate exists in image space
    if ix < 0 or ix >= width or iy < 0 or iy >= height:
        return None

    # 4. Apply Flips (O(1) Index Arithmetic)
    # Instead of transforming the whole array, we just transform the index.
    if flip_x:
        ix = width - 1 - ix

    if flip_y:
        iy = height - 1 - iy

    # 5. Safe Indexing
    try:
        return data[iy, ix]
    except IndexError:
        # Catch-all for rare edge cases (e.g., ragged arrays)
        return None


def get_roi(
        data: npt.NDArray[Any],
        x: int,
        y: int,
        width: int,
        height: int,
        copy: bool = True
) -> Optional[npt.NDArray[Any]]:
    """
    Extract a rectangular region of interest (ROI) from image data (optimized).

    Args:
        data: Image data array (H, W) or (H, W, C) of any dtype
        x: Left edge (column) of ROI
        y: Top edge (row) of ROI
        width: ROI width in pixels
        height: ROI height in pixels
        copy: If True (default), return a copy. If False, return read-only view.

    Returns:
        ROI array or None if invalid region.
        Return type matches the input array's dtype.

    Note:
        - Uses numpy slicing for optimal performance
        - Automatically handles grayscale and multi-channel images
        - Clamps to valid bounds
        - Supports all numpy dtypes (uint8, uint16, float32, etc.)
    """
    if data is None or data.size == 0:
        return None

    # Get image dimensions
    img_height, img_width = data.shape[:2]

    # Clamp coordinates to valid bounds
    x1 = max(0, min(x, img_width))
    y1 = max(0, min(y, img_height))
    x2 = max(0, min(x + width, img_width))
    y2 = max(0, min(y + height, img_height))

    # Check if region is valid
    if x2 <= x1 or y2 <= y1:
        return None

    try:
        # NumPy slicing: [row_start:row_end, col_start:col_end]
        # Works for both 2D and 3D arrays automatically
        roi = data[y1:y2, x1:x2]

        if copy:
            return roi.copy()
        else:
            # Return read-only view
            roi = roi.view()
            roi.flags.writeable = False
            return roi

    except (IndexError, ValueError):
        return None

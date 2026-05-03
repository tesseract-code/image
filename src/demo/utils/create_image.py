from typing import Tuple, Union, Optional

import numpy as np


def create_mono_checkered(
        shape: Tuple[int, int] = (1024, 1024),
        square_size: int = 64,
        dtype: np.dtype = np.float32
) -> np.ndarray:
    """Create a grayscale checkerboard pattern with intensity borders.

    Generates a checkerboard of alternating dark and light gray squares,
    overlaid with white, mid-gray, black, and light-gray borders on the
    top, bottom, left, and right edges, respectively. Useful for testing
    histograms and pixel value inspection.

    Parameters
    ----------
    shape : Tuple[int, int]
        Height and width of the output image. Default is (1024, 1024).
    square_size : int
        Side length (in pixels) of each checker square. Default is 64.
    dtype : np.dtype
        Data type of the output array. Default is ``np.float32``.

    Returns
    -------
    np.ndarray
        A 2D array of shape `shape` containing the checkerboard pattern.

    Notes
    -----
    The checkerboard uses intensity values 0.8 (light gray) and 0.2 (dark gray).
    Border widths are auto‑scaled to at least 1 pixel and at most 10 pixels,
    or approximately 2% of the smaller image dimension.
    """
    h, w = shape

    # 1. Vectorized Checkerboard Generation
    # Create coordinate grids
    y, x = np.mgrid[:h, :w]
    # Determine parity of the grid
    # (y // size + x // size) % 2 gives the checkerboard pattern
    checkers = ((y // square_size) + (x // square_size)) % 2

    # Map 0/1 to specific gray levels (0.2 dark gray, 0.8 light gray)
    img = np.where(checkers == 0, 0.8, 0.2).astype(dtype)

    # 2. Add Borders
    # Calculate border width (min 1px, max 10px, or scale with image)
    b_width = max(1, min(10, min(h, w) // 50))

    img[:b_width, :] = 1.0  # White Top
    img[-b_width:, :] = 0.5  # Mid Gray Bottom
    img[:, :b_width] = 0.0  # Black Left
    img[:, -b_width:] = 0.75  # Light Gray Right

    return img


def create_rgb_checkered(
        shape: Tuple[int, int] = (1024, 1024),
        square_size: int = 64,
        dtype: Union[np.dtype, str] = np.float32,
        out: Optional[np.ndarray] = None
) -> np.ndarray:
    """Create an RGB checkerboard pattern.

    Builds a color checkerboard with light and dark gray squares and
    colored borders (red top, green bottom, blue left, yellow right).

    Parameters
    ----------
    shape : Tuple[int, int]
        Height and width of the output image, in pixels. Default is (1024, 1024).
    square_size : int
        Side length (in pixels) of each checker square. Default is 64.
    dtype : Union[np.dtype, str]
        Data type of the output array. Default is ``np.float32``.
    out : Optional[np.ndarray]
        If provided, the image data is written into this array in‑place
        (zero‑copy). Its shape must be ``(H, W, 3)``. If ``None`` (default),
        a new array is allocated.

    Returns
    -------
    np.ndarray
        A 3D array of shape ``(H, W, 3)`` containing the RGB checkerboard.
        If `out` was supplied, the same array is returned.

    Raises
    ------
    ValueError
        If `out` is provided but its shape does not match ``(H, W, 3)``.

    Notes
    -----
    Light gray squares have RGB ``[0.8, 0.8, 0.8]``; dark gray squares have
    RGB ``[0.2, 0.2, 0.2]``. The border widths are automatically scaled
    between 1 and 10 pixels based on image size.
    """
    h, w = shape

    # 1. Prepare Output Buffer
    if out is None:
        img = np.ones((h, w, 3), dtype=dtype)
        img[:] = 0.2  # Initialize Dark Gray
    else:
        # Verify shape matches
        expected_shape = (h, w, 3)
        if out.shape != expected_shape:
            raise ValueError(
                f"Output buffer mismatch. Got {out.shape}, expected {expected_shape}")
        img = out
        img[:] = 0.2  # Fill in-place

    # 2. Vectorized Checkerboard Mask
    # (Pre-calculation of this mask outside the loop is recommended for
    # pure upload benchmarking, but we keep it here to simulate 'work')
    y, x = np.mgrid[:h, :w]
    mask = ((y // square_size) + (x // square_size)) % 2 == 0

    # Apply Light Gray to mask
    img[mask] = [0.8, 0.8, 0.8]

    # 3. Add Colored Borders
    b_width = max(1, min(10, min(h, w) // 50))

    img[:b_width, :, :] = [1.0, 0.0, 0.0]  # Red Top
    img[-b_width:, :, :] = [0.0, 1.0, 0.0]  # Green Bottom
    img[:, :b_width, :] = [0.0, 0.0, 1.0]  # Blue Left
    img[:, -b_width:, :] = [1.0, 1.0, 0.0]  # Yellow Right

    return img


def create_target_regions(
        shape: Tuple[int, int] = (1024, 1024),
        dtype: np.dtype = np.float32
) -> np.ndarray:
    """Create a black image with white target regions in corners and center.

    Places a central white square (about 20% of the image size) and four
    smaller white squares near each corner (about 10% of the image size).
    Useful for testing coordinate mapping, crosshairs, and overlay alignment.

    Parameters
    ----------
    shape : Tuple[int, int]
        Height and width of the output image. Default is (1024, 1024).
    dtype : np.dtype
        Data type of the output array. Default is ``np.float32``.

    Returns
    -------
    np.ndarray
        A 2D array of shape `shape` with value 1.0 inside the target regions
        and 0.0 elsewhere.
    """
    h, w = shape
    img = np.zeros((h, w), dtype=dtype)

    # 1. Center Square
    # Scale center size relative to image (approx 20% of smallest dim)
    center_size = min(h, w) // 5
    cy, cx = h // 2, w // 2
    half_sz = center_size // 2

    # Safe slicing handles boundaries automatically
    img[cy - half_sz: cy + half_sz, cx - half_sz: cx + half_sz] = 1.0

    # 2. Corner Squares
    # Scale corner size (approx 10% of smallest dim)
    corner_size = min(h, w) // 10
    margin = max(1, min(h, w) // 20)  # Gap from the edge

    # Define Top-Left coordinates for the 4 boxes
    positions = [
        (margin, margin),  # TL
        (margin, w - margin - corner_size),  # TR
        (h - margin - corner_size, margin),  # BL
        (h - margin - corner_size, w - margin - corner_size)  # BR
    ]

    for (pos_y, pos_x) in positions:
        # Ensure we don't go out of bounds if image is extremely small
        if pos_y < h and pos_x < w:
            end_y = min(pos_y + corner_size, h)
            end_x = min(pos_x + corner_size, w)
            img[pos_y:end_y, pos_x:end_x] = 1.0

    return img
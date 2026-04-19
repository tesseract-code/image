from typing import Any

import numpy as np
from numpy.typing import NDArray

# Type Definitions

type IntegerArray = NDArray[np.integer]
type FloatArray = NDArray[np.floating]

type ImageDType = np.uint8 | np.uint16 | np.int32 | np.float32 | np.float64 | np.bool_
type ImageLike = NDArray[ImageDType]
type Images = list[ImageLike]


# Type-check Logic

def is_image(target: Any) -> bool:
    """
    Checks if target is a valid non-empty 2D or 3D numpy array representing an image.
    """
    # Type Check
    if not isinstance(target, np.ndarray):
        return False

    # Shape Check
    if target.ndim not in (2, 3):
        return False

    if target.size == 0:
        return False

    # Dtype Check
    return np.issubdtype(target.dtype, np.number) or np.issubdtype(target.dtype,
                                                                   np.bool_)


def is_standard_image(target: Any) -> bool:
    if not is_image(target):
        return False

    # If 3D, check if channel count is "sane" for standard rendering (1, 2, 3, 4)
    if target.ndim == 3:
        # Assuming (H, W, C) layout
        return target.shape[2] <= 4

    return True

from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np

from image.settings.pixels import PixelFormat
from image.utils.types import ImageLike
from image.utils.channel import PixelType


def ensure_contiguity(arr: ImageLike,
                      dtype: Optional[np.dtype] = None
                      ) -> ImageLike:
    """
    Ensures memory is C-contiguous (Row-Major).
    """
    if not arr.flags['C_CONTIGUOUS']:
        return np.ascontiguousarray(arr, dtype=dtype)
    return arr


@dataclass(frozen=True, slots=True)
class PixelBuffer:
    """
    Immutable, standardized container for raw pixel data in CPU memory.
    Guarantees:
    1. Data is always (H, W, C) - even for grayscale.
    2. Data is strictly numeric.
    3. Metadata matches the array content.
    """
    data: ImageLike
    width: int
    height: int
    pixel_fmt: PixelFormat

    def __post_init__(self):
        # Strict validation on creation
        _ = self.pixel_type

    @property
    def pixel_type(self) -> PixelType:
        """
        Derives the PxelType from the underlying data.
        """
        dtype = self.data.dtype

        # Fast mapping
        if dtype == np.uint8: return PixelType.CHAR
        if dtype == np.float32: return PixelType.FLOAT
        if dtype == np.int32: return PixelType.INT
        if dtype == np.float64: return PixelType.DOUBLE

        raise TypeError(f"PixelBuffer holds unsupported dtype: {dtype}")

    @property
    def shape(self) -> Tuple[int, int, int]:
        """Always returns (Height, Width, Channels)."""
        return self.height, self.width, self.channels

    @property
    def size(self) -> Tuple[int, int]:
        """Returns (Width, Height)."""
        return self.width, self.height

    @property
    def nbytes(self) -> int:
        return self.data.nbytes

    @property
    def is_scalar(self) -> bool:
        """True if the image effectively has 1 channel."""
        return self.channels == 1

    @property
    def is_rgb(self) -> bool:
        return self.pixel_fmt == PixelFormat.RGB

    @property
    def is_bgr(self) -> bool:
        return self.pixel_fmt == PixelFormat.BGR

    def copy_data(self) -> ImageLike:
        """Returns a deep copy of the underlying array."""
        return self.data.copy()

    def as_contiguous(self) -> ImageLike:
        """Returns a C-Contiguous view (or copy) for OpenGL."""
        return ensure_contiguity(self.data)

    def __repr__(self) -> str:
        return (f"PixelBuffer("
                f"shape={self.width}x{self.height}x{self.channels}, "
                f"fmt={self.pixel_fmt.name}, "
                f"dtype={self.data.dtype})")

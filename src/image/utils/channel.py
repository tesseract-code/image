from enum import IntEnum, unique
from typing import Union

import numpy as np


@unique
class PixelType(IntEnum):
    """
    Specifies the primitive data type used for each channel component.
    """
    CHAR = 0  # 1 byte  (uint8)  - Standard images (0-255)
    INT = 1  # 4 bytes (int32)  - Segmentation masks / IDs
    LONG = 2  # 8 bytes (int64)  - Large counters
    FLOAT = 3  # 4 bytes (float32)- HDR, ML tensors (0.0-1.0)
    DOUBLE = 4  # 8 bytes (float64)- High precision scientific data

    @property
    def numpy_dtype(self) -> np.dtype:
        """Maps enum to numpy dtype."""
        match self:
            case PixelType.CHAR:
                return np.dtype(np.uint8)
            case PixelType.INT:
                return np.dtype(np.int32)
            case PixelType.LONG:
                return np.dtype(np.int64)
            case PixelType.FLOAT:
                return np.dtype(np.float32)
            case PixelType.DOUBLE:
                return np.dtype(np.float64)

    @property
    def bytes_per_channel(self) -> int:
        """Returns size in bytes of a single channel component."""
        return self.numpy_dtype.itemsize

    @classmethod
    def from_dtype(cls, dtype: Union[np.dtype, type]) -> "PixelType":
        """
        Infers ChannelDType from a numpy array's dtype.
        Uses (kind, itemsize) for endian-safe and alias-safe matching.
        """
        # Ensure we have a dtype instance (handles input like np.float32 vs np.dtype("float32"))
        if not isinstance(dtype, np.dtype):
            try:
                dtype = np.dtype(dtype)
            except TypeError:
                raise ValueError(
                    f"Could not convert input to numpy dtype: {dtype}")

        kind = dtype.kind  # 'u', 'i', 'f', etc.
        itemsize = dtype.itemsize  # Bytes (1, 4, 8)

        match kind:
            case 'u':  # Unsigned Integer
                if itemsize == 1: return cls.CHAR

            case 'i':  # Signed Integer
                if itemsize == 4: return cls.INT
                if itemsize == 8: return cls.LONG

            case 'f':  # Floating Point
                if itemsize == 4: return cls.FLOAT
                if itemsize == 8: return cls.DOUBLE

            case _:
                # Handle specific failures or pass through to error
                pass

        raise ValueError(
            f"Unsupported dtype: {dtype} (kind='{kind}', size={itemsize})")

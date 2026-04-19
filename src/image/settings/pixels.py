from enum import IntEnum, unique
from typing import Tuple

import numpy as np


@unique
class PixelFormat(IntEnum):
    """
    Image pixel layout constants.
    Describes the semantic composition of pixel data.
    """
    RGB = 0
    BGR = 1
    RGBA = 2
    BGRA = 3
    MONOCHROME = 4
    RG = 5  # 2-channel (e.g., Red-Green, UV maps, Optical Flow)
    YUV420 = 6  # Planar, Subsampled
    YUV422 = 7  # Planar/Packed, Subsampled
    YUV444 = 8  # Usually Packed, No subsampling
    NV12 = 9  # Semi-Planar (Y Plane + UV Interleaved Plane)
    NV21 = 10  # Semi-Planar (Y Plane + VU Interleaved Plane)

    @property
    def channels(self) -> int:
        """
        Returns the number of components per pixel for PACKED formats.
        Raises ValueError for planar/subsampled formats where stride is non-uniform.
        """
        match self:
            case PixelFormat.MONOCHROME:
                return 1
            case PixelFormat.RG:
                return 2
            case PixelFormat.RGB | PixelFormat.BGR | PixelFormat.YUV444:
                return 3
            case PixelFormat.RGBA | PixelFormat.BGRA:
                return 4
            case _:
                # YUV420, NV12, etc. do not have a constant byte-stride per pixel.
                # Calculating 'width * height * channels' is invalid for these.
                raise NotImplementedError(
                    f"Channel count is not uniform/atomic for planar format: {self.name}"
                )

    @staticmethod
    def infer_from_shape(shape: Tuple[int, ...]) -> "PixelFormat":
        """
        Infers format from array shape (H, W, C).

        Note: Ambiguous cases (BGR vs RGB) default to standard RGB/RGBA.
        """
        ndim = len(shape)

        # Case: (H, W) -> Monochrome
        if ndim == 2:
            return PixelFormat.MONOCHROME

        # Case: (H, W, C)
        if ndim == 3:
            c = shape[2]
            match c:
                case 1:
                    return PixelFormat.MONOCHROME
                case 2:
                    return PixelFormat.RG
                case 3:
                    return PixelFormat.RGB  # Assumption: RGB is more common than BGR/YUV
                case 4:
                    return PixelFormat.RGBA  # Assumption: RGBA is more common than BGRA

        raise ValueError(f"Cannot infer PixelFormat from shape {shape}")

    @classmethod
    def from_channels(cls, channels: int) -> 'PixelFormat':
        """
        Fast O(1) mapping from atomic channel count to PixelFormat.
        """
        match channels:
            case 1:
                return PixelFormat.MONOCHROME
            case 2:
                return PixelFormat.RG
            case 3:
                return PixelFormat.RGB
            case 4:
                return PixelFormat.RGBA
            case _:
                raise ValueError(
                    f"Unsupported channel count for auto-mapping: {channels}")

    @property
    def is_planar(self) -> bool:
        """Helper to determine if format requires special plane handling."""
        return self in (
            PixelFormat.YUV420,
            PixelFormat.YUV422,
            PixelFormat.NV12,
            PixelFormat.NV21
        )


def broadcast_to_format(
        image: np.ndarray,
        pixel_fmt: PixelFormat,
        copy: bool = False
) -> np.ndarray:
    """
    Broadcast image array to match pixel format channel requirements.

    Handles common broadcasting cases:
    - Grayscale (H, W) → RGB (H, W, 3)
    - Grayscale (H, W) → RGBA (H, W, 4)
    - Grayscale (H, W) → Monochrome (H, W, 1)
    - Already matching → pass-through

    Args:
        image: Input array, shape (H, W) or (H, W, C)
        pixel_fmt: Target pixel format specifying required channels
        copy: If True, always return a copy. If False, returns view when possible.

    Returns:
        Array with shape (H, W, channels) matching pixel_fmt

    Raises:
        ValueError: If image cannot be broadcast to target format

    Examples:
        >>> gray = np.random.rand(480, 640)
        >>> rgb = broadcast_to_format(gray, PixelFormat.RGB)
        >>> rgb.shape
        (480, 640, 3)

        >>> rgb_img = np.random.rand(480, 640, 3)
        >>> result = broadcast_to_format(rgb_img, PixelFormat.RGB)
        >>> result is rgb_img  # No-op for matching shapes
        True
    """
    if image.ndim < 2 or image.ndim > 3:
        raise ValueError(f"Image must be 2D or 3D, got shape {image.shape}")

    h, w = image.shape[:2]
    target_channels = pixel_fmt.channels

    # Case 1: Already has correct shape
    if image.ndim == 3 and image.shape[2] == target_channels:
        return image.copy() if copy else image

    # Case 2: Grayscale to multi-channel (RGB, RGBA, etc.)
    if image.ndim == 2 and target_channels > 1:
        # Use stride tricks for zero-copy broadcast
        broadcasted = np.lib.stride_tricks.as_strided(
            image,
            shape=(h, w, target_channels),
            strides=(*image.strides, 0)  # Zero stride on channel axis
        )
        return broadcasted.copy() if copy else broadcasted

    # Case 3: Grayscale to monochrome (add channel dimension)
    if image.ndim == 2 and target_channels == 1:
        result = image.reshape(h, w, 1)
        return result.copy() if copy else result

    # Case 4: Multi-channel source to different multi-channel target
    if image.ndim == 3:
        src_channels = image.shape[2]
        if src_channels == 1 and target_channels > 1:
            # Squeeze and broadcast
            squeezed = image[:, :, 0]
            return broadcast_to_format(squeezed, pixel_fmt, copy)
        else:
            raise ValueError(
                f"Cannot broadcast {image.shape} (channels={src_channels}) "
                f"to {pixel_fmt.name} (channels={target_channels})"
            )

    raise ValueError(
        f"Unsupported broadcast from {image.shape} to {pixel_fmt.name}")
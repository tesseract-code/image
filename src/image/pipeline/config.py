from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from pycore.settings.provider import SettingsProvider
from image.settings.pixels import PixelFormat
from image.utils.types import FloatArray


@dataclass(slots=True, frozen=False)
class ProcessingConfig:
    """
    Immutable, thread-safe configuration for image processing.
    Uses slots for memory efficiency and frozen for immutability.
    """
    auto_contrast: bool = False
    normalize: bool = False
    normalize_min: float | None = None
    normalize_max: float | None = None
    gain: float = 1.0
    offset: float = 0.0
    colormap_enabled: bool = False
    colormap_name: str = 'gray'
    colormap_lut: FloatArray | None = None
    colormap_reverse: bool = False
    roi: Tuple | None = None
    fmt: PixelFormat | None = None

    def get_output_format(self, input_format: PixelFormat) -> PixelFormat:
        """Determines the resulting format."""
        # 1. Colormaps force RGB (or RGBA depending on LUT)
        if self.colormap_enabled:
            return PixelFormat.RGB

        # 2. Explicit conversion requested
        if self.fmt is not None:
            return self.fmt

        # 3. Passthrough
        return input_format

    def get_output_shape(self, input_shape: tuple,
                         input_format: PixelFormat) -> tuple:
        """
        Calculates (H, W, C) based on ROI, Input, and Target Format.
        """
        # (Assuming input_shape is (H, W) or (H, W, C))
        height, width = input_shape[:2]

        # 1. Determine Target Format
        final_fmt = self.get_output_format(input_format)

        # 2. Get Channels
        try:
            channels = final_fmt.channels
        except NotImplementedError:
            # Handle YUV cases or default to input 3rd dim
            channels = input_shape[2] if len(input_shape) > 2 else 1

        # 3. Return Standard Shape
        if channels == 1:
            return height, width
        return height, width, channels

    def get_output_dtype(self, input_dtype: np.dtype) -> np.dtype:
        """Determines if we need float32 for math or uint8 for display."""
        if self.colormap_enabled:
            return np.dtype(np.uint8)

        math_active = abs(self.gain - 1.0) > 1e-6 or abs(self.offset) > 1e-6
        if math_active:
            return np.dtype(np.float32)

        return input_dtype

    @classmethod
    def from_settings(cls,
                      data: dict[str, Any] | SettingsProvider):
        """
        Create configuration from external settings.
        Handles Dict, QObject, or any object with __dict__.
        """
        match data:
            case obj if hasattr(obj, 'get_copy'):
                settings_dict = obj.get_copy()
            case obj if hasattr(obj, '__dict__'):
                settings_dict = obj.__dict__
            case obj if hasattr(obj, '_asdict'):
                # NamedTuple handling (Standard Python API)
                settings_dict = obj._asdict()
            case _:
                settings_dict = data

        # Build configuration using structural pattern matching
        normalize = False
        normalize_min = None
        normalize_max = None

        # Handle LUT-based normalization (legacy UI naming)
        if settings_dict.get('lut_enabled', False):
            normalize = True
            normalize_min = float(settings_dict.get('lut_min', 0.0))
            normalize_max = float(settings_dict.get('lut_max', 1.0))
        elif settings_dict.get('normalize', False):
            normalize = True
            normalize_min = settings_dict.get('normalize_min')
            normalize_max = settings_dict.get('normalize_max')
            if normalize_min is not None:
                normalize_min = float(normalize_min)
            if normalize_max is not None:
                normalize_max = float(normalize_max)

        return cls(
            normalize=normalize,
            normalize_min=normalize_min,
            normalize_max=normalize_max,
            gain=float(settings_dict.get('gain', 1.0)),
            offset=float(settings_dict.get('offset', 0.0)),
            colormap_enabled=bool(settings_dict.get('colormap_enabled', False)),
            colormap_name=str(settings_dict.get('colormap_name', 'gray')),
            colormap_lut=settings_dict.get('colormap_lut'),
            colormap_reverse=bool(settings_dict.get('colormap_reverse', False)),
            fmt=settings_dict.get("img_format", None)
        )

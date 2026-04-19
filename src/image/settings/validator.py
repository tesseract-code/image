import logging
from typing import Any

from image.settings.base import LUTType

logger = logging.getLogger(__name__)

class ImageSettingsValidator:
    """
    Validation rules for image settings.
    Integrates with generic SettingsValidator via registration.
    """

    # Define valid ranges
    RANGES = {
        'zoom': (0.01, 100.0),
        'rotation': (-360.0, 360.0),
        'brightness': (-1.0, 1.0),
        'contrast': (0.1, 3.0),
        'gamma': (0.1, 3.0),
        'gain': (0.1, 10.0),
        'offset': (-1.0, 1.0),
        'color_balance_r': (0.0, 2.0),
        'color_balance_g': (0.0, 2.0),
        'color_balance_b': (0.0, 2.0),
        'lut_min': (0.0, 1.0),
        'lut_max': (0.0, 1.0),
    }

    VALID_COLORMAPS = {
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        'gray', 'hot', 'cool', 'spring', 'summer', 'autumn', 'winter',
        'bone', 'copper', 'pink', 'jet', 'turbo'
    }

    @classmethod
    def register_validators(cls, validator):
        """
        Register all image-specific validators with the generic validator.

        Args:
            validator: SettingsValidator instance from SettingsServer
        """
        # Range validators
        for field_name, (min_val, max_val) in cls.RANGES.items():
            validator.register(
                field_name,
                cls._make_range_validator(field_name, min_val, max_val)
            )

        # LUT type validator
        validator.register('lut_type', cls._validate_lut_type)

        # Colormap validator
        validator.register('colormap_name', cls._validate_colormap)

        # Boolean validators
        for field in ['invert', 'lut_enabled', 'interpolation',
                      'colormap_enabled', 'colormap_reverse']:
            validator.register(field, cls._make_bool_validator(field))

        logger.info("Image settings validators registered")

    @staticmethod
    def _make_range_validator(name: str, min_val: float, max_val: float):
        """Create a range validator with custom error message."""

        def validator(value: Any) -> bool:
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{name} must be numeric, got {type(value).__name__}")
            if not (min_val <= value <= max_val):
                raise ValueError(
                    f"{name}={value} out of range [{min_val}, {max_val}]"
                )
            return True

        return validator

    @staticmethod
    def _validate_lut_type(value: Any) -> bool:
        """Validate LUT type."""
        if not isinstance(value, LUTType):
            raise ValueError(
                f"lut_type must be LUTType enum, got {type(value).__name__}")
        return True

    @classmethod
    def _validate_colormap(cls, value: Any) -> bool:
        """Validate colormap name."""
        if not isinstance(value, str):
            raise ValueError(
                f"colormap_name must be string, got {type(value).__name__}")
        if value not in cls.VALID_COLORMAPS:
            raise ValueError(
                f"Invalid colormap '{value}'. "
                f"Valid options: {', '.join(sorted(cls.VALID_COLORMAPS))}"
            )
        return True

    @staticmethod
    def _make_bool_validator(name: str):
        """Create a boolean validator."""

        def validator(value: Any) -> bool:
            if not isinstance(value, bool):
                raise ValueError(
                    f"{name} must be boolean, got {type(value).__name__}")
            return True

        return validator

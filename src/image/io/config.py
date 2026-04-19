from dataclasses import dataclass
from enum import IntEnum, unique
from typing import Optional, Tuple, Literal

from image.settings.pixels import PixelFormat


@unique
class ImageReadFlags(IntEnum):
    """Standardized image read flags across backends."""
    DEFAULT = 0
    IGNORE_EXIF_ORIENTATION = 1
    UNCHANGED = 2  # Load as-is, no conversions
    GRAYSCALE = 4  # Force grayscale
    ANYDEPTH = 8  # Respect original bit depth
    ANYCOLOR = 16  # Respect original color space


@dataclass(frozen=False, slots=True)
class LoadConfig:
    """
    Comprehensive configuration for image loading operations.
    Provides high-level control over loading behavior.
    """
    # Core parameters
    target_format: Optional[PixelFormat] = None
    flip_vertically: bool = False

    # EXIF handling
    apply_exif_orientation: bool = True  # Apply EXIF rotation
    strip_exif: bool = False  # Remove EXIF after loading

    # Optimization
    max_dimension: Optional[int] = None  # Downsample on load if exceeds
    thumbnail_size: Optional[Tuple[int, int]] = None  # Generate thumbnail
    reducing_gap: float = 2.0  # Pillow resize optimization (2.0-3.0 recommended)

    # Quality & Integrity
    validate_integrity: bool = True  # Check for corruption
    compute_hash: bool = False  # Compute image hash
    hash_algorithm: Literal['md5', 'sha256'] = 'md5'

    # Advanced
    icc_profile_handling: Literal['ignore', 'apply', 'preserve'] = 'ignore'
    preserve_transparency: bool = True
    flags: ImageReadFlags = ImageReadFlags.DEFAULT

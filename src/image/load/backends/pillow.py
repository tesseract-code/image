import logging
from enum import StrEnum
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

from image.load.config import LoadConfig, ImageReadFlags
from image.load.interface import ImageLoaderAdapter
from image.load.metadata import ImageMetadata
from image.settings.pixels import PixelFormat

try:
    from PIL import Image, ImageOps, ExifTags
    from PIL.Image import Resampling

    HAS_PILLOW = True
except ImportError:
    HAS_PILLOW = False

logger = logging.getLogger(__name__)

# Constants
# ~420MB at 8-bit RGB. Prevents decompression bombs.
PILLOW_MAX_SAFE_PIXELS = 178_956_970


# ==========================================
# Enhanced Configuration Classes
# ==========================================

class PillowMode(StrEnum):
    """Explicit mapping of Pillow mode strings."""
    RGB = "RGB"
    RGBA = "RGBA"
    GRAYSCALE = "L"
    CMYK = "CMYK"
    LAB = "LAB"
    PALETTE = "P"


class PillowAdapter(ImageLoaderAdapter):
    """
    Enhanced Pillow adapter with EXIF handling, optimization,
    and comprehensive metadata support.
    """

    def __init__(self):
        if not HAS_PILLOW:
            raise ImportError("Pillow is not installed.")

        # Configure Pillow limits (security & performance)
        Image.MAX_IMAGE_PIXELS = PILLOW_MAX_SAFE_PIXELS

    def load(self,
             path: Path,
             config: LoadConfig) -> Tuple[
        np.ndarray, PixelFormat, ImageMetadata]:

        # Validate first if requested
        if config.validate_integrity:
            is_valid, error = self.validate_image(path)
            if not is_valid:
                raise ValueError(f"Corrupted image: {error}")

        with Image.open(path) as img:
            # Store original format
            file_fmt = img.format

            # --- EXIF Orientation Handling (Critical Feature) ---
            if config.apply_exif_orientation and not (
                    config.flags & ImageReadFlags.IGNORE_EXIF_ORIENTATION
            ):
                # Use Pillow's built-in EXIF orientation correction
                img = ImageOps.exif_transpose(img) or img

            # --- Optimization: Draft Mode for JPEG ---
            # Draft mode allows JPEG decoder to downsample during decode
            if config.max_dimension and file_fmt == 'JPEG':
                draft_size = (config.max_dimension, config.max_dimension)
                img.draft('RGB', draft_size)

            # --- Optimization: Thumbnail Generation ---
            if config.thumbnail_size:
                # Pillow's thumbnail is very efficient (uses reducing_gap internally)
                img.thumbnail(config.thumbnail_size, Resampling.LANCZOS)

            # --- Extract metadata before conversion ---
            metadata = self._extract_metadata(img, path, config)

            # --- Format Conversion Logic ---
            final_fmt = self._determine_format(img, config)
            pil_mode = self._get_pil_mode(final_fmt, config)

            if img.mode != pil_mode.value:
                # Preserve transparency if requested
                if config.preserve_transparency and 'transparency' in img.info:
                    if pil_mode in (PillowMode.RGB, PillowMode.RGBA):
                        img = img.convert(PillowMode.RGBA.value)
                    else:
                        img = img.convert(str(pil_mode.value))
                else:
                    img = img.convert(str(pil_mode.value))

            # --- Apply flip if requested ---
            if config.flip_vertically:
                img = img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

            # --- Convert to numpy ---
            data = np.array(img, dtype=np.uint8)

            # --- Handle BGR swaps (OpenCV compatibility) ---
            if config.target_format == PixelFormat.BGR:
                data = data[..., ::-1]  # RGB -> BGR
            elif config.target_format == PixelFormat.BGRA:
                data = data[..., [2, 1, 0, 3]]  # RGBA -> BGRA

            # --- Compute hash if requested ---
            # if config.compute_hash:
            #     metadata.file_hash = self._compute_hash_safe(
            #         data, config.hash_algorithm
            #     )

        return data, final_fmt, metadata

    def _determine_format(self, img: Image.Image,
                          config: LoadConfig) -> PixelFormat:
        """Determine target pixel format."""
        if config.target_format:
            return config.target_format

        # Inference based on image mode
        mode_map = {
            PillowMode.GRAYSCALE.value: PixelFormat.MONOCHROME,
            PillowMode.RGB.value: PixelFormat.RGB,
            PillowMode.RGBA.value: PixelFormat.RGBA,
        }

        return mode_map.get(img.mode, PixelFormat.RGB)

    def _get_pil_mode(self, layout: PixelFormat,
                      config: LoadConfig) -> PillowMode:
        """Map PixelFormat to Pillow mode."""
        if layout == PixelFormat.BGR:
            return PillowMode.RGB  # Convert later
        elif layout == PixelFormat.BGRA:
            return PillowMode.RGBA  # Convert later
        elif layout == PixelFormat.MONOCHROME:
            return PillowMode.GRAYSCALE
        elif layout == PixelFormat.RGB:
            return PillowMode.RGB
        elif layout == PixelFormat.RGBA:
            return PillowMode.RGBA
        else:
            raise ValueError(f"Unsupported layout: {layout}")

    def _extract_metadata(self, img: Image.Image,
                          path: Path,
                          config: LoadConfig) -> ImageMetadata:
        """Extract comprehensive metadata."""
        metadata = ImageMetadata(
            width=img.width,
            height=img.height,
            format=img.format,
            mode=img.mode,
        )

        # EXIF data
        if not config.strip_exif:
            try:
                exif = img.getexif()
                # if exif:
                #     metadata.exif_orientation = exif.get(
                #         ExifTags.Base.Orientation, 1
                #     )
                #     # Extract common EXIF tags
                #     for tag_id, value in exif.items():
                #         tag = ExifTags.TAGS.get(tag_id, tag_id)
                #         metadata.exif_data[str(tag)] = value
            except Exception as e:
                logger.debug(f"EXIF extraction failed: {e}")

        # ICC Profile
        # if 'icc_profile' in img.info:
        #     metadata.has_icc_profile = True
        #     if config.icc_profile_handling == 'preserve':
        #         metadata.icc_profile = img.info['icc_profile']
        #
        # # Animation detection
        # if hasattr(img, 'n_frames'):
        #     metadata.frame_count = img.n_frames
        #     metadata.is_animated = img.n_frames > 1

        return metadata

    def get_metadata(self, path: Path,
                     include_exif: bool = True) -> ImageMetadata:
        """Fast metadata extraction without loading pixels."""
        with Image.open(path) as img:
            config = LoadConfig()
            return self._extract_metadata(img, path, config)

    def validate_image(self, path: Path) -> Tuple[bool, Optional[str]]:
        """Validate image can be opened and decoded."""
        try:
            with Image.open(path) as img:
                img.verify()  # Verify without decoding
            # Re-open for load test (verify closes the file)
            with Image.open(path) as img:
                img.load()  # Actually decode
            return True, None
        except Exception as e:
            return False, str(e)

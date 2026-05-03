import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

from image.load.config import LoadConfig, ImageReadFlags
from image.load.interface import ImageLoaderAdapter
from image.load.metadata import ImageMetadata
from image.settings.pixels import PixelFormat

try:
    import cv2

    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

logger = logging.getLogger(__name__)

class Cv2Adapter(ImageLoaderAdapter):
    """
    Enhanced OpenCV adapter with proper EXIF handling,
    optimization flags, and robust error handling.
    """

    def __init__(self):
        if not HAS_OPENCV:
            raise ImportError("opencv-python is not installed.")

        # Check OpenCV version for EXIF support
        self._has_exif_support = self._check_exif_support()

    def _check_exif_support(self) -> bool:
        """Check if OpenCV version supports EXIF orientation."""
        version = cv2.__version__
        major, minor = map(int, version.split('.')[:2])
        return major >= 3 and minor >= 1  # EXIF support since 3.1

    def load(self,
             path: Path,
             config: LoadConfig) -> Tuple[
        np.ndarray, PixelFormat, ImageMetadata]:

        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")

        # --- Determine OpenCV read flags ---
        cv_flags = self._build_cv_flags(config)

        # --- Load image ---
        raw_img = cv2.imread(str(path), cv_flags)

        if raw_img is None:
            raise ValueError(f"OpenCV failed to decode: {path}")

        # --- Determine current format ---
        current_fmt = PixelFormat.infer_from_shape(raw_img.shape)

        # --- EXIF Orientation (OpenCV handles automatically in 3.1+) ---
        # Note: OpenCV applies EXIF by default, but doesn't expose the data
        # If user wants to ignore, must reload with IMREAD_IGNORE_ORIENTATION
        if not config.apply_exif_orientation and self._has_exif_support:
            # Reload without EXIF correction
            raw_img = cv2.imread(
                str(path),
                cv_flags | cv2.IMREAD_IGNORE_ORIENTATION
            )
            current_fmt = PixelFormat.infer_from_shape(raw_img.shape)

        data = raw_img
        final_fmt = current_fmt

        # --- Format conversion ---
        if config.target_format and config.target_format != current_fmt:
            data = self._convert_color(data, current_fmt, config.target_format)
            final_fmt = config.target_format

        # --- Optimization: Resize if max dimension specified ---
        if config.max_dimension:
            h, w = data.shape[:2]
            if max(h, w) > config.max_dimension:
                scale = config.max_dimension / max(h, w)
                new_size = (int(w * scale), int(h * scale))
                data = cv2.resize(data, new_size, interpolation=cv2.INTER_AREA)

        # --- Thumbnail generation ---
        if config.thumbnail_size:
            data = cv2.resize(
                data,
                config.thumbnail_size,
                interpolation=cv2.INTER_AREA
            )

        # --- Flip ---
        if config.flip_vertically:
            data = cv2.flip(data, 0)

        # --- Extract metadata ---
        metadata = self._extract_metadata(path, data, config)

        # --- Compute hash ---
        # if config.compute_hash:
        #     metadata.file_hash = self._compute_hash_safe(
        #         data, config.hash_algorithm
        #     )

        return data, final_fmt, metadata

    def _build_cv_flags(self, config: LoadConfig) -> int:
        """Build OpenCV imread flags from config."""
        if config.flags & ImageReadFlags.UNCHANGED:
            return cv2.IMREAD_UNCHANGED

        if config.flags & ImageReadFlags.GRAYSCALE:
            return cv2.IMREAD_GRAYSCALE

        if config.flags & ImageReadFlags.ANYDEPTH:
            return cv2.IMREAD_ANYDEPTH | cv2.IMREAD_ANYCOLOR

        # Default: load as color
        return cv2.IMREAD_COLOR

    def _convert_color(self, img: np.ndarray,
                       src: PixelFormat,
                       dst: PixelFormat) -> np.ndarray:
        """Enhanced color conversion with proper mapping."""
        if src == dst:
            return img

        # Comprehensive conversion matrix
        conversions = {
            # BGR source
            (PixelFormat.BGR, PixelFormat.RGB): cv2.COLOR_BGR2RGB,
            (PixelFormat.BGR, PixelFormat.RGBA): cv2.COLOR_BGR2RGBA,
            (PixelFormat.BGR, PixelFormat.BGRA): cv2.COLOR_BGR2BGRA,
            (PixelFormat.BGR, PixelFormat.MONOCHROME): cv2.COLOR_BGR2GRAY,
            # BGRA source
            (PixelFormat.BGRA, PixelFormat.BGR): cv2.COLOR_BGRA2BGR,
            (PixelFormat.BGRA, PixelFormat.RGB): cv2.COLOR_BGRA2RGB,
            (PixelFormat.BGRA, PixelFormat.RGBA): cv2.COLOR_BGRA2RGBA,
            (PixelFormat.BGRA, PixelFormat.MONOCHROME): cv2.COLOR_BGRA2GRAY,
            # RGB source
            (PixelFormat.RGB, PixelFormat.BGR): cv2.COLOR_RGB2BGR,
            (PixelFormat.RGB, PixelFormat.BGRA): cv2.COLOR_RGB2BGRA,
            (PixelFormat.RGB, PixelFormat.RGBA): cv2.COLOR_RGB2RGBA,
            (PixelFormat.RGB, PixelFormat.MONOCHROME): cv2.COLOR_RGB2GRAY,
            # RGBA source
            (PixelFormat.RGBA, PixelFormat.RGB): cv2.COLOR_RGBA2RGB,
            (PixelFormat.RGBA, PixelFormat.BGR): cv2.COLOR_RGBA2BGR,
            (PixelFormat.RGBA, PixelFormat.BGRA): cv2.COLOR_RGBA2BGRA,
            (PixelFormat.RGBA, PixelFormat.MONOCHROME): cv2.COLOR_RGBA2GRAY,
            # Grayscale source
            (PixelFormat.MONOCHROME, PixelFormat.BGR): cv2.COLOR_GRAY2BGR,
            (PixelFormat.MONOCHROME, PixelFormat.RGB): cv2.COLOR_GRAY2RGB,
            (PixelFormat.MONOCHROME, PixelFormat.BGRA): cv2.COLOR_GRAY2BGRA,
            (PixelFormat.MONOCHROME, PixelFormat.RGBA): cv2.COLOR_GRAY2RGBA,
        }

        code = conversions.get((src, dst))
        if code is None:
            raise ValueError(
                f"Conversion not supported: {src.name} -> {dst.name}"
            )

        return cv2.cvtColor(img, code)

    def _extract_metadata(self, path: Path,
                          img: np.ndarray,
                          config: LoadConfig) -> ImageMetadata:
        """Extract metadata (OpenCV has limited metadata access)."""
        metadata = ImageMetadata(
            width=img.shape[1],
            height=img.shape[0],
            format=path.suffix.strip('.').upper() or None,
            mode="BGR" if img.ndim == 3 and img.shape[2] == 3 else "GRAY",
            bit_depth=img.dtype.itemsize * 8,
        )

        # Note: OpenCV doesn't provide easy EXIF access
        # For comprehensive EXIF, must use Pillow or external library
        logger.debug(
            "OpenCV has limited EXIF support; use Pillow for full metadata")

        return metadata

    def get_metadata(self, path: Path,
                     include_exif: bool = True) -> ImageMetadata:
        """
        Extract metadata using OpenCV.

        WARNING: Unlike Pillow, OpenCV decodes the full image to determine
        dimensions. This operation may be slow for large files.
        """
        # OpenCV must load full image for metadata
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise ValueError("Failed to read image")

        return self._extract_metadata(path, img, LoadConfig())

    def validate_image(self, path: Path) -> Tuple[bool, Optional[str]]:
        """Validate image integrity."""
        try:
            img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if img is None:
                return False, "Failed to decode image"
            return True, None
        except Exception as e:
            return False, str(e)



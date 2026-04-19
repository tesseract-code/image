# ==========================================
# Adapter 3: Numpy (Scientific Data)
# ==========================================
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

from image.io.config import LoadConfig
from image.io.metadata import ImageMetadata
from image.settings.pixels import PixelFormat


class ImageLoaderAdapter(ABC):
    """
    Enhanced protocol for image loading backends with optimization,
    validation, and advanced metadata support.
    """

    @abstractmethod
    def load(self,
             path: Path,
             config: LoadConfig) -> Tuple[
        np.ndarray, PixelFormat, ImageMetadata]:
        """
        Loads image data with comprehensive configuration support.

        Args:
            path: Path to image file
            config: Loading configuration with all options

        Returns:
            Tuple containing:
            1. The raw numpy array (H, W, C)
            2. The actual PixelFormat of the returned array
            3. Extended metadata including EXIF, ICC, etc.
        """
        pass

    @abstractmethod
    def get_metadata(self, path: Path,
                     include_exif: bool = True) -> ImageMetadata:
        """
        Fast metadata extraction without loading full pixel data.

        Args:
            path: Path to image file
            include_exif: Whether to parse EXIF data
        """
        pass

    @abstractmethod
    def validate_image(self, path: Path) -> Tuple[bool, Optional[str]]:
        """
        Validate image integrity without full load.

        Returns:
            Tuple of (is_valid, error_message)
        """
        pass

    def _compute_hash_safe(self, data: np.ndarray, alg: str) -> str:
        """
        Helper: Computes hash ensuring memory consistency.
        Crucial: Different backends produce different strides/padding.
        Ensures data is C-contiguous before hashing to guarantee consistent results.
        """
        # Ensure contiguous memory layout (C-order) for deterministic hashing
        if not data.flags['C_CONTIGUOUS']:
            data = np.ascontiguousarray(data)

        hasher = hashlib.new(alg)
        hasher.update(data.tobytes())
        return hasher.hexdigest()


# ==========================================
# Enhanced Pillow Implementation
# ==========================================


# ==========================================
# Enhanced OpenCV Implementation
# ==========================================


# ==========================================
# Factory & API
# ==========================================

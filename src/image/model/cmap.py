import logging
from enum import unique, IntEnum, auto
from typing import Dict, Tuple
from typing import Optional, Union, Any

import numpy as np
import numpy.typing as npt
from PyQt6.QtGui import QColor
from matplotlib import cm
from matplotlib import colormaps
from matplotlib.colors import Colormap

from image.pipeline.operations.transform import apply_lut

logger = logging.getLogger(__name__)

# At top of file
LUMA_R = 0.299  # ITU-R BT.601 Red weight
LUMA_G = 0.587  # ITU-R BT.601 Green weight
LUMA_B = 0.114  # ITU-R BT.601 Blue weight


@unique
class LUTType(IntEnum):
    """Lookup table transformation types"""
    LINEAR = auto()
    LOG = auto()
    SQRT = auto()
    SQUARE = auto()


def normalize_value_for_lut(
        value: Union[np.generic, npt.NDArray[Any]],
        dtype: np.dtype
) -> Union[float, npt.NDArray[np.floating]]:
    """
    Normalize value to [0, 1] range based on dtype.

    Args:
        value: Input value(s) to normalize
        dtype: Data type of the value

    Returns:
        Normalized value(s) in [0, 1] range
    """
    if np.issubdtype(dtype, np.floating):
        # Assume float data is already [0, 1], clamp to be safe
        return np.clip(value, 0.0, 1.0)
    elif np.issubdtype(dtype, np.unsignedinteger):
        # Handle all unsigned integer types uniformly
        info = np.iinfo(dtype)
        return np.asarray(value, dtype=np.float64) / float(info.max)
    elif np.issubdtype(dtype, np.signedinteger):
        # For signed integers, normalize to full range
        info = np.iinfo(dtype)
        return (np.asarray(value, dtype=np.float64) - info.min) / (
                info.max - info.min)
    else:
        # Fallback: try to normalize as float
        return np.clip(np.asarray(value, dtype=np.float32), 0.0, 1.0)


def apply_colormap_to_value(
        value: Union[np.generic, npt.NDArray[Any]],
        lut: npt.NDArray[np.uint8],
        data_dtype: np.dtype
) -> Optional[npt.NDArray[np.uint8]]:
    """
    Apply colormap LUT to a single value or pixel (O(1) lookup).

    Args:
        value: Scalar value or 1D array (for multichannel, uses Luma conversion)
        lut: Colormap LUT array of shape (N, 3) where N is typically 256
        data_dtype: Original data type for proper normalization

    Returns:
        RGB triplet as uint8 array [R, G, B], or None if value is None

    Note:
        - For grayscale input: maps scalar to RGB
        - For multichannel input: converts to Luma (Y = 0.299*R + 0.587*G + 0.114*B)
        - LUT index = int(normalized_value * (len(lut) - 1))
    """
    if value is None:
        return None

    # Extract scalar from array if needed
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None

        # For multichannel (RGB), convert to luminance using ITU-R BT.601 weights
        if value.size >= 3:
            # Standard Luma coefficients (same as used in image processing)
            scalar_value = float(
                LUMA_R * value.flat[0] +
                LUMA_G * value.flat[1] +
                LUMA_B * value.flat[2]
            )
        else:
            # Grayscale or single channel
            scalar_value = float(value.flat[0])
    else:
        scalar_value = float(value)

    # Normalize to [0, 1]
    normalized = normalize_value_for_lut(scalar_value, data_dtype)

    # Map to LUT index
    lut_size = len(lut)
    index = int(np.clip(normalized * (lut_size - 1), 0, lut_size - 1))

    return lut[index]


def apply_colormap_to_region(
        region: npt.NDArray[Any],
        lut: npt.NDArray[np.uint8],
        out: Optional[npt.NDArray[np.uint8]] = None
) -> npt.NDArray[np.uint8]:
    """
    Apply colormap LUT to entire image region (vectorized).

    Args:
        region: Image region array. Shapes supported:
                - (H, W) for grayscale
                - (H, W, 1) for single-channel
                - (H, W, C) for multichannel (uses first channel)
        lut: Colormap LUT array of shape (N, 3) where N is typically 256
        out: output buffer

    Returns:
        RGB image array of shape (H, W, 3) as uint8

    Note:
        - Automatically normalizes based on input dtype
        - For multichannel input, uses only first channel
        - Output is always RGB uint8
        - Vectorized for performance
    """
    if region is None or region.size == 0:
        return np.array([], dtype=np.uint8)

    h, w = region.shape[:2]

    # --- 1. Memory Management (The "In-Place" Alternative) ---
    # If the user provided a buffer, use it. Otherwise, allocate new.
    if out is not None:
        if out.shape != (h, w, 3) or out.dtype != np.uint8:
            raise ValueError(
                f'Output buffer mismatch. Expected ({h},{w},3) uint8, got {out.shape} {out.dtype}')
        output_buffer = out
    else:
        output_buffer = np.zeros((h, w, 3), dtype=np.uint8)
    apply_lut(region, lut, output_buffer)

    return output_buffer


class ColormapModel:
    """
    Caches pre-computed colormap lookup tables.
    LUTs are 256x3 uint8 arrays for fast indexing.
    """

    def __init__(self, resolution: int = 256):
        self.resolution = resolution
        self._cache: Dict[Tuple[str, bool], np.ndarray] = {}

    def get_lut(self, cmap_name: str, reverse: bool = False) -> np.ndarray:
        """
        Get cached LUT or generate if not cached.

        Args:
            cmap_name: Matplotlib colormap name
            reverse: Whether to reverse the colormap

        Returns:
            uint8 array of shape (256, 3)
        """
        key = (cmap_name, reverse)

        if key not in self._cache:
            self._cache[key] = self._generate_lut(cmap_name, reverse)
            logger.debug(
                f"Generated and cached LUT: {cmap_name} (reverse={reverse})")

        return self._cache[key]

    def _generate_lut(self, cmap_name: str, reverse: bool) -> np.ndarray:
        """Generate a 256x3 uint8 LUT from matplotlib colormap"""
        try:
            cmap = cm.get_cmap(cmap_name)
        except ValueError:
            logger.warning(
                f"Unknown colormap '{cmap_name}', falling back to 'gray'")
            cmap = cm.get_cmap('gray')

        # Sample colormap at 256 points
        indices = np.linspace(0, 1, self.resolution)
        rgba = cmap(indices)  # (256, 4)

        # Extract RGB, convert to uint8
        lut = (rgba[:, :3] * 255).astype(np.uint8)

        if reverse:
            lut = lut[::-1].copy()  # Ensure contiguous after reversal

        return np.ascontiguousarray(lut)

    def clear(self):
        """Clear the cache"""
        self._cache.clear()

    def preload(self, cmap_names: list, reverse: bool = False):
        """Preload commonly used colormaps"""
        for name in cmap_names:
            self.get_lut(name, False)
            if reverse:
                self.get_lut(name, True)


"""
High-contrast color finder for matplotlib colormaps.
Optimized for overlay elements like crosshairs and ROI rectangles.
"""


class ColorOptimizer:
    """Find high-contrast colors for matplotlib colormap overlays."""

    # Pre-defined high-contrast candidates
    CANDIDATES = [
        (1.0, 1.0, 1.0),  # white
        (0.0, 0.0, 0.0),  # black
        (1.0, 0.0, 0.0),  # red
        (0.0, 1.0, 0.0),  # green
        (0.0, 0.0, 1.0),  # blue
        (1.0, 1.0, 0.0),  # yellow
        (1.0, 0.0, 1.0),  # magenta
        (0.0, 1.0, 1.0),  # cyan
    ]

    def get_contrasting_color(self, cmap, sample_points=50):
        """
        Find the best contrasting color for a colormap.

        Parameters
        ----------
        cmap : str or Colormap
            Matplotlib colormap to analyze
        sample_points : int
            Number of points to sample from colormap

        Returns
        -------
        tuple
            RGB color tuple (r, g, b) with values in [0, 1]
        """
        if isinstance(cmap, str):
            cmap = colormaps[cmap]

        # Sample colormap evenly
        samples = cmap(np.linspace(0, 1, sample_points))[:, :3]

        # Find candidate with maximum minimum contrast
        best_color = None
        best_min_contrast = -1

        for candidate in self.CANDIDATES:
            # Calculate contrast ratio against all samples
            contrasts = [self._contrast_ratio(candidate, sample)
                         for sample in samples]
            min_contrast = min(contrasts)

            if min_contrast > best_min_contrast:
                best_min_contrast = min_contrast
                best_color = candidate

        return best_color

    def get_contrasting_color_qt(self, cmap, sample_points=50):
        """
        Get contrasting color as QColor for Qt applications.

        Returns
        -------
        QColor
            Qt color object
        """
        r, g, b = self.get_contrasting_color(cmap, sample_points)
        return QColor(int(r * 255), int(g * 255), int(b * 255))

    def _contrast_ratio(self, color1, color2):
        """
        Calculate WCAG contrast ratio between two colors.

        Uses relative luminance for perceptually accurate contrast.
        Higher values = better contrast (21:1 is maximum).
        """
        lum1 = self._relative_luminance(color1)
        lum2 = self._relative_luminance(color2)

        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)

        return (lighter + 0.05) / (darker + 0.05)

    @staticmethod
    def _relative_luminance(rgb):
        """Calculate relative luminance using sRGB formula."""
        r, g, b = rgb[:3]

        # Apply gamma correction
        def gamma(c):
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

        r, g, b = gamma(r), gamma(g), gamma(b)

        # ITU-R BT.709 weights
        return 0.2126 * r + 0.7152 * g + 0.0722 * b

    def analyze_colormap(self, cmap, sample_points=50):
        """
        Get detailed contrast analysis for a colormap.

        Returns
        -------
        dict
            Analysis results including best color and contrast scores
        """
        if isinstance(cmap, str):
            cmap_name = cmap
            cmap = colormaps[cmap]
        else:
            cmap_name = getattr(cmap, 'name', 'unknown')

        samples = cmap(np.linspace(0, 1, sample_points))[:, :3]

        results = {}
        for candidate in self.CANDIDATES:
            contrasts = [self._contrast_ratio(candidate, s) for s in samples]
            results[candidate] = {
                'min': min(contrasts),
                'max': max(contrasts),
                'mean': np.mean(contrasts),
                'median': np.median(contrasts)
            }

        best = max(results.items(), key=lambda x: x[1]['min'])

        return {
            'colormap': cmap_name,
            'best_color': best[0],
            'best_min_contrast': best[1]['min'],
            'all_candidates': results
        }

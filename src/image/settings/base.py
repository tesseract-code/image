import logging
import threading
from typing import Any, NamedTuple, Optional, Tuple

from PyQt6.QtCore import QObject, pyqtSignal

from image.model.cmap import LUTType
from image.settings.pixels import PixelFormat
from image.settings.roi import ROI

logger = logging.getLogger(__name__)


class ImageSettingsSnapshot(NamedTuple):
    zoom: float
    pan_x: float
    pan_y: float
    rotation: float
    brightness: float
    contrast: float
    gamma: float
    gain: float
    offset: float
    color_balance_r: float
    color_balance_g: float
    color_balance_b: float
    invert: bool
    lut_enabled: bool
    lut_min: float
    lut_max: float
    lut_type: LUTType
    interpolation: bool
    colormap_enabled: bool
    colormap_name: str
    colormap_reverse: bool
    roi: Optional[Tuple[int, int, int, int]]
    img_format: PixelFormat
    norm_vmin: Optional[float]
    norm_vmax: Optional[float]


class ImageSettings(QObject):
    """Thread-safe image settings with validation"""
    changed = pyqtSignal()

    def __init__(self, img_format: PixelFormat = PixelFormat.RGB):
        super().__init__()
        self._lock = threading.RLock()

        self.roi: Optional[ROI] = None

        self.colormap_enabled: bool = False
        self.colormap_name: str = 'viridis'  # matplotlib colormap name
        self.colormap_reverse: bool = False
        # Normalization
        self.norm_vmin: Optional[float] = None
        self.norm_vmax: Optional[float] = None

        # Transform settings
        self.zoom_enabled: bool = False
        self.zoom: float = 1.0
        self.panning_enabled: bool = False
        self.pan_x: float = 0.0
        self.pan_y: float = 0.0
        self.rotation: float = 0.0

        # Processing settings
        self.brightness: float = 0.0
        self.contrast: float = 1.0
        self.gamma: float = 1.0
        self.gain: float = 1.0
        self.offset: float = 0.0

        # Color settings
        self.color_balance_r: float = 1.0
        self.color_balance_g: float = 1.0
        self.color_balance_b: float = 1.0
        self.invert: bool = False

        # LUT settings
        self.lut_enabled: bool = False
        self.lut_min: float = 0.0
        self.lut_max: float = 1.0
        self.lut_type: LUTType = LUTType.LINEAR

        # Display settings
        self.interpolation: bool = False

        self.format: PixelFormat = img_format

    def update_setting(self, name: str, value: Any) -> bool:
        """Thread-safe setting update with validation. Use if object is
        multithreaded over directly setting the data"""
        with self._lock:
            if not hasattr(self, name):
                logger.warning(f"Setting {name} does not exist")
                return False

            # Validate ranges
            if name == 'zoom' and not (0.01 <= value <= 100.0):
                logger.warning(f"Zoom value {value} out of range [0.01, 100.0]")
                return False
            elif name == 'gamma' and not (0.1 <= value <= 3.0):
                logger.warning(f"Gamma value {value} out of range [0.1, 3.0]")
                return False
            elif name == 'contrast' and not (0.1 <= value <= 3.0):
                logger.warning(
                    f"Contrast value {value} out of range [0.1, 3.0]")
                return False

            old_value = getattr(self, name)
            if old_value != value:
                setattr(self, name, value)
                self.changed.emit()
                return True
        return False

    def get_copy(self) -> ImageSettingsSnapshot:
        """
        Returns a lightweight, immutable snapshot of settings.
        Optimized for fast pickling over Multiprocessing queues.
        """
        with self._lock:
            # Using positionals (by order) is slightly faster than keywords,
            # but keywords are safer. We use the constructor directly.
            return ImageSettingsSnapshot(
                zoom=self.zoom,
                pan_x=self.pan_x,
                pan_y=self.pan_y,
                rotation=self.rotation,
                brightness=self.brightness,
                contrast=self.contrast,
                gamma=self.gamma,
                gain=self.gain,
                offset=self.offset,
                color_balance_r=self.color_balance_r,
                color_balance_g=self.color_balance_g,
                color_balance_b=self.color_balance_b,
                invert=self.invert,
                lut_enabled=self.lut_enabled,
                lut_min=self.lut_min,
                lut_max=self.lut_max,
                lut_type=self.lut_type,
                interpolation=self.interpolation,
                colormap_enabled=self.colormap_enabled,
                colormap_name=self.colormap_name,
                colormap_reverse=self.colormap_reverse,
                roi=self.roi.as_tupe() if self.roi else None,
                img_format=self.format,
                norm_vmin=self.norm_vmin,
                norm_vmax=self.norm_vmax,
            )


def create_default_settings_snapshot() -> ImageSettingsSnapshot:
    """Returns a default configuration for ImageSettingsSnapshot."""
    return ImageSettingsSnapshot(
        zoom=1.0,
        pan_x=0.0,
        pan_y=0.0,
        rotation=0.0,
        brightness=0.0,
        contrast=1.0,
        gamma=1.0,
        gain=1.0,
        offset=0.0,
        color_balance_r=1.0,
        color_balance_g=1.0,
        color_balance_b=1.0,
        invert=False,
        lut_enabled=False,
        lut_min=0.0,
        lut_max=255.0,
        lut_type=LUTType.LINEAR,  # Assumes LINEAR is a valid member of LUTType
        interpolation=True,
        colormap_enabled=False,
        colormap_name="gray",
        colormap_reverse=False,
        roi=None,
        img_format=PixelFormat.RGB,  # Assumes a standard default format
        norm_vmin=None,
        norm_vmax=None
    )

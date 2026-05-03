from dataclasses import dataclass, field
from typing import Optional, Dict, Any


@dataclass(frozen=True, slots=True)
class ImageMetadata:
    """Extended metadata extracted from image files."""
    # Basic info
    width: int
    height: int
    format: Optional[str]
    mode: str
    bit_depth: Optional[int] = None

    # EXIF
    exif_orientation: Optional[int] = None
    exif_data: Dict[str, Any] = field(default_factory=dict)

    # Color management
    has_icc_profile: bool = False
    icc_profile: Optional[bytes] = None
    color_space: Optional[str] = None

    # Multi-frame
    is_animated: bool = False
    frame_count: int = 1

    # Integrity
    file_hash: Optional[str] = None
    is_corrupt: bool = False

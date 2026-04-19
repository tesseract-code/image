import dataclasses
import struct
from typing import ClassVar, Dict, Any, NamedTuple, Optional

import numpy as np

from image.pipeline.stats import FrameStats
from image.settings.pixels import PixelFormat

type FrameSettings = Dict[str, Any] | NamedTuple | None


@dataclasses.dataclass(slots=True, frozen=True)
class FrameHeader:
    frame_id: float
    timestamp: float
    width: int
    height: int
    channels: int
    _pad: int = 0

    _STRUCT_FMT: ClassVar[str] = "<ddIIII"
    SIZE_BYTES: ClassVar[int] = struct.calcsize(_STRUCT_FMT)

    def pack(self) -> bytes:
        return struct.pack(
            self._STRUCT_FMT,
            self.frame_id,
            self.timestamp,
            self.width,
            self.height,
            self.channels,
            0  # Padding byte
        )

    @classmethod
    def unpack(cls, data: bytes) -> 'FrameHeader':
        if len(data) < cls.SIZE_BYTES:
            raise ValueError(
                f"Data buffer too small for FrameHeader. Expected {cls.SIZE_BYTES}, got {len(data)}")

        unpacked = struct.unpack(cls._STRUCT_FMT, data[:cls.SIZE_BYTES])

        return cls(*unpacked[:-1])


@dataclasses.dataclass(slots=True, frozen=True)
class RenderFrame:
    """
    The transport envelope.
    'image_view' is a VOLATILE view of the SHM Ring Buffer.
    """
    image_view: np.ndarray
    metadata: FrameStats
    format: Optional[PixelFormat] = None

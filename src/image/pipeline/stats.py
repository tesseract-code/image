import time
from dataclasses import dataclass, field
from typing import Tuple, Optional

import numpy as np

from image.pipeline.operations.transform import sample_image_stats


@dataclass(slots=True, frozen=True)
class FrameStats:
    """Computed analysis of the frame."""
    shape: Tuple[int, ...]
    dtype_str: str
    vmin: float
    vmax: float
    mean: float
    std: float
    dmin: float
    dmax: float
    processing_time_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)


def get_frame_stats(image: np.ndarray,
                    processing_time_ms:float = 0.0):
    timestamp_epoch = time.time()
    stats = sample_image_stats(image)
    dmin = float(stats.min)
    dmax = float(stats.max)
    return FrameStats(
        timestamp=timestamp_epoch,
        shape=image.shape,
        dtype_str=image.dtype.str,
        mean=float(stats.mean),
        std=float(stats.std),
        dmin=dmin,
        dmax=dmax,
        vmin=dmin,
        vmax=dmax,
        processing_time_ms=processing_time_ms,
    )

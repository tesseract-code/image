import logging
import time
from typing import Optional

import numpy as np

from cross_platform.qt6_utils.image.pipeline.operations.stats import (
    compute_image_stats)
from cross_platform.qt6_utils.image.pipeline.operations.transform import (
    apply_transformations)
from cross_platform.qt6_utils.image.pipeline.config import ProcessingConfig
from cross_platform.qt6_utils.image.pipeline.metadata import FrameStats

logger = logging.getLogger(__name__)

def noop_pipeline(
        image: np.ndarray,
        output_buffer: np.ndarray,
        config: ProcessingConfig
) -> Optional[FrameStats]:
    timestamp_epoch = time.time()
    start_ns = time.perf_counter_ns()
    stats = compute_image_stats(image)
    np.copyto(output_buffer, image, casting='unsafe')
    end_ns = time.perf_counter_ns()
    duration_ms = (end_ns - start_ns) / 1_000_000.0

    return FrameStats(
        timestamp=timestamp_epoch,
        shape=image.shape,
        dtype_str=image.dtype.str,
        mean=float(stats.mean),
        std=float(stats.std),
        dmin=float(stats.min),
        dmax=float(stats.max),
        vmin=float(stats.min),
        vmax=float(stats.max),
        processing_time_ms=duration_ms
    )

def image_pipeline(
        image: np.ndarray,
        output_buffer: np.ndarray,
        config: ProcessingConfig
) -> Optional[FrameStats]:
    """
    Executes the pipeline stage and captures telemetry.
    """
    # 1. Capture Timing & Context (Low overhead)
    # Use time() for the record timestamp, perf_counter_ns() for delta measurement
    timestamp_epoch = time.time()
    start_ns = time.perf_counter_ns()

    if image is None:
        return None

    try:
        # 2. Compute Statistics (Bottleneck A)
        # We need stats for both normalization AND metadata.
        stats = compute_image_stats(image)
        # 4. Execute Pipeline Stage (Bottleneck B)
        # Dispatches to either Float/Analysis or Int/Visual path based on LUT presence
        final_min, final_max = apply_transformations(
            image=image,
            output_buffer=output_buffer,
            config=config,
            stats=stats,
            lut=config.colormap_lut
        )

        # 5. Calculate Duration (High Precision)
        # Integer math is faster and safer here
        end_ns = time.perf_counter_ns()
        duration_ms = (end_ns - start_ns) / 1_000_000.0

        # 6. Construct Metadata
        # Using slots=True class makes this instantiation cheap enough for 300FPS
        return FrameStats(
            timestamp=timestamp_epoch,
            shape=output_buffer.shape,
            dtype_str=output_buffer.dtype.str,
            mean=float(stats.mean),
            std=float(stats.std),
            dmin=float(stats.min),
            dmax=float(stats.max),
            vmin=final_min,
            vmax=final_max,
            processing_time_ms=duration_ms
        )

    except Exception as e:
        # In production, we catch here to prevent the whole pipeline thread from dying,
        # log the error, and return None so the display can skip this frame.
        logger.error(f"Frame processing failed: {e}", exc_info=True)
        return None

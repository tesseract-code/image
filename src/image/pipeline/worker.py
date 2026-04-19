import logging
import multiprocessing
import queue
from multiprocessing import shared_memory
from typing import Dict, Optional, Tuple, Any, Callable

import numpy as np

from image.model.cmap import ColormapModel
from image.pipeline.config import ProcessingConfig
from image.pipeline.frame import FrameHeader
from image.pipeline.operations.process import noop_pipeline
from image.pipeline.stats import FrameStats
from pycore.cpu import set_high_priority
from pycore.shm import SharedMemoryRingBuffer, cleanup_shm_cache

logger = logging.getLogger(__name__)

# Type Alias
type ProcessCallable = Callable[
    [np.ndarray, np.ndarray, ProcessingConfig], Optional[FrameStats]]


class ImageWorkerContext:
    __slots__ = ('output_ring', 'input_shm_cache', 'config', 'cmap_cache',
                 '_last_output_full_view')

    def __init__(self, buffer_count: int):
        self.output_ring = SharedMemoryRingBuffer(
            buffer_count=max(1, buffer_count))
        self.input_shm_cache: Dict[str, shared_memory.SharedMemory] = {}
        self.config = ProcessingConfig()
        self.cmap_cache = ColormapModel()
        self._last_output_full_view: Optional[np.ndarray] = None
        self._preload_colormaps()

    def _preload_colormaps(self):
        try:
            import matplotlib.pyplot as plt
            common_cmaps = sorted([c for c in plt.colormaps() if "_r" not in c])
            self.cmap_cache.preload(common_cmaps)
        except Exception:
            pass

    def update_config(self, settings: Optional[dict]):
        if settings is None: return
        self.config = ProcessingConfig.from_settings(settings)
        if self.config.colormap_enabled:
            self.config.colormap_lut = self.cmap_cache.get_lut(
                self.config.colormap_name, self.config.colormap_reverse
            )

    def acquire_input_image(self, name: str, shape: tuple, dtype: Any) -> Tuple[
        Optional[np.ndarray], Optional[FrameHeader]]:
        if name not in self.input_shm_cache:
            try:
                self.input_shm_cache[name] = shared_memory.SharedMemory(
                    name=name)
            except FileNotFoundError:
                return None, None

        shm = self.input_shm_cache[name]
        pixel_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)

        if shm.size < (pixel_bytes + FrameHeader.SIZE_BYTES):
            return None, None

        try:
            header = FrameHeader.unpack(shm.buf[:FrameHeader.SIZE_BYTES])
            # Use offset to skip header
            img_view = np.ndarray(
                shape, dtype=dtype, buffer=shm.buf,
                offset=FrameHeader.SIZE_BYTES
            )
            return img_view, header
        except Exception:
            return None, None

    def prepare_output_buffer(self,
                              in_shape: tuple,
                              in_dtype: np.dtype
                              ) -> Tuple[Optional[str], Optional[np.ndarray]]:
        """
        Prepares output buffer.
        FIX: Now accepts in_dtype to preserve type identity when processing logic allows.
        """
        self._last_output_full_view = None

        out_shape = self.config.get_output_shape(in_shape, self.config.fmt)
        out_dtype = self.config.get_output_dtype(in_dtype)

        pixel_nbytes = int(np.prod(out_shape) * np.dtype(out_dtype).itemsize)
        total_alloc_size = pixel_nbytes + FrameHeader.SIZE_BYTES

        out_name, output_flat_bytes = self.output_ring.alloc_buffer(
            total_alloc_size)
        self._last_output_full_view = output_flat_bytes

        # Strict slicing to exclude Jitter Padding
        start_idx = FrameHeader.SIZE_BYTES
        end_idx = start_idx + pixel_nbytes

        try:
            active_bytes = output_flat_bytes[start_idx:end_idx]
            typed_view = active_bytes.view(out_dtype)
            output_view = typed_view.reshape(out_shape)
            return out_name, output_view
        except ValueError as e:
            logger.error(f"Output Reshape Error: {e}")
            return None, None

    def write_output_header(self, metadata: FrameStats,
                            base_header: Optional[FrameHeader] = None):
        if self._last_output_full_view is None: return

        h, w = metadata.shape[:2]
        c = metadata.shape[2] if len(metadata.shape) > 2 else 1
        fid = base_header.frame_id if base_header else 0.0
        ts = base_header.timestamp if base_header else 0.0

        header = FrameHeader(
            frame_id=fid, timestamp=ts, width=w, height=h, channels=c
        )
        # Write header to offset 0
        self._last_output_full_view[:FrameHeader.SIZE_BYTES] = np.frombuffer(
            header.pack(), dtype=np.uint8
        )

    def close(self):
        cleanup_shm_cache(self.input_shm_cache, unlink=True)
        self.input_shm_cache.clear()
        if self.output_ring:
            self.output_ring.cleanup()
            self.output_ring = None


def image_worker_entry(input_queue: multiprocessing.Queue,
                       output_queue: multiprocessing.Queue,
                       buffer_count: int,
                       stop_event: multiprocessing.Event,
                       is_high_priority: bool = False,
                       processing_callable: ProcessCallable = noop_pipeline):
    if is_high_priority:
        set_high_priority("ImageProcessingPipeline")

    ctx = ImageWorkerContext(buffer_count)

    try:
        while not stop_event.is_set():
            try:
                task = input_queue.get(timeout=0.05)
                if task is None: continue
            except queue.Empty:
                continue

            try:
                shm_name, shape, dtype, settings = task
                ctx.update_config(settings)

                raw_image, in_header = ctx.acquire_input_image(shm_name, shape,
                                                               dtype)
                if raw_image is None or raw_image.size < 1: continue

                # Copy input to ensure stable processing source
                working_image = raw_image.copy()

                out_name, output_view = ctx.prepare_output_buffer(shape,
                                                                  raw_image.dtype)
                if output_view is None: continue

                metadata = processing_callable(working_image, output_view,
                                               ctx.config)

                if metadata is not None:
                    ctx.write_output_header(metadata, in_header)
                    output_queue.put((out_name, metadata))

            except Exception as e:
                logger.error(f"Worker Loop Error: {e}")

    finally:
        ctx.close()

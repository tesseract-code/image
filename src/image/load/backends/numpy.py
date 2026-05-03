from pathlib import Path
from typing import override, Tuple, Optional

import numpy as np
from image.load.config import LoadConfig

from image.load.interface import ImageLoaderAdapter
from image.load.metadata import ImageMetadata
from image.settings.pixels import PixelFormat


class NumpyAdapter(ImageLoaderAdapter):
    """
    Adapter for .npy / .npz files.
    Allows treating tensor files as images for visualization/processing.
    """

    @override
    def validate_image(self, path: Path) -> Tuple[bool, Optional[str]]:
        try:
            # np.load with mmap_mode='r' reads header only
            np.load(str(path), mmap_mode='r')
            return True, None
        except Exception as e:
            return False, str(e)

    @override
    def get_metadata(self, path: Path) -> ImageMetadata:
        # Read header via memory map
        arr = np.load(str(path), mmap_mode='r')
        if isinstance(arr, np.lib.npyio.NpzFile):
            # Inspect first array in archive
            arr = arr[arr.files[0]]

        return ImageMetadata(
            width=arr.shape[1] if arr.ndim > 1 else arr.shape[0],
            height=arr.shape[0] if arr.ndim > 1 else 1,
            format='NUMPY',
            mode='N/A',
            dtype_str=str(arr.dtype)
        )

    @override
    def load(self, path: Path, config: LoadConfig) -> Tuple[
        np.ndarray, PixelFormat, ImageMetadata]:
        try:
            raw = np.load(str(path))

            # Handle .npz archives
            if isinstance(raw, np.lib.npyio.NpzFile):
                # Heuristic: Look for common keys, else take first
                keys = raw.files
                target_key = next(
                    (k for k in ['image', 'data', 'arr_0'] if k in keys),
                    keys[0])
                data = raw[target_key]
            else:
                data = raw

            # Validations
            if data.ndim not in (2, 3):
                raise ValueError(
                    f"Numpy array must be 2D or 3D, got {data.ndim}")

            # Infer Format
            current_fmt = PixelFormat.infer_from_shape(data.shape)

            # Conversions (Manual, since no CV2/PIL helpers)
            # If target format requested is different, simple channel swaps or replication
            # can be implemented here. For now, we assume strict adherence or fail.
            if config.target_format and config.target_format != current_fmt:
                # Basic grayscale -> RGB expansion
                if (current_fmt == PixelFormat.MONOCHROME and
                        config.target_format == PixelFormat.RGB):
                    data = np.stack((data,) * 3, axis=-1)
                    current_fmt = PixelFormat.RGB

            if config.flip_vertically:
                data = data[::-1, ...]

            meta = self.get_metadata(path)
            # if config.compute_hash:
            #     meta.file_hash = self._compute_hash_safe(data,
            #                                              config.hash_algorithm)

            return data, current_fmt, meta

        except Exception as e:
            raise ValueError(f"Failed to load numpy file: {e}")


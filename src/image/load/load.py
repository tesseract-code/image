"""
Image loading utilities for OpenGL texture pipeline.
Python equivalent of stb_image (stbi_load) functionality.

Supports: JPEG, PNG, BMP, TGA, GIF, TIFF, WebP and more.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Sequence, TypeAlias, Tuple
from typing import Optional, Union

import numpy as np

from image.settings.pixels import PixelFormat
from image.load.config import LoadConfig, ImageReadFlags
from image.load.factory import get_adapter, Backend
from image.load.metadata import ImageMetadata
from image.utils.data import PixelBuffer

logger = logging.getLogger(__name__)

PathLike: TypeAlias = Union[str, Path]

_GLOBAL_EXECUTOR: Optional[ThreadPoolExecutor] = None
_DEFAULT_MAX_WORKERS: int = 8


def _get_executor(max_workers: int = 4) -> ThreadPoolExecutor:
    """Lazy safe initialization of global thread pool."""
    global _GLOBAL_EXECUTOR
    if _GLOBAL_EXECUTOR is None:
        _GLOBAL_EXECUTOR = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="ImgLoad"
        )
    return _GLOBAL_EXECUTOR


def shutdown_executor(wait: bool = True):
    """Shutdown the global thread pool executor."""
    global _GLOBAL_EXECUTOR
    if _GLOBAL_EXECUTOR:
        _GLOBAL_EXECUTOR.shutdown(wait=wait)
        _GLOBAL_EXECUTOR = None


def load_image(
        file_path: PathLike,
        config: Optional[LoadConfig] = None,
        backend: Optional[Backend] = None,
        ensure_contiguous: bool = True
) -> tuple[PixelBuffer, ImageMetadata]:
    """
    Primary entry point for loading images into PixelBuffer.

    Args:
        file_path: Path to image file
        config: Loading configuration. If None, uses defaults:
                - Infers format from image
                - Applies EXIF orientation
                - Validates integrity
        backend: Force specific backend ('pillow', 'opencv', 'auto')
                 If None, uses global default
        ensure_contiguous: Force C-contiguous memory layout

    Returns:
        PixelBuffer, or (PixelBuffer, ImageMetadata) if return_metadata=True
    """
    path = Path(file_path).resolve()

    # Use defaults if no config provided
    if config is None:
        config = LoadConfig(
            target_format=None,  # Infer from image
            flip_vertically=False,
            apply_exif_orientation=True,
            validate_integrity=True,
        )

    # Delegate to adapter
    adapter = get_adapter(backend)
    raw_data, fmt, metadata = adapter.load(path=path, config=config)

    # Enforce memory contiguity for buffer requirements
    if ensure_contiguous and not raw_data.flags['C_CONTIGUOUS']:
        raw_data = np.ascontiguousarray(raw_data)

    # Wrap in PixelBuffer
    buffer = PixelBuffer(data=raw_data,
                         width=metadata.width,
                         height=metadata.height,
                         pixel_fmt=fmt)

    return buffer, metadata


# ==========================================
# Convenience Wrappers
# ==========================================

def load_image_rgba(
        path: PathLike,
        flip: bool = True,
        apply_exif: bool = True,
        backend: Optional[Backend] = None
) -> Tuple[PixelBuffer, ImageMetadata]:
    """
    Load image as RGBA format.

    Args:
        path: Path to image
        flip: Flip vertically (OpenGL convention)
        apply_exif: Apply EXIF orientation
        backend: Override default backend
    """
    config = LoadConfig(
        target_format=PixelFormat.RGBA,
        flip_vertically=flip,
        apply_exif_orientation=apply_exif,
        preserve_transparency=True,
    )
    return load_image(path, config=config, backend=backend)


def load_image_rgb(
        path: PathLike,
        flip: bool = True,
        apply_exif: bool = True,
        backend: Optional[Backend] = None
) -> Tuple[PixelBuffer, ImageMetadata]:
    """
    Load image as RGB format.

    Args:
        path: Path to image
        flip: Flip vertically (OpenGL convention)
        apply_exif: Apply EXIF orientation
        backend: Override default backend
    """
    config = LoadConfig(
        target_format=PixelFormat.RGB,
        flip_vertically=flip,
        apply_exif_orientation=apply_exif,
    )
    return load_image(path, config=config, backend=backend)


def load_image_bgr(
        path: PathLike,
        flip: bool = False,
        apply_exif: bool = False,
        backend: Optional[Backend] = None
) -> Tuple[PixelBuffer, ImageMetadata]:
    """
    Load image as BGR format (OpenCV native).

    Args:
        path: Path to image
        flip: Flip vertically
        apply_exif: Apply EXIF orientation (usually False for CV pipelines)
        backend: Override default backend (consider 'opencv' for BGR)
    """
    config = LoadConfig(
        target_format=PixelFormat.BGR,
        flip_vertically=flip,
        apply_exif_orientation=apply_exif,
    )
    return load_image(path, config=config, backend=backend)


def load_image_gray(
        path: PathLike,
        flip: bool = True,
        apply_exif: bool = True,
        backend: Optional[Backend] = None
) -> Tuple[PixelBuffer, ImageMetadata]:
    """
    Load image as grayscale/monochrome.

    Args:
        path: Path to image
        flip: Flip vertically (OpenGL convention)
        apply_exif: Apply EXIF orientation
        backend: Override default backend
    """
    config = LoadConfig(
        target_format=PixelFormat.MONOCHROME,
        flip_vertically=flip,
        apply_exif_orientation=apply_exif,
    )
    return load_image(path, config=config, backend=backend)


def load_thumbnail(
        path: PathLike,
        size: tuple[int, int] = (256, 256),
        fmt: Optional[PixelFormat] = None,
        backend: Optional[Backend] = None
) -> Tuple[PixelBuffer, ImageMetadata]:
    """
    Efficiently load image thumbnail.

    Uses draft mode for JPEG when possible (Pillow backend).

    Args:
        path: Path to image
        size: Maximum dimensions (width, height)
        fmt: Target format, or None to infer
        backend: Override default backend
    """
    config = LoadConfig(
        target_format=fmt,
        thumbnail_size=size,
        apply_exif_orientation=True,
        reducing_gap=2.0,  # Quality optimization
    )
    return load_image(path, config=config, backend=backend)


def load_validated(
        path: PathLike,
        config: Optional[LoadConfig] = None,
        compute_hash: bool = True,
        backend: Optional[Backend] = None
) -> tuple[PixelBuffer, ImageMetadata]:
    """
    Load with full validation and integrity checking.

    Suitable for user uploads or security-sensitive contexts.

    Args:
        path: Path to image
        config: Optional config (validation forced regardless)
        compute_hash: Compute SHA256 hash of pixel data
        backend: Override default backend

    Returns:
        (buffer, metadata) tuple with hash and validation info
    """
    if config is None:
        config = LoadConfig()

    # Force validation options
    config.validate_integrity = True
    config.compute_hash = compute_hash
    config.hash_algorithm = 'sha256' if compute_hash else 'md5'

    return load_image(path, config=config, backend=backend)


# ==========================================
# Batch Loading API
# ==========================================

def batch_load_images(
        file_paths: Sequence[PathLike],
        config: Optional[LoadConfig] = None,
        backend: Optional[Backend] = None,
        executor: Optional[ThreadPoolExecutor] = None,
        return_errors: bool = False,
        max_workers: int = 4
) -> List[Union[Tuple[PixelBuffer, ImageMetadata], Exception]]:
    """
    Concurrent batch image loading with thread pool.

    Args:
        file_paths: Sequence of image paths to load
        config: Shared loading configuration for all images
        backend: Backend to use for all loads
        executor: Custom thread pool, or None to use global pool
        return_errors: If True, exceptions are returned in list.
                       If False, failed loads are logged and omitted.
        max_workers: Number of worker threads if using global pool

    Returns:
        List of PixelBuffer objects (or Exceptions if return_errors=True)

    Example:
        paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
        config = LoadConfig(max_dimension=1024, apply_exif_orientation=True)
        buffers = batch_load_images(paths, config=config, max_workers=8)
    """
    if not file_paths:
        return []

    exec_instance = executor or _get_executor(max_workers)

    # Use defaults if no config provided
    if config is None:
        config = LoadConfig(
            apply_exif_orientation=True,
            validate_integrity=True,
        )

    # Capture closure for the task
    def _task(p: PathLike
              ) -> Union[Tuple[PixelBuffer, ImageMetadata], Exception, None]:
        try:
            return load_image(p, config=config, backend=backend)
        except Exception as e:
            if return_errors:
                return e
            logger.warning(f"Failed to load {p}: {e}")
            return None

    # Map ensures preservation of order matching 'file_paths'
    results = []
    for res in exec_instance.map(_task, file_paths):
        if res is not None:
            results.append(res)

    return results


def extract_image_file_metadata(
        path: PathLike,
        include_exif: bool = True,
        backend: Optional[Backend] = None
) -> ImageMetadata:
    """
    Extract image metadata without loading full pixel data.

    Fast operation suitable for scanning image directories.

    Args:
        path: Path to image file
        include_exif: Parse EXIF data
        backend: Override default backend

    Returns:
        ImageMetadata with dimensions, format, EXIF, etc.

    Example:
        metadata = get_image_metadata('photo.jpg')
        print(f"{metadata.width}x{metadata.height}")
        print(f"Camera: {metadata.exif_data.get('Model')}")
    """
    adapter = get_adapter(backend)
    return adapter.get_metadata(Path(path), include_exif=include_exif)


def validate_image_file(
        path: PathLike,
        backend: Optional[Backend] = None
) -> tuple[bool, Optional[str]]:
    """
    Validate image file integrity without full load.

    Args:
        path: Path to image file
        backend: Override default backend

    Returns:
        (is_valid, error_message) tuple
    """
    adapter = get_adapter(backend)
    return adapter.validate_image(Path(path))


# ==========================================
# Utility Functions
# ==========================================

def create_config_for_web_upload(
        max_dimension: int = 2048,
        compute_hash: bool = True
) -> LoadConfig:
    """
    Create optimal configuration for web upload handling.

    Args:
        max_dimension: Maximum image dimension
        compute_hash: Compute SHA256 for deduplication

    Returns:
        LoadConfig optimized for web uploads
    """
    return LoadConfig(
        max_dimension=max_dimension,
        apply_exif_orientation=True,
        validate_integrity=True,
        compute_hash=compute_hash,
        hash_algorithm='sha256',
        preserve_transparency=True,
    )


def create_config_for_cv_pipeline(
        target_format: PixelFormat = PixelFormat.BGR,
        max_dimension: Optional[int] = None
) -> LoadConfig:
    """
    Create optimal configuration for computer vision pipeline.

    Args:
        target_format: Pixel format (usually BGR for OpenCV)
        max_dimension: Optional size constraint

    Returns:
        LoadConfig optimized for CV processing
    """
    return LoadConfig(
        target_format=target_format,
        apply_exif_orientation=False,  # Want raw sensor data
        flip_vertically=False,
        validate_integrity=False,  # Speed over safety
        max_dimension=max_dimension,
        flags=ImageReadFlags.UNCHANGED,
    )


def create_config_for_photography(
        max_dimension: Optional[int] = None
) -> LoadConfig:
    """
    Create optimal configuration for photography/editing workflows.

    Args:
        max_dimension: Optional size constraint

    Returns:
        LoadConfig optimized for professional photography
    """
    return LoadConfig(
        target_format=PixelFormat.RGB,
        max_dimension=max_dimension,
        apply_exif_orientation=True,
        strip_exif=False,  # Preserve metadata
        icc_profile_handling='preserve',  # Color accuracy
        preserve_transparency=True,
        reducing_gap=3.0,  # Highest quality
        validate_integrity=True,
    )


# ==========================================
# Context Manager for Batch Operations
# ==========================================

class ImageBatchLoader:
    """
    Context manager for efficient batch loading with custom thread pool.

    Example:
        with ImageBatchLoader(max_workers=8) as loader:
            buffers = loader.load(paths, config=config)
    """

    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.executor: Optional[ThreadPoolExecutor] = None

    def __enter__(self):
        self.executor = ThreadPoolExecutor(
            max_workers=self.max_workers,
            thread_name_prefix="BatchImgLoad"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.executor:
            self.executor.shutdown(wait=True)
        return False

    def load(
            self,
            file_paths: Sequence[PathLike],
            config: Optional[LoadConfig] = None,
            backend: Optional[Backend] = None,
            return_errors: bool = False
    ) -> List[Union[PixelBuffer, Exception]]:
        """Load images using this batch loader's thread pool."""
        return batch_load_images(
            file_paths,
            config=config,
            backend=backend,
            executor=self.executor,
            return_errors=return_errors
        )

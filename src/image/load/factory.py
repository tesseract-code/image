import logging
from enum import unique, StrEnum
from typing import Dict, Optional

from image.load.backends.numpy import NumpyAdapter
from image.load.backends.opencv import Cv2Adapter
from image.load.backends.pillow import PillowAdapter
from image.load.interface import ImageLoaderAdapter

logger = logging.getLogger(__name__)


@unique
class Backend(StrEnum):
    PILLOW = "pillow"
    OPENCV = "opencv"
    NUMPY = "numpy"
    AUTO = "auto"


_ADAPTERS: Dict[Backend, ImageLoaderAdapter] = {}
_DEFAULT_BACKEND: Backend = Backend.OPENCV
_CURRENT_BACKEND: Optional[Backend] = None


def get_adapter(backend: Optional[Backend] = None) -> ImageLoaderAdapter:
    """
    Factory to get the requested or default adapter singleton.

    Args:
        backend: Specific backend to use, or None for current default

    Returns:
        ImageLoaderAdapter instance (singleton per backend)
    """
    target = backend or _CURRENT_BACKEND

    if target == Backend.AUTO:
        try:
            target = Backend.OPENCV
            Cv2Adapter()
        except ImportError:
            target = Backend.PILLOW

    if target in _ADAPTERS:
        return _ADAPTERS[target]

    if target == Backend.PILLOW:
        adapter = PillowAdapter()
    elif target == Backend.OPENCV:
        adapter = Cv2Adapter()
    elif target == Backend.NUMPY:
        adapter = NumpyAdapter()
    else:
        raise ValueError(f"Unknown backend: {target}")

    _ADAPTERS[target] = adapter
    logger.debug(f"Initialized {target.value} adapter")
    return adapter


def set_default_backend(backend: Backend):
    """
    Sets the global default backend for load_image.

    Args:
        backend: Backend to use as default
    """
    global _CURRENT_BACKEND
    get_adapter(backend)
    _CURRENT_BACKEND = backend
    logger.info(f"Image loader backend switched to: {backend.value}")

from __future__ import annotations

import ctypes
import logging
from typing import Optional

import numpy as np

from image.gl.backend import GL
from image.gl.errors import GLUploadError
from image.gl.pbo.constants import _SENTINEL
from image.gl.types import GLsizeiptr, GLintptr, GLbitfield
from image.settings.pixels import PixelFormat, broadcast_to_format
from image.utils.data import ensure_contiguity as _ensure_contiguity
from pycore.log.ctx import ContextAdapter
from pycore.mtcopy import tuned_parallel_copy as _fast_copy

logger = ContextAdapter(logging.getLogger(__name__), {})


def calculate_pixel_alignment(gl_type: int, gl_format: int) -> int:
    """
    Return the GL_UNPACK_ALIGNMENT value appropriate for a format/type pair.

    ==================  ===========  ===========
    gl_type             gl_format    alignment
    ==================  ===========  ===========
    GL_FLOAT (4 bytes)  any          4
    GL_UNSIGNED_BYTE    GL_RGBA      4
    GL_UNSIGNED_BYTE    anything     1
    anything else       any          1
    ==================  ===========  ===========
    """
    if gl_type == GL.GL_FLOAT:
        return 4
    if gl_type == GL.GL_UNSIGNED_BYTE:
        return 4 if gl_format == GL.GL_RGBA else 1
    return 1


def configure_pixel_storage(
        gl_type: int,
        gl_format: int,
        row_length: int = 0,
        skip_pixels: int = 0,
        skip_rows: int = 0,
) -> None:
    """
    Set glPixelStorei parameters to match the layout of source pixel data.

    Alignment is derived automatically via calculate_pixel_alignment.
    """
    alignment = calculate_pixel_alignment(gl_type, gl_format)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, alignment)
    GL.glPixelStorei(GL.GL_UNPACK_ROW_LENGTH, row_length)
    GL.glPixelStorei(GL.GL_UNPACK_SKIP_PIXELS, skip_pixels)
    GL.glPixelStorei(GL.GL_UNPACK_SKIP_ROWS, skip_rows)


def extract_pointer(ptr_obj: object) -> Optional[int]:
    """
    Normalise a PyOpenGL mapped-buffer pointer to a plain int.

    Returns the pointer address, or None if extraction fails.
    """
    value = getattr(ptr_obj, "value", _SENTINEL)
    if value is not _SENTINEL:
        return value  # may be None for a null pointer — caller checks
    try:
        return ctypes.cast(ptr_obj, ctypes.c_void_p).value
    except Exception:
        return None


def memmove_pbo(pbo_id: int, data: np.ndarray) -> bool:
    """
    Copy a CPU array into a PBO using the map/copy/unmap path.

    Pipeline: CPU array → glMapBufferRange → fast_copy → glUnmapBuffer

    The PBO is left bound to GL_PIXEL_UNPACK_BUFFER on return.
    The caller is responsible for unbinding after glTexSubImage2D.

    Returns:
        True on success; False if the PBO could not be mapped.
    """
    nbytes = data.nbytes
    GL.glBindBuffer(GL.GL_PIXEL_UNPACK_BUFFER, pbo_id)
    GL.glBufferData(GL.GL_PIXEL_UNPACK_BUFFER, GLsizeiptr(nbytes), None,
                    GL.GL_STREAM_DRAW)

    ptr_obj = GL.glMapBufferRange(
        GL.GL_PIXEL_UNPACK_BUFFER,
        GLintptr(0),
        GLsizeiptr(nbytes),
        GLbitfield(GL.GL_MAP_WRITE_BIT | GL.GL_MAP_INVALIDATE_BUFFER_BIT),
    )
    if not ptr_obj:
        logger.error("memmove_pbo: glMapBufferRange returned NULL for PBO %d",
                     pbo_id)
        return False

    ptr_int = extract_pointer(ptr_obj)
    if ptr_int is None:
        logger.error("memmove_pbo: could not extract pointer for PBO %d",
                     pbo_id)
        return False

    _fast_copy(ptr_int, _ensure_contiguity(data))
    GL.glUnmapBuffer(GL.GL_PIXEL_UNPACK_BUFFER)
    return True


def write_pbo_buffer(
        pbo_array: np.ndarray,
        image: np.ndarray,
        pixel_fmt: "PixelFormat",
) -> None:
    """
    Write image data into a mapped PBO buffer, broadcasting channels if needed.

    Requires the optional image.settings.pixels dependency.

    Args:
        pbo_array : Mapped buffer array from PBOUploadManager.acquire_next_writeable.
        image     : Source image (HxW or HxWxC).
        pixel_fmt : PixelFormat describing the channel layout of pbo_array.

    Raises:
        ImportError   : If the image.settings.pixels package is not available.
        GLUploadError : If the broadcast shape does not match pbo_array.shape.
    """
    broadcasted = broadcast_to_format(image, pixel_fmt, copy=False)
    if broadcasted.shape != pbo_array.shape:
        raise GLUploadError(
            "Broadcast image shape %s does not match PBO buffer shape %s. "
            "Ensure the PBO was acquired with the correct width, height, and "
            "channel count for this image." % (
                broadcasted.shape, pbo_array.shape)
        )
    pbo_array[:] = broadcasted

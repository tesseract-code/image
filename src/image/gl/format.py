"""
gl_format.py
============
OpenGL texture format resolution for PyOpenGL/NumPy image pipelines.

Maps :class:`~cross_platform.qt6_utils.image.settings.pixels.PixelFormat`
enum members and NumPy dtype strings to the three GL token tuple required by
``glTexImage2D`` / ``glTexSubImage2D``:

    ``(gl_format, gl_internal_format, gl_type)``

All resolution is performed via cached helpers so repeated calls for the same
``(format, dtype)`` combination cost a single dict lookup.

Supported dtype range
---------------------
``uint8``, ``int8``, ``uint16``, ``int16``, ``float16``, ``float32``.

``float64`` is explicitly **not** supported: OpenGL has no 64-bit texture
storage format.  Callers holding ``float64`` arrays must downcast to
``float32`` before upload::

    image = image.astype(np.float32)

Planar YUV formats (YUV420, YUV422, NV12, NV21)
------------------------------------------------
These are multi-plane formats that cannot be fully represented as a single
``GL_TEXTURE_2D``.  They are mapped to ``GL_RED`` (luma plane only) as a
best-effort approximation.  Callers that need correct chroma reproduction
must convert to RGB/RGBA on the CPU before calling :func:`get_gl_texture_spec`.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import TypeAlias, Union

import numpy as np

from cross_platform.qt6_utils.image.gl.backend import GL
from cross_platform.qt6_utils.image.gl.error import GLTextureError
from cross_platform.qt6_utils.image.settings.pixels import PixelFormat

__all__ = [
    "GLTextureSpec",
    "get_gl_texture_spec",
]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

#: The three-token tuple consumed by ``glTexImage2D`` and ``glTexSubImage2D``:
#: ``(gl_format, gl_internal_format, gl_type)``.
GLTextureSpec: TypeAlias = tuple[int, int, int]


# ---------------------------------------------------------------------------
# Internal resolution helpers (cached)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=32)
def _resolve_gl_dtype_params(dtype_name: str) -> tuple[int, str]:
    """
    Map a canonical NumPy dtype name to ``(GL_TYPE token, internal-format suffix)``.

    The suffix is the string appended to the base internal-format name to form
    the final GL constant name, e.g. ``"16F"`` → ``"GL_RGB16F"``.

    Results are cached by ``dtype_name`` so the match is executed only once
    per dtype seen during the process lifetime.

    Args:
        dtype_name: Canonical string from ``np.dtype.name`` (e.g. ``"float32"``).

    Returns:
        ``(gl_type, suffix)`` where ``gl_type`` is the GL transfer-type token
        and ``suffix`` is appended to the base internal-format string.

    Raises:
        ValueError: For ``"float64"`` (no 64-bit GL texture storage exists)
            or any other unsupported dtype.

    Note:
        ``"int8"`` maps to the ``_SNORM`` suffix family.  Signed-normalised
        textures interpret ``[-128, 127]`` as ``[-1.0, 1.0]`` in the shader.
        If unsigned normalised storage is needed, convert to ``uint8`` first.
    """
    match dtype_name:
        case "uint8":
            return GL.GL_UNSIGNED_BYTE, "8"
        case "int8":
            # Signed normalised: shader samples in [-1.0, 1.0]
            return GL.GL_BYTE, "8_SNORM"
        case "uint16":
            return GL.GL_UNSIGNED_SHORT, "16"
        case "int16":
            return GL.GL_SHORT, "16_SNORM"
        case "float16":
            return GL.GL_HALF_FLOAT, "16F"
        case "float32":
            return GL.GL_FLOAT, "32F"
        case "float64":
            # GL defines no 64-bit texture internal formats.  GL_DOUBLE exists
            # as a vertex-attribute type only.  Silently mapping float64 to
            # GL_DOUBLE with a 32F suffix would produce a transfer/storage
            # mismatch, corrupting data.  The caller must downcast.
            raise ValueError(
                "float64 is not a supported GL texture dtype.  "
                "Downcast to float32 before upload: image.astype(np.float32)"
            )
        case _:
            raise ValueError(
                "Unsupported NumPy dtype for GL texture upload: %r.  "
                "Supported dtypes: uint8, int8, uint16, int16, float16, float32."
                % dtype_name
            )


@lru_cache(maxsize=64)
def _resolve_gl_format_base(fmt_name: str) -> tuple[int, str]:
    """
    Map a :class:`PixelFormat` name to ``(GL_FORMAT token, base internal-format string)``.

    ``GL_FORMAT`` is the *transfer* format — it describes how pixel bytes are
    laid out in the CPU-side buffer.  The *internal* format (e.g. ``GL_RGB8``)
    describes how the GPU stores the data; it is always the canonical
    ``GL_RGB``/``GL_RGBA``/``GL_R``/``GL_RG`` family regardless of whether
    the transfer format is byte-swapped (e.g. ``BGR``).

    Args:
        fmt_name: The ``.name`` attribute of a :class:`PixelFormat` member
                  (e.g. ``"RGB"``, ``"BGRA"``).

    Returns:
        ``(gl_format, base_internal)`` where ``base_internal`` is a string
        such as ``"GL_RGB"`` that is combined with a dtype suffix to form the
        final GL internal-format constant name.

    Raises:
        ValueError: If ``fmt_name`` does not match any known PixelFormat.

    Note on planar YUV formats
    --------------------------
    ``YUV420``, ``YUV422``, ``NV12``, and ``NV21`` are multi-plane formats
    that cannot be faithfully represented as a single ``GL_TEXTURE_2D``.
    They are mapped to ``GL_RED`` (luma plane only) so that at least the
    luminance channel is uploaded correctly.  Full chroma reproduction
    requires CPU-side conversion to RGB/RGBA before calling
    :func:`get_gl_texture_spec`.
    """
    match fmt_name:
        # 3-channel transfer formats
        case "RGB" | "YUV444":
            return GL.GL_RGB, "GL_RGB"
        case "BGR":
            # Transfer is byte-swapped; GPU internal storage is still RGB.
            return GL.GL_BGR, "GL_RGB"

        # 4-channel transfer formats
        case "RGBA":
            return GL.GL_RGBA, "GL_RGBA"
        case "BGRA":
            return GL.GL_BGRA, "GL_RGBA"

        # 1-channel: greyscale and planar YUV (luma plane only — see docstring)
        case "MONOCHROME" | "GRAY" | "YUV420" | "YUV422" | "NV12" | "NV21":
            return GL.GL_RED, "GL_R"

        # 2-channel
        case "RG":
            return GL.GL_RG, "GL_RG"

        case _:
            raise ValueError(
                "Unsupported PixelFormat for GL texture upload: %r." % fmt_name
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@lru_cache(maxsize=128)
def get_gl_texture_spec(
    fmt: PixelFormat | str,
    dtype: Union[str, np.dtype, type],
) -> GLTextureSpec:
    """
    Resolve the complete OpenGL texture parameter triple for an image upload.

    Combines the results of :func:`_resolve_gl_format_base` and
    :func:`_resolve_gl_dtype_params` to construct the GL internal-format
    constant name (e.g. ``"GL_RGB16F"``), then looks it up on the ``GL``
    module.

    Results are cached by ``(fmt, dtype)`` — the first call for a given
    combination performs the resolution; all subsequent calls return the
    cached triple with no computation.

    Args:
        fmt:   A :class:`PixelFormat` enum member or its string ``.name``
               (e.g. ``PixelFormat.RGB`` or ``"RGB"``).  Enum members are
               preferred; bare strings are accepted for convenience.
        dtype: The NumPy element type of the image array.  Accepts:

               * A dtype string: ``"float32"``
               * A NumPy dtype object: ``np.dtype("float32")``
               * A NumPy scalar type: ``np.float32``

               Any value accepted by ``np.dtype()`` is valid.

    Returns:
        ``(gl_format, gl_internal_format, gl_type)`` — the three integer GL
        tokens required by ``glTexImage2D`` and ``glTexSubImage2D``.

    Raises:
        TypeError:       If ``dtype`` cannot be coerced to a ``np.dtype``
                         (e.g. a non-numeric Python type was passed).
        ValueError:      If the dtype or pixel format is not supported
                         (see :func:`_resolve_gl_dtype_params` and
                         :func:`_resolve_gl_format_base`).
        GLTextureError:  If the constructed GL internal-format constant
                         (e.g. ``GL_RGB16_SNORM``) is not present in the
                         active driver's GL namespace.  This indicates a
                         driver or profile capability gap, not a caller error.

    Example::

        gl_fmt, gl_int_fmt, gl_type = get_gl_texture_spec(
            PixelFormat.RGB, np.float32
        )
        # → (GL_RGB, GL_RGB32F, GL_FLOAT)

        gl_fmt, gl_int_fmt, gl_type = get_gl_texture_spec(
            PixelFormat.GRAY, "uint16"
        )
        # → (GL_RED, GL_R16, GL_UNSIGNED_SHORT)
    """
    # ------------------------------------------------------------------ #
    # 1. Normalise dtype → canonical name string                          #
    # ------------------------------------------------------------------ #
    # np.dtype() accepts strings, numpy scalar types, and dtype objects.
    # Passing a non-numeric Python type (e.g. list) raises TypeError, which
    # we let propagate unchanged — it is an unambiguous caller error.
    try:
        dtype_str = np.dtype(dtype).name
    except TypeError as e:
        raise TypeError(
            "Cannot resolve GL dtype: %r is not a valid NumPy dtype.  "
            "Pass a dtype string ('float32'), np.dtype object, or NumPy "
            "scalar type (np.float32)." % (dtype,)
        ) from e

    # ------------------------------------------------------------------ #
    # 2. Normalise fmt → name string                                      #
    # ------------------------------------------------------------------ #
    # PixelFormat members expose .name; bare strings are passed through.
    # Any other type with a .name attribute (e.g. an enum from another
    # module) is also handled correctly.
    fmt_name: str = fmt.name if hasattr(fmt, "name") else str(fmt)

    # ------------------------------------------------------------------ #
    # 3. Resolve cached parameters                                        #
    # ------------------------------------------------------------------ #
    gl_type, type_suffix = _resolve_gl_dtype_params(dtype_str)
    gl_format, base_internal = _resolve_gl_format_base(fmt_name)

    # ------------------------------------------------------------------ #
    # 4. Construct and look up the internal-format GL constant            #
    # ------------------------------------------------------------------ #
    # The constant name is built by concatenation, e.g.:
    #   base="GL_RGB"  suffix="16F"  →  "GL_RGB16F"
    #   base="GL_R"    suffix="8"    →  "GL_R8"
    internal_attr = "%s%s" % (base_internal, type_suffix)

    gl_internal = getattr(GL, internal_attr, None)
    if gl_internal is None:
        # Do NOT fall back silently to the base format.  Falling back from
        # GL_RGB16_SNORM to GL_RGB would change the storage precision and
        # sign convention without the caller's knowledge, corrupting data.
        # This is a capability gap — raise with an actionable message.
        raise GLTextureError(
            "GL internal format %r (derived from fmt=%r, dtype=%r) is not "
            "available in the current driver/profile.  Consider using a "
            "different dtype (e.g. float32 instead of int16) or converting "
            "the image on the CPU before upload." % (internal_attr, fmt_name, dtype_str)
        )

    logger.debug(
        "Resolved GL texture spec: fmt=%s dtype=%s → "
        "gl_format=0x%04x gl_internal=%s gl_type=0x%04x",
        fmt_name, dtype_str, gl_format, internal_attr, gl_type,
    )

    return gl_format, gl_internal, gl_type
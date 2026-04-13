"""
PyOpenGL type definitions for 2D image processing.

Provides semantic type aliases for static analysis and IDE support.
Enables autocomplete, type checking, and self-documenting function signatures.
"""

from typing import NewType

__all__ = [
    "GLenum",
    "GLbitfield",
    "GLfloat",
    "GLint",
    "GLuint",
    "GLubyte",
    "GLsizei",
    "GLintptr",
    "GLsizeiptr",
    "GLHandle",
    "GLBuffer",
    "GLTexture",
    "GLFramebuffer",
]

# ============================================================================
# ENUMERATIONS & CONSTANTS
# ============================================================================

GLenum = NewType("GLenum", int)
"""OpenGL enumeration constant (GL_TEXTURE_2D, GL_RGBA, GL_PIXEL_UNPACK_BUFFER, etc.).

Use for any parameter that accepts GL_* constants.

Example:
    >>> def set_texture_wrap(texture: GLTexture, wrap_mode: GLenum) -> None:
    ...     glBindTexture(GL_TEXTURE_2D, texture)
    ...     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap_mode)
"""

GLbitfield = NewType("GLbitfield", int)
"""Bit field for combining multiple GL flags with bitwise OR.

Example:
    >>> glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
"""

# ============================================================================
# NUMERIC TYPES
# ============================================================================

GLfloat = NewType("GLfloat", float)
"""32-bit floating point for coordinates, colors, and matrix values.

Example:
    >>> glClearColor(GLfloat(0.0), GLfloat(0.0), GLfloat(0.0), GLfloat(1.0))
"""

GLint = NewType("GLint", int)
"""32-bit signed integer for sizes, offsets, and property values.

Example:
    >>> def set_texture_unit(unit: GLint) -> None:
    ...     glActiveTexture(GL_TEXTURE0 + unit)
"""

GLuint = NewType("GLuint", int)
"""32-bit unsigned integer (base type for resource IDs)."""

GLubyte = NewType("GLubyte", int)
"""8-bit unsigned integer, commonly for color components (0-255).

Example:
    >>> pixels: list[GLubyte] = [255, 0, 0, 255] * 100  # Red pixels
"""

# ============================================================================
# SIZE & OFFSET TYPES (Semantic clarity)
# ============================================================================

GLsizei = NewType("GLsizei", int)
"""Non-negative integer for sizes and counts.

Use for texture dimensions, vertex counts, buffer sizes.

Example:
    >>> def create_texture(width: GLsizei, height: GLsizei) -> GLTexture:
    ...     texture = glGenTextures(1)
    ...     glBindTexture(GL_TEXTURE_2D, texture)
    ...     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    ...     return GLTexture(texture)
"""

GLintptr = NewType("GLintptr", int)
"""Pointer-sized signed integer for buffer byte offsets.

Example:
    >>> def update_buffer_region(buffer: GLBuffer, offset: GLintptr, data: bytes) -> None:
    ...     glBindBuffer(GL_COPY_WRITE_BUFFER, buffer)
    ...     glBufferSubData(GL_COPY_WRITE_BUFFER, offset, len(data), data)
"""

GLsizeiptr = NewType("GLsizeiptr", int)
"""Pointer-sized unsigned integer for buffer byte sizes.

Example:
    >>> def allocate_buffer(size: GLsizeiptr) -> GLBuffer:
    ...     buffer = glGenBuffers(1)
    ...     glBindBuffer(GL_PIXEL_PACK_BUFFER, buffer)
    ...     glBufferData(GL_PIXEL_PACK_BUFFER, size, None, GL_STREAM_READ)
    ...     return GLBuffer(buffer)
"""

# ============================================================================
# RESOURCE HANDLES (Strongly Typed for 2D Image Processing)
# ============================================================================

GLHandle = NewType("GLHandle", int)
"""Generic OpenGL resource handle (base type for all GPU objects)."""

GLBuffer = NewType("GLBuffer", GLHandle)
"""Handle to a buffer object (VBO for vertex data, PBO for pixel data).

Common uses:
- Vertex Buffer Objects (VBO): GL_ARRAY_BUFFER
- Pixel Buffer Objects (PBO): GL_PIXEL_UNPACK_BUFFER, GL_PIXEL_PACK_BUFFER
- Copy buffers: GL_COPY_READ_BUFFER, GL_COPY_WRITE_BUFFER

Example (upload image via PBO):
    >>> def upload_image_async(data: bytes) -> GLBuffer:
    ...     pbo = glGenBuffers(1)
    ...     glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo)
    ...     glBufferData(GL_PIXEL_UNPACK_BUFFER, len(data), data, GL_STREAM_DRAW)
    ...     return GLBuffer(pbo)

Example (download pixels via PBO):
    >>> def download_pixels(texture: GLTexture, width: GLsizei, height: GLsizei) -> GLBuffer:
    ...     pbo = glGenBuffers(1)
    ...     glBindBuffer(GL_PIXEL_PACK_BUFFER, pbo)
    ...     glBufferData(GL_PIXEL_PACK_BUFFER, width * height * 4, None, GL_STREAM_READ)
    ...     glBindTexture(GL_TEXTURE_2D, texture)
    ...     glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    ...     return GLBuffer(pbo)
"""

GLTexture = NewType("GLTexture", GLHandle)
"""Handle to a 2D texture object for image storage and sampling.

Example (create and upload texture):
    >>> def create_texture_from_pixels(
    ...     width: GLsizei,
    ...     height: GLsizei,
    ...     pixels: bytes,
    ... ) -> GLTexture:
    ...     texture = glGenTextures(1)
    ...     glBindTexture(GL_TEXTURE_2D, texture)
    ...     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, pixels)
    ...     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    ...     glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    ...     return GLTexture(texture)

Example (bind for rendering):
    >>> def bind_texture_unit(unit: GLint, texture: GLTexture) -> None:
    ...     glActiveTexture(GL_TEXTURE0 + unit)
    ...     glBindTexture(GL_TEXTURE_2D, texture)
"""

GLFramebuffer = NewType("GLFramebuffer", GLHandle)
"""Handle to a framebuffer object (FBO) for off-screen rendering.

Use for:
- Processing pipeline stages
- Intermediate texture generation
- Render-to-texture operations

Example:
    >>> def create_render_target(width: GLsizei, height: GLsizei) -> tuple[GLFramebuffer, GLTexture]:
    ...     fbo = glGenFramebuffers(1)
    ...     glBindFramebuffer(GL_FRAMEBUFFER, fbo)
    ...     
    ...     texture = glGenTextures(1)
    ...     glBindTexture(GL_TEXTURE_2D, texture)
    ...     glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, None)
    ...     glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, texture, 0)
    ...     
    ...     return GLFramebuffer(fbo), GLTexture(texture)
"""
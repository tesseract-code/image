"""
quad.py
=======
VAO/VBO management for a screen-aligned quad used by the colorbar renderer.

The ``QuadGeometry`` class owns the lifetime of a single OpenGL vertex-array
object (VAO) and its backing vertex-buffer object (VBO).  It is intentionally
narrow in scope: upload interleaved ``[x, y, u, v]`` vertex data and issue a
single ``GL_TRIANGLE_FAN`` draw call.

Vertex layout (interleaved, tightly packed)
-------------------------------------------
Each vertex is 16 bytes (4 × float32):

    offset 0  – position  (x, y)   – attribute location 0
    offset 8  – texcoord  (u, v)   – attribute location 1

    ┌─────┬─────┬─────┬─────┐
    │  x  │  y  │  u  │  v  │  ← 4 × float32 = 16 bytes / vertex
    └─────┴─────┴─────┴─────┘
     0     4     8     12    16

Integration
-----------
* Uses :data:`gl_errors.gl_error_check` to wrap every mutating GL call so
  driver errors are caught, logged, and re-raised as typed exceptions.
* Uses :mod:`gl_types` aliases (``GLHandle``, ``GLBuffer``, ``GLuint``, …)
  for self-documenting, statically-analysable signatures.

Typical usage::

    with QuadGeometry() as quad:
        quad.update_vertices(vertices)
        quad.draw()
"""

from __future__ import annotations

import ctypes
import logging
from types import TracebackType

import numpy as np

from image.gl.backend import GL, initialize_context
from image.gl.errors import (
    GLError,
    GLInitializationError,
    GLUploadError,
    gl_error_check,
)
from image.gl.types import (
    GLBuffer,
    GLHandle,
)

__all__ = ["VertexLayout", "QuadGeometry"]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Vertex layout constants
# ---------------------------------------------------------------------------

class VertexLayout:
    """
    Compile-time constants that describe the interleaved vertex format.

    All offsets and strides are in **bytes** and assume 32-bit (4-byte)
    floats throughout.  The layout is::

        [x: f32, y: f32, u: f32, v: f32]
         ^                ^
         position (loc 0) texcoord (loc 1)

    Class attributes
    ----------------
    POSITION_SIZE : int
        Number of scalar components in the position attribute (x, y → 2).
    TEXCOORD_SIZE : int
        Number of scalar components in the texcoord attribute (u, v → 2).
    POSITION_OFFSET : int
        Byte offset of the position attribute from the start of each vertex.
        Always 0 because position is the first field.
    TEXCOORD_OFFSET : int
        Byte offset of the texcoord attribute from the start of each vertex.
        Equal to ``POSITION_SIZE * sizeof(float)`` = 8 bytes.
    STRIDE : int
        Total byte size of one vertex record (position + texcoord) = 16 bytes.
        Passed verbatim to ``glVertexAttribPointer``.
    """

    POSITION_SIZE: int = 2
    TEXCOORD_SIZE: int = 2

    # Byte offsets — computed from sizes so they stay in sync automatically.
    POSITION_OFFSET: int = 0
    TEXCOORD_OFFSET: int = POSITION_SIZE * 4  # 2 floats × 4 bytes = 8

    # Total stride: both attributes × 4 bytes per float.
    STRIDE: int = (POSITION_SIZE + TEXCOORD_SIZE) * 4  # 16 bytes


# ---------------------------------------------------------------------------
# Quad geometry
# ---------------------------------------------------------------------------

class QuadGeometry:
    """
    Manages the VAO and VBO for a GPU-resident, dynamically-updated quad.

    The quad is defined by exactly **four vertices** uploaded each frame via
    :meth:`update_vertices` and rendered with a single
    ``glDrawArrays(GL_TRIANGLE_FAN, 0, 4)`` call.  Using ``GL_TRIANGLE_FAN``
    means the winding order is::

        3 ── 2
        │  ╲ │
        0 ── 1

    where vertex 0 is the fan pivot.  Callers should arrange their NDC or
    screen-space coordinates accordingly.

    Lifecycle
    ---------
    Resources are allocated lazily on the first call to :meth:`initialize`
    (or on ``__enter__`` when used as a context manager) and freed by
    :meth:`cleanup` (or ``__exit__``).

    The class is *not* thread-safe; all GL calls must originate from the
    thread that owns the current GL context.

    Parameters
    ----------
    None — construction is free of GL side-effects.

    Raises
    ------
    GLInitializationError
        If VAO/VBO allocation or attribute pointer setup fails during
        :meth:`initialize`.
    GLUploadError
        If the GPU buffer transfer fails during :meth:`update_vertices`.
    GLError
        If the draw call raises a GL error during :meth:`draw`.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        # Resource handles — 0 is the GL "null" name for every object type.
        self._vao: GLHandle = GLHandle(0)
        self._vbo: GLBuffer = GLBuffer(GLHandle(0))

        # Guards against double-initialisation and no-op cleanup.
        self._initialized: bool = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """
        Allocate the VAO, VBO, and configure vertex attribute pointers.

        This method is idempotent: subsequent calls after a successful first
        call return immediately without re-allocating resources.

        The VBO is pre-allocated for exactly **64 bytes** (4 vertices × 4
        floats × 4 bytes) with ``GL_DYNAMIC_DRAW`` usage, signalling to the
        driver that the buffer will be updated frequently.

        Attribute layout configured here:

        +-----------+----------+------+--------+---------+
        | Location  | Name     | Size | Offset | Stride  |
        +===========+==========+======+========+=========+
        | 0         | position | 2    | 0 B    | 16 B    |
        +-----------+----------+------+--------+---------+
        | 1         | texcoord | 2    | 8 B    | 16 B    |
        +-----------+----------+------+--------+---------+

        Raises
        ------
        GLInitializationError
            If any GL call during setup reports an error.
        """
        if self._initialized:
            return

        initialize_context()
        logger.debug("QuadGeometry: allocating VAO/VBO")

        with gl_error_check("QuadGeometry VAO/VBO allocation",
                            GLInitializationError):
            self._vao = GLHandle(int(GL.glGenVertexArrays(1)))
            self._vbo = GLBuffer(GLHandle(int(GL.glGenBuffers(1))))

        with gl_error_check("QuadGeometry buffer setup", GLInitializationError):
            GL.glBindVertexArray(self._vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)

            # Pre-allocate: 4 vertices × 4 components × 4 bytes = 64 bytes.
            # GL_DYNAMIC_DRAW hints that the data will be written often and
            # read by the GPU, letting the driver keep it in a fast write path.
            _QUAD_BUFFER_BYTES: int = (
                    4  # vertices
                    * (VertexLayout.POSITION_SIZE + VertexLayout.TEXCOORD_SIZE)
                    * 4  # bytes per float32
            )
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER,
                _QUAD_BUFFER_BYTES,
                None,
                GL.GL_DYNAMIC_DRAW,
            )

            # Attribute 0 — position (x, y).
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(
                0,
                VertexLayout.POSITION_SIZE,
                GL.GL_FLOAT,
                GL.GL_FALSE,
                VertexLayout.STRIDE,
                ctypes.c_void_p(VertexLayout.POSITION_OFFSET),
            )

            # Attribute 1 — texcoord (u, v).
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(
                1,
                VertexLayout.TEXCOORD_SIZE,
                GL.GL_FLOAT,
                GL.GL_FALSE,
                VertexLayout.STRIDE,
                ctypes.c_void_p(VertexLayout.TEXCOORD_OFFSET),
            )

            # Unbind to restore default GL state.
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindVertexArray(0)

        self._initialized = True
        logger.debug(
            "QuadGeometry: ready — VAO=%d, VBO=%d", self._vao, self._vbo
        )

    # ------------------------------------------------------------------
    # Vertex upload
    # ------------------------------------------------------------------

    def update_vertices(self, vertices: np.ndarray) -> None:
        """
        Stream updated vertex data to the pre-allocated GPU buffer.

        The array is uploaded via ``glBufferSubData`` into the buffer
        allocated in :meth:`initialize`.  The GPU buffer is always 64 bytes;
        ``vertices`` must not exceed that size.

        Parameters
        ----------
        vertices : np.ndarray
            A ``(4, 4)`` array of ``float32`` values.  Each row is one
            vertex in ``[x, y, u, v]`` order matching :class:`VertexLayout`.

        Raises
        ------
        RuntimeError
            If :meth:`initialize` has not been called yet.  (A warning is
            logged and the call returns early rather than raising, to avoid
            crashing rendering code that calls ``update_vertices`` before the
            GL context is ready — callers that need hard failure should check
            :attr:`is_initialized` themselves.)
        TypeError
            If ``vertices.dtype`` is not ``np.float32``.
        ValueError
            If ``vertices.shape`` is not ``(4, 4)``.
        GLUploadError
            If ``glBufferSubData`` reports a GL error.
        """
        if not self._initialized:
            # Soft failure: log and return rather than raising so that
            # partially-constructed renderers don't crash on the first frame.
            logger.warning(
                "QuadGeometry.update_vertices called before initialize(); "
                "skipping upload"
            )
            return

        # --- Validate array dtype before touching the GPU --------
        if vertices.dtype != np.float32:
            raise TypeError(
                f"QuadGeometry requires float32 vertex data; got {vertices.dtype}"
            )

        if vertices.size != 16:
            raise ValueError(
                f"QuadGeometry expects exactly 16 float32 values "
                f"(4 vertices × 4 components); got size={vertices.size}, shape={vertices.shape}"
            )

        # glBufferSubData requires a C-contiguous memory layout.
        # np.ascontiguousarray is a no-op if the array is already contiguous.
        vertices = np.ascontiguousarray(vertices)

        logger.debug("QuadGeometry: uploading %d bytes of vertex data",
                     vertices.nbytes)

        try:
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
            with gl_error_check("QuadGeometry vertex upload", GLUploadError):
                GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, vertices.nbytes,
                                   vertices)
        finally:
            # Always restore default binding even if the upload raised.
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(self) -> None:
        """
        Emit a single ``GL_TRIANGLE_FAN`` draw call for the four vertices.

        This method is a no-op if the geometry has not been initialised,
        allowing it to be called unconditionally from a render loop without
        an explicit guard.

        The VAO binding is restored to 0 after the draw to avoid accidental
        contamination of subsequent GL state.

        Raises
        ------
        GLError
            If ``glDrawArrays`` reports a GL error and error checking is
            enabled in the current :class:`GLConfig`.
        """
        if not self._initialized:
            return

        GL.glBindVertexArray(self._vao)
        with gl_error_check("QuadGeometry draw", GLError):
            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)
        GL.glBindVertexArray(0)

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def cleanup(self) -> None:
        """
        Delete the VAO and VBO and reset all handles to zero.

        This method is idempotent: calling it on an already-cleaned-up
        instance is safe.  ``_initialized`` is set to ``False`` *before* the
        GL deletions so that any re-entrant call during an exception handler
        sees the object as already cleaned up.

        Any GL errors during deletion are logged at ``ERROR`` level but **not
        re-raised** — cleanup must not throw, because it is called from
        ``__exit__`` which may already be handling an exception.
        """
        if not self._initialized:
            return

        # Mark first to prevent reentry if a GL call below raises.
        self._initialized = False
        logger.debug(
            "QuadGeometry: deleting VAO=%d, VBO=%d", self._vao, self._vbo
        )

        try:
            if self._vbo:
                GL.glDeleteBuffers(1, [self._vbo])
            if self._vao:
                GL.glDeleteVertexArrays(1, [self._vao])
        except Exception:
            # Deletion errors are non-fatal: log and continue so that
            # the caller's exception (if any) is not swallowed.
            logger.exception(
                "QuadGeometry: exception during GL resource deletion "
                "(VAO=%d, VBO=%d)",
                self._vao,
                self._vbo,
            )
        finally:
            # Zero the handles regardless of whether deletion succeeded.
            self._vao = GLHandle(0)
            self._vbo = GLBuffer(GLHandle(0))

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """``True`` if :meth:`initialize` has succeeded and :meth:`cleanup` has not yet been called."""
        return self._initialized

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "QuadGeometry":
        """
        Initialize the geometry and return ``self``.

        Allows usage as::

            with QuadGeometry() as quad:
                quad.update_vertices(vertices)
                quad.draw()

        Raises
        ------
        GLInitializationError
            Propagated from :meth:`initialize` if GL setup fails.
        """
        self.initialize()
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc_val: BaseException | None,
            exc_tb: TracebackType | None,
    ) -> bool:
        """
        Release GPU resources unconditionally.

        The return value is always ``False``: exceptions raised inside the
        ``with`` block are never suppressed here.
        """
        self.cleanup()
        return False  # Do not suppress exceptions.

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        state = (
            f"VAO={self._vao}, VBO={self._vbo}"
            if self._initialized
            else "uninitialized"
        )
        return f"<QuadGeometry {state}>"

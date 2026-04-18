"""
gl_quad.py
==========
Fullscreen quad geometry management for PyOpenGL render pipelines.

Owns a single VAO/VBO/EBO triple that encodes a unit quad covering
normalised device coordinates ``[-1, 1]``.  Each vertex carries a 3-component
position and a 2-component UV coordinate:

    attribute layout
    ----------------
    location 0  –  position  (vec3, offset 0)
    location 1  –  texcoord  (vec2, offset 12)

The geometry is static (``GL_STATIC_DRAW``) and uploaded once in
:meth:`~GeometryManager.initialize`.  The VAO records all buffer bindings
and attribute pointers so that subsequent frames only need
``glBindVertexArray`` + ``glDrawElements``.

Usage
-----

    geo = GeometryManager()
    if not geo.initialize():
        raise GLInitializationError("Geometry setup failed")

    # inside paintGL:
    with geo:
        geo.draw()

    # or manually:
    geo.bind()
    geo.draw()
    geo.unbind()

    # at shutdown:
    geo.cleanup()
"""

from __future__ import annotations

import ctypes
from typing import Optional

import numpy as np

from pycore.log.ctx import with_logger
from image.gl.backend import GL
from image.gl.errors import GLError

__all__ = ["GeometryManager"]

# Byte size of one float32 element.  Named constant avoids magic-number
# repetition in the stride / offset calculations below.
_FLOAT_BYTES = 4


@with_logger
class GeometryManager:
    """
    Owns the VAO/VBO/EBO for a fullscreen quad and manages their lifetime.

    The quad spans the entire clip-space viewport (positions in ``[-1, 1]``)
    and maps UV ``(0, 0)`` to the bottom-left corner and ``(1, 1)`` to the
    top-right corner, matching OpenGL's default texture-coordinate convention.

    Binding is separated from drawing so that callers can manage the VAO
    scope themselves (e.g. to interleave multiple draw calls inside one
    bind/unbind pair without redundant state changes).

    Attributes:
        vao: VAO handle, or ``None`` before :meth:`initialize` succeeds.
        vbo: VBO handle, or ``None`` before :meth:`initialize` succeeds.
        ebo: EBO handle, or ``None`` before :meth:`initialize` succeeds.
    """

    # -----------------------------------------------------------------------
    # Geometry data (class-level, shared across all instances)
    # -----------------------------------------------------------------------

    # Each row: X  Y  Z  U  V
    VERTICES: np.ndarray = np.array(
        [
            -1.0, -1.0, 0.0,  0.0, 0.0,   # bottom-left
             1.0, -1.0, 0.0,  1.0, 0.0,   # bottom-right
             1.0,  1.0, 0.0,  1.0, 1.0,   # top-right
            -1.0,  1.0, 0.0,  0.0, 1.0,   # top-left
        ],
        dtype=np.float32,
    )

    # Two triangles (CCW winding): bottom-left, bottom-right, top-right,
    # top-right, top-left, bottom-left.
    INDICES: np.ndarray = np.array([0, 1, 2, 2, 3, 0], dtype=np.uint32)

    # Vertex layout constants derived from VERTICES structure.
    _COMPONENTS_PER_VERTEX = 5               # position(3) + texcoord(2)
    _STRIDE = _COMPONENTS_PER_VERTEX * _FLOAT_BYTES   # 20 bytes
    _POSITION_OFFSET  = 0                    # bytes from start of vertex
    _TEXCOORD_OFFSET  = 3 * _FLOAT_BYTES     # 12 bytes (after 3 position floats)

    __slots__ = ("vao", "vbo", "ebo")

    def __init__(self) -> None:
        self.vao: Optional[int] = None
        self.vbo: Optional[int] = None
        self.ebo: Optional[int] = None

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def initialize(self) -> bool:
        """
        Allocate and configure the VAO, VBO, and EBO on the GPU.

        Creates the vertex array object, uploads static geometry data into a
        vertex buffer and an element array buffer, and records the attribute
        layout in the VAO so that rendering only requires a single
        ``glBindVertexArray`` call per frame.

        Must be called from the thread that owns the active GL context,
        typically inside ``initializeGL()``.

        Returns:
            ``True`` on success; ``False`` if any GL object could not be
            created.  On failure, :meth:`cleanup` is called internally to
            release any partially allocated resources.

        Raises:
            Does **not** raise — all errors are caught, logged, and signalled
            via the boolean return value so that the caller's initialisation
            path remains simple.
        """
        try:
            self.vao = GL.glGenVertexArrays(1)
            if self.vao == 0:
                self._logger.error("glGenVertexArrays returned 0 — VAO creation failed")
                return False

            self.vbo = GL.glGenBuffers(1)
            self.ebo = GL.glGenBuffers(1)
            if self.vbo == 0 or self.ebo == 0:
                self._logger.error("glGenBuffers returned 0 — buffer creation failed")
                self.cleanup()
                return False

            # All buffer bindings and attribute pointer calls made while the
            # VAO is bound are recorded into it.
            GL.glBindVertexArray(self.vao)

            # Upload static vertex data.
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER,
                self.VERTICES.nbytes,
                self.VERTICES,
                GL.GL_STATIC_DRAW,
            )

            # Upload static index data.
            # IMPORTANT: the EBO must be bound BEFORE the VAO is unbound.
            # Unbinding the EBO while the VAO is still active would detach it
            # from the VAO, causing glDrawElements to draw from no buffer.
            GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
            GL.glBufferData(
                GL.GL_ELEMENT_ARRAY_BUFFER,
                self.INDICES.nbytes,
                self.INDICES,
                GL.GL_STATIC_DRAW,
            )

            # Attribute 0 — position (vec3)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(
                0,                                       # layout location
                3,                                       # component count
                GL.GL_FLOAT,                             # element type
                GL.GL_FALSE,                             # normalise
                self._STRIDE,                            # stride in bytes
                ctypes.c_void_p(self._POSITION_OFFSET),  # byte offset
            )

            # Attribute 1 — texture coordinate (vec2)
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(
                1,                                       # layout location
                2,                                       # component count
                GL.GL_FLOAT,                             # element type
                GL.GL_FALSE,                             # normalise
                self._STRIDE,                            # stride in bytes
                ctypes.c_void_p(self._TEXCOORD_OFFSET),  # byte offset
            )

            # Unbind the VAO first.  The EBO association is recorded inside
            # the VAO, so it remains valid after this unbind.
            GL.glBindVertexArray(0)

            # Unbind the VBO (cosmetic — does not affect the VAO's recorded state).
            # Do NOT unbind the EBO before unbinding the VAO (see comment above).
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)

            self._logger.info("Geometry initialised successfully")
            return True

        except GLError as e:
            self._logger.error("GL error during geometry initialisation: %s", e)
            self.cleanup()
            return False
        except Exception as e:
            self._logger.error("Geometry initialisation failed: %s", e)
            self.cleanup()
            return False

    # -----------------------------------------------------------------------
    # Render interface
    # -----------------------------------------------------------------------

    def bind(self) -> None:
        """
        Bind the VAO for subsequent GL calls.

        A no-op when :meth:`initialize` has not yet been called or has failed,
        so callers do not need to guard against an uninitialised manager.
        """
        if self.vao is not None:
            GL.glBindVertexArray(self.vao)

    def unbind(self) -> None:
        """Unbind any VAO, restoring the default (no-VAO) state."""
        GL.glBindVertexArray(0)

    def draw(self) -> None:
        """
        Issue ``glDrawElements`` for the quad geometry.

        The VAO **must** already be bound before this method is called.
        This method does not call :meth:`bind` or :meth:`unbind` itself;
        the caller owns the bind scope so that multiple draw calls can share
        a single bind/unbind pair without redundant state changes.

        A warning is logged and the call is skipped if the VAO handle has not
        been allocated (i.e. :meth:`initialize` was never called or failed).
        """
        if self.vao is None:
            self._logger.warning(
                "draw() called on uninitialised geometry — skipping"
            )
            return

        GL.glDrawElements(
            GL.GL_TRIANGLES,
            len(self.INDICES),
            GL.GL_UNSIGNED_INT,
            ctypes.c_void_p(0),  # offset 0 into the bound EBO
        )

    # -----------------------------------------------------------------------
    # Context manager  (bind / unbind scope)
    # -----------------------------------------------------------------------

    def __enter__(self) -> "GeometryManager":
        """Bind the VAO on entry."""
        self.bind()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Unbind the VAO on exit, even if the block raised."""
        self.unbind()

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def cleanup(self) -> None:
        """
        Delete all GPU objects and reset handles to ``None``.

        Safe to call multiple times and safe to call when initialisation
        only partially succeeded — each handle is checked independently.

        Deletion order: VAO first, then VBO and EBO.  Deleting the VAO
        first releases its recorded references to the buffers, after which
        the buffers themselves can be freed without leaving a dangling
        reference in any surviving VAO.

        Must be called from the thread that owns the GL context.
        """
        if self.vao is not None:
            # FIX: original code passed self.vbo here — the VAO handle was
            # leaked and the wrong object ID was sent to the driver.
            GL.glDeleteVertexArrays(1, np.array([self.vao], dtype=np.uint32))
            self.vao = None
            self._logger.debug("VAO deleted")

        if self.vbo is not None:
            GL.glDeleteBuffers(1, np.array([self.vbo], dtype=np.uint32))
            self.vbo = None
            self._logger.debug("VBO deleted")

        if self.ebo is not None:
            GL.glDeleteBuffers(1, np.array([self.ebo], dtype=np.uint32))
            self.ebo = None
            self._logger.debug("EBO deleted")
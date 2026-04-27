"""
uniform.py
=============
Shader uniform location cache and type-safe upload manager for PyOpenGL.

Provides two public interfaces:

* `UniformManager` — owns the location cache and dispatches each
  ``glUniform*`` call based on the GL type token.
* `UniformType` — ``IntEnum`` mirroring every GL uniform type constant
  used by the dispatch table.

Design notes
------------
Location caching
    ``glGetUniformLocation`` is a driver round-trip.  All locations are queried
    once in `~UniformManager.register_uniforms` and stored in a dict and
    a `~types.SimpleNamespace` (``manager.locs.brightness``).  Hot-path
    callers use `~UniformManager.set_fast` with a pre-fetched
    ``GLint`` location to avoid even the dict lookup.

Type dispatch
    `~UniformManager.set_fast` requires an explicit `UniformType`.
    `~UniformManager.set` accepts ``UniformType.AUTO``, which first
    checks the introspection cache (populated during registration) and falls
    back to `~UniformManager._infer_type` for values set before
    introspection ran.

Error handling
    ``glGetError`` is drained once after each ``glUniform*`` call inside
    `~UniformManager._set_uniform_by_type`.  A non-zero code raises
    `~image.gl.error.GLError`, which is caught
    one level up in `~UniformManager._set_by_loc_and_type` and logged
    rather than propagated — a failed uniform upload should degrade rendering
    quality, not crash the frame loop.
"""

from __future__ import annotations

import logging
from enum import IntEnum, StrEnum, unique
from types import SimpleNamespace
from typing import Any, Type, Union, cast

import numpy as np

from image.gl.backend import GL
from image.gl.errors import GLError
from image.gl.types import GLenum, GLint, GLuint
from image.utils.data import ensure_contiguity
from pycore.log.ctx import ContextAdapter

__all__ = [
    "UniformType",
    "VertexShaderUniforms",
    "FragmentShaderUniforms",
    "UniformManager",
]

logger = ContextAdapter(logging.getLogger(__name__), {})


class UniformType(IntEnum):
    """
    OpenGL uniform type tokens as a Python ``IntEnum``.

    Each member's value is the corresponding ``GL_*`` integer constant so that
    a ``UniformType`` can be passed directly wherever a ``GLenum`` is expected.

    ``AUTO`` (``-1``) is a sentinel that triggers type inference in
    `UniformManager.set`; it is not a valid GL token and must never be
    passed to `UniformManager.set_fast`.
    """
    # Scalars
    INT = GL.GL_INT
    UNSIGNED_INT = GL.GL_UNSIGNED_INT
    FLOAT = GL.GL_FLOAT
    BOOL = GL.GL_BOOL

    # Float vectors
    VEC2 = GL.GL_FLOAT_VEC2
    VEC3 = GL.GL_FLOAT_VEC3
    VEC4 = GL.GL_FLOAT_VEC4

    # Integer vectors
    IVEC2 = GL.GL_INT_VEC2
    IVEC3 = GL.GL_INT_VEC3
    IVEC4 = GL.GL_INT_VEC4

    # Boolean vectors
    BVEC2 = GL.GL_BOOL_VEC2
    BVEC3 = GL.GL_BOOL_VEC3
    BVEC4 = GL.GL_BOOL_VEC4

    # Square matrices
    MAT2 = GL.GL_FLOAT_MAT2
    MAT3 = GL.GL_FLOAT_MAT3
    MAT4 = GL.GL_FLOAT_MAT4

    # Non-square matrices
    MAT2x3 = GL.GL_FLOAT_MAT2x3
    MAT2x4 = GL.GL_FLOAT_MAT2x4
    MAT3x2 = GL.GL_FLOAT_MAT3x2
    MAT3x4 = GL.GL_FLOAT_MAT3x4
    MAT4x2 = GL.GL_FLOAT_MAT4x2
    MAT4x3 = GL.GL_FLOAT_MAT4x3

    # Samplers — all uploaded as int (texture unit index)
    SAMPLER_1D = GL.GL_SAMPLER_1D
    SAMPLER_2D = GL.GL_SAMPLER_2D
    SAMPLER_3D = GL.GL_SAMPLER_3D
    SAMPLER_CUBE = GL.GL_SAMPLER_CUBE

    # Sentinel — triggers automatic type detection
    AUTO = -1


@unique
class VertexShaderUniforms(StrEnum):
    TRANSFORM_MATRIX = "u_transform"
    PROJECTION_MATRIX = "u_projection"


@unique
class FragmentShaderUniforms(StrEnum):
    """
    Canonical uniform variable names for the image fragment shader.

    Values are the exact GLSL identifiers as they appear in the shader source.
    Used by `UniformManager.register_members` to batch-register all
    uniforms from a single enum class.
    """

    IMAGE_TEXTURE = "imageTexture"
    COLORMAP_TEXTURE = "colormapTexture"
    USE_CMAP = "use_cmap"
    BRIGHTNESS = "brightness"
    CONTRAST = "contrast"
    INV_GAMMA = "inv_gamma"
    COLOR_BALANCE = "color_balance"
    INVERT = "invert"
    LUT_ENABLED = "lut_enabled"
    LUT_MIN = "lut_min"
    LUT_NORM_FACTOR = "lut_norm_factor"
    LUT_TYPE = "lut_type"
    NORM_VMIN = "norm_vmin"
    NORM_VMAX = "norm_vmax"


class UniformManager:
    """
    Cache shader uniform locations and dispatch typed uploads.

    Typical lifecycle::

        manager = UniformManager(program_id)
        manager.register_members(_FragmentShaderUniforms)

        # Inside the render loop — zero dict overhead:
        manager.set_fast(manager.locs.brightness, 1.2, UniformType.FLOAT)

    Attributes:
        locs: `~types.SimpleNamespace` populated by
              `register_uniforms`.  Each active uniform gets an
              attribute whose name matches the GLSL identifier and whose value
              is the cached :data:`GLint` location.  Inactive uniforms
              (location ``-1``) are omitted so that attribute access raises
              ``AttributeError`` rather than silently passing ``-1`` to the
              driver.
    """

    __slots__ = ("_program", "_locations", "_types", "locs", "_logger")

    def __init__(self, program: GLuint) -> None:
        """
        Initialize the manager for ``program``.

        Args:
            program: A linked OpenGL program handle (:data:`GLuint`).

        Raises:
            ValueError: If ``program`` is ``0`` (the GL sentinel for "no program").
        """
        if program == 0:
            raise ValueError(
                "UniformManager requires a valid program handle; received 0."
            )

        self._program: GLuint = program
        self._locations: dict[str, GLint] = {}
        self._types: dict[str, GLenum] = {}
        self._logger = logger

        # Public dot-notation access to cached locations.
        # Only active uniforms (location != -1) receive an attribute here.
        self.locs = SimpleNamespace()

    @property
    def program(self) -> GLuint:
        """The OpenGL program handle this manager is bound to."""
        return self._program

    def register_members(
            self,
            enum_cls: Type[StrEnum],
            introspect: bool = True,
    ) -> None:
        """
        Register all uniforms declared in a `~enum.StrEnum` class.

        Extracts the string values from each enum member and forwards them to
        `register_uniforms`.

        Args:
            enum_cls:   A `~enum.StrEnum` whose member values are GLSL
                        uniform identifiers.
            introspect: When ``True``, also queries the driver for each
                        uniform's type and size via ``glGetActiveUniform``.
        """
        names: list[str] = [
            cast(str, member.value) for member in enum_cls.__members__.values()
        ]
        self.register_uniforms(names, introspect=introspect)

    def register_uniforms(
            self,
            names: list[str],
            introspect: bool = True,
    ) -> None:
        """
        Query and cache locations for the given uniform names.

        Binds the program for the duration of the query, then restores the
        unbound state.  Uniforms that the GLSL compiler has optimized away
        (location ``-1``) are logged at ``WARNING`` level but do not raise;
        they are excluded from :attr:`locs` so accidental ``set_fast`` calls
        with a stale ``-1`` location are caught at the ``if location == -1``
        guard.

        Args:
            names:      GLSL uniform identifiers to register.
            introspect: When ``True``, calls ``glGetActiveUniform`` for every
                        active uniform and caches the type token so that
                        `set` can dispatch correctly without an explicit
                        `UniformType` argument.
        """
        GL.glUseProgram(self._program)
        self._check_gl_error("glUseProgram before uniform registration")

        uniform_info: dict[str, dict[str, Any]] = {}
        if introspect:
            uniform_info = self._introspect_uniforms()

        for name in names:
            location = GL.glGetUniformLocation(self._program, name)
            self._locations[name] = GLint(location)

            if location == -1:
                # Inactive uniforms are legal — the GLSL compiler removes
                # variables that have no observable effect on the output.
                self._logger.warning(
                    "Uniform '%s' not found in program %d "
                    "(may be unused/optimised out)",
                    name, self._program,
                )
            else:
                # Expose via dot-notation only for active uniforms.
                setattr(self.locs, name, GLint(location))

                if name in uniform_info:
                    self._types[name] = GLenum(uniform_info[name]["type"])
                    self._logger.debug(
                        "Uniform '%s' cached: loc=%d type=%s",
                        name,
                        location,
                        self._gl_type_name(uniform_info[name]["type"]),
                    )

        GL.glUseProgram(0)
        self._check_gl_error("glUseProgram after uniform registration")

    def set(
            self,
            name: str,
            value: Any,
            uniform_type: Union[UniformType, GLenum] = UniformType.AUTO,
    ) -> bool:
        """
        Upload a uniform value by GLSL name.

        Looks up the cached location for ``name`` and delegates to
        `_set_by_loc_and_type`.  Slightly slower than `set_fast`
        due to the dict lookup; use `set_fast` inside the render loop.

        Args:
            name:         GLSL uniform identifier (must have been registered).
            value:        Value to upload.
            uniform_type: Explicit type, or :attr:`UniformType.AUTO` to infer.

        Returns:
            ``True`` on success, ``False`` if the uniform is not active or if
            an error occurs during upload.
        """
        location = self._locations.get(name, GLint(-1))
        if location == -1:
            return False
        return self._set_by_loc_and_type(location, value, uniform_type, name)

    def set_fast(
            self,
            location: GLint,
            value: Any,
            uniform_type: Union[UniformType, GLenum],
    ) -> bool:
        """
        Upload a uniform value by pre-cached location.

        Fastest upload path: no string lookup, no type inference.  Call this
        inside the render loop with locations obtained from :attr:`locs`.

        Args:
            location:     Pre-queried location from :attr:`locs` or
                          `get_location`.  A value of ``-1`` is a
                          no-op that returns ``False``.
            value:        Value to upload.
            uniform_type: Explicit type — must not be :attr:`UniformType.AUTO`.

        Returns:
            ``True`` on success, ``False`` if ``location`` is ``-1`` or if an
            error occurs.
        """
        if location == -1:
            return False
        return self._set_by_loc_and_type(location, value, uniform_type, "<loc>")

    def get_location(self, name: str) -> GLint:
        """
        Return the cached location for ``name``, or ``GLint(-1)`` if unknown.

        Args:
            name: GLSL uniform identifier.

        Returns:
            Cached location as :data:`GLint`.
        """
        return self._locations.get(name, GLint(-1))

    def _set_by_loc_and_type(
            self,
            location: GLint,
            value: Any,
            uniform_type: Union[UniformType, GLenum],
            debug_name: str,
    ) -> bool:
        """
        Resolve the type (if needed) and call `_set_uniform_by_type`.

        When ``uniform_type`` is :attr:`UniformType.AUTO`, the type is taken
        from the introspection cache (populated at registration time) if
        available, otherwise inferred from the Python type of ``value``.

        Args:
            location:     Pre-resolved uniform location.
            value:        Value to upload.
            uniform_type: Explicit type or ``AUTO``.
            debug_name:   Name used in error log messages.

        Returns:
            ``True`` on success, ``False`` on any error (GL or Python-level).
        """
        if uniform_type == UniformType.AUTO:
            if debug_name in self._types:
                # Prefer the driver-reported type from introspection.
                uniform_type = self._types[debug_name]
            else:
                uniform_type = self._infer_type(value)

        # Normalize to a plain GLenum int so the dispatch table only checks
        # against GL constants, never against UniformType wrappers.
        if isinstance(uniform_type, UniformType):
            uniform_type = GLenum(uniform_type.value)
        elif isinstance(uniform_type, int):
            uniform_type = GLenum(uniform_type)

        try:
            return self._set_uniform_by_type(location, value, uniform_type)
        except GLError as e:
            self._logger.error(
                "GL error setting uniform '%s': %s", debug_name, e
            )
            return False
        except Exception as e:
            self._logger.error(
                "Error setting uniform '%s': %s", debug_name, e
            )
            return False

    def _set_uniform_by_type(
            self,
            location: GLint,
            value: Any,
            gl_type: GLenum,
    ) -> bool:
        """
        Dispatch a ``glUniform*`` call based on ``gl_type``.

        Calls ``glGetError`` once after the upload.  A non-zero error code
        raises `~image.gl.error.GLError`, which
        `_set_by_loc_and_type` catches and logs.

        Args:
            location: Pre-resolved uniform location.
            value:    Value to upload.
            gl_type:  GL uniform type token (e.g. ``GL_FLOAT_VEC3``).

        Returns:
            ``True`` on success, ``False`` if ``gl_type`` is not handled by
            the dispatch table.

        Raises:
            GLError: If the driver reports an error after the upload.
        """

        if gl_type in (
                GL.GL_INT,
                GL.GL_BOOL,
                GL.GL_SAMPLER_1D,
                GL.GL_SAMPLER_2D,
                GL.GL_SAMPLER_3D,
                # all sampler types bind as a texture-unit int
                GL.GL_SAMPLER_CUBE,
        ):
            GL.glUniform1i(location, int(value))

        elif gl_type == GL.GL_FLOAT:
            GL.glUniform1f(location, float(value))

        elif gl_type == GL.GL_UNSIGNED_INT:
            GL.glUniform1ui(location, int(value))
        elif gl_type == GL.GL_FLOAT_VEC2:
            GL.glUniform2fv(location, 1, np.asarray(value, np.float32))
        elif gl_type == GL.GL_FLOAT_VEC3:
            GL.glUniform3fv(location, 1, np.asarray(value, np.float32))
        elif gl_type == GL.GL_FLOAT_VEC4:
            GL.glUniform4fv(location, 1, np.asarray(value, np.float32))
        elif gl_type == GL.GL_INT_VEC2:
            GL.glUniform2iv(location, 1, np.asarray(value, np.int32))
        elif gl_type == GL.GL_INT_VEC3:
            GL.glUniform3iv(location, 1, np.asarray(value, np.int32))
        elif gl_type == GL.GL_INT_VEC4:
            GL.glUniform4iv(location, 1, np.asarray(value, np.int32))
        elif gl_type == GL.GL_BOOL_VEC2:
            GL.glUniform2iv(location, 1, np.asarray(value, np.int32))
        elif gl_type == GL.GL_BOOL_VEC3:
            GL.glUniform3iv(location, 1, np.asarray(value, np.int32))
        elif gl_type == GL.GL_BOOL_VEC4:
            GL.glUniform4iv(location, 1, np.asarray(value, np.int32))
        elif gl_type == GL.GL_FLOAT_MAT2:
            mat = ensure_contiguity(np.asarray(value, np.float32))
            GL.glUniformMatrix2fv(location, 1, GL.GL_FALSE, mat)
        elif gl_type == GL.GL_FLOAT_MAT3:
            mat = ensure_contiguity(np.asarray(value, np.float32))
            GL.glUniformMatrix3fv(location, 1, GL.GL_FALSE, mat)
        elif gl_type == GL.GL_FLOAT_MAT4:
            mat = ensure_contiguity(np.asarray(value, np.float32))
            GL.glUniformMatrix4fv(location, 1, GL.GL_FALSE, mat)
        elif gl_type == GL.GL_FLOAT_MAT2x3:
            mat = ensure_contiguity(np.asarray(value, np.float32))
            GL.glUniformMatrix2x3fv(location, 1, GL.GL_FALSE, mat)
        elif gl_type == GL.GL_FLOAT_MAT3x2:
            mat = ensure_contiguity(np.asarray(value, np.float32))
            GL.glUniformMatrix3x2fv(location, 1, GL.GL_FALSE, mat)

        else:
            self._logger.warning(
                "Unsupported uniform type in dispatch: 0x%x", gl_type
            )
            return False

        # Drain one error from the queue.  A non-zero code means the upload
        # was rejected by the driver; raise so the caller can log and decide
        # how to handle it.
        error = GL.glGetError()
        if error != GL.GL_NO_ERROR:
            raise GLError(
                "glUniform* error 0x%x at location %d" % (error, location))

        return True

    def _introspect_uniforms(self) -> dict[str, dict[str, Any]]:
        """
        Query all active uniforms from the program via ``glGetActiveUniform``.

        Called once during `register_uniforms` when ``introspect=True``.
        The result is used to populate the type cache so that
        `_set_by_loc_and_type` can dispatch correctly without an
        explicit type argument.

        Returns:
            ``{name: {"index": int, "type": GLenum, "size": GLint}}`` for every
            active uniform reported by the driver.  Array uniforms have their
            ``[0]`` suffix stripped from the name.
        """
        num_uniforms = GL.glGetProgramiv(self._program, GL.GL_ACTIVE_UNIFORMS)
        uniform_info: dict[str, dict[str, Any]] = {}

        for i in range(num_uniforms):
            name, size, type_ = GL.glGetActiveUniform(self._program, i)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            # Strip the ``[0]`` suffix the driver appends to array uniforms.
            clean_name = name.split("[")[0]
            uniform_info[clean_name] = {
                "index": i,
                "type": GLenum(type_),
                "size": GLint(size),
            }

        return uniform_info

    @staticmethod
    def _infer_type(value: Any) -> GLenum:
        """
        Infer the best-fit GL uniform type from a Python value.

        Used as a fallback when no introspection data is available for the
        uniform being uploaded.

        Type mapping
        ------------
        =======================================  ==============
        Python type                              GL token
        =======================================  ==============
        ``bool`` (checked first — bool < int)   ``GL_BOOL``
        ``int``                                  ``GL_INT``
        ``float``                                ``GL_FLOAT``
        ``ndarray`` / ``list`` shape ``(4, 4)``  ``GL_FLOAT_MAT4``
        ``ndarray`` / ``list`` shape ``(3, 3)``  ``GL_FLOAT_MAT3``
        ``ndarray`` / ``list`` shape ``(2, 2)``  ``GL_FLOAT_MAT2``
        ``ndarray`` / ``list`` shape ``(4,)``    ``GL_FLOAT_VEC4``
        ``ndarray`` / ``list`` shape ``(3,)``    ``GL_FLOAT_VEC3``
        ``ndarray`` / ``list`` shape ``(2,)``    ``GL_FLOAT_VEC2``
        Anything else                            ``GL_FLOAT`` (safe default)

        Args:
            value: The Python value to classify.

        Returns:
            A :data:`GLenum` token for the inferred type.
        """
        # IMPORTANT: bool must be checked before int.
        # In Python, bool is a subclass of int, so isinstance(True, int) is
        # True — an unguarded int check would silently treat booleans as ints.
        if isinstance(value, bool):
            return GLenum(GL.GL_BOOL)
        if isinstance(value, int):
            return GLenum(GL.GL_INT)
        if isinstance(value, float):
            return GLenum(GL.GL_FLOAT)
        if isinstance(value, (np.ndarray, list)):
            arr = np.asarray(value)
            shape_map = {
                (4, 4): GL.GL_FLOAT_MAT4,
                (3, 3): GL.GL_FLOAT_MAT3,
                (2, 2): GL.GL_FLOAT_MAT2,
                (4,): GL.GL_FLOAT_VEC4,
                (3,): GL.GL_FLOAT_VEC3,
                (2,): GL.GL_FLOAT_VEC2,
            }
            return GLenum(shape_map.get(arr.shape, GL.GL_FLOAT))

        # Unknown type — default to float; the driver will generate
        # GL_INVALID_OPERATION if the type truly mismatches.
        return GLenum(GL.GL_FLOAT)

    @staticmethod
    def _gl_type_name(gl_type: GLenum) -> str:
        """
        Return a human-readable GLSL type name for debug logging.

        Args:
            gl_type: A GL uniform type token.

        Returns:
            The GLSL type keyword string, or ``"0x<hex>"`` for unknown tokens.
        """
        _NAMES: dict[int, str] = {
            GL.GL_FLOAT: "float",
            GL.GL_FLOAT_VEC2: "vec2",
            GL.GL_FLOAT_VEC3: "vec3",
            GL.GL_FLOAT_VEC4: "vec4",
            GL.GL_INT: "int",
            GL.GL_INT_VEC2: "ivec2",
            GL.GL_INT_VEC3: "ivec3",
            GL.GL_INT_VEC4: "ivec4",
            GL.GL_UNSIGNED_INT: "uint",
            GL.GL_BOOL: "bool",
            GL.GL_BOOL_VEC2: "bvec2",
            GL.GL_BOOL_VEC3: "bvec3",
            GL.GL_BOOL_VEC4: "bvec4",
            GL.GL_FLOAT_MAT2: "mat2",
            GL.GL_FLOAT_MAT3: "mat3",
            GL.GL_FLOAT_MAT4: "mat4",
            GL.GL_SAMPLER_1D: "sampler1D",
            GL.GL_SAMPLER_2D: "sampler2D",
            GL.GL_SAMPLER_3D: "sampler3D",
            GL.GL_SAMPLER_CUBE: "samplerCube",
        }
        return _NAMES.get(int(gl_type), "0x%x" % gl_type)

    def _check_gl_error(self, operation: str) -> None:
        """
        Drain one error from the GL queue and raise if non-zero.

        Called at the bookend of `register_uniforms` to verify that
        the program bind and unbind operations succeeded.

        Args:
            operation: Human-readable label for the operation being checked,
                       included in the exception and log message.

        Raises:
            GLError: If ``glGetError`` returns anything other than
                     ``GL_NO_ERROR``.
        """
        error = GL.glGetError()
        if error != GL.GL_NO_ERROR:
            msg = "OpenGL error 0x%x during %s" % (error, operation)
            self._logger.error(msg)
            raise GLError(msg)

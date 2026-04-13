"""
gl_program.py
=============
Shader compilation, linking, and program management for PyOpenGL pipelines.

Provides both low-level free functions for compiling and linking individual
shaders and a :class:`ShaderProgramManager` that owns the program handle and
its associated :class:`~cross_platform.qt6_utils.image.gl.uniform.UniformManager`.

Error strategy
--------------
All functions in this module raise :exc:`GLShaderError` on failure rather than
returning ``None``.  Silent ``None`` returns make failure invisible to callers
that do not check the return value; a typed exception surfaces the problem
immediately with a driver-supplied log message attached.

``glValidateProgram`` is intentionally kept as a ``(bool, str | None)`` return
rather than a raise, because validation failure is state-dependent (it checks
the *current* draw state, not the program itself) and is used diagnostically
rather than as a hard gate.

Program binary caching
----------------------
``glCreateProgram`` compiles GLSL source every time.  For production builds
with many shader variants, consider caching compiled binaries using
``glGetProgramBinary`` / ``glProgramBinary`` (available from GL 4.1 Core).
This module does not implement caching; it remains the caller's responsibility.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional, Union, Any

import numpy as np

from cross_platform.qt6_utils.image.gl.backend import GL
from cross_platform.qt6_utils.image.gl.error import GLShaderError
from cross_platform.qt6_utils.image.gl.types import GLenum, GLint, GLuint
from cross_platform.qt6_utils.image.gl.uniform import UniformManager

__all__ = [
    "compile_shader",
    "link_program",
    "create_program",
    "validate_program",
    "delete_program",
    "set_uniform",
    "ShaderProgramManager",
]

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compilation
# ---------------------------------------------------------------------------

def compile_shader(
        source: str,
        shader_type: GLenum,
        path: str = "",
) -> GLuint:
    """
    Compile a single shader stage from GLSL source.

    Creates a shader object, uploads the source, triggers compilation, and
    checks the result.  If the driver reports any info-log content after a
    successful compile (e.g. vendor-specific warnings) it is logged at
    ``WARNING`` level.

    Args:
        source:      Complete GLSL source string for the shader stage.
        shader_type: Stage token — ``GL.GL_VERTEX_SHADER`` or
                     ``GL.GL_FRAGMENT_SHADER`` (wrapped as :data:`GLenum`).
        path:        Filesystem path to the source file, used only in error
                     messages to help locate the offending shader.

    Returns:
        The compiled shader handle as :data:`GLuint`.  The caller is
        responsible for deleting it with ``glDeleteShader`` once it has been
        attached to a program (or if subsequent steps fail).

    Raises:
        GLShaderError: If shader object creation fails or if the GLSL compiler
                       reports a compilation error.  The driver info-log is
                       included in the exception message.
    """
    stage_name = "Vertex" if shader_type == GL.GL_VERTEX_SHADER else "Fragment"

    shader = GL.glCreateShader(shader_type)
    if shader == 0:
        # glCreateShader returns 0 only when no context is current or the
        # shader type token is invalid — both are hard precondition failures.
        raise GLShaderError(
            "%s shader object creation failed (glCreateShader returned 0)" % stage_name)

    GL.glShaderSource(shader, source)
    GL.glCompileShader(shader)

    compile_status = GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS)
    if not compile_status:
        # Retrieve the compiler log before deleting the shader so the
        # error message contains the actionable GLSL diagnostic.
        log_length = GL.glGetShaderiv(shader, GL.GL_INFO_LOG_LENGTH)
        if log_length > 0:
            error_log = GL.glGetShaderInfoLog(shader).decode("utf-8")
        else:
            error_log = "(no compiler log available)"

        GL.glDeleteShader(shader)
        raise GLShaderError(
            "%s shader compilation failed (%s):\n%s" % (
            stage_name, path or "<source>", error_log)
        )

    # Compilation succeeded.  Some drivers still populate the info log with
    # non-fatal warnings (e.g. precision qualifiers, extension usage).
    # A log_length of 1 means only the null terminator is present — skip it.
    log_length = GL.glGetShaderiv(shader, GL.GL_INFO_LOG_LENGTH)
    if log_length > 1:
        info_log = GL.glGetShaderInfoLog(shader).decode("utf-8").strip()
        if info_log:
            logger.warning(
                "%s shader compiled with warnings (%s):\n%s",
                stage_name, path or "<source>", info_log,
            )

    return GLuint(shader)


# ---------------------------------------------------------------------------
# Linking
# ---------------------------------------------------------------------------

def link_program(
        vertex_shader: GLuint,
        fragment_shader: GLuint,
) -> GLuint:
    """
    Link a compiled vertex and fragment shader into an executable program.

    Attaches both shaders, links the program, then detaches them.  Detaching
    after a successful link is deliberate: the driver has copied the compiled
    code into the program object and the shader handles are no longer needed
    by the program.  The caller is still responsible for deleting the shader
    handles with ``glDeleteShader``.

    Args:
        vertex_shader:   Handle to a compiled vertex shader (:data:`GLuint`).
        fragment_shader: Handle to a compiled fragment shader (:data:`GLuint`).

    Returns:
        The linked program handle as :data:`GLuint`.

    Raises:
        GLShaderError: If program object creation fails or if the linker
                       reports an error.  The driver info-log is included in
                       the exception message.
    """
    program = GL.glCreateProgram()
    if program == 0:
        raise GLShaderError(
            "Program object creation failed (glCreateProgram returned 0)")

    GL.glAttachShader(program, vertex_shader)
    GL.glAttachShader(program, fragment_shader)
    GL.glLinkProgram(program)

    # Detach immediately after linking: the compiled code is now inside the
    # program object.  Keeping shaders attached prevents driver optimisations
    # on some implementations and wastes GPU memory.
    GL.glDetachShader(program, vertex_shader)
    GL.glDetachShader(program, fragment_shader)

    link_status = GL.glGetProgramiv(program, GL.GL_LINK_STATUS)
    if not link_status:
        log_length = GL.glGetProgramiv(program, GL.GL_INFO_LOG_LENGTH)
        if log_length > 0:
            error_log = GL.glGetProgramInfoLog(program).decode("utf-8")
        else:
            error_log = "(no linker log available)"

        GL.glDeleteProgram(program)
        raise GLShaderError("Shader program link failed:\n%s" % error_log)

    # Log non-fatal linker messages (interface mismatch hints, etc.).
    log_length = GL.glGetProgramiv(program, GL.GL_INFO_LOG_LENGTH)
    if log_length > 1:
        info_log = GL.glGetProgramInfoLog(program).decode("utf-8").strip()
        if info_log:
            logger.warning("Shader program linked with warnings:\n%s", info_log)

    return GLuint(program)


# ---------------------------------------------------------------------------
# High-level creation
# ---------------------------------------------------------------------------

def create_program(
        vertex_path: Union[str, Path],
        fragment_path: Union[str, Path],
) -> GLuint | None:
    """
    Load, compile, and link a shader program from two GLSL source files.

    This is the primary entry point for building a program from disk.  It
    manages the intermediate shader handle lifetimes internally: shader
    objects are deleted whether linking succeeds or fails, so the caller only
    receives a clean program handle.

    Args:
        vertex_path:   Path to the vertex shader GLSL source file.
        fragment_path: Path to the fragment shader GLSL source file.

    Returns:
        The linked program handle as :data:`GLuint`.

    Raises:
        GLShaderError: If either source file cannot be found or read, if
                       compilation fails for either stage, or if linking fails.
                       The exception message includes the driver log and the
                       file path of the offending shader.

    Note:
        Each call recompiles from source.  For applications with many shader
        variants, cache compiled binaries using ``glGetProgramBinary`` /
        ``glProgramBinary`` (GL 4.1 Core) to avoid per-launch compile cost.
    """
    v_path = Path(vertex_path).resolve()
    f_path = Path(fragment_path).resolve()

    if not v_path.exists():
        raise GLShaderError("Vertex shader file not found: %s" % v_path)
    if not f_path.exists():
        raise GLShaderError("Fragment shader file not found: %s" % f_path)

    logger.info(
        "Loading shaders — vertex: %s | fragment: %s",
        v_path, f_path,
    )

    try:
        vertex_source = v_path.read_text(encoding="utf-8")
        fragment_source = f_path.read_text(encoding="utf-8")
    except OSError as e:
        raise GLShaderError("Failed to read shader source files: %s" % e) from e

    # Compile both stages.  compile_shader raises GLShaderError on failure, so
    # the vertex handle is guaranteed valid when we reach the fragment step.
    vertex_shader = compile_shader(vertex_source, GLenum(GL.GL_VERTEX_SHADER),
                                   str(v_path))

    try:
        fragment_shader = compile_shader(
            fragment_source, GLenum(GL.GL_FRAGMENT_SHADER), str(f_path)
        )
    except GLShaderError:
        # Vertex shader compiled successfully but fragment failed.
        # Delete the vertex handle before re-raising to avoid a leak.
        GL.glDeleteShader(vertex_shader)
        raise

    try:
        program = link_program(vertex_shader, fragment_shader)
    finally:
        # Delete both shader handles unconditionally.  If linking raised,
        # the finally block still runs and the handles are freed.
        GL.glDeleteShader(vertex_shader)
        GL.glDeleteShader(fragment_shader)

    logger.info("Shader program created successfully (id=%d)", program)
    return program


# ---------------------------------------------------------------------------
# Validation (diagnostic only)
# ---------------------------------------------------------------------------

def validate_program(program: Optional[GLuint]) -> tuple[bool, Optional[str]]:
    """
    Run ``glValidateProgram`` against the current GL draw state.

    Validation checks whether the program can execute given the *current*
    state of the pipeline — bound framebuffers, active textures, etc.  It is
    a purely diagnostic operation and does not modify any GL state.

    This function intentionally returns a ``(bool, str | None)`` pair rather
    than raising on failure.  Validation failure is context-dependent and does
    not imply the program is broken; calling it before the render loop (before
    a valid FBO is bound, for example) will almost always report failure.  The
    return value lets the caller decide whether to treat it as an error.

    Args:
        program: Program handle to validate, or ``None`` / ``0`` for a safe
                 no-op that returns ``(False, "Program ID is None or 0")``.

    Returns:
        ``(True, None)`` if validation passes.
        ``(False, log_message)`` if validation fails, where ``log_message``
        is the driver-supplied info-log string.

    Warning:
        Do not call during initialisation.  Validation succeeds only when the
        full pipeline state (FBO, samplers, etc.) is configured as it will be
        at actual draw time.  Calling it from ``initializeGL`` will almost
        certainly report "Current draw framebuffer is invalid".
    """
    if program is None or program == 0:
        return False, "Program ID is None or 0"

    GL.glValidateProgram(program)
    status = GL.glGetProgramiv(program, GL.GL_VALIDATE_STATUS)

    if status == GL.GL_TRUE:
        return True, None

    raw_log = GL.glGetProgramInfoLog(program)
    error_log = raw_log.decode(
        "utf-8") if raw_log else "Unknown validation error"
    logger.warning("Shader validation failed:\n%s", error_log)
    return False, error_log


# ---------------------------------------------------------------------------
# Deletion
# ---------------------------------------------------------------------------

def delete_program(program: Optional[GLuint]) -> None:
    """
    Delete a program object and release its GPU resources.

    A no-op when ``program`` is ``None`` or ``0``, so callers do not need to
    guard against uninitialized handles.

    Args:
        program: Program handle to delete, or ``None`` for a safe no-op.
    """
    if program:
        GL.glDeleteProgram(program)


# ---------------------------------------------------------------------------
# Uniform setter (standalone utility)
# ---------------------------------------------------------------------------

def set_uniform(
        program: GLuint,
        name: str,
        value: Any,
        location: GLint,
        uniform_type: str = "auto",
) -> None:
    """
    Set a shader uniform by pre-queried location with optional type inference.

    Binds ``program`` for the duration of the call and unbinds it afterward.
    This makes the function self-contained but means it must **not** be called
    from inside a :meth:`ShaderProgramManager.batch_update_uniforms` block or
    any other scope that already has a program bound — doing so will unbind
    the active program when this function returns, causing subsequent draw
    calls to use no program.

    For hot-path uniform updates, prefer
    :meth:`ShaderProgramManager.batch_update_uniforms` which keeps the program
    bound across multiple uniform writes.

    Args:
        program:      Program handle that owns the uniform.
        name:         Uniform variable name — used only in the warning log
                      when ``location == -1``; no GL lookup is performed.
        value:        Value to upload.  Accepted types depend on
                      ``uniform_type``; see below.
        location:     Pre-queried uniform location (from ``glGetUniformLocation``).
                      Pass ``-1`` to silently skip (inactive uniforms that the
                      GLSL compiler has optimised away return ``-1``).
        uniform_type: Explicit type tag or ``"auto"`` for inference.  Supported
                      tags: ``"int"``, ``"float"``, ``"vec2"``, ``"vec3"``,
                      ``"vec4"``, ``"mat3"``, ``"mat4"``.

    Type inference (``uniform_type="auto"``)
    ----------------------------------------
    * ``int`` / ``bool`` / ``np.integer``  → ``glUniform1i``
    * ``float`` / ``np.floating``          → ``glUniform1f``
    * ``np.ndarray`` shape ``(4, 4)``      → ``glUniformMatrix4fv``
    * ``np.ndarray`` shape ``(3, 3)``      → ``glUniformMatrix3fv``
    * sequence of length 2 / 3 / 4        → ``glUniform{2,3,4}fv``

    If inference cannot determine a type, the call is a silent no-op and
    ``glUniform*`` is never called.
    """
    if location == -1:
        # location == -1 is the GL sentinel for "uniform not found / optimised
        # away".  This is normal for conditionally-compiled uniforms and should
        # not be an error — log at DEBUG, not WARNING, to avoid log spam on
        # shaders that declare uniforms for future use.
        logger.debug(
            "Uniform '%s' not active in program (location=-1); skipping", name)
        return

    # --- Type inference --------------------------------------------------- #
    if uniform_type == "auto":
        if isinstance(value, (int, bool, np.integer)):
            uniform_type = "int"
        elif isinstance(value, (float, np.floating)):
            uniform_type = "float"
        elif isinstance(value, (list, tuple, np.ndarray)):
            if hasattr(value, "shape"):
                # NumPy array: use shape for unambiguous dispatch.
                shape = value.shape
                if shape == (4, 4):
                    uniform_type = "mat4"
                elif shape == (3, 3):
                    uniform_type = "mat3"
                elif len(shape) == 1:
                    uniform_type = {2: "vec2", 3: "vec3", 4: "vec4"}.get(
                        shape[0], "auto")
            else:
                # Plain list/tuple: dispatch on length.
                uniform_type = {2: "vec2", 3: "vec3", 4: "vec4"}.get(len(value),
                                                                     "auto")

    # --- Upload ----------------------------------------------------------- #
    # Bind the program for this call only.  See the docstring warning about
    # not nesting this inside an already-bound program scope.
    GL.glUseProgram(program)
    try:
        if uniform_type == "int":
            GL.glUniform1i(location, int(value))
        elif uniform_type == "float":
            GL.glUniform1f(location, float(value))
        elif uniform_type == "vec2":
            GL.glUniform2fv(location, 1, np.asarray(value, dtype=np.float32))
        elif uniform_type == "vec3":
            GL.glUniform3fv(location, 1, np.asarray(value, dtype=np.float32))
        elif uniform_type == "vec4":
            GL.glUniform4fv(location, 1, np.asarray(value, dtype=np.float32))
        elif uniform_type == "mat3":
            GL.glUniformMatrix3fv(location, 1, GL.GL_FALSE,
                                  np.asarray(value, dtype=np.float32))
        elif uniform_type == "mat4":
            GL.glUniformMatrix4fv(location, 1, GL.GL_FALSE, value)
        else:
            logger.error(
                "set_uniform: could not resolve type for uniform '%s' "
                "(value=%r, inferred type=%r); upload skipped.",
                name, value, uniform_type,
            )
    finally:
        # Always unbind, even if the glUniform* call raised, to leave the
        # GL state clean for subsequent draw calls.
        GL.glUseProgram(0)


# ---------------------------------------------------------------------------
# Program manager
# ---------------------------------------------------------------------------

class ShaderProgramManager:
    """
    Owns a single compiled and linked OpenGL program and its uniform interface.

    Wraps the program handle lifecycle (creation via :func:`create_program`,
    deletion via :func:`delete_program`) and exposes the program as a context
    manager for scoped binding.

    The associated :class:`~cross_platform.qt6_utils.image.gl.uniform.UniformManager`
    is created alongside the program and shares its lifetime.  Access it via
    :attr:`uniform_manager` or use :meth:`batch_update_uniforms` to update
    multiple uniforms while the program is bound.

    Slots are used to prevent accidental attribute creation and to give the
    interpreter a slight attribute-lookup advantage on high-frequency calls.
    """

    __slots__ = ("_program_id", "_uniform_manager")

    def __init__(self) -> None:
        # Zero is the GL sentinel for "no program"; used as the uninitialised
        # state so is_valid returns False before initialize() is called.
        self._program_id: GLuint = GLuint(0)
        self._uniform_manager: Optional[UniformManager] = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def handle(self) -> GLuint:
        """The raw OpenGL program handle. ``0`` when uninitialised."""
        return self._program_id

    @property
    def is_valid(self) -> bool:
        """``True`` when a program has been successfully linked and not yet cleaned up."""
        return self._program_id is not None and self._program_id > 0

    @property
    def uniform_manager(self) -> Optional[UniformManager]:
        """
        The :class:`~cross_platform.qt6_utils.image.gl.uniform.UniformManager`
        bound to this program, or ``None`` before :meth:`initialize` is called.
        """
        return self._uniform_manager

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def initialize(
            self,
            vertex_path: Union[str, Path],
            fragment_path: Union[str, Path],
    ) -> None:
        """
        Compile, link, and store the shader program.

        Idempotent: a second call while a valid program is already held is a
        no-op.  To replace the program, call :meth:`cleanup` first.

        Args:
            vertex_path:   Path to the vertex shader GLSL source.
            fragment_path: Path to the fragment shader GLSL source.

        Raises:
            GLShaderError: Propagated from :func:`create_program` if any stage
                           of compilation or linking fails.
        """
        if self._program_id:
            # Already initialised — guard against accidental double-init which
            # would leak the existing program handle.
            return

        # create_program raises GLShaderError on any failure; no None check needed.
        prog = create_program(vertex_path=vertex_path,
                              fragment_path=fragment_path)
        self._program_id = prog
        self._uniform_manager = UniformManager(self._program_id)

    def cleanup(self) -> None:
        """
        Delete the program object and release GPU resources.

        Resets the handle to ``0`` so :attr:`is_valid` returns ``False`` and
        a subsequent :meth:`initialize` call can re-create the program.  Safe
        to call multiple times.
        """
        if self._program_id:
            delete_program(self._program_id)
            self._program_id = GLuint(0)
            self._uniform_manager = None

    # ------------------------------------------------------------------
    # Context manager (program binding)
    # ------------------------------------------------------------------

    def __enter__(self) -> GLuint:
        """
        Bind the program and return its handle for the duration of the block.

        Raises:
            GLShaderError: If the program has not been initialised
                           (``is_valid`` is ``False``).
        """
        if not self.is_valid:
            raise GLShaderError(
                "ShaderProgramManager.__enter__: program is not initialised. "
                "Call initialize() before using the program as a context manager."
            )
        GL.glUseProgram(self._program_id)
        return self._program_id

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Unbind the program unconditionally, even if the block raised."""
        GL.glUseProgram(0)

    # ------------------------------------------------------------------
    # Batch uniform update
    # ------------------------------------------------------------------

    @contextmanager
    def batch_update_uniforms(self) -> Iterator[UniformManager]:
        """
        Bind the program and yield the :class:`UniformManager` for bulk updates.

        Keeps ``glUseProgram`` active across all uniform writes in the block,
        avoiding the repeated bind/unbind overhead of calling
        :func:`set_uniform` individually for each uniform.

        Usage::

            with manager.batch_update_uniforms() as um:
                um.set_fast(location=um.locs.MVP, value=mvp, uniform_type=UniformType.MAT4)
                um.set_fast(location=um.locs.COLOR, value=color, uniform_type=UniformType.VEC3)

        Yields:
            The :class:`~cross_platform.qt6_utils.image.gl.uniform.UniformManager`
            associated with this program.

        Raises:
            GLShaderError: If the program is not initialised (propagated from
                           :meth:`__enter__`).
        """
        if self._uniform_manager is None:
            raise GLShaderError(
                "batch_update_uniforms: UniformManager is not available. "
                "Call initialize() before updating uniforms."
            )
        # Delegate binding/unbinding to __enter__/__exit__ via the `with self` block.
        with self:
            yield self._uniform_manager

"""
Immutable GL configuration presets and runtime capability validation.

Defines GLConfig, a frozen dataclass that captures all knobs controlling
GL behaviour (immutable texture storage, error checking, debug output, and
pixel-unpack alignment). Two presets are provided via GL_CONFIGS:

    "default"  — optimised for production; all checks disabled.
    "debug"    — enables glGetError checks and KHR_debug output.

Typical usage after the GL context has been created::

    from gl_config import get_gl_config

    cfg = get_gl_config("debug", gl_version=(4, 6))
    if cfg.USE_IMMUTABLE_STORAGE:
        GL.glTexStorage2D(...)

Requires Python >= 3.9, PyOpenGL, OpenGL >= 4.1.
"""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class GLConfig:
    """Immutable snapshot of GL feature flags for a single runtime configuration.

    All fields default to the most conservative (lowest-requirement) value so
    that a plain ``GLConfig()`` is always safe to construct before the GL
    context version is known.  Call :func:`validate` once the context is live
    to confirm that the chosen flags are supported.

    Attributes:
        USE_IMMUTABLE_STORAGE: Allocate textures with ``glTexStorage*``
            (GL 4.2+ / ``GL_ARB_texture_storage``) instead of
            ``glTexImage*``.  Immutable textures allow the driver to validate
            completeness once at allocation time rather than at every draw
            call, reducing per-frame overhead.  Requires GL >= 4.2; raises
            at :meth:`validate` time if the context does not meet that
            requirement.
        CHECK_GL_ERRORS: Call ``glGetError`` after each GL operation to
            surface driver-level errors.  Carries a measurable CPU cost on
            some drivers; leave ``False`` in production builds.
        DEBUG_MODE: Request ``GL_KHR_debug`` callback output.  Requires the
            context to have been created with the ``GL_CONTEXT_FLAG_DEBUG_BIT``
            flag set.  Has no effect on macOS, which does not expose
            ``KHR_debug``.
        FORCE_UNPACK_ALIGNMENT_1: Set ``GL_UNPACK_ALIGNMENT`` to 1 before
            texture uploads.  Ensures correct pixel addressing for images
            whose row stride is not a multiple of the default 4-byte
            alignment.  Disable only when all uploaded images are guaranteed
            to be 4-byte-aligned.
    """

    USE_IMMUTABLE_STORAGE: bool = False
    CHECK_GL_ERRORS: bool = False
    DEBUG_MODE: bool = False
    FORCE_UNPACK_ALIGNMENT_1: bool = True

    def validate(self, gl_version: tuple[int, int]) -> None:
        """Assert that this configuration is compatible with a live GL context.

        Should be called once inside ``initialize_context()``, after the
        context version has been queried, before any rendering takes place.

        Args:
            gl_version: The ``(major, minor)`` version reported by the active
                GL context, e.g. ``(4, 6)``.

        Raises:
            ValueError: If ``USE_IMMUTABLE_STORAGE`` is ``True`` and
                ``gl_version`` is below ``(4, 2)``.
        """
        if self.USE_IMMUTABLE_STORAGE and gl_version < (4, 2):
            major, minor = gl_version
            raise ValueError(
                f"USE_IMMUTABLE_STORAGE requires GL >= 4.2, "
                f"but the active context reports {major}.{minor}."
            )


GL_CONFIGS: dict[str, GLConfig | None] = {
    "default": None,  # populated by initialize_context() from backend
    "debug": GLConfig(CHECK_GL_ERRORS=True, DEBUG_MODE=True),
}


def get_gl_config(
        preset: Literal["default", "debug"] = "default",
        gl_version: tuple[int, int] | None = None,
) -> GLConfig:
    """
    Return a preset GLConfig, optionally validated against the active context.

    Args:
        preset: Which preset to return.  ``"default"`` is optimised for
            production; ``"debug"`` enables error checking and KHR_debug
            output.
        gl_version: When provided, :meth:`GLConfig.validate` is called
            immediately and raises ``ValueError`` if the preset's requirements
            are not met by the context.  Pass ``None`` to skip validation
            (e.g. before the context is initialised).

    Returns:
        The frozen :class:`GLConfig` for the requested preset.

    Raises:
        KeyError: If ``preset`` is not a recognised key in :data:`GL_CONFIGS`.
        ValueError: Propagated from :meth:`GLConfig.validate` when
            ``gl_version`` is provided and the preset's requirements are not
            met.
    """
    cfg = GL_CONFIGS.get(preset)

    if cfg is None:
        if preset == "default":
            raise RuntimeError(
                "The 'default' GL config is not available until "
                "initialize_context() has been called."
            )
        raise KeyError(f"Unknown GL config preset: {preset!r}")

    if gl_version is not None:
        cfg.validate(gl_version)

    return cfg

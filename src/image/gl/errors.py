"""
errors.py
============
OpenGL exception hierarchy, error-code mapping, and a context manager for
safe, structured GL error checking with detailed logging.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Iterator

from image.gl.backend import GL
from image.gl.config import get_gl_config

__all__ = [
    "GLError",
    "GLInitializationError",
    "GLTextureError",
    "GLUploadError",
    "GLShaderError",
    "GLFramebufferError",
    "GLMemoryError",
    "GLSyncTimeout",
    "gl_error_check",
    "clear_gl_errors",
    "GL_ERROR_CODES",
]

logger = logging.getLogger(__name__)


def _build_error_code_map() -> dict[int, str]:
    """
    Build the GL error-code → name mapping defensively at import time.

    ``GL_STACK_OVERFLOW`` and ``GL_STACK_UNDERFLOW`` were part of the fixed-
    function pipeline that was removed from the Core profile in GL 3.1.
    Accessing them directly on a Core-profile context raises ``AttributeError``
    and crashes the import.  ``getattr`` with a sentinel of ``None`` lets us
    include them only when the driver actually exposes them.

    Returns:
        A ``dict`` mapping integer GL error codes to their canonical string
        names, containing only codes present on the current driver.
    """
    # Mandatory error codes: present in every GL profile >= 2.0.
    codes: dict[int, str] = {
        GL.GL_INVALID_ENUM: "GL_INVALID_ENUM",
        GL.GL_INVALID_VALUE: "GL_INVALID_VALUE",
        GL.GL_INVALID_OPERATION: "GL_INVALID_OPERATION",
        GL.GL_OUT_OF_MEMORY: "GL_OUT_OF_MEMORY",
        GL.GL_INVALID_FRAMEBUFFER_OPERATION: "GL_INVALID_FRAMEBUFFER_OPERATION",
    }

    # Legacy fixed-function codes: absent in Core profile (GL 3.1+).
    # Include them only when the driver exports the enum token.
    _optional: list[tuple[str, str]] = [
        ("GL_STACK_OVERFLOW", "GL_STACK_OVERFLOW"),
        ("GL_STACK_UNDERFLOW", "GL_STACK_UNDERFLOW"),
    ]
    for attr, name in _optional:
        token = getattr(GL, attr, None)
        if token is not None:
            codes[token] = name

    return codes


# Maps GL error-code integers to their canonical ``GL_*`` string names.
GL_ERROR_CODES: dict[int, str] = _build_error_code_map()


class GLError(Exception):
    """
    Base exception for all OpenGL-related errors raised by this package.

    Catch this class to handle any GL failure regardless of category:

    .. code-block:: python

        with gl_error_check("draw call"):
            GL.glDrawArrays(...)
    """


class GLInitializationError(GLError):
    """Raised when the OpenGL context or associated resources fail to
    initialize."""


class GLTextureError(GLError):
    """
    Raised when a texture operation fails (creation, parameter setup, binding).

    ``GLUploadError`` is a specialization of this class for data-transfer
    failures specifically.
    """


class GLUploadError(GLTextureError):
    """
    Raised when transferring image data to the GPU fails.

    Subclasses ``GLTextureError`` because upload is a texture operation;
    callers that catch ``GLTextureError`` will also catch this.
    """


class GLShaderError(GLError):
    """Raised when shader compilation or program linking fails."""


class GLFramebufferError(GLError):
    """Raised when framebuffer attachment configuration or completeness check fails."""


class GLMemoryError(GLError):
    """
    Raised when the GL driver reports ``GL_OUT_OF_MEMORY``.

    Named ``GLMemoryError`` (not ``MemoryError``) to avoid shadowing the
    Python built-in ``MemoryError``, which signals host-side allocation
    failures unrelated to the GPU.
    """


class GLSyncTimeout(Exception):
    """
    Raised when a ``glClientWaitSync`` call exceeds the configured timeout.

    This exception indicates that the GPU did not signal the fence sync object
    within the allowed wait period (``_SYNC_TIMEOUT_NS``).  It does **not**
    mean the transfer failed — the DMA may still complete after this point —
    but the CPU can no longer safely read the PBO without risking a data race.

    To reduce the likelihood of spurious timeouts, increase
    ``_SYNC_TIMEOUT_NS`` or ensure the upstream render pass completes before
    the fence is waited on.
    """


@contextmanager
def gl_error_check(
        operation: str,
        exception_class: type[GLError] = GLError,
) -> Iterator[None]:
    """
    Context manager that drains the GL error queue after a block of GL calls.

    Usage:

        with gl_error_check("texture upload", GLTextureError):
            GL.glTexImage2D(...)

    Behavior
    ---------
    * When ``GLConfig.CHECK_GL_ERRORS`` is ``False`` the block executes with
      zero overhead and no logging — disabled error checking in a production
      build is intentional, not a warning condition.
    * When enabled, **all** pending errors are drained from the GL error queue
      after the block exits (whether normally or via exception), then a single
      consolidated exception is raised if any were found.  This prevents the
      error queue from accumulating stale codes that would misattribute errors
      to a later operation.
    * Each individual error is logged at ``ERROR`` level exactly **once**.  The
      combined exception message repeats the list for convenience, but no
      second ``logger.error`` call is made, avoiding duplicate log entries.

    Args:
        operation:       Human-readable label for the guarded block.
                         Shown in log messages and the exception string.
                         Examples: ``"texture upload"``, ``"shader link"``.
        exception_class: The ``GLError`` subclass to raise when errors are
                         found.  Defaults to the base ``GLError``.  Pass a
                         more specific class (e.g. ``GLTextureError``) so
                         callers can catch narrowly.

    Yields:
        ``None`` — the body of the ``with`` block.

    Raises:
        ``exception_class``: When one or more GL errors are pending after the
            block exits and ``CHECK_GL_ERRORS`` is enabled.  The message
            lists every error that was drained from the queue.

    Note:
        ``glGetError`` is not free — each call performs a driver round-trip.
        Keeping ``CHECK_GL_ERRORS`` disabled in release builds avoids this
        overhead entirely.
    """
    gl_cfg = get_gl_config()

    if not gl_cfg.CHECK_GL_ERRORS:
        # Error checking is deliberately disabled (e.g. release build).
        # Do not log — this fires on every call site and is not anomalous.
        yield
        return

    try:
        yield
    finally:
        # Drain the full error queue.
        errors: list[str] = []

        while (error_code := GL.glGetError()) != GL.GL_NO_ERROR:
            name = GL_ERROR_CODES.get(error_code, f"UNKNOWN_GL_ERROR")
            entry = "%s (0x%04x)" % (name, error_code)
            # Log each error individually for granularity; the combined
            # raise below does NOT log again to avoid duplicate records.
            logger.error("GL error during '%s': %s", operation, entry)
            errors.append(entry)

        if errors:
            # Build one consolidated exception message and raise once.
            # Deliberately not calling logger.error here — every individual
            # error was already logged in the loop above.
            n = len(errors)
            detail = "\n  ".join(errors)
            logger.error(f"{n} OpenGL error{'s' if n != 1 else ''} during '{operation}':\n  {detail}")
            raise exception_class(
                f"{n} OpenGL error{'s' if n != 1 else ''} during '{operation}':\n  {detail}"
            )

def clear_gl_errors(label: str = "") -> list[str]:
    """
    Drain the GL error queue and discard any pending errors.

    Useful for resetting error state before a guarded block when you know
    prior GL calls may have left errors in the queue that are irrelevant to
    the operation you are about to check.  For example:

        clear_gl_errors("pre-upload flush")
        with gl_error_check("texture upload", GLTextureError):
            GL.glTexImage2D(...)

    Unlike ``gl_error_check``, this function does **not** raise — it only
    drains and logs.  Each discarded error is logged at ``WARNING`` level
    so that stale errors are visible without aborting the caller.

    This function is a no-op when ``GLConfig.CHECK_GL_ERRORS`` is ``False``,
    matching the behavior of ``gl_error_check`` and avoiding driver
    round-trips in release builds.

    Args:
        label: Optional human-readable label used in log messages to identify
               the call site.  Examples: ``"pre-upload flush"``,
               ``"context init"``.  Omit for a generic message.

    Returns:
        A list of error strings that were drained from the queue (e.g.
        ``["GL_INVALID_ENUM (0x0500)"]``).  Returns an empty list when the
        queue was already clean or ``CHECK_GL_ERRORS`` is disabled.
    """
    gl_cfg = get_gl_config()

    if not gl_cfg.CHECK_GL_ERRORS:
        return []

    site = f" before '{label}'" if label else ""
    drained: list[str] = []

    while (error_code := GL.glGetError()) != GL.GL_NO_ERROR:
        name = GL_ERROR_CODES.get(error_code, "UNKNOWN_GL_ERROR")
        entry = "%s (0x%04x)" % (name, error_code)
        logger.warning("Stale GL error discarded%s: %s", site, entry)
        drained.append(entry)

    return drained
"""Shader management module for loading and validating shader files."""

from pathlib import Path
from types import MappingProxyType

SHADER_DIR = Path(__file__).parent

type ShaderMap = MappingProxyType[str, Path]

IMAGE_SHADERS: ShaderMap = MappingProxyType({
    "image_vertex": SHADER_DIR / "image.vert",
    "image_fragment": SHADER_DIR / "image.frag",
})

COLORBAR_SHADERS: ShaderMap = MappingProxyType({
    "colorbar_vertex": SHADER_DIR / "colorbar.vert",
    "colorbar_fragment": SHADER_DIR / "colorbar.frag",
})

SHADERS: ShaderMap = MappingProxyType(IMAGE_SHADERS | COLORBAR_SHADERS)


def validate_shader_paths(shaders: dict[str, Path]) -> None:
    """
    Validate that all shader files exist.

    Args:
        shaders: Dictionary mapping shader names to file paths.

    Raises:
        FileNotFoundError: If any shader file is missing, with details on all missing files.
    """
    missing = {name: path for name, path in shaders.items() if
               not path.exists()}

    if missing:
        details = "\n".join(
            f"  {name}: {path}" for name, path in missing.items())
        raise FileNotFoundError(
            f"Shader files not found:\n{details}"
        )

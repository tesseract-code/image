from pathlib import Path

import numpy as np

from image.gl_imshow import imshow
from image.io.factory import Backend
from image.io.load import load_image

# Path is resolved relative to this file, not the working directory
_TEST_IMG_PATH = Path(__file__).parent / "img" / "nebula.jpg"


def load_test_img() -> np.ndarray:
    if not _TEST_IMG_PATH.exists():
        raise FileNotFoundError(f"Test image not found: {_TEST_IMG_PATH}")

    buf, meta = load_image(_TEST_IMG_PATH, backend=Backend.PILLOW)

    # np.flipud corrects for OpenGL's bottom-left origin convention
    return np.flipud(buf.data)


def main():
    # --- RGB demo ---
    img_rgb = load_test_img()
    imshow(img_rgb, title="Image – RGB", cmap="gray", block=False)

    # --- Normalize to [0, 1] float32 before any arithmetic ---
    img_norm = img_rgb.astype(np.float32)
    if img_norm.max() > 1.0:
        img_norm /= 255.0

    # --- Grayscale / colormap demo ---
    if img_norm.ndim == 3 and img_norm.shape[2] >= 3:
        # Slice to RGB, discarding alpha if present, then apply BT.709 luma coefficients
        gray = img_norm[..., :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    else:
        gray = img_norm  # already grayscale and float32

    imshow(gray, title="Image – Viridis colormap", cmap="viridis", block=False)
    imshow(gray, title="Image – Magma colormap", cmap="magma", block=False)

    # --- Explicit vmin / vmax clipping demo (values in [0, 1] normalized range) ---
    imshow(
        gray,
        title="Image – contrast-clipped [0.2, 0.8]",
        cmap="plasma",
        vmin=0.2,
        vmax=0.8,
        block=True,  # blocks here → closes all windows on Esc / Q
    )


if __name__ == "__main__":
    main()
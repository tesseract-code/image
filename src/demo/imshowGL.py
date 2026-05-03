from pathlib import Path

import numpy as np

from image.gl_imshow import imshow
from image.load.factory import Backend
from image.load.load import load_image

_TEST_IMG_PATH = Path(__file__).parent / "img" / "nebula.jpg"


def load_test_img() -> np.ndarray:
    """Load the test image ``nebula.jpg`` and flip it vertically.

    The test image is loaded relative to this source file, not the
    current working directory. The image is flipped vertically to
    correct for OpenGL's bottom‑left origin convention used by the
    display backend.

    Returns
    -------
    np.ndarray
        The loaded image array, flipped upside‑down.

    Raises
    ------
    FileNotFoundError
        If the test image file is missing.
    """
    if not _TEST_IMG_PATH.exists():
        raise FileNotFoundError(f"Test image not found: {_TEST_IMG_PATH}")

    buf, meta = load_image(_TEST_IMG_PATH, backend=Backend.PILLOW)
    return np.flipud(buf.data)


def main():
    """Run a demonstration of image display with different colormaps.

    Displays the original RGB image, a grayscale version using the
    viridis and magma colormaps, and a contrast‑clipped version
    (vmin=0.2, vmax=0.8) with the plasma colormap. The final window
    blocks until closed (by pressing Esc or Q), closing all windows
    at once.
    """
    img_rgb = load_test_img()
    imshow(img_rgb, title="Image – RGB", cmap="gray", block=False)

    img_norm = img_rgb.astype(np.float32)
    if img_norm.max() > 1.0:
        img_norm /= 255.0

    if img_norm.ndim == 3 and img_norm.shape[2] >= 3:
        gray = img_norm[..., :3] @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    else:
        gray = img_norm

    imshow(gray, title="Image – Viridis colormap", cmap="viridis", block=False)
    imshow(gray, title="Image – Magma colormap", cmap="magma", block=False)

    imshow(
        gray,
        title="Image – contrast-clipped [0.2, 0.8]",
        cmap="plasma",
        vmin=0.2,
        vmax=0.8,
        block=True,
    )


if __name__ == "__main__":
    main()
import logging
import sys

import numpy as np

from image.gui.imageVTK import ImageViewerWidget, _NUMPY_AVAILABLE

if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication

    logging.basicConfig(level=logging.DEBUG)

    app = QApplication(sys.argv)
    viewer = ImageViewerWidget()
    viewer.resize(800, 600)
    viewer.show()

    h, w = 256, 256
    grad = np.tile(np.linspace(0, 1, w), (h, 1))
    rgb = np.stack([grad, grad, grad], axis=-1)
    viewer.set_data(rgb)

    sys.exit(app.exec())
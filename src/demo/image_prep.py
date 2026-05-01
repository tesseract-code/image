import logging
import sys
from multiprocessing import set_start_method
from typing import Callable, Tuple

from PyQt6.QtCore import QThread
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QMainWindow

from demo import create_rgb_checkered
from demo import ImageGeneratorWorker
from image.gl.utils import get_surface_format
from image.gui.controller.overlay import OverlayImageWidgetController
from qtcore.event import run_qt_app

logger = logging.getLogger(__name__)


class ImagePrep(QMainWindow):
    def __init__(self,
                 generator_func: Callable = create_rgb_checkered,
                 shape: Tuple[int, int] = (1024, 1024),
                 fps: int = 60):
        super().__init__()
        self.resize(800, 800)

        # UI Setup
        QSurfaceFormat.setDefaultFormat(get_surface_format())
        self.overlay_view_ctrl = OverlayImageWidgetController()
        self.setCentralWidget(self.overlay_view_ctrl.central_widget())

        # --- THREAD SETUP ---
        self.thread = QThread()

        # Instantiate Worker with desired settings
        self.worker = ImageGeneratorWorker(
            generator_func=generator_func,  # Pass the function ref
            shape=shape,  # Size
            fps=fps,  # Target FPS
            square_size=64  # Generator specific arg,
        )

        # Move worker to thread
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.overlay_view_ctrl.set_image)
        self.worker.finished.connect(self.thread.quit)
        self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)

        # Start
        self.thread.start()

    def closeEvent(self, event):
        # Clean shutdown
        self.worker.stop()
        self.thread.quit()
        self.thread.wait()
        self.overlay_view_ctrl.cleanup()
        event.accept()


def main():
    set_start_method('spawn', force=True)
    return sys.exit(run_qt_app(ImagePrep, ))


if __name__ == '__main__':
    logging.basicConfig()
    logger.root.setLevel(logging.INFO)
    main()

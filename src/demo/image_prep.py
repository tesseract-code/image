import logging
import sys
from multiprocessing import set_start_method
from typing import Callable, Tuple

from PyQt6.QtCore import QThread, QDeadlineTimer
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QMainWindow

from demo.utils.create_image import create_rgb_checkered
from demo.utils.image_generator import ImageGeneratorWorker
from image.gl.utils import get_surface_format
from image.gui.controller.overlay import OverlayImageWidgetController
from qtcore.event import run_qt_app
from qtcore.reference import has_qt_cpp_binding

logger = logging.getLogger(__name__)


class ImagePrep(QMainWindow):
    """A QMainWindow that displays generated images in real‑time.

    Uses a background thread to generate frames at a constant frame rate and
    forwards them to an overlay widget for display. Designed for testing
    real‑time image throughput and GUI responsiveness.

    Parameters
    ----------
    generator_func : Callable
        Factory function that returns an image array.
        Default is :func:`create_rgb_checkered`.
    shape : Tuple[int, int]
        Height and width of the generated images. Default is (1024, 1024).
    fps : int
        Target frames per second. Must be positive. Default is 60.
    """

    def __init__(self,
                 generator_func: Callable = create_rgb_checkered,
                 shape: Tuple[int, int] = (1024, 1024),
                 fps: int = 60):
        """
        Initialize the window, set up the image generator thread, and begin
        streaming frames.

        Parameters
        ----------
        generator_func : Callable
            See class documentation.
        shape : Tuple[int, int]
            See class documentation.
        fps : int
            See class documentation.
        """
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
            fps=fps,  # Target FPS,
            regenerate_each_frame=False,
            max_frames=1000,
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

    def cleanup(self):
        """Stop the worker and safely shut down the background thread.

        Requests the worker to stop, then quits the thread with a 3‑second
        deadline. If the thread does not finish in time, it is forcefully
        terminated. All resources are then released.

        Notes
        -----
        This method is safe to call multiple times; after the first call
        ``self.worker`` and ``self.thread`` are set to ``None``.
        """
        if self.worker is not None:
            self.worker.stop()
            self.worker = None

        if (self.thread is not None
                and has_qt_cpp_binding(self.thread)
                and self.thread.isRunning()):
            self.thread.quit()
            deadline = QDeadlineTimer(3000)
            self.thread.wait(deadline=deadline)

            if not self.thread.isFinished():
                self.thread.terminate()

            self.thread = None

        if self.overlay_view_ctrl is not None:
            self.overlay_view_ctrl.cleanup()
            self.overlay_view_ctrl = None

    def closeEvent(self, event):
        """Handle the window close event by performing a clean shutdown.

        Calls :meth:`cleanup` to stop the generator thread and release
        resources, then calls the parent class implementation.
        """
        self.cleanup()
        super().closeEvent(event)


def main():
    """Entry point for the image‑streaming application.

    Forces the ``spawn`` start method for multiprocessing compatibility and
    launches the Qt application with an :class:`ImagePrep` window.

    Returns
    -------
    int
        The exit code returned by the Qt event loop.
    """
    set_start_method('spawn', force=True)
    return sys.exit(run_qt_app(ImagePrep, ))


if __name__ == '__main__':
    logging.basicConfig()
    logger.root.setLevel(logging.INFO)
    main()
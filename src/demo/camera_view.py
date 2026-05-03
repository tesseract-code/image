import logging
import sys
import time
from multiprocessing import set_start_method

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, QThread
from PyQt6.QtGui import QSurfaceFormat
from PyQt6.QtWidgets import QMainWindow

from demo.utils.camera_capture import frame_generator
from image.gl.utils import get_surface_format
from image.gui.controller.overlay import OverlayImageWidgetController
from qtcore.event import run_qt_app
from qtcore.worker import WorkerSignals

logger = logging.getLogger(__name__)


class CameraWorker(QObject):
    frame_ready = pyqtSignal(np.ndarray)
    finished = pyqtSignal()

    def __init__(self, camera_index: int = 0, fps: int = 30, parent=None):
        super().__init__(parent=parent)
        self.cam_index = camera_index
        self.fps = fps
        self.target_frame_time = 1.0 / self.fps
        self._is_running = False
        self.signals = WorkerSignals()

    def run(self):
        self._is_running = True
        next_frame_time = time.perf_counter()
        generator = frame_generator(self.cam_index)

        for frame in generator:
            if self._is_running:
                self.frame_ready.emit(np.flipud(frame))

                now = time.perf_counter()
                next_frame_time += self.target_frame_time
                time_to_wait = next_frame_time - now

                if time_to_wait > 0:
                    # Hybrid Strategy:
                    # 1. Sleep to save CPU if we have a "long" wait (> 2ms)
                    #    We sleep a bit less than needed to wake up safely before the target.
                    if time_to_wait > 0.002:
                        time.sleep(time_to_wait - 0.001)

                    # 2. Busy Wait (Spinlock) for the final millisecond(s)
                    #    This provides microsecond precision that time.sleep() cannot guarantee.
                    while time.perf_counter() < next_frame_time:
                        pass
                else:
                    # We are behind schedule (lagging)
                    # Reset the schedule to now prevent a burst of catch-up frames
                    next_frame_time = time.perf_counter()
            else:
                break

        self.signals.finished.emit("CameraWorker", None)

    def stop(self):
        self._is_running = False


class CameraViewer(QMainWindow):
    def __init__(self,
                 camera_index: int = 0,
                 fps: int = 30):
        super().__init__()
        self.resize(800, 800)

        # UI Setup
        QSurfaceFormat.setDefaultFormat(get_surface_format())
        self.overlay_view_ctrl = OverlayImageWidgetController()
        self.setCentralWidget(self.overlay_view_ctrl.central_widget())

        # --- THREAD SETUP ---
        self.thread = QThread()

        # Instantiate Worker with desired settings
        self.worker = CameraWorker(camera_index=camera_index, fps=fps)

        # Move worker to thread
        self.worker.moveToThread(self.thread)

        # Connect signals
        self.thread.started.connect(self.worker.run)
        self.worker.frame_ready.connect(self.overlay_view_ctrl.set_image)
        self.worker.signals.finished.connect(self.thread.quit)
        self.worker.signals.finished.connect(self.worker.deleteLater)
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
    return sys.exit(run_qt_app(CameraViewer, ))


if __name__ == '__main__':
    logger.root.setLevel(logging.DEBUG)
    main()

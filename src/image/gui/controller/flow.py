import numpy as np
from PyQt6.QtCore import QObject, QTimer, pyqtSignal, Qt
from PyQt6.QtGui import QGuiApplication
from math import ceil

from image.pipeline.mailbox import FrameMailbox
from image.pipeline.stats import FrameStats


class FlowController(QObject):
    # Signal now carries both image and metadata
    update_event = pyqtSignal(np.ndarray, FrameStats)

    def __init__(self, mailbox: FrameMailbox, fps=90):
        super().__init__()
        self.mailbox = mailbox

        screen = QGuiApplication.instance().primaryScreen()
        self._vsync_time_ms: int = int(ceil(1 / screen.refreshRate()))

        self._update_delay_ms = max(self._vsync_time_ms, int(1000 / fps))

        self.timer = QTimer()
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        # Use a slightly faster poll rate than refresh rate to minimize latency
        self.timer.setInterval(self._update_delay_ms)
        self.timer.timeout.connect(self._check_mailbox)
        self.timer.start()

    def _check_mailbox(self):
        # Poll the mailbox (Thread-safe read)
        result = self.mailbox.read()

        if result:
            image, meta = result
            # Emit to UI (This is the only signal going to the Main Loop)
            self.update_event.emit(image, meta)

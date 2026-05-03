import threading
import time
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal, pyqtSlot


class ImageGeneratorWorker(QObject):
    """Worker class intended to run in a separate QThread.

    Generates an image based on the provided generator function and emits
    it at a precise target FPS.

    Attributes
    ----------
    frame_ready : pyqtSignal(np.ndarray)
        Signal emitted with the generated frame as a NumPy array.
    finished : pyqtSignal()
        Signal emitted when the generation loop finishes (normally or after stop).
    error : pyqtSignal(str)
        Signal emitted with an error description if an exception occurs.

    Notes
    -----
    The hybrid sleep + spinlock timing strategy used here trades CPU time for
    precision. The spinlock consumes ~100% of one core during the final
    millisecond before each frame deadline. At 30 fps this amounts to roughly
    30 ms/s of full-core spin — acceptable for a background thread, but
    callers should be aware of the CPU cost.
    """

    frame_ready = pyqtSignal(np.ndarray)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(
            self,
            generator_func: Callable[..., np.ndarray],
            shape: Tuple[int, int] = (1024, 1024),
            fps: int = 30,
            regenerate_each_frame: bool = False,
            max_frames: Optional[int] = None,
            **generator_kwargs: Any
    ):
        """
        Initialize the ImageGeneratorWorker.

        Parameters
        ----------
        generator_func : Callable[..., np.ndarray]
            One of the refactored creator functions that returns an image array.
        shape : Tuple[int, int], optional
            Height and width of the generated image. Default is (1024, 1024).
        fps : int, optional
            Target frames per second. Must be positive. Default is 30.
        regenerate_each_frame : bool, optional
            If True, regenerates the image every frame (CPU/memory stress test).
            If False, generates once and re-emits (pure bandwidth/GUI test).
            Default is False.
        max_frames : Optional[int], optional
            Stop after emitting this many frames. ``None`` means run until
            :meth:`stop` is called. Default is None.
        **generator_kwargs : Any
            Additional keyword arguments forwarded to `generator_func`.

        Raises
        ------
        ValueError
            If `fps` is not a positive integer.
        """
        super().__init__()

        # Fix #7: validate fps before the division so the error message is clear
        if fps <= 0:
            raise ValueError(f"fps must be a positive integer, got {fps!r}")

        self.generator_func = generator_func
        self.shape = shape
        self.fps = fps
        self.target_frame_time = 1.0 / self.fps
        self.regenerate_each_frame = regenerate_each_frame
        self.max_frames = max_frames
        self.generator_kwargs = generator_kwargs

        self._stop_event = threading.Event()

    @pyqtSlot()
    def run(self):
        """Main loop that emits frames at the target FPS.

        Generates images using `generator_func` and emits them via the
        ``frame_ready`` signal. Timing is controlled to maintain a steady
        frame rate. Emits ``finished`` when done or ``error`` if an exception
        occurs.
        """
        self._stop_event.clear()
        frames_emitted = 0

        try:
            current_frame: Optional[np.ndarray] = None
            if not self.regenerate_each_frame:
                current_frame = self.generator_func(
                    shape=self.shape,
                    **self.generator_kwargs
                )

            next_frame_time = time.perf_counter()

            while not self._stop_event.is_set():
                if self.max_frames is not None and frames_emitted >= self.max_frames:
                    break

                if self.regenerate_each_frame:
                    current_frame = self.generator_func(
                        shape=self.shape,
                        **self.generator_kwargs
                    )

                self.frame_ready.emit(current_frame)
                frames_emitted += 1

                # --- PRECISION TIMING LOGIC ---
                now = time.perf_counter()
                next_frame_time += self.target_frame_time
                time_to_wait = next_frame_time - now

                if time_to_wait > 0:
                    if time_to_wait > 0.002:
                        time.sleep(time_to_wait - 0.001)

                    while time.perf_counter() < next_frame_time:
                        pass
                else:
                    # Behind schedule — reset baseline to avoid a burst of
                    # back-to-back catch-up frames
                    next_frame_time = time.perf_counter()

        except Exception as exc:
            self.error.emit(f"{type(exc).__name__}: {exc}")

        self.finished.emit()

    def stop(self):
        """Signal the worker loop to exit gracefully."""
        self._stop_event.set()
import multiprocessing
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import List
from typing import Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtCore import (
    Qt, QPointF, QPoint, pyqtSlot
)
from PyQt6.QtGui import (
    QPainter, QPen, QColor, QPaintEvent, QPolygonF, QMouseEvent
)
from PyQt6.QtWidgets import (
    QWidget, QSizePolicy, QLabel
)


# --- Data Structures ---


@dataclass(slots=True, frozen=True)
class IntensityProfile:
    horizontal: np.ndarray
    vertical: np.ndarray
    is_rgb: bool
    data_range: Tuple[float, float]

    def __eq__(self, other):
        if self is other: return True
        if not isinstance(other, IntensityProfile): return False
        return (np.array_equal(self.horizontal, other.horizontal) and
                np.array_equal(self.vertical, other.vertical))


def compute_intensity_profile(img_view: np.ndarray,
                              target_res: int) -> IntensityProfile:
    """Performs mathematically perfect reduction (average of ALL pixels)."""
    if img_view.size == 0:
        return IntensityProfile(np.array([]), np.array([]), False, (0.0, 1.0))

    is_rgb = False
    work_img = img_view

    if work_img.ndim == 3:
        depth = work_img.shape[2]
        if depth >= 3:
            work_img = work_img[:, :, :3]
            is_rgb = True
        elif depth == 1:
            work_img = work_img[:, :, 0]

    h_raw = np.mean(work_img, axis=0, dtype=np.float32)
    v_raw = np.mean(work_img, axis=1, dtype=np.float32)

    def resample(arr, count):
        if arr.ndim == 1:
            current = arr.shape[0]
        else:
            current = arr.shape[0] if arr.shape[1] == 3 else arr.shape[1]

        if current == count:
            if is_rgb:
                return arr if arr.ndim == 2 else arr.reshape(count, 3)
            else:
                return arr.flatten() if arr.ndim > 1 else arr

        method = cv2.INTER_AREA if current > count else cv2.INTER_LINEAR

        if is_rgb:
            if arr.ndim == 1:
                src = arr.reshape(1, -1,
                                  3) if arr.size % 3 == 0 else arr.reshape(1,
                                                                           -1)
            else:
                src = arr.reshape(1, current, 3)
            res = cv2.resize(src, (count, 1), interpolation=method)
            return res.reshape(count, 3)
        else:
            src = arr.reshape(1, -1)
            res = cv2.resize(src, (count, 1), interpolation=method)
            return res.flatten()

    h_prof = resample(h_raw, target_res)
    v_prof = resample(v_raw, target_res)

    if h_prof.size > 0:
        d_min = float(min(h_prof.min(), v_prof.min()))
        d_max = float(max(h_prof.max(), v_prof.max()))
        rng = (d_min, d_max) if d_min != d_max else (0.0, 255.0)
    else:
        rng = (0.0, 255.0)

    return IntensityProfile(h_prof, v_prof, is_rgb, rng)


def _worker_entry(shm_name, max_shape, dtype, cmd_conn, res_conn):
    """Worker processes ALL requests in order."""
    shm = None
    try:
        shm = shared_memory.SharedMemory(name=shm_name)
        full_buffer = np.ndarray(max_shape, dtype=dtype, buffer=shm.buf)
        target_res = 1024

        while True:
            try:
                if not cmd_conn.poll(timeout=0.1):
                    continue

                msg = cmd_conn.recv()
                if msg is None:
                    break

                op, payload = msg

                if op == 'config':
                    target_res = payload
                elif op == 'process':
                    job_id, h, w, d = payload

                    if d == 1:
                        valid_view = full_buffer[:h, :w, 0]
                    else:
                        valid_view = full_buffer[:h, :w, :d]

                    res = compute_intensity_profile(valid_view, target_res)
                    res_conn.send((job_id, res))

            except EOFError:
                break
            except Exception as e:
                print(f"Worker error: {e}")
                continue

    finally:
        if shm is not None:
            shm.close()
        try:
            cmd_conn.close()
        except:
            pass
        try:
            res_conn.close()
        except:
            pass


class AsyncIntensityProfiler(QObject):
    """
    Queue-based profiler: Processes every request, emits signal when done.
    No dropped frames.
    """
    profileReady = pyqtSignal(object)  # Emits IntensityProfile

    def __init__(self, max_width=4096, max_height=4096, target_resolution=1024):
        super().__init__()
        self.target_resolution = target_resolution
        self._shutdown_done = False
        self._next_job_id = 0

        # Allocate Shared Memory
        self._shm_shape = (max_height, max_width, 3)
        self._shm_dtype = np.uint8
        dummy = np.zeros(self._shm_shape, dtype=self._shm_dtype)
        self._shm_size = dummy.nbytes

        self._shm = shared_memory.SharedMemory(create=True, size=self._shm_size)
        self._shm_array = np.ndarray(self._shm_shape, dtype=self._shm_dtype,
                                     buffer=self._shm.buf)

        # IPC
        self._cmd_send, cmd_recv = multiprocessing.Pipe()
        res_send, self._res_recv = multiprocessing.Pipe()

        # Start Worker
        self._process = multiprocessing.Process(
            target=_worker_entry,
            args=(self._shm.name, self._shm_shape, self._shm_dtype,
                  cmd_recv, res_send)
        )
        self._process.daemon = True
        self._process.start()

        self._cmd_send.send(('config', target_resolution))

        # Result polling timer
        from PyQt6.QtCore import QTimer
        self._poll_timer = QTimer()
        self._poll_timer.timeout.connect(self._check_results)
        self._poll_timer.start(10)  # Check every 10ms

    def _check_results(self):
        """Poll for results and emit signal."""
        while self._res_recv.poll():
            try:
                job_id, result = self._res_recv.recv()
                self.profileReady.emit(result)
            except EOFError:
                break

    def process(self, image: np.ndarray):
        """Queue image for processing. Result comes via profileReady signal."""
        h, w = image.shape[:2]
        if h > self._shm_shape[0] or w > self._shm_shape[1]:
            return

        # Copy to shared memory
        if image.ndim == 2:
            self._shm_array[:h, :w, 0] = image
            depth = 1
        else:
            depth = image.shape[2]
            self._shm_array[:h, :w, :depth] = image

        # Queue job
        job_id = self._next_job_id
        self._next_job_id += 1

        try:
            self._cmd_send.send(('process', (job_id, h, w, depth)))
        except:
            pass

    def shutdown(self):
        """Clean shutdown."""
        if self._shutdown_done:
            return

        self._shutdown_done = True
        self._poll_timer.stop()

        try:
            if self._process.is_alive():
                try:
                    self._cmd_send.send(None)
                except:
                    pass

            self._process.join(timeout=2.0)

            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=1.0)

        finally:
            try:
                self._cmd_send.close()
            except:
                pass
            try:
                self._res_recv.close()
            except:
                pass
            try:
                self._shm.close()
            except:
                pass
            try:
                self._shm.unlink()
            except:
                pass

    def __del__(self):
        self.shutdown()


# --- Corrected Worker Entry Point ---
# You must overwrite the previous _worker_entry with this one
# that handles dynamic sub-regions of the SHM buffer.

# --- THE CUSTOM WIDGET ---
class OverlayTip(QLabel):
    """
    A high-performance, custom tooltip that floats above the UI.
    Updates instantly without the standard OS tooltip delays.
    """

    def __init__(self, parent=None):
        super().__init__(parent,
                         Qt.WindowType.ToolTip | Qt.WindowType.FramelessWindowHint)
        # Style: Semi-transparent black background, white text, rounded corners
        self.setStyleSheet("""
            QLabel {
                color: palette(tooltip);
                border: 1px solid #555;
                border-radius: 3px;
                padding: 4px;
                font-size: 11px;
            }
        """)
        # self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        # Ensure it doesn't steal focus or capture mouse events
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)


class HistogramWidget(QWidget):
    """
    High-performance histogram with interactive floating tooltip.
    """
    jump_to_index = pyqtSignal(int)

    def __init__(self, orientation=Qt.Orientation.Horizontal, parent=None):
        super().__init__(parent)
        self.orientation = orientation

        # --- CONFIGURATION ---
        self.bg_color = QColor("#2b2b2b")
        self.overlay_color = QColor(255, 255, 255, 150)  # Brighter line

        # Optimized Pens
        self.pens = {
            'r': QPen(QColor(255, 50, 50), 1.5),
            'g': QPen(QColor(50, 255, 50), 1.5),
            'b': QPen(QColor(80, 80, 255), 1.5),
            'mono': QPen(QColor(100, 200, 255), 1.5)
        }

        # --- OPTIMIZATIONS ---
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent)
        self.setMouseTracking(True)  # Essential for hover

        # --- STATE ---
        self._data: List[np.ndarray] = []
        self._range: Tuple[float, float] = (0, 255)
        self._mouse_pos: Optional[QPoint] = None
        self._hover_index: int = -1

        # --- THE FLOATING TOOLTIP ---
        # Parented to self so it is destroyed with the widget,
        # but acts as a Window so it can draw outside bounds.
        self._tooltip = OverlayTip(self)

        # --- LAYOUT ---
        if orientation == Qt.Orientation.Horizontal:
            self.setSizePolicy(QSizePolicy.Policy.Expanding,
                               QSizePolicy.Policy.Fixed)
            self.setMinimumHeight(60)
        else:
            self.setSizePolicy(QSizePolicy.Policy.Fixed,
                               QSizePolicy.Policy.Expanding)
            self.setMinimumWidth(60)

    def set_data(self, data: np.ndarray, data_range: Tuple[float, float],
                 is_rgb: bool):
        self._range = data_range
        self._data.clear()

        if data.size == 0:
            self.update()
            return

        if is_rgb and data.ndim > 1 and data.shape[1] == 3:
            self._data = [data[:, 0], data[:, 1], data[:, 2]]
        else:
            self._data = [data]

        self.update()

    # --- INPUT EVENTS ---

    def mouseMoveEvent(self, event: QMouseEvent):
        self._mouse_pos = event.pos()
        self.update()  # Repaint crosshair
        self._update_tooltip()  # Move/Update text

    def leaveEvent(self, event):
        self._mouse_pos = None
        self._hover_index = -1
        self._tooltip.hide()
        self.update()

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.MouseButton.LeftButton and self._hover_index != -1:
            self.jump_to_index.emit(self._hover_index)

    def _update_tooltip(self):
        """Updates position and text of the floating label."""
        if not self._mouse_pos or not self._data:
            self._tooltip.hide()
            return

        # 1. Calculate Index (Logic duplicated from paintEvent for robust decoupling)
        w, h = self.width(), self.height()
        count = len(self._data[0])
        idx = 0

        if self.orientation == Qt.Orientation.Horizontal:
            idx = int(round(self._mouse_pos.x() / w * (count - 1)))
        else:
            idx = int(round(self._mouse_pos.y() / h * (count - 1)))

        idx = max(0, min(idx, count - 1))
        self._hover_index = idx

        # 2. Get Values
        vals = []
        for arr in self._data:
            if idx < len(arr):
                vals.append(arr[idx])

        if not vals:
            self._tooltip.hide()
            return

        # 3. Format Text
        if len(vals) == 3:
            txt = f"Idx: {idx} | R:{vals[0]:.0f} G:{vals[1]:.0f} B:{vals[2]:.0f}"
        else:
            txt = f"Idx: {idx} | Val:{vals[0]:.0f}"

        self._tooltip.setText(txt)
        self._tooltip.adjustSize()

        # 4. Smart Positioning (Global Coordinates)
        # Map local mouse pos to Global Screen pos
        global_pos = self.mapToGlobal(self._mouse_pos)

        # Offset so cursor doesn't cover text
        offset_x = 15
        offset_y = 15

        # Simple boundary check could go here if needed,
        # but Qt ToolTips handle screen edges mostly auto-magically.
        self._tooltip.move(global_pos.x() + offset_x, global_pos.y() + offset_y)

        if self._tooltip.isHidden():
            self._tooltip.show()

    # --- RENDER LOGIC ---

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.bg_color)

        if not self._data:
            return

        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        count = len(self._data[0])
        if count < 2: return

        val_min, val_max = self._range
        val_span = val_max - val_min if val_max > val_min else 1.0

        if self.orientation == Qt.Orientation.Horizontal:
            x_scale = w / (count - 1)
            y_scale = h / val_span
        else:
            x_scale = w / val_span
            y_scale = h / (count - 1)

        # Draw Polylines
        for i, channel_data in enumerate(self._data):
            if len(self._data) == 3:
                key = ['r', 'g', 'b'][i]
            else:
                key = 'mono'
            painter.setPen(self.pens[key])

            points = QPolygonF()

            if self.orientation == Qt.Orientation.Horizontal:
                for idx, val in enumerate(channel_data):
                    px = idx * x_scale
                    py = h - ((val - val_min) * y_scale)
                    points.append(QPointF(px, py))
            else:
                for idx, val in enumerate(channel_data):
                    px = (val - val_min) * x_scale
                    py = idx * y_scale
                    points.append(QPointF(px, py))

            painter.drawPolyline(points)

        # Draw Hover Line (Crosshair) only - Text handled by Overlay
        if self._mouse_pos:
            painter.setPen(QPen(self.overlay_color, 1, Qt.PenStyle.DashLine))

            # Snap logic
            if self.orientation == Qt.Orientation.Horizontal:
                # We reuse the cached index logic implicitly via coordinate math
                # Or re-calculate for perfect visual snap
                idx = int(round(self._mouse_pos.x() / w * (count - 1)))
                idx = max(0, min(idx, count - 1))
                line_x = idx * x_scale
                painter.drawLine(QPointF(line_x, 0), QPointF(line_x, h))
            else:
                idx = int(round(self._mouse_pos.y() / h * (count - 1)))
                idx = max(0, min(idx, count - 1))
                line_y = idx * y_scale
                painter.drawLine(QPointF(0, line_y), QPointF(w, line_y))


class HistogramController(QObject):
    """
    Acts as the Presenter/Controller.
    Owns the widget instances and the logic to populate them.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        # 1. Views
        self._h_widget = HistogramWidget(Qt.Orientation.Horizontal)
        self._v_widget = HistogramWidget(Qt.Orientation.Vertical)

        # 2. State for Caching
        self._last_profile: IntensityProfile | None = None
        self._is_visible = True
        self._async_profile = AsyncIntensityProfiler(max_width=6000,
                                                     max_height=6000)
        self._async_profile.setParent(self)

        self._async_profile.profileReady.connect(self._update_histogram)

    # --- Public API for Layouts ---
    def get_horizontal_widget(self) -> QWidget:
        return self._h_widget

    def get_vertical_widget(self) -> QWidget:
        return self._v_widget

    @pyqtSlot(object)
    def _update_histogram(self, new_profile):

        # 2. Caching Guard
        # If the profile hasn't changed, DO NOT touch the views.
        if self._last_profile == new_profile:
            return

        self._last_profile = new_profile

        # 3. Push to Views (Passive View Pattern)
        # We pass the raw numpy array directly.
        # The Widget's set_data method handles the splitting of channels.

        # Horizontal
        self._h_widget.set_data(
            new_profile.horizontal,
            new_profile.data_range,
            new_profile.is_rgb
        )

        # Vertical
        self._v_widget.set_data(
            new_profile.vertical,
            new_profile.data_range,
            new_profile.is_rgb
        )

    # --- Public API for Data ---
    def process_image(self, image: np.ndarray):
        """
        The main entry point.
        Calculates, checks cache, and pushes to views.
        """
        # 1. Logic (Model)
        # compute_intensity_profile returns 'horizontal' and 'vertical'
        # as np.ndarray with shape (N,) or (N, 3).
        self._async_profile.process(image)

    def set_visible(self, visible: bool):
        """Optimization: Stop computing if tab/window is hidden."""
        self._is_visible = visible
        self._h_widget.setVisible(visible)
        self._v_widget.setVisible(visible)

        if not visible:
            # Clear cache so when we become visible again, we force an update
            self._last_profile = None

    def shutdown(self):
        """Explicit lifecycle management."""
        self.set_visible(False)
        self._last_profile = None
        self._async_profile.shutdown()

        # Pass empty numpy arrays to clear the widget
        empty = np.array([])
        self._h_widget.set_data(empty, (0, 1), False)
        self._v_widget.set_data(empty, (0, 1), False)

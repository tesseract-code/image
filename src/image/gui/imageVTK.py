"""
Embeddable 2D Image Viewer Widget for PyQt6 using VTK.

Classes
-------
ImageViewerWidget
    A QWidget that displays a 2D image with zoom, pan, and window/level
    controls.  Uses the SafeVTK widget to avoid macOS OpenGL paint crashes.

Examples
--------
>>> from image.gui.imageVTK import ImageViewerWidget
>>> viewer = ImageViewerWidget()
>>> viewer.load_image("path/to/image.png")
>>> viewer.show()
"""

__all__ = ["ImageViewerWidget"]

import logging
import os
from typing import Optional

import vtkmodules.all as vtk
from PyQt6.QtCore import QSize, Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkmodules.vtkInteractionStyle import vtkInteractorStyleImage
from vtkmodules.vtkRenderingCore import (
    vtkImageActor,
    vtkRenderer,
)

from pycore.platform import IS_MACOS
from qtcore.vtk_utils import SafeVTKWidget

try:
    import numpy as np
    from vtkmodules.util.numpy_support import numpy_to_vtk

    _NUMPY_AVAILABLE = True
except ImportError:
    _NUMPY_AVAILABLE = False

log = logging.getLogger(__name__)


class ImageViewerWidget(QWidget):
    """Embeddable 2D image viewer with window/level and zoom/pan.

    Parameters
    ----------
    parent : QWidget, optional
        Parent widget.
    show_controls : bool, default True
        If False, the toolbar and window/level sliders are hidden.

    Attributes
    ----------
    image_loaded : pyqtSignal(str)
        Emitted with the absolute file path when an image is successfully
        loaded from a file via ``load_image``.
    image_changed : pyqtSignal()
        Emitted whenever the displayed image changes, regardless of source
        (``load_image``, ``set_image_data``, or ``set_data``).
    """

    image_loaded = pyqtSignal(str)
    image_changed = pyqtSignal()

    WINDOW_SLIDER_MIN = 1
    WINDOW_SLIDER_MAX = 4000
    LEVEL_SLIDER_MIN = -1000
    LEVEL_SLIDER_MAX = 4000

    def __init__(self, parent=None, show_controls=True) -> None:
        super().__init__(parent)

        self._reader = None
        self._image_actor: Optional[vtkImageActor] = None
        self._window_level_filter: Optional[
            vtk.vtkImageMapToWindowLevelColors] = None
        self._current_file: Optional[str] = None

        self._window: float = 255.0
        self._level: float = 127.5
        self._auto_window_level: bool = True

        self._file_filter = (
            "Image Files (*.png *.jpg *.jpeg *.tif *.tiff *.bmp *.dcm *.vti *.mha *.mhd);;"
            "All Files (*)"
        )
        self._file_dir = os.getcwd()

        self._cleaned_up = False
        self._setup_ui()

        # Apply initial control visibility after the UI is fully constructed
        # so _show_controls always reflects the live widget state.
        if not show_controls:
            self._set_controls_visible(False)

    def sizeHint(self) -> QSize:
        return QSize(800, 600)

    def _setup_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(2)

        toolbar_layout = QHBoxLayout()
        toolbar_layout.setContentsMargins(4, 4, 4, 4)

        self._btn_open = QPushButton("Open")
        self._btn_open.setToolTip("Open image file")
        self._btn_open.clicked.connect(self.open_file)
        toolbar_layout.addWidget(self._btn_open)

        self._btn_reset = QPushButton("Reset")
        self._btn_reset.setToolTip("Reset view and window/level")
        self._btn_reset.clicked.connect(self.reset_view)
        toolbar_layout.addWidget(self._btn_reset)

        toolbar_layout.addStretch()

        self._status_label = QLabel("No image")
        toolbar_layout.addWidget(self._status_label)

        self._toolbar_container = QWidget()
        self._toolbar_container.setLayout(toolbar_layout)
        main_layout.addWidget(self._toolbar_container)

        wl_widget = QWidget()
        wl_layout = QHBoxLayout(wl_widget)

        wl_layout.addWidget(QLabel("Window:"))
        self._window_slider = QSlider(Qt.Orientation.Horizontal)
        self._window_slider.setRange(self.WINDOW_SLIDER_MIN,
                                     self.WINDOW_SLIDER_MAX)
        self._window_slider.setValue(int(self._window))
        self._window_slider.valueChanged.connect(self._on_window_changed)
        wl_layout.addWidget(self._window_slider)

        wl_layout.addWidget(QLabel("Level:"))
        self._level_slider = QSlider(Qt.Orientation.Horizontal)
        self._level_slider.setRange(self.LEVEL_SLIDER_MIN,
                                    self.LEVEL_SLIDER_MAX)
        self._level_slider.setValue(int(self._level))
        self._level_slider.valueChanged.connect(self._on_level_changed)
        wl_layout.addWidget(self._level_slider)

        self._auto_wl_btn = QPushButton("Auto W/L")
        self._auto_wl_btn.setCheckable(True)
        self._auto_wl_btn.setChecked(self._auto_window_level)
        self._auto_wl_btn.toggled.connect(self._on_auto_wl_toggled)
        wl_layout.addWidget(self._auto_wl_btn)

        self._wl_container = wl_widget
        main_layout.addWidget(self._wl_container)

        self._vtk_widget = SafeVTKWidget(self)
        main_layout.addWidget(self._vtk_widget, 1)

        self._renderer = vtkRenderer()
        self._renderer.SetBackground(0.1, 0.1, 0.1)
        self._vtk_widget.GetRenderWindow().AddRenderer(self._renderer)

        style = vtkInteractorStyleImage()
        style.SetInteractionModeToImage2D()
        self._interactor = self._vtk_widget.GetRenderWindow().GetInteractor()
        self._interactor.SetInteractorStyle(style)
        self._interactor.Initialize()
        self._vtk_widget.GetRenderWindow().Render()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def open_file(self) -> None:
        """Open an image file via a file dialog."""
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Image",
            self._file_dir,
            self._file_filter,
        )
        if path:
            self.load_image(path)

    def set_file_filter(self, filter_str: str) -> None:
        """Set the file filter used by the Open dialog.

        Parameters
        ----------
        filter_str : str
            File filter string, e.g., ``"Images (*.png *.jpg)"``.
        """
        self._file_filter = filter_str

    def set_file_directory(self, directory: str) -> None:
        """Set the starting directory for the Open dialog.

        Parameters
        ----------
        directory : str
            Absolute path to an existing directory.  Non-existent paths are
            rejected with a warning and the previous value is kept.
        """
        if os.path.isdir(directory):
            self._file_dir = directory
        else:
            log.warning("set_file_directory: not a directory: %s", directory)

    def set_controls_visible(self, visible: bool) -> None:
        """Show or hide the toolbar and window/level controls.

        Parameters
        ----------
        visible : bool
            If True, controls are shown; otherwise hidden.
        """
        self._set_controls_visible(visible)

    def load_image(self, path: str) -> bool:
        """Load and display a 2D image from a file.

        Parameters
        ----------
        path : str
            Path to the image file.

        Returns
        -------
        bool
            True on success, False on failure.
        """
        path = os.path.abspath(path)
        if not os.path.isfile(path):
            log.error("File not found: %s", path)
            self._set_status("File not found.")
            return False

        self._set_status(f"Loading {os.path.basename(path)}...")

        try:
            reader = vtk.vtkImageReader2Factory.CreateImageReader2(path)
            if reader is None:
                raise RuntimeError("Unsupported image format")
            reader.SetFileName(path)
            reader.Update()
        except Exception as e:
            log.exception("Failed to read image: %s", e)
            self._set_status(f"Error: {e}")
            self._clear_current_image()
            return False

        self._reader = None
        self._reader = reader

        try:
            self._install_image_data(reader.GetOutput(), source=reader)
        except Exception as e:
            log.exception("Failed to install image data: %s", e)
            self._set_status(f"Error: {e}")
            self._clear_current_image()
            return False

        self._current_file = path
        self._file_dir = os.path.dirname(path)
        self._set_status(f"Loaded: {os.path.basename(path)}")
        self.image_loaded.emit(path)
        return True

    def set_image_data(self, image_data: vtk.vtkImageData) -> None:
        """Display a VTK image dataset directly.

        This bypasses file reading and is useful for in-memory images.
        The previous image (if any) is replaced.

        Parameters
        ----------
        image_data : vtkImageData
            The image data to display.
        """
        if image_data is None:
            log.warning("set_image_data called with None; ignoring.")
            return

        self._reader = None
        self._install_image_data(image_data, source=None)
        self._current_file = None

    def set_data(self, X: "np.ndarray") -> bool:
        """Display a 2-D NumPy array as an image.

        Supports grayscale (2-D) and colour (3-D with 3 or 4 channels).
        The array data is copied into the VTK pipeline.

        Parameters
        ----------
        X : np.ndarray
            A 2-D (height, width) or 3-D (height, width, channels) array.
            Common numeric types (uint8, float32, …) are accepted.

        Returns
        -------
        bool
            True if the array was displayed successfully, False on failure.

        Raises
        ------
        ImportError
            If NumPy is not installed.
        TypeError
            If *X* is not a ``numpy.ndarray``.
        ValueError
            If *X* has an unsupported shape.
        """
        if not _NUMPY_AVAILABLE:
            raise ImportError(
                "NumPy is required to use set_data. "
                "Install it with `pip install numpy`."
            )

        if not isinstance(X, np.ndarray):
            raise TypeError("Expected a NumPy ndarray")

        if X.ndim not in (2, 3):
            raise ValueError("Array must be 2-D or 3-D (with channels)")

        if X.ndim == 2:
            h, w = X.shape
            num_components = 1
        else:
            h, w, c = X.shape
            if c not in (1, 3, 4):
                raise ValueError(
                    f"3-D array last dimension must be 1, 3 or 4, got {c}"
                )
            num_components = c

        flat_array = np.ascontiguousarray(X.reshape(-1, num_components))
        vtk_array = numpy_to_vtk(flat_array, deep=True)

        image_data = vtk.vtkImageData()
        image_data.SetDimensions(w, h, 1)
        image_data.GetPointData().SetScalars(vtk_array)

        # set_image_data emits image_changed and writes a status message;
        # no separate _set_status call is needed here.
        self.set_image_data(image_data)
        self._set_status(f"NumPy array {X.shape} displayed")
        return True

    def reset_view(self) -> None:
        """Reset camera and restore auto window/level."""
        if self._image_actor is None:
            return
        self._renderer.ResetCamera()
        self._renderer.ResetCameraClippingRange()
        self._auto_window_level = True
        self._auto_wl_btn.setChecked(True)
        self._compute_auto_window_level()
        self._apply_window_level()
        self._vtk_widget.GetRenderWindow().Render()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _install_image_data(
            self,
            image_data: vtk.vtkImageData,
            source: Optional[vtk.vtkAlgorithm] = None,
    ) -> None:
        """Wire *image_data* into the VTK pipeline and refresh the view.

        This is the single authoritative path for replacing the displayed
        image.  Both ``load_image`` and ``set_image_data`` delegate here to
        avoid duplication.

        Parameters
        ----------
        image_data : vtkImageData
            The image to display.
        source : vtkAlgorithm, optional
            When provided the filter is connected via ``SetInputConnection``
            (preserving the upstream pipeline); otherwise
            ``SetInputDataObject`` is used.
        """
        if self._image_actor is not None:
            self._renderer.RemoveActor(self._image_actor)

        self._window_level_filter = None
        self._window_level_filter = vtk.vtkImageMapToWindowLevelColors()

        if source is not None:
            self._window_level_filter.SetInputConnection(
                source.GetOutputPort()
            )
        else:
            self._window_level_filter.SetInputDataObject(image_data)

        self._window_level_filter.Update()

        self._image_actor = vtkImageActor()
        self._image_actor.GetMapper().SetInputConnection(
            self._window_level_filter.GetOutputPort()
        )
        self._renderer.AddActor(self._image_actor)

        self._configure_sliders_for(image_data)

        self._renderer.ResetCamera()
        self._renderer.ResetCameraClippingRange()
        self._auto_window_level = True
        self._auto_wl_btn.setChecked(True)
        self._compute_auto_window_level()
        self._apply_window_level()
        self._vtk_widget.GetRenderWindow().Render()

        self.image_changed.emit()

    def _configure_sliders_for(self, image_data: vtk.vtkImageData) -> None:
        """Adjust slider ranges to match *image_data*'s scalar range."""
        scalar_range = image_data.GetScalarRange()
        # Use the actual data width as the window slider maximum, falling back
        # to WINDOW_SLIDER_MAX when the range is narrower than the default.
        data_width = max(
            self.WINDOW_SLIDER_MIN,
            int(scalar_range[1] - scalar_range[0]),
        )
        self._window_slider.setRange(self.WINDOW_SLIDER_MIN, data_width)
        self._level_slider.setRange(
            int(scalar_range[0]) - 1000,
            int(scalar_range[1]) + 1000,
        )

    def _compute_auto_window_level(self) -> None:
        if self._window_level_filter is None:
            return
        input_data = self._window_level_filter.GetInputDataObject(0, 0)
        if input_data is None:
            return
        scalar_range = input_data.GetScalarRange()
        self._level = (scalar_range[0] + scalar_range[1]) / 2.0
        raw_window = scalar_range[1] - scalar_range[0]
        if raw_window == 0.0:
            log.debug(
                "_compute_auto_window_level: scalar range is zero "
                "(constant image); clamping window to 1.0"
            )
        self._window = max(1.0, raw_window)

    def _apply_window_level(self) -> None:
        if self._window_level_filter is None:
            return
        self._window_level_filter.SetWindow(self._window)
        self._window_level_filter.SetLevel(self._level)

        self._window_slider.blockSignals(True)
        self._window_slider.setValue(int(self._window))
        self._window_slider.blockSignals(False)

        self._level_slider.blockSignals(True)
        self._level_slider.setValue(int(self._level))
        self._level_slider.blockSignals(False)

        self._vtk_widget.GetRenderWindow().Render()

    def _on_window_changed(self, value: int) -> None:
        self._window = float(value)
        self._auto_window_level = False
        self._auto_wl_btn.setChecked(False)
        self._apply_window_level()

    def _on_level_changed(self, value: int) -> None:
        self._level = float(value)
        self._auto_window_level = False
        self._auto_wl_btn.setChecked(False)
        self._apply_window_level()

    def _on_auto_wl_toggled(self, checked: bool) -> None:
        self._auto_window_level = checked
        if checked:
            self._compute_auto_window_level()
            self._apply_window_level()

    def _set_status(self, message: str) -> None:
        self._status_label.setText(message)

    def _set_controls_visible(self, visible: bool) -> None:
        self._toolbar_container.setVisible(visible)
        self._wl_container.setVisible(visible)

    def _clear_current_image(self) -> None:
        if self._image_actor is not None:
            self._renderer.RemoveActor(self._image_actor)
            self._image_actor = None
        self._reader = None
        self._window_level_filter = None
        self._current_file = None
        self._vtk_widget.GetRenderWindow().Render()
        self._set_status("No image")

    def cleanup(self) -> None:
        """Release VTK resources safely (called by closeEvent)."""
        if self._cleaned_up:
            return
        self._cleaned_up = True

        try:
            if self._renderer is not None:
                self._renderer.RemoveAllViewProps()
                self._renderer.Clear()

            win = self._vtk_widget.GetRenderWindow()
            if win is not None:
                if self._interactor is not None:
                    self._interactor.TerminateApp()
                win.Finalize()
            self._vtk_widget.close()
        except Exception:
            log.exception("Error during cleanup")

    def closeEvent(self, event) -> None:
        self.cleanup()
        super().closeEvent(event)

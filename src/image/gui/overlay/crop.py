import logging
from typing import Optional

import numpy as np
from PyQt6.QtCore import Qt, pyqtSlot
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QLabel, QFrame,
                             QComboBox, QScrollArea, QSizePolicy, QToolBar)

from image.gui.imageQt import GraphicsImageView
from qtgui.drop_down import Dropdown
from qtgui.joystick import JoystickWidget
from qtgui.pixmap import colorize_pixmap

logger = logging.getLogger(__name__)


class CropView(QFrame):
    """Widget that displays a high-fidelity magnified view of active ROI area.

    Supports all ROI types (Rect, Ellipse, Line, Polygon) and provides
    real-time preview with controllable magnification and joystick control.
    """

    def __init__(self, layered_view, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.layered_view = layered_view

        # State
        self._current_crop: Optional[np.ndarray] = None
        self._current_bounds: Optional[dict] = None
        self._auto_update = True

        self._setup_ui()
        self._connect_signals()

    def _setup_ui(self):
        """Setup the user interface."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Info label
        self.info_label = QLabel("No active ROI")
        self.info_label.setWordWrap(False)

        toolbar = QToolBar()

        toolbar.addWidget(self.info_label)

        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Expanding,
                                         QSizePolicy.Policy.Preferred))
        toolbar.addWidget(spacer)

        # Magnification control
        toolbar.addWidget(QLabel("Mag:"))
        self.mag_combo = QComboBox()
        self.mag_combo.addItems(["1×", "2×", "4×", "8×"])
        self.mag_combo.setCurrentIndex(0)
        self.mag_combo.setToolTip("Magnification factor (powers of 2)")
        self.mag_combo.currentIndexChanged.connect(
            self._on_magnification_changed)
        toolbar.addWidget(self.mag_combo)

        layout.addWidget(toolbar)

        graphics_scroll = QScrollArea()
        # High-fidelity image viewer
        self.image_viewer = GraphicsImageView()
        graphics_scroll.setWidget(self.image_viewer)
        graphics_scroll.setWidgetResizable(True)
        graphics_scroll.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(graphics_scroll, 1)

        # Joystick control
        joystick_dropdown = Dropdown(title="Positioner", title_icon=QIcon(
            colorize_pixmap(QPixmap("line-icons:gamepad-line.svg"),
                            self.palette().buttonText().color())),
                                     scroll_area=True)
        self.joystick = JoystickWidget()
        joystick_dropdown.add_content_widget(self.joystick)
        layout.addWidget(joystick_dropdown)

    def _connect_signals(self):
        """Connect to layered view signals."""
        self.layered_view.roiCropChanged.connect(self._on_crop_changed)

    @pyqtSlot(bool)
    def _on_auto_update_toggled(self, checked: bool):
        """Handle auto-update toggle."""
        self._auto_update = checked
        self.refresh_btn.setEnabled(not checked)

    @pyqtSlot(int)
    def _on_magnification_changed(self, _index: int):
        """Handle magnification change."""
        self._update_display()

    @pyqtSlot()
    def _manual_refresh(self):
        """Manually refresh the crop display."""
        if self._current_crop is not None:
            self._update_display()

    @pyqtSlot(np.ndarray, object)
    def _on_crop_changed(self, crop_array: np.ndarray, roi_bounds: dict):
        """Handle new crop data from ROI changes.

        Args:
            crop_array: Cropped image data
            roi_bounds: Dictionary with keys: x, y, width, height, type
        """
        if not self._auto_update:
            return

        self._current_crop = crop_array
        self._current_bounds = roi_bounds
        self._update_display()

    def _update_display(self):
        """Update the displayed crop with current magnification."""
        if self._current_crop is None:
            self.image_viewer.clear_image()
            self.info_label.setText("No active ROI")
            return

        try:
            bounds = self._current_bounds

            # Get magnification from combo box
            mag_index = self.mag_combo.currentIndex()
            mag_values = [1, 2, 4, 8]
            mag = mag_values[mag_index]

            # Update info with ROI type
            self.info_label.setText(
                f"<b>Size:</b> {bounds['width']} × {bounds['height']} px<br>"
                f"<b>Position:</b> ({bounds['x']:.0f}, {bounds['y']:.0f})"
            )

            # Display in high-fidelity viewer
            success = self.image_viewer.set_image_from_array(
                self._current_crop,
                magnification=mag
            )

            if not success:
                self.info_label.setText(
                    f"{self.info_label.text()}<br><font color='red'>Display error</font>"
                )

        except Exception as e:
            logger.error(f"Error updating crop display: {e}")
            self.info_label.setText(f"Display error: {e}")

    def get_current_crop(self) -> Optional[np.ndarray]:
        """Get the current crop array.

        Returns:
            Current crop as numpy array or None
        """
        return self._current_crop

    def get_current_bounds(self) -> Optional[dict]:
        """Get the current ROI bounds.

        Returns:
            Dictionary with x, y, width, height, type or None
        """
        return self._current_bounds

    def set_magnification(self, value: int):
        """Set magnification programmatically.

        Args:
            value: Magnification factor (will snap to nearest power of 2: 1, 2, 4, 8)
        """
        powers_of_2 = [1, 2, 4, 8]
        closest = min(powers_of_2, key=lambda x: abs(x - value))
        index = powers_of_2.index(closest)
        self.mag_combo.setCurrentIndex(index)

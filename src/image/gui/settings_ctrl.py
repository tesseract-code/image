import time
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSlot, pyqtSignal
from PyQt6.QtWidgets import (QScrollArea, QWidget, QVBoxLayout, QFrame, QLabel,
                             QSlider, QDoubleSpinBox, QPushButton, QComboBox)

from cross_platform.qt6_utils.qtgui.src.qtgui.switch import Switch
from cross_platform.svg_icons.svg_path import get_icon, IconType
from image.gui.navigation import NavigablePanel, create_group_frame, \
    create_parameter_row
from image.model.cmap import LUTType
from image.pipeline.stats import FrameStats
from image.settings.roi import ROI


class ControlPanel(NavigablePanel):
    """
    Control panel for OpenGL image viewer settings.

    Organizes settings into logical pages with immediate visual feedback.
    Settings changes are propagated through the settings object and trigger
    shader recompilation as needed.
    """
    roi_changed = pyqtSignal(ROI)
    roi_visibility_changed = pyqtSignal(bool)
    colorbar_visibility_changed = pyqtSignal(bool)

    def __init__(self, settings, image_viewer, parent=None):
        self.settings = settings
        self.image_viewer = image_viewer
        self.updating_ui = False  # Prevent circular updates
        self.info_labels = {}
        self._current_metadata: Optional[FrameStats] = None

        super().__init__(parent=parent)
        self._connect_signals()

    def get_title_text(self) -> str:
        return "Image Viewer Controls"

    def get_title_icon(self):
        return get_icon(
            IconType.LINE_IMAGE,
            (256, 256),
            self.palette().text().color()
        )

    def add_pages(self):
        """Add all control pages in logical order."""
        self.add_page(IconType.INFO, "Info", self._create_info_page())
        self.add_page(IconType.LINE_EQUALIZER, "Adjustments",
                      self._create_adjustments_page())
        self.add_page(IconType.LINE_COLOR_FILTER, "Colormap",
                      self._create_colormap_page())
        self.add_page(IconType.LINE_ZOOM_IN, "Transform",
                      self._create_transform_page())
        self.add_page(IconType.LINE_IMAGE_EDIT, "Overlays",
                      self._create_overlays_page())
        self.add_page(IconType.LINE_IMAGE, "Display",
                      self._create_display_page())

    def _create_scrollable_page(self) -> tuple[QScrollArea, QVBoxLayout]:
        """Create a scrollable page container for content."""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        content = QWidget()
        layout = QVBoxLayout(content)

        scroll.setWidget(content)
        return scroll, layout

    # ========================================================================
    # Page 1: Info
    # ========================================================================

    def _create_info_page(self) -> QWidget:
        """
        Information page: Frame metadata and performance metrics.

        Displays current frame properties (dimensions, statistics) and
        real-time performance data (FPS, frame time).
        """
        scroll, layout = self._create_scrollable_page()

        # Frame Properties
        props_frame = create_group_frame(
            "Frame Properties",
            "Current frame metadata and statistics"
        )

        property_specs = [
            ('time', "Timestamp"),
            ('shape', 'Dimensions'),
            ('min_val', 'Min Value'),
            ('max_val', 'Max Value'),
            ('mean', 'Mean'),
            ('std', 'Std Dev'),
        ]

        for i, (key, label) in enumerate(property_specs):
            value_label = QLabel("—")
            value_label.setMinimumWidth(45)
            value_label.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
            )
            is_last = (i == len(property_specs) - 1)
            row = create_parameter_row(label, value_label,
                                       show_border=not is_last)
            self.info_labels[key] = value_label
            props_frame.layout().addWidget(row)

        layout.addWidget(props_frame)

        # Performance Metrics
        perf_frame = create_group_frame(
            "Performance",
            "Real-time rendering performance"
        )

        self.fps_label = QLabel("—")
        perf_frame.layout().addWidget(
            create_parameter_row("FPS", self.fps_label)
        )

        self.frame_time_label = QLabel("—")
        perf_frame.layout().addWidget(
            create_parameter_row("Frame Time", self.frame_time_label,
                                 show_border=False)
        )

        layout.addWidget(perf_frame)
        layout.addStretch()

        return scroll

    # ========================================================================
    # Page 2: Adjustments
    # ========================================================================

    def _create_adjustments_page(self) -> QWidget:
        """
        Adjustments page: Image tone and level controls.

        Provides real-time adjustment of brightness, contrast, gamma,
        gain, offset, and color inversion.
        """
        scroll, layout = self._create_scrollable_page()

        # Tone Controls
        tone_frame = create_group_frame(
            "Tone",
            "Adjust image brightness, contrast, and gamma"
        )
        tone_layout = tone_frame.layout()

        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        tone_layout.addWidget(
            create_parameter_row("Brightness", self.brightness_slider)
        )

        self.contrast_slider = QSlider(Qt.Orientation.Horizontal)
        self.contrast_slider.setRange(10, 300)
        self.contrast_slider.setValue(100)
        tone_layout.addWidget(
            create_parameter_row("Contrast", self.contrast_slider)
        )

        self.gamma_spin = QDoubleSpinBox()
        self.gamma_spin.setRange(0.1, 3.0)
        self.gamma_spin.setSingleStep(0.1)
        self.gamma_spin.setValue(1.0)
        self.gamma_spin.setKeyboardTracking(False)
        tone_layout.addWidget(
            create_parameter_row("Gamma", self.gamma_spin, show_border=False)
        )

        layout.addWidget(tone_frame)

        # Level Controls
        levels_frame = create_group_frame(
            "Levels",
            "Fine-tune intensity scaling and inversion"
        )
        levels_layout = levels_frame.layout()

        self.gain_spin = QDoubleSpinBox()
        self.gain_spin.setRange(0.1, 10.0)
        self.gain_spin.setSingleStep(0.1)
        self.gain_spin.setValue(1.0)
        self.gain_spin.setKeyboardTracking(False)
        levels_layout.addWidget(create_parameter_row("Gain", self.gain_spin))

        self.offset_spin = QDoubleSpinBox()
        self.offset_spin.setRange(-1.0, 1.0)
        self.offset_spin.setSingleStep(0.01)
        self.offset_spin.setValue(0.0)
        self.offset_spin.setKeyboardTracking(False)
        levels_layout.addWidget(
            create_parameter_row("Offset", self.offset_spin)
        )

        self.invert_switch = Switch(parent=levels_frame)
        levels_layout.addWidget(
            create_parameter_row("Invert Colors", self.invert_switch,
                                 show_border=False)
        )

        layout.addWidget(levels_frame)
        layout.addStretch()

        return scroll

    # ========================================================================
    # Page 3: Colormap
    # ========================================================================

    def _create_colormap_page(self) -> QWidget:
        """
        Colormap page: Color mapping and LUT configuration.

        Controls colormap application, selection, reversal, and lookup
        table transformations for specialized visualization.
        """
        scroll, layout = self._create_scrollable_page()

        # Colormap Settings
        cmap_frame = create_group_frame(
            "Colormap",
            "Apply color mapping to grayscale images"
        )
        cmap_layout = cmap_frame.layout()

        self.colormap_enabled_switch = Switch()
        cmap_layout.addWidget(create_parameter_row("Enable Colormap",
                                                   self.colormap_enabled_switch))

        # Get matplotlib colormaps (exclude reversed versions)
        import matplotlib.pyplot as plt
        cmap_names = sorted([c for c in plt.colormaps() if "_r" not in c])

        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(cmap_names)
        self.colormap_combo.setCurrentText("viridis")
        cmap_layout.addWidget(
            create_parameter_row("Colormap", self.colormap_combo)
        )

        self.colormap_reverse_switch = Switch()
        cmap_layout.addWidget(create_parameter_row("Reverse Colormap",
                                                   self.colormap_reverse_switch))

        self.colorbar_visible_switch = Switch()
        self.colorbar_visible_switch.setEnabled(False)
        cmap_layout.addWidget(create_parameter_row("Show Colorbar",
                                                   self.colorbar_visible_switch,
                                                   show_border=False))

        layout.addWidget(cmap_frame)

        # LUT Settings
        lut_frame = create_group_frame(
            "Lookup Table",
            "Apply non-linear intensity transformations"
        )
        lut_layout = lut_frame.layout()

        self.lut_enabled_switch = Switch()
        lut_layout.addWidget(create_parameter_row("Enable LUT",
                                                  self.lut_enabled_switch))

        self.lut_type_combo = QComboBox()
        self.lut_type_combo.addItems(
            ["Linear", "Logarithmic", "Square Root", "Square"]
        )
        lut_layout.addWidget(
            create_parameter_row("Type", self.lut_type_combo)
        )

        self.lut_min_spin = QDoubleSpinBox()
        self.lut_min_spin.setRange(0.0, 1.0)
        self.lut_min_spin.setSingleStep(0.01)
        self.lut_min_spin.setValue(0.0)
        self.lut_min_spin.setKeyboardTracking(False)
        lut_layout.addWidget(
            create_parameter_row("Min", self.lut_min_spin)
        )

        self.lut_max_spin = QDoubleSpinBox()
        self.lut_max_spin.setRange(0.0, 1.0)
        self.lut_max_spin.setSingleStep(0.01)
        self.lut_max_spin.setValue(1.0)
        self.lut_max_spin.setKeyboardTracking(False)
        lut_layout.addWidget(
            create_parameter_row("Max", self.lut_max_spin, show_border=False)
        )

        layout.addWidget(lut_frame)
        layout.addStretch()

        return scroll

    # ========================================================================
    # Page 4: Transform
    # ========================================================================

    def _create_transform_page(self) -> QWidget:
        """
        Transform page: View transformation controls.

        Adjust zoom level and reset view transformations.
        """
        scroll, layout = self._create_scrollable_page()

        frame = create_group_frame(
            "View Transform",
            "Adjust zoom and reset view"
        )
        frame_layout = frame.layout()

        self.zoom_spin = QDoubleSpinBox()
        self.zoom_spin.setRange(0.01, 100.0)
        self.zoom_spin.setSingleStep(0.1)
        self.zoom_spin.setDecimals(2)
        self.zoom_spin.setValue(1.0)
        self.zoom_spin.setKeyboardTracking(False)
        self.zoom_spin.setSuffix("x")
        frame_layout.addWidget(
            create_parameter_row("Zoom", self.zoom_spin, show_border=True)
        )

        reset_btn = QPushButton("Reset View")
        reset_btn.clicked.connect(self.reset_view)
        frame_layout.addWidget(reset_btn)

        layout.addWidget(frame)
        layout.addStretch()

        return scroll

    # ========================================================================
    # Page 5: Overlays
    # ========================================================================

    def _create_overlays_page(self) -> QWidget:
        """
        Overlays page: Crosshairs, ROI, and overlay controls.

        Enable/disable tracking and center crosshairs, configure ROI
        cropping with power-of-2 sizes.
        """
        scroll, layout = self._create_scrollable_page()

        # Crosshair Controls
        crosshair_frame = create_group_frame(
            "Crosshairs",
            "Toggle tracking and center crosshairs"
        )
        crosshair_layout = crosshair_frame.layout()

        self.tracking_crosshair_switch = Switch(parent=crosshair_frame)
        crosshair_layout.addWidget(
            create_parameter_row("Tracking Crosshair",
                                 self.tracking_crosshair_switch)
        )

        self.center_crosshair_switch = Switch(parent=crosshair_frame)
        crosshair_layout.addWidget(
            create_parameter_row("Center Crosshair",
                                 self.center_crosshair_switch)
        )

        layout.addWidget(crosshair_frame)

        # ROI Controls
        roi_frame = create_group_frame(
            "Region of Interest",
            "Enable ROI crop with configurable size"
        )
        roi_layout = roi_frame.layout()

        self.roi_enabled_switch = Switch(parent=roi_frame)
        roi_layout.addWidget(
            create_parameter_row("Enable ROI Crop", self.roi_enabled_switch)
        )

        # Crop size selector (powers of 2)
        self.crop_size_combo = QComboBox()
        crop_sizes = [2 ** i for i in range(1, 9)]  # 2, 4, 8, 16, ..., 256
        self.crop_size_combo.addItems([f"{size}x{size}" for size in crop_sizes])
        self.crop_size_combo.setCurrentText("128x128")
        roi_layout.addWidget(
            create_parameter_row("Crop Size", self.crop_size_combo,
                                 show_border=False)
        )

        layout.addWidget(roi_frame)
        layout.addStretch()

        return scroll

    # ========================================================================
    # Page 6: Display
    # ========================================================================

    def _create_display_page(self) -> QWidget:
        """
        Display page: Visual elements and rendering options.

        Toggle histogram, axes, and interpolation settings.
        """
        scroll, layout = self._create_scrollable_page()

        # Image Elements
        elements_frame = create_group_frame(
            "Image Elements",
            "Toggle histogram and axes visibility"
        )
        elements_layout = elements_frame.layout()

        self.histogram_enabled_switch = Switch(parent=elements_frame)
        elements_layout.addWidget(
            create_parameter_row("Show Histogram",
                                 self.histogram_enabled_switch)
        )

        self.axes_enabled_switch = Switch(parent=elements_frame)
        elements_layout.addWidget(
            create_parameter_row("Show Axes", self.axes_enabled_switch)
        )

        layout.addWidget(elements_frame)

        # Rendering Options
        rendering_frame = create_group_frame(
            "Rendering",
            "Configure texture filtering and interpolation"
        )
        rendering_layout = rendering_frame.layout()

        self.interpolation_switch = Switch(parent=rendering_frame)
        self.interpolation_switch.setChecked(True)
        rendering_layout.addWidget(
            create_parameter_row("Smooth Interpolation",
                                 self.interpolation_switch)
        )

        layout.addWidget(rendering_frame)
        layout.addStretch()

        return scroll

    # ========================================================================
    # Signal Connections
    # ========================================================================

    def _connect_signals(self):
        """Connect all UI controls to settings and handlers."""
        # Adjustments
        self.brightness_slider.valueChanged.connect(self._on_brightness_changed)
        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        self.gamma_spin.valueChanged.connect(self._on_gamma_changed)
        self.gain_spin.valueChanged.connect(self._on_gain_changed)
        self.offset_spin.valueChanged.connect(self._on_offset_changed)
        self.invert_switch.toggled.connect(self._on_invert_changed)

        # Colormap
        self.colormap_enabled_switch.toggled.connect(
            self._on_colormap_enabled_changed)
        self.colormap_combo.currentTextChanged.connect(
            self._on_colormap_changed)
        self.colormap_reverse_switch.toggled.connect(
            self._on_colormap_reverse_changed)
        self.colorbar_visible_switch.toggled.connect(
            self._on_colorbar_visible_changed)

        # LUT
        self.lut_enabled_switch.toggled.connect(self._on_lut_enabled_changed)
        self.lut_type_combo.currentTextChanged.connect(
            self._on_lut_type_changed)
        self.lut_min_spin.valueChanged.connect(self._on_lut_min_changed)
        self.lut_max_spin.valueChanged.connect(self._on_lut_max_changed)

        # Transform
        self.zoom_spin.valueChanged.connect(self._on_zoom_changed)

        # Overlays
        self.tracking_crosshair_switch.toggled.connect(
            self._on_tracking_crosshair_changed)
        self.center_crosshair_switch.toggled.connect(
            self._on_center_crosshair_changed)
        self.roi_enabled_switch.toggled.connect(self._on_roi_enabled_changed)
        self.crop_size_combo.currentTextChanged.connect(
            self._on_crop_size_changed)

        # Display
        self.histogram_enabled_switch.toggled.connect(
            self._on_histogram_enabled_changed)
        self.axes_enabled_switch.toggled.connect(self._on_axes_enabled_changed)
        self.interpolation_switch.toggled.connect(
            self._on_interpolation_changed)

        # Settings feedback
        self.settings.changed.connect(self._on_settings_changed)

    # ========================================================================
    # Adjustment Handlers
    # ========================================================================

    def _on_brightness_changed(self, value: int):
        if not self.updating_ui:
            self.settings.update_setting('brightness', value / 100.0)

    def _on_contrast_changed(self, value: int):
        if not self.updating_ui:
            self.settings.update_setting('contrast', value / 100.0)

    def _on_gamma_changed(self, value: float):
        if not self.updating_ui:
            self.settings.update_setting('gamma', value)

    def _on_gain_changed(self, value: float):
        if not self.updating_ui:
            self.settings.update_setting('gain', value)

    def _on_offset_changed(self, value: float):
        if not self.updating_ui:
            self.settings.update_setting('offset', value)

    def _on_invert_changed(self, checked: bool):
        if not self.updating_ui:
            self.settings.update_setting('invert', checked)

    # ========================================================================
    # Colormap Handlers
    # ========================================================================

    def _on_colormap_enabled_changed(self, checked: bool):
        if not self.updating_ui:
            self.settings.update_setting('colormap_enabled', checked)
            if not checked:
                self.colorbar_visible_switch.setChecked(False)
            self.colorbar_visible_switch.setEnabled(checked)

    def _on_colormap_changed(self, name: str):
        if not self.updating_ui:
            self.settings.update_setting('colormap_name', name)

    def _on_colormap_reverse_changed(self, checked: bool):
        if not self.updating_ui:
            self.settings.update_setting('colormap_reverse', checked)

    def _on_colorbar_visible_changed(self, checked: bool):
        if not self.updating_ui:
            self.colorbar_visibility_changed.emit(checked)

    # ========================================================================
    # LUT Handlers
    # ========================================================================

    def _on_lut_enabled_changed(self, checked: bool):
        if not self.updating_ui:
            self.settings.update_setting('lut_enabled', checked)

    def _on_lut_type_changed(self, text: str):
        if not self.updating_ui:
            type_map = {
                "Linear": LUTType.LINEAR,
                "Logarithmic": LUTType.LOG,
                "Square Root": LUTType.SQRT,
                "Square": LUTType.SQUARE
            }
            self.settings.update_setting('lut_type',
                                         type_map.get(text, LUTType.LINEAR))

    def _on_lut_min_changed(self, value: float):
        if not self.updating_ui:
            if value >= self.lut_max_spin.value():
                value = self.lut_max_spin.value() - 0.01
                self.lut_min_spin.setValue(value)
            self.settings.update_setting('lut_min', value)

    def _on_lut_max_changed(self, value: float):
        if not self.updating_ui:
            if value <= self.lut_min_spin.value():
                value = self.lut_min_spin.value() + 0.01
                self.lut_max_spin.setValue(value)
            self.settings.update_setting('lut_max', value)

    # ========================================================================
    # Transform Handlers
    # ========================================================================

    def _on_zoom_changed(self, value: float):
        if not self.updating_ui:
            self.settings.update_setting('zoom', value)

    def reset_view(self):
        """Reset all view transformations to defaults."""
        self.settings.update_setting('zoom', 1.0)
        self.settings.update_setting('pan_x', 0.0)
        self.settings.update_setting('pan_y', 0.0)
        self.settings.update_setting('rotation', 0.0)

    # ========================================================================
    # Overlay Handlers
    # ========================================================================

    def _on_tracking_crosshair_changed(self, checked: bool):
        if not self.updating_ui:
            self.tracking_crosshair_visibility_changed.emit(checked)

    def _on_center_crosshair_changed(self, checked: bool):
        if not self.updating_ui:
            self.center_crosshair_visibility_changed.emit(checked)

    def _on_roi_enabled_changed(self, checked: bool):
        if not self.updating_ui:
            self.roi_visibility_changed.emit(checked)

    def _on_crop_size_changed(self, text: str):
        if not self.updating_ui:
            # Extract size from "128x128" format
            size = int(text.split('x')[0]) - 1
            roi = ROI(x=0, y=0, width=size, height=size)
            self.roi_changed.emit(roi)

    # ========================================================================
    # Display Handlers
    # ========================================================================

    def _on_histogram_enabled_changed(self, checked: bool):
        if not self.updating_ui:
            self.settings.update_setting('histogram_enabled', checked)

    def _on_axes_enabled_changed(self, checked: bool):
        if not self.updating_ui:
            self.settings.update_setting('axes_enabled', checked)

    def _on_interpolation_changed(self, checked: bool):
        if not self.updating_ui:
            self.settings.update_setting('interpolation', checked)

    # ========================================================================
    # Settings Feedback
    # ========================================================================

    def _on_settings_changed(self):
        """Update UI when settings change externally."""
        self.updating_ui = True
        try:
            self.zoom_spin.setValue(self.settings.zoom)
            # Add other UI updates as needed
        except:
            pass
        finally:
            self.updating_ui = False

    # ========================================================================
    # Info Updates
    # ========================================================================

    @pyqtSlot(FrameStats)
    def _update_image_info_helper(self, metadata: FrameStats):
        """Update frame information display."""
        try:
            self.info_labels['shape'].setText(str(metadata.shape))
            self.info_labels['min_val'].setText(f"{metadata.vmin:.4f}")
            self.info_labels['max_val'].setText(f"{metadata.vmax:.4f}")
            self.info_labels['mean'].setText(f"{metadata.mean:.4f}")
            self.info_labels['std'].setText(f"{metadata.std:.4f}")
            self.info_labels['time'].setText(
                time.strftime('%Y-%m-%d %H:%M:%S',
                              time.localtime(metadata.timestamp))
            )
        except:
            pass

    @pyqtSlot(FrameStats)
    def update_image_metadata(self, metadata: FrameStats):
        """
        Update displayed metadata and performance stats.

        Args:
            metadata: Current frame statistics
        """
        self._current_metadata = metadata
        self._update_img_perf_stats()

    def _update_img_perf_stats(self):
        """Refresh performance metrics display."""
        try:
            self._update_image_info_helper(self._current_metadata)
            perf = self.image_viewer.get_performance_stats()
            if 'fps' in perf:
                self.fps_label.setText(f"{perf['fps']:.1f}")
            if 'avg_ms' in perf:
                self.frame_time_label.setText(f"{perf['avg_ms']:.2f} ms")
        except:
            pass

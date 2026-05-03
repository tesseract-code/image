"""
Image processing controller with overlay support, histogram analysis, and ROI
functionality.
"""

from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, QRectF, QSize, Qt, pyqtSlot
from PyQt6.QtWidgets import (QFrame, QGraphicsRectItem, QGridLayout,
                             QScrollArea, QSizePolicy, QSplitter, QWidget)

from image.gl.colorbar.view import ColorbarWidget, TickPosition
from image.gui.controller.base import BasePipelineController, PipelineState
from image.gui.controller.flow import FlowController
from image.gui.controller.sync import SyncPipelineController
from image.gui.item.axes import HorizontalAxis, VerticalAxis
from image.gui.item.histogram import HistogramController
from image.gui.overlay.crop import CropView
from image.gui.settings_ctrl import ControlPanel
from image.gui.stack_view import OverlayStack, logger
from image.model.cmap import apply_colormap_to_value
from image.model.model import ImageDataModel
from image.pipeline.stats import FrameStats
from image.settings.base import ImageSettings
from image.settings.pixels import PixelFormat
from image.settings.roi import ROI
from pycore.log.ctx import with_logger

# Default ROI dimensions on first load
_DEFAULT_ROI = ROI(0, 0, 127, 127)


@with_logger
class OverlayImageWidgetController(QObject):
    """
    Main controller for image viewer with overlay support.
    """

    def __init__(self, parent: Optional[QObject] = None):
        super().__init__(parent=parent)

        # Model
        self._model = ImageDataModel()

        # Controllers
        self._pipeline_ctrl: BasePipelineController | None = None
        self._flow_control: FlowController | None = None
        self._histogram_ctrl = HistogramController()

        # UI Components
        self._central_widget: QSplitter | None = None
        self._settings: ImageSettings | None = None
        self._overlay_stack: OverlayStack | None = None
        self._roi_rect_item: QGraphicsRectItem | None = None
        self._crop_view: CropView | None = None
        self._colorbar: ColorbarWidget | None = None
        self._axisX: HorizontalAxis | None = None
        self._axisY: VerticalAxis | None = None
        self._control_panel: ControlPanel | None = None

        # State
        self._show_colorbar: bool = False
        self._current_cmap = None
        self._current_roi_settings = _DEFAULT_ROI

        self._setup_ui()
        self._connect_ui()
        self._initialize_pipeline()

    def _setup_ui(self):
        """Construct the UI hierarchy and layout."""
        self._central_widget = QSplitter(Qt.Orientation.Horizontal)

        # Main scroll area for image display
        self.img_scroll_area = QScrollArea()
        self.img_scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.img_scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.img_scroll_area.setContentsMargins(0, 0, 0, 0)
        self.img_scroll_area.setStyleSheet("background-color: palette(mid);")

        # Image frame container
        self.img_frame = QFrame()
        self.img_frame.setSizePolicy(QSizePolicy(QSizePolicy.Policy.Fixed,
                                                 QSizePolicy.Policy.Fixed))
        self.img_frame_layout = QGridLayout(self.img_frame)

        # Image viewer with OpenGL backend.
        # NOTE: _overlay_stack must be initialised before _control_panel,
        # which takes gl_viewer as a constructor argument.
        self._settings = ImageSettings(img_format=PixelFormat.RGB)
        self._overlay_stack = OverlayStack(self._settings)
        self._overlay_stack.setMinimumSize(QSize(600, 600))

        # Histograms (intensity profiles)
        v_histogram = self._histogram_ctrl.get_vertical_widget()
        h_histogram = self._histogram_ctrl.get_horizontal_widget()
        v_histogram.setFixedWidth(40)
        h_histogram.setFixedHeight(40)
        v_histogram.setVisible(False)
        h_histogram.setVisible(False)

        # Colorbar
        self._colorbar = ColorbarWidget()
        self._colorbar.set_orientation(Qt.Orientation.Vertical)
        self._colorbar.set_tick_count(10)
        self._colorbar.set_tick_position(TickPosition.END)
        self._colorbar.set_range(0, 255)
        self._colorbar.set_colorbar_width(40)
        self._colorbar.setVisible(False)

        # Axes
        self._axisX = HorizontalAxis()
        self._axisX.setLabelPrecision(0)
        self._axisX.setVisible(False)
        self._axisY = VerticalAxis()
        self._axisY.setLabelPrecision(0)
        self._axisY.setVisible(False)

        # Layout
        self.img_frame_layout.addWidget(self._axisY, 0, 0)
        self.img_frame_layout.addWidget(v_histogram, 0, 1)
        self.img_frame_layout.addWidget(self._overlay_stack, 0, 2)
        self.img_frame_layout.addWidget(self._colorbar, 0, 3)
        self.img_frame_layout.addWidget(h_histogram, 1, 2)
        self.img_frame_layout.addWidget(self._axisX, 2, 2)
        self.img_frame_layout.setColumnStretch(2, 1)
        self.img_frame_layout.setRowStretch(0, 1)

        self.img_scroll_area.setWidget(self.img_frame)
        self.img_scroll_area.setWidgetResizable(True)
        self.img_scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Control panel
        self._control_panel = ControlPanel(self._settings,
                                           self._overlay_stack.gl_viewer)
        self._control_panel.setMaximumWidth(500)

        self._crop_view = CropView(self._overlay_stack)
        self._control_panel.set_fixed_widget(self._crop_view)
        self._control_panel._toggle_navigation()

        # Main layout
        self._central_widget.addWidget(self._control_panel)
        self._central_widget.addWidget(self.img_scroll_area)
        self._central_widget.setCollapsible(0, True)

        # Configure overlays
        self._overlay_stack.crosshair_manager.set_crosshair_color("orange")
        self._overlay_stack.overlay.enable_center_crosshair(False)
        self._overlay_stack.overlay.lock_center_crosshair(True)
        self._overlay_stack.overlay.enable_tracking_crosshair(False)

    def _connect_ui(self):
        """Connect UI signals to controller slots."""
        self._control_panel.roi_changed.connect(self._on_roi_changed)
        self._control_panel.roi_visibility_changed.connect(
            self._on_roi_visibility_changed)
        self._control_panel.colorbar_visibility_changed.connect(
            self._on_colorbar_visibility_changed)

        # Colormap signals
        self._control_panel.colormap_changed.connect(
            self._on_colorbar_colormap_changed)
        self._control_panel.colormap_enabled_changed.connect(
            self._on_colormap_state_changed)

        # Crosshair signals
        self._control_panel.tracking_crosshair_visibility_changed.connect(
            self._on_tracking_crosshair_changed)
        self._control_panel.center_crosshair_visibility_changed.connect(
            self._on_center_crosshair_changed)

        # Display element signals
        self._control_panel.histogram_visibility_changed.connect(
            self._on_histogram_visibility_changed)
        self._control_panel.axes_visibility_changed.connect(
            self._on_axes_visibility_changed)

        self._overlay_stack.gl_viewer.frameChanged.connect(
            self._on_gl_texture_upload_event)
        self._crop_view.joystick.movements_batched.connect(
            self._on_jostick_moves)
        self._overlay_stack.mouse_pos_changed.connect(
            self._on_mouse_pos_changed)

    def _initialize_pipeline(self):
        """Initialize the async rendering pipeline."""
        self._pipeline_ctrl = SyncPipelineController(
            viewer=self._overlay_stack.gl_viewer,
            settings=self._settings,
            parent=self
        )
        self._pipeline_ctrl.signals.status_changed.connect(
            self._on_pipeline_state_change
        )
        self._pipeline_ctrl.start_pipeline()
        logger.debug("Image pipeline initialized")

    def central_widget(self) -> Optional[QWidget]:
        """Return the main widget for embedding in parent UI."""
        return self._central_widget

    @pyqtSlot(str, object)
    def _on_pipeline_state_change(self, pipeline_id: str, state: PipelineState):
        """Handle pipeline state transitions."""
        if state == PipelineState.RUNNING:
            if not self._flow_control:
                self._logger.debug(
                    "Starting FlowController (60fps update loop)")
                self._flow_control = FlowController(
                    self._pipeline_ctrl.mailbox, fps=60
                )
                self._flow_control.update_event.connect(
                    self._on_flow_control_event,
                    Qt.ConnectionType.DirectConnection
                )
        else:
            self._logger.debug(f"Pipeline state: {state}")

    @pyqtSlot(list)
    def _on_jostick_moves(self, moves: list):
        """Handle joystick movements for ROI control."""
        active_roi = self._overlay_stack.overlay.active_roi
        if (self._roi_rect_item is not None and
                active_roi is self._roi_rect_item and
                self._roi_rect_item.isVisible()):
            self._roi_rect_item.move_by_joystick_moves(moves)

    def _ensure_roi(self):
        """Lazy initialisation of ROI rectangle."""
        if self._roi_rect_item is None:
            self._roi_rect_item = self._overlay_stack.overlay.addRectROI(
                QRectF(
                    self._current_roi_settings.x,
                    self._current_roi_settings.y,
                    self._current_roi_settings.width - 1,
                    self._current_roi_settings.height - 1
                )
            )

    def _on_roi_visibility_changed(self, visible: bool):
        """Toggle ROI visibility."""
        if visible:
            self._ensure_roi()

        if self._roi_rect_item:
            self._roi_rect_item.setVisible(visible)
            self._control_panel.set_fixed_widget_visible(visible)

    def _on_roi_changed(self, roi: ROI):
        """Apply new ROI geometry — position and size — to the rect item."""
        self._current_roi_settings = roi
        if self._roi_rect_item is not None and self._roi_rect_item.isVisible():
            self._roi_rect_item.setRect(
                QRectF(
                    self._current_roi_settings.x,
                    self._current_roi_settings.y,
                    self._current_roi_settings.width,
                    self._current_roi_settings.height
                )
            )

    def _on_colorbar_visibility_changed(self, visible: bool):
        self._colorbar.setVisible(visible)
        if visible:
            self._colorbar.set_colormap(
                name=self._settings.colormap_name,
                reverse=self._settings.colormap_reverse,
            )

    @pyqtSlot(int, int)
    def _on_mouse_pos_changed(self, x: int, y: int):
        """Update tooltip with pixel value under mouse cursor."""
        if self._model.has_data():
            value = self._model.get_value_at(x, y, flip_x=False, flip_y=True)
            if value is not None:
                if self._settings.colormap_enabled:
                    value = apply_colormap_to_value(
                        value,
                        self._overlay_stack.gl_viewer._cmap_cache.get_lut(
                            self._settings.colormap_name),
                        data_dtype=self._model.get_dtype())

        self._overlay_stack.update_tooltip(x, y, value)

    def _update_histogram(self):
        """
        Update histogram widget with current image data.

        The image is flipped vertically to correct for OpenGL's bottom-left
        origin convention before histogram processing.
        """
        self._histogram_ctrl.process_image(np.flip(self._model.get_view(), 0))

    def _update_colorbar(self):
        """Update colorbar range to match current image metadata."""
        metadata = self._model.get_metadata()
        self._colorbar.set_range(metadata.vmin, metadata.vmax)
        # Colormap name/reverse is now handled by the dedicated signal slot

    @pyqtSlot(FrameStats)
    def _on_gl_texture_upload_event(self, metadata: FrameStats):
        """Called when OpenGL texture upload completes. Updates dependent UI."""
        self._control_panel.update_image_metadata(metadata)

        # shape is (height, width[, channels]) — Y maps to height, X to width
        self._axisY.setRange(0, metadata.shape[0] - 1)
        self._axisX.setRange(0, metadata.shape[1] - 1)

        self._update_histogram()

        if self._settings.colormap_enabled:
            self._update_colorbar()

    @pyqtSlot(np.ndarray, FrameStats)
    def _on_flow_control_event(self, image: np.ndarray, metadata: FrameStats):
        """
        Handle flow controller update events (60fps loop).

        Uses the frame and metadata delivered by the signal directly to avoid
        a TOCTOU race with the model being updated mid-handler.
        """
        self._logger.debug("Updating GL image view")

        self._overlay_stack.gl_viewer.present(
            image,
            metadata,
            self._settings.format,
        )

        # Resize viewer if image dimensions changed
        height, width = image.shape[:2]
        overlay_size = self._overlay_stack.size()
        if overlay_size.width() != width or overlay_size.height() != height:
            image_size = QSize(width, height)
            self._overlay_stack.setFixedSize(image_size)
            self._overlay_stack.gl_viewer.setFixedSize(image_size)

    @pyqtSlot(np.ndarray)
    def set_image(self, img: np.ndarray):
        """
        Set new image data and trigger pipeline processing.

        Args:
            img: Image data as numpy array.
        """
        self._model.set_data(img)

        if not self._pipeline_ctrl.is_running:
            self._logger.warning(
                "Frame received while pipeline is not running — frame dropped."
            )
            return

        self._pipeline_ctrl.ingest_frame(image=img)

    @pyqtSlot()
    def cleanup(self):
        """Clean shutdown of all components."""
        if self._pipeline_ctrl:
            self._pipeline_ctrl.stop_pipeline()
        if self._histogram_ctrl:
            self._histogram_ctrl.shutdown()
        if self._overlay_stack:
            self._overlay_stack.cleanup()

    # ========================================================================
    # NEW Slots for colormap changes
    # ========================================================================

    @pyqtSlot(str, bool)
    def _on_colorbar_colormap_changed(self, name: str, reverse: bool):
        """Update the colorbar colormap immediately when changed in UI."""
        if self._colorbar is not None:
            self._colorbar.set_colormap(name=name, reverse=reverse)
            self._current_cmap = name

    @pyqtSlot(bool)
    def _on_colormap_state_changed(self, enabled: bool):
        """Hide the colorbar when the colormap is disabled."""
        if not enabled:
            self._colorbar.setVisible(False)
            self._update_colorbar()

    # ========================================================================
    # Crosshair slots
    # ========================================================================

    @pyqtSlot(bool)
    def _on_tracking_crosshair_changed(self, visible: bool):
        """Toggle the tracking crosshair on the overlay."""
        self._overlay_stack.overlay.enable_tracking_crosshair(visible)

    @pyqtSlot(bool)
    def _on_center_crosshair_changed(self, visible: bool):
        """Toggle the fixed center crosshair on the overlay."""
        self._overlay_stack.overlay.enable_center_crosshair(visible)

    # ========================================================================
    # Display element slots
    # ========================================================================

    @pyqtSlot(bool)
    def _on_histogram_visibility_changed(self, visible: bool):
        """Show or hide the intensity histogram widgets."""
        self._histogram_ctrl.get_vertical_widget().setVisible(visible)
        self._histogram_ctrl.get_horizontal_widget().setVisible(visible)

    @pyqtSlot(bool)
    def _on_axes_visibility_changed(self, visible: bool):
        """Show or hide the X/Y axis rulers."""
        self._axisX.setVisible(visible)
        self._axisY.setVisible(visible)
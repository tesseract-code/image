"""
Image processing controller with overlay support, histogram analysis, and ROI
functionality.
"""

from typing import Optional

import numpy as np
from PyQt6.QtCore import QObject, QRectF, QSize, Qt, pyqtSlot
from PyQt6.QtWidgets import (QFrame, QGraphicsRectItem, QGridLayout,
                             QScrollArea, QSizePolicy, QSplitter, QWidget)

from image.gl.colorbar.view import (ColorbarWidget)
from image.gl.colorbar.view import TickPosition
from image.gui.controller.base import (
    BasePipelineController, PipelineState)
from image.gui.controller.flow import FlowController
from image.gui.controller.sync import (
    SyncPipelineController)
from image.gui.item.axes import (HorizontalAxis,
                                                          VerticalAxis)
from image.gui.item.histogram import (
    HistogramController)
from image.gui.overlay.crop import CropView
from image.gui.settings_ctrl import ControlPanel
from image.gui.stack_view import OverlayStack, logger
from image.model.model import ImageDataModel
from image.pipeline.stats import FrameStats
from image.settings.base import ImageSettings
from image.settings.pixels import PixelFormat
from image.settings.roi import ROI
from image.model.cmap import apply_colormap_to_value
from pycore.log.ctx import with_logger


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
        self._img_settings: ImageSettings | None = None
        self._overlay_stack: OverlayStack | None = None
        self._roi_rect_item: QGraphicsRectItem | None = None
        self._img_crop_view: CropView | None = None
        self._img_colorbar_widget: ColorbarWidget | None = None
        self._img_axisX: HorizontalAxis | None = None
        self._img_axisY: VerticalAxis | None = None
        self._control_panel: ControlPanel | None = None

        # State
        self._show_colorbar: bool = False
        self._current_cmap = None

        self._current_roi_settings = ROI(0, 0, 127, 127)

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
        # Image viewer with OpenGL backend
        self._img_settings = ImageSettings(img_format=PixelFormat.RGB)
        self._overlay_stack = OverlayStack(self._img_settings)
        self._overlay_stack.setMinimumSize(QSize(600, 600))
        # Histograms (intensity profiles)
        v_histogram = self._histogram_ctrl.get_vertical_widget()
        h_histogram = self._histogram_ctrl.get_horizontal_widget()
        v_histogram.setFixedWidth(40)
        h_histogram.setFixedHeight(40)
        v_histogram.setVisible(False)
        h_histogram.setVisible(False)
        # Colorbar
        self._img_colorbar_widget = ColorbarWidget()
        self._img_colorbar_widget.set_orientation(Qt.Orientation.Vertical)
        self._img_colorbar_widget.set_tick_count(10)
        self._img_colorbar_widget.set_tick_position(TickPosition.END)
        self._img_colorbar_widget.set_range(0, 255)
        self._img_colorbar_widget.set_colorbar_width(40)
        self._img_colorbar_widget.setVisible(False)
        # Axes
        self._img_axisX = HorizontalAxis()
        self._img_axisX.setLabelPrecision(0)
        self._img_axisX.setVisible(False)
        self._img_axisY = VerticalAxis()
        self._img_axisY.setLabelPrecision(0)
        self._img_axisY.setVisible(False)
        # Layout
        self.img_frame_layout.addWidget(self._img_axisY, 0, 0)
        self.img_frame_layout.addWidget(v_histogram, 0, 1)
        self.img_frame_layout.addWidget(self._overlay_stack, 0, 2)
        self.img_frame_layout.addWidget(self._img_colorbar_widget, 0, 3)
        self.img_frame_layout.addWidget(h_histogram, 1, 2)
        self.img_frame_layout.addWidget(self._img_axisX, 2, 2)
        self.img_frame_layout.setColumnStretch(2, 1)
        self.img_frame_layout.setRowStretch(0, 1)

        self.img_scroll_area.setWidget(self.img_frame)
        self.img_scroll_area.setWidgetResizable(True)
        self.img_scroll_area.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # Control panel
        self._control_panel = ControlPanel(self._img_settings,
                                           self._overlay_stack.gl_viewer)
        self._control_panel.setMaximumWidth(500)

        self._img_crop_view = CropView(self._overlay_stack)
        self._control_panel.set_fixed_widget(self._img_crop_view)
        self._control_panel._toggle_navigation()
        # Main layout
        self._central_widget.addWidget(self._control_panel)
        self._central_widget.addWidget(self.img_scroll_area)
        self._central_widget.setCollapsible(0, True)
        # Configure overlays
        self._overlay_stack.crosshair_manager.set_crosshair_color("orange")
        self._overlay_stack.overlay.enable_center_crosshair(True)
        self._overlay_stack.overlay.lock_center_crosshair(True)
        self._overlay_stack.overlay.enable_tracking_crosshair(True)

    def _connect_ui(self):
        """Connect UI signals to controller slots."""
        self._control_panel.roi_changed.connect(
            self._on_roi_changed
        )
        self._control_panel.roi_visibility_changed.connect(
            self._on_roi_visibility_changed
        )

        self._control_panel.colorbar_visibility_changed.connect(
            self._on_colorbar_visibility_changed
        )

        self._overlay_stack.gl_viewer.frameChanged.connect(
            self._on_gl_texture_upload_event
        )
        self._img_crop_view.joystick.movements_batched.connect(
            self._on_jostick_moves
        )
        self._overlay_stack.mouse_pos_changed.connect(
            self._on_mouse_pos_changed
        )

    def _initialize_pipeline(self):
        """Initialize the async rendering pipeline."""
        self._pipeline_ctrl = SyncPipelineController(
            viewer=self._overlay_stack.gl_viewer,
            settings=self._img_settings,
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
    def _on_pipeline_state_change(self, id: str, state: PipelineState):
        """Handle pipeline state transitions."""
        if state == PipelineState.RUNNING:
            if not self._flow_control:
                logger.debug("Starting FlowController (60fps update loop)")
                self._flow_control = FlowController(
                    self._pipeline_ctrl.mailbox, fps=60
                )
                self._flow_control.update_event.connect(
                    self._on_flow_control_event,
                    Qt.ConnectionType.DirectConnection
                )
        else:
            logger.debug(f"Pipeline state: {state}")

    @pyqtSlot(list)
    def _on_jostick_moves(self, moves: list):
        """Handle joystick movements for ROI control."""
        active_roi = self._overlay_stack.overlay.active_roi
        if (self._roi_rect_item is not None and
                active_roi is self._roi_rect_item and
                self._roi_rect_item.isVisible()):
            self._roi_rect_item.move_by_joystick_moves(moves)

    def _ensure_roi(self):
        """Lazy initialization of ROI rectangle."""
        if self._roi_rect_item is None:
            self._roi_rect_item = self._overlay_stack.overlay.addRectROI(
                QRectF(
                    self._current_roi_settings.x,
                    self._current_roi_settings.y,
                    self._current_roi_settings.width,
                    self._current_roi_settings.height
                )
            )

    def _on_roi_visibility_changed(self, visible: bool):
        """Toggle ROI visibility."""
        if visible:
            self._ensure_roi()

        if self._roi_rect_item:
            self._roi_rect_item.setVisible(visible)
            self._control_panel.set_fixed_widget_visible(visible)

    def _on_roi_changed(self, roi):
        self._current_roi_settings = roi
        if self._roi_rect_item is not None and self._roi_rect_item.isVisible():
            self._roi_rect_item.setRect(
                QRectF(
                    self._roi_rect_item.x(),
                    self._roi_rect_item.y(),
                    self._current_roi_settings.width,
                    self._current_roi_settings.height
                )
            )

    def _on_colorbar_visibility_changed(self, visible: bool):
        self._img_colorbar_widget.setVisible(visible)
        if visible:
            self._img_colorbar_widget.set_colormap(
                name=self._img_settings.colormap_name,
            reverse=self._img_settings.colormap_reverse)

    @pyqtSlot(int, int)
    def _on_mouse_pos_changed(self, x: int, y: int):
        """Update tooltip with pixel value under mouse cursor."""
        if self._model.has_data():
            value = self._model.get_value_at(x, y, flip_x=False, flip_y=True)
            print("RAW VALUE: ", value)
            if value is not None:
                if self._img_settings.colormap_enabled:
                    print("CMAP ENABLED")
                    value = apply_colormap_to_value(
                        value,
                        self._overlay_stack.gl_viewer._cmap_cache.get_lut(
                            self._img_settings.colormap_name),
                    data_dtype=self._model.get_dtype())

                    print("VALUE WITH CMAP: ", value)

                self._overlay_stack.update_tooltip(x, y, value)

    def _update_histogram(self):
        """
        Update histogram widgets with current image data.

        Note: Image is flipped vertically because OpenGL uses bottom-left origin
        while image coordinates use top-left origin.
        """
        # Get flipped view for histogram processing
        # The flip happens during copy to shared memory in the histogram controller
        self._histogram_ctrl.process_image(np.flip(self._model.get_view(), 0))

        # Show histogram widgets if hidden
        v_histogram = self._histogram_ctrl.get_vertical_widget()
        h_histogram = self._histogram_ctrl.get_horizontal_widget()

        if not v_histogram.isVisible():
            v_histogram.setVisible(True)
        if not h_histogram.isVisible():
            h_histogram.setVisible(True)

    def _update_colorbar(self):
        """Update colorbar range and colormap to match current image."""
        metadata = self._model.get_metadata()
        self._img_colorbar_widget.set_range(metadata.vmin, metadata.vmax)

        # Update colormap if changed
        if self._current_cmap != self._img_settings.colormap_name:
            self._current_cmap = self._img_settings.colormap_name
            self._img_colorbar_widget.set_colormap(
                self._img_settings.colormap_name
            )

    @pyqtSlot(FrameStats)
    def _on_gl_texture_upload_event(self, metadata: FrameStats):
        """Called when OpenGL texture upload completes. Updates dependent UI."""

        # Update control panel with new metadata
        self._control_panel.update_image_metadata(metadata)

        # Update axes ranges
        self._img_axisY.setRange(0, metadata.shape[1] - 1)
        self._img_axisX.setRange(0, metadata.shape[0] - 1)

        # Show axes if hidden
        if not self._img_axisY.isVisible():
            self._img_axisY.setVisible(True)
        if not self._img_axisX.isVisible():
            self._img_axisX.setVisible(True)

        # Update histogram and colorbar
        self._update_histogram()

        if self._img_settings.colormap_enabled:
            self._update_colorbar()

    @pyqtSlot(np.ndarray, FrameStats)
    def _on_flow_control_event(self, image: np.ndarray, metadata: FrameStats):
        """Handle flow controller update events (60fps loop)."""
        logger.debug("Updating GL image view")

        if not self._model.has_data():
            return

        metadata = self._model.get_metadata()

        # Upload texture to GPU
        self._overlay_stack.gl_viewer.present(
            self._model.get_view(),
            metadata,
            self._img_settings.format
        )

        # Resize viewer if dimensions changed
        height, width = metadata.shape[:2]  # TODO: review logical consistency
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
            img: Image data as numpy array
        """
        self._model.set_data(img)

        if self._model.has_data() and self._pipeline_ctrl.is_running:
            self._pipeline_ctrl.ingest_frame(image=img)

    @pyqtSlot()
    def cleanup(self):
        """Clean shutdown of all components."""
        if self._pipeline_ctrl:
            self._pipeline_ctrl.stop_pipeline()
        if self._histogram_ctrl:
            self._histogram_ctrl.shutdown()
        self._overlay_stack.cleanup()

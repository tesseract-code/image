import random
import sys

from PyQt6.QtCore import QRectF
from PyQt6.QtGui import QColor, QPixmap, QImage, \
    QPainter
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QPushButton, QGraphicsScene,
                             QCheckBox,
                             QSpinBox, QLabel, QGroupBox,
                             QGraphicsLineItem, QGraphicsItemGroup)
from PyQt6.QtWidgets import QGraphicsPixmapItem

from image.gui.overlay.view import SynchableGraphicsView


class TestWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("GraphicsView with Image Display - PyQt6")
        self.setGeometry(100, 100, 1200, 800)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create scene and view
        self.scene = QGraphicsScene()
        self.scene.setSceneRect(0, 0, 800, 600)  # Set initial scene size

        self.graphics_view = SynchableGraphicsView(self.scene, self)
        self.graphics_view.setRenderHint(
            QPainter.RenderHint.SmoothPixmapTransform)
        layout.addWidget(self.graphics_view)

        # Create control panel
        self.create_crosshair_controls(layout)
        self.create_roi_controls(layout)
        self.create_image_controls(layout)

        # Status bar
        self.statusBar().showMessage("Ready - Load an image to test")

        # Connect signals
        self.graphics_view.roiClicked.connect(self.on_roi_clicked)
        self.graphics_view.roiChanged.connect(self.on_roi_changed)

        # Enable center crosshair by default
        self.graphics_view.enable_center_crosshair(True)
        self.graphics_view.lock_center_crosshair(True)

    def create_crosshair_controls(self, parent_layout):
        """Create controls for crosshair settings with clear separation"""
        crosshair_group = QGroupBox("Crosshair Controls")
        crosshair_layout = QHBoxLayout(crosshair_group)

        # Tracking crosshair controls
        self.chk_tracking = QCheckBox("Tracking Crosshair (Full XY)")
        self.chk_tracking.setChecked(False)
        self.chk_tracking.toggled.connect(
            self.graphics_view.enable_tracking_crosshair)

        # Center crosshair controls
        self.chk_center = QCheckBox("Center Crosshair (Plus)")
        self.chk_center.setChecked(True)
        self.chk_center.toggled.connect(
            self.graphics_view.enable_center_crosshair)

        # Center crosshair length controls
        self.spin_length = QSpinBox()
        self.spin_length.setRange(10, 100)
        self.spin_length.setValue(20)
        self.spin_length.valueChanged.connect(
            self.graphics_view.set_center_crosshair_length)

        crosshair_layout.addWidget(self.chk_tracking)
        crosshair_layout.addWidget(self.chk_center)
        crosshair_layout.addWidget(QLabel("Center Length:"))
        crosshair_layout.addWidget(self.spin_length)
        crosshair_layout.addStretch()

        parent_layout.addWidget(crosshair_group)

    def create_roi_controls(self, parent_layout):
        """Create controls for ROI testing"""
        roi_layout = QHBoxLayout()

        self.btn_add_ellipse = QPushButton("Add Ellipse ROI")
        self.btn_add_rect = QPushButton("Add Rectangle ROI")
        self.btn_clear_rois = QPushButton("Clear All ROIs")

        self.btn_add_ellipse.clicked.connect(self.add_ellipse_roi)
        self.btn_add_rect.clicked.connect(self.add_rect_roi)
        self.btn_clear_rois.clicked.connect(self.graphics_view.clearROIs)

        roi_layout.addWidget(QLabel("ROI Testing:"))
        roi_layout.addWidget(self.btn_add_ellipse)
        roi_layout.addWidget(self.btn_add_rect)
        roi_layout.addWidget(self.btn_clear_rois)
        roi_layout.addStretch()

        parent_layout.addLayout(roi_layout)

    def create_image_controls(self, parent_layout):
        """Create controls for image testing"""
        image_layout = QHBoxLayout()

        self.btn_load_test_image = QPushButton("Load Test Image")
        self.btn_fit_to_view = QPushButton("Fit to View")

        self.btn_load_test_image.clicked.connect(self.load_test_image)
        # self.btn_fit_to_view.clicked.connect(self.graphics_view.fit_to_scene)

        image_layout.addWidget(QLabel("Image Controls:"))
        image_layout.addWidget(self.btn_load_test_image)
        image_layout.addWidget(self.btn_fit_to_view)
        image_layout.addStretch()

        parent_layout.addLayout(image_layout)

    def load_test_image(self):
        """Load a test image into the scene"""
        try:
            # Create a test image (you can replace this with actual image loading)
            width, height = 800, 600
            image = QImage(width, height, QImage.Format.Format_RGB32)

            # Fill with a gradient for testing
            for y in range(height):
                for x in range(width):
                    r = int(255 * x / width)
                    g = int(255 * y / height)
                    b = 128
                    image.setPixelColor(x, y, QColor(r, g, b))

            pixmap = QPixmap.fromImage(image)
            pixmap_item = QGraphicsPixmapItem(pixmap)

            # Clear existing items (except crosshairs)
            for item in self.scene.items():
                if not isinstance(item,
                                  (QGraphicsLineItem, QGraphicsItemGroup)):
                    self.scene.removeItem(item)

            self.scene.addItem(pixmap_item)
            self.scene.setSceneRect(0, 0, width, height)

            # Fit the view to the image
            self.graphics_view.fit_to_scene()

            self.statusBar().showMessage(
                f"Loaded test image ({width}x{height})")

        except Exception as e:
            self.statusBar().showMessage(f"Error loading image: {e}")

    def add_ellipse_roi(self):
        try:
            roi = self.graphics_view.addEllipseROI(QRectF(0, 0, 120, 80))
            x = random.randint(100, 500)
            y = random.randint(100, 400)
            roi.setPos(x, y)
            self.statusBar().showMessage(f"Added Ellipse ROI at ({x}, {y})")
        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")

    def add_rect_roi(self):
        try:
            roi = self.graphics_view.addRectROI(QRectF(0, 0, 100, 100))
            x = random.randint(100, 500)
            y = random.randint(100, 400)
            roi.setPos(x, y)
            self.statusBar().showMessage(f"Added Rectangle ROI at ({x}, {y})")
        except Exception as e:
            self.statusBar().showMessage(f"Error: {e}")

    def on_roi_clicked(self, roi):
        roi_type = type(roi).__name__
        pos = roi.pos()
        self.statusBar().showMessage(
            f"{roi_type} clicked at ({pos.x():.1f}, {pos.y():.1f})")

    def on_roi_changed(self, roi):
        roi_type = type(roi).__name__
        pos = roi.pos()
        self.statusBar().showMessage(
            f"{roi_type} moved to ({pos.x():.1f}, {pos.y():.1f})")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TestWindow()
    window.show()
    sys.exit(app.exec())

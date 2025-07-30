from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QGraphicsView, QGraphicsScene
)
from PySide6.QtGui import QPainter, QBrush, QColor
from PySide6.QtCore import Qt

from GUI.rotation_panel import RotationControlPanel
from Controller.viewer_controller import ViewerController

class DicomViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("manual image fusion example")

        # Setup scene and view
        self.scene = QGraphicsScene()
        self.graphics_view = QGraphicsView(self.scene)
        self.graphics_view.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        self.graphics_view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        # Setup viewer controller (logic)
        self.viewer_controller = ViewerController(self.scene)

        # Setup UI components
        self.layer_list = QListWidget()
        self.layer_list.currentRowChanged.connect(self.on_layer_selected)

        self.load_btn = QPushButton("Load DICOM Folder")
        self.load_btn.clicked.connect(self.load_dicom)

        self.remove_button = QPushButton("Remove Current Layer")
        self.remove_button.clicked.connect(self.remove_current_layer)

        self.toggle_visibility_button = QPushButton("Hide Current Layer")
        # TODO: Connect toggle visibility logic if needed

        self.rotation_panel = RotationControlPanel()
        self.rotation_panel.set_rotation_changed_callback(self.on_rotation_changed)

        self.slice_slider = None  # will be set in setup_ui

        self.setup_ui()

    def setup_ui(self):
        # Create slice slider
        from PySide6.QtWidgets import QSlider
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(100)
        self.slice_slider.setValue(50)

        # Connect slice slider to controller
        self.viewer_controller.set_slice_slider(self.slice_slider)

        # Create slider container for opacity and offset sliders
        from PySide6.QtWidgets import QVBoxLayout
        self.slider_container = QVBoxLayout()
        self.viewer_controller.set_slider_container(self.slider_container)

        # Compose controls layout
        controls = QVBoxLayout()
        controls.addWidget(self.load_btn)
        controls.addWidget(self.toggle_visibility_button)
        controls.addWidget(self.remove_button)
        controls.addWidget(QLabel("Select Layer:"))
        controls.addWidget(self.layer_list)
        controls.addWidget(QLabel("Rotation Controls"))
        controls.addWidget(self.rotation_panel)
        controls.addLayout(self.slider_container)
        controls.addWidget(QLabel("Global Slice"))
        controls.addWidget(self.slice_slider)

        # Compose main layout
        layout = QHBoxLayout()
        layout.addLayout(controls, 2)
        layout.addWidget(self.graphics_view, 4)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_dicom(self):
        from PySide6.QtWidgets import QFileDialog
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if not folder:
            return

        name = self.viewer_controller.load_dicom_folder(folder)
        if name is not None:
            self.layer_list.addItem(name)
            # Select the newly added layer
            new_index = self.layer_list.count() - 1
            self.layer_list.setCurrentRow(new_index)
            self.update_rotation_sliders()

    def on_layer_selected(self, index):
        self.viewer_controller.select_layer(index)
        self.update_rotation_sliders()

    def update_rotation_sliders(self):
        if self.viewer_controller.selected_layer_index is None:
            self.rotation_panel.set_rotations([0, 0, 0])
        else:
            layer = self.viewer_controller.volume_layers[self.viewer_controller.selected_layer_index]
            self.rotation_panel.set_rotations(layer.rotation)

    def on_rotation_changed(self, axis_index, value):
        self.viewer_controller.update_rotation(axis_index, value)

    def remove_current_layer(self):
        index = self.viewer_controller.selected_layer_index
        if index is None:
            return
        self.viewer_controller.remove_current_layer()
        self.layer_list.takeItem(index)
        # Update layer selection after removal
        count = self.layer_list.count()
        if count > 0:
            self.layer_list.setCurrentRow(min(index, count - 1))
        self.update_rotation_sliders()

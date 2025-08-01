from PySide6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QGraphicsView, QGraphicsScene
)
from PySide6.QtGui import QPainter, QBrush, QColor
from PySide6.QtCore import Qt

from GUI.rotation_panel import RotationControlPanel
from Controller.viewer_controller import ViewerController
from GUI.translation_panel import TranslationControlPanel
from GUI.extra_controls import ZoomControlPanel

class DicomViewer(QMainWindow):
    """
     Main window for the manual image fusion DICOM viewer application.

     This class sets up the user interface, manages user interactions, and coordinates with the
     viewer controller to handle DICOM image layers, visualization, and controls.
     """
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

        self.translation_panel = TranslationControlPanel()
        self.translation_panel.set_offset_changed_callback(self.on_offset_changed)

        self.zoom_panel = ZoomControlPanel()
        self.zoom_panel.set_zoom_changed_callback(self.on_zoom_changed)

        self.slice_slider = None  # will be set in setup_ui

        self.setup_ui()

    def setup_ui(self):
        """
             Sets up the user interface for the DICOM viewer main window.

             This method creates and arranges all UI components, including sliders,
             panels, and control buttons, and connects them to the viewer controller
             and the main window layout.
        """
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

        #Rotation sliders
        controls.addWidget(QLabel("Rotation Controls"))
        controls.addWidget(self.rotation_panel)
        controls.addLayout(self.slider_container)

        #translation sliders
        controls.addWidget(QLabel("Translation Controls"))
        controls.addWidget(self.translation_panel)

        #Zoom in extra container
        controls.addWidget(QLabel("Zoom"))
        controls.addWidget(self.zoom_panel)

        #global slice slider
        controls.addWidget(QLabel("Global Slice"))
        controls.addWidget(self.slice_slider)

        # Compose main layout
        layout = QHBoxLayout()
        layout.addLayout(controls, 2)
        layout.addWidget(self.graphics_view, 4)

        #Adding to the controller
        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def load_dicom(self):
        """
            Loads a DICOM folder and adds it as a new layer to the viewer.

            Prompts the user to select a DICOM folder, loads the volume using the viewer controller,
            and updates the layer list and controls if successful.

            Returns:
                None
            """

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
            self.update_layer_controls()

    def on_layer_selected(self, index):
        """
              Handles the event when a new layer is selected in the layer list.

              Updates the selected layer in the viewer controller and refreshes the layer controls accordingly.

              Args:
                  index: The index of the newly selected layer.
              """
        self.viewer_controller.select_layer(index)
        self.update_layer_controls()

    def update_layer_controls(self):
        """
                Updates the rotation and translation controls to match the currently selected layer.

                If no layer is selected, resets the controls to default values. Otherwise, sets the controls
                to reflect the rotation and offset of the selected image layer.
        """
        if self.viewer_controller.selected_layer_index is None:
            self.rotation_panel.set_rotations([0, 0, 0])
            self.translation_panel.set_offsets([0, 0])
        else:
            layer = self.viewer_controller.volume_layers[self.viewer_controller.selected_layer_index]
            self.rotation_panel.set_rotations(layer.rotation)
            self.translation_panel.set_offsets(layer.offset)

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
        self.update_layer_controls()


    def on_offset_changed(self, offset):
        """
            This method updates the translation of the currently selected image layer in the viewer
            controller when the translation panel's offset is changed.
        """
        self.viewer_controller.update_translation(offset)

    def on_zoom_changed(self, scale_factor):
        self.graphics_view.resetTransform()
        self.graphics_view.scale(scale_factor, scale_factor)

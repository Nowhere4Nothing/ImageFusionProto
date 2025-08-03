from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QPushButton,
    QLabel, QListWidget, QGraphicsView, QGraphicsScene, QFileDialog, QVBoxLayout, QSlider
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
        self.slider_container = None
        self.setWindowTitle("manual image fusion example")

        # Setup scene and view
        self.scene = QGraphicsScene()
        self.graphics_view = QGraphicsView(self.scene)
        self.graphics_view.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        self.graphics_view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        # Setup viewer controller (logic)
        self.viewer_controller = ViewerController(self.scene)

        # Track sliders for cleanup
        self.layer_slider_rows = {}

        # Setup UI components
        self.layer_list = QListWidget()
        self.layer_list.currentRowChanged.connect(self.on_layer_selected)

        self.load_btn = QPushButton("Load DICOM Folder")
        self.load_btn.clicked.connect(self.load_dicom)

        self.remove_button = QPushButton("Remove Current Layer")
        self.remove_button.clicked.connect(self.remove_current_layer)

        self.toggle_visibility_button = QPushButton("Hide Current Layer")
        # TODO: Connect toggle visibility logic if needed

        self.reset_sliders_button = QPushButton("Reset Sliders")
        self.reset_sliders_button.clicked.connect(self.reset_layer_controls)

        self.rotation_panel = RotationControlPanel()
        self.rotation_panel.set_rotation_changed_callback(self.on_rotation_changed)

        self.translation_panel = TranslationControlPanel()
        self.translation_panel.set_offset_changed_callback(self.on_offset_changed)

        self.zoom_panel = ZoomControlPanel()
        self.zoom_panel.set_zoom_changed_callback(self.on_zoom_changed)
        self.current_zoom = 1.0
        self.zoom_panel.set_zoom_changed_callback(self.on_zoom_changed)

        self.rt_dose_layer = None

        self.slice_slider = None  # will be set in setup_ui

        self.setup_ui()

    def setup_ui(self):
        """
             Sets up the user interface for the DICOM viewer main window.

             This method creates and arranges all UI components, including sliders,
             panels, and control buttons, and connects them to the viewer controller
             and the main window layout.
        """
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(100)
        self.slice_slider.setValue(50)

        # Connect slice slider to controller
        self.viewer_controller.set_slice_slider(self.slice_slider)

        # Create slider container for opacity and offset sliders
       
        self.slider_container = QVBoxLayout()
        self.viewer_controller.set_slider_container(self.slider_container)

        # Compose controls layout
        controls = QVBoxLayout()
        controls.addWidget(self.load_btn)
        controls.addWidget(self.toggle_visibility_button)
        controls.addWidget(self.remove_button)
        controls.addWidget(QLabel("Select Layer:"))
        controls.addWidget(self.layer_list)
        controls.addWidget(self.reset_sliders_button)

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
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if not folder:
            return

        result = self.viewer_controller.load_dicom_folder(folder)
        if result is not None:
            name, layer, slider_rows = result
            self.layer_list.addItem(name)  # Add layer name (string) to list widget
            self.layer_list.setCurrentRow(self.layer_list.count() - 1)
            self.layer_slider_rows[name] = slider_rows
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
        """
        Removes the currently selected image layer from the viewer and updates
        the UI.

        This method ensures both the internal data and UI elements (like sliders)
        are cleaned up properly.
        """
        index = self.viewer_controller.selected_layer_index
        if index is None:
            return

        layer_name = self.layer_list.item(index).text()

        # Remove sliders (frames) for this layer
        slider_frames = self.layer_slider_rows.pop(layer_name, [])
        for frame in slider_frames:
            try:
                #throws error if not here when deleting layers
                if frame is not None:
                    self.slider_container.removeWidget(frame)
                    frame.setParent(None)
                    frame.deleteLater()
            except RuntimeError:
                # Frame already deleted — skip
                continue

        # Remove the image layer
        self.viewer_controller.remove_current_layer()

        # Remove the name from the list
        self.layer_list.takeItem(index)

        remaining = self.layer_list.count()

        # Update selected_layer_index safely
        if remaining == 0:
            self.viewer_controller.selected_layer_index = None
        else:
            # If removed last item, select previous, else select same index
            new_index = min(index, remaining - 1)
            self.layer_list.setCurrentRow(new_index)
            self.viewer_controller.selected_layer_index = new_index

        # Refresh controls
        self.update_layer_controls()


    def on_offset_changed(self, offset):
        """
            This method updates the translation of the currently selected image layer
            in the viewer controller when the translation panel's offset is changed.
        """
        self.viewer_controller.update_translation(offset)

    def reset_zoom(self):
        self.graphics_view.resetTransform()
        self.current_zoom = 1.0
        self.zoom_panel.set_zoom(1.0)

    def on_zoom_changed(self, new_zoom):
        """
            Updates the zoom level of the graphics view based on the provided zoom factor.

            This method resets the current transformation and applies the new zoom,
            updating the internal zoom state.

            Args:
                new_zoom: The new zoom factor to apply to the graphics view.
        """
        # Always reset the view's transform before applying new zoom
        self.graphics_view.resetTransform()

        # Apply the absolute zoom level
        self.graphics_view.scale(new_zoom, new_zoom)

        # Track the current zoom
        self.current_zoom = new_zoom

    def reset_layer_controls(self):
        index = self.viewer_controller.selected_layer_index
        if index is None:
            return

        layer = self.viewer_controller.volume_layers[index]

        # Reset internal layer values
        layer.rotation = [0, 0, 0]
        layer.offset = (0, 0)
        layer.slice_offset = 0
        layer.opacity = 1.0
        layer.cached_rotated_volume = None

        # Reset UI controls
        self.rotation_panel.reset_rotation()
        self.translation_panel.reset_trans()
        self.zoom_panel.set_zoom(1.0)
        self.on_zoom_changed(1.0)

        # Reset opacity and slice offset sliders & update their value labels
        slider_frames = self.layer_slider_rows.get(layer.name, [])
        for frame in slider_frames:
            layout = frame.layout()
            if not layout:
                continue

            label_item = layout.itemAt(0)
            slider_item = layout.itemAt(1)
            value_label_item = layout.itemAt(2)

            if label_item is None or slider_item is None:
                continue

            label = label_item.widget()
            slider = slider_item.widget()
            value_label = value_label_item.widget() if value_label_item is not None else None

            if isinstance(label, QLabel) and isinstance(slider, QSlider):
                slider.blockSignals(True)
                if "Opacity" in label.text():
                    slider.setValue(100)
                    if isinstance(value_label, QLabel):
                        value_label.setText("100%")
                elif "Slice Offset" in label.text():
                    slider.setValue(0)
                    if isinstance(value_label, QLabel):
                        value_label.setText("0")
                slider.blockSignals(False)

        # Update the display
        self.viewer_controller.update_global_slice_slider_range()
        # index is still valid and active
        self.viewer_controller.selected_layer_index = index

        #re-render
        self.viewer_controller.update_display()

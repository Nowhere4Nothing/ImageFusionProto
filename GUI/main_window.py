from PySide6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QPushButton,
    QLabel, QListWidget,  QFileDialog, QVBoxLayout, QSlider,
)
from PySide6.QtCore import Qt

from GUI.MultiViewWidget import MultiViewWidget
from GUI.rotation_panel import RotationControlPanel
from GUI.translation_panel import TranslationControlPanel
from GUI.extra_controls import ZoomControlPanel
from utils.layer_loader import reset_opacity_and_offset, highlight_selected_layer


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

        # Setup scene and multiview
        self.multi_view = MultiViewWidget()

        self.axial_controller = self.multi_view.axial_viewer.controller
        self.coronal_controller = self.multi_view.coronal_viewer.controller
        self.sagittal_controller = self.multi_view.sagittal_viewer.controller

        # Central widget
        main_layout = QVBoxLayout()

        # Top row: axial + coronal side-by-side
        top_row = QHBoxLayout()
        top_row.addWidget(self.multi_view.axial_viewer)
        top_row.addWidget(self.multi_view.coronal_viewer)

        bottom_row = QHBoxLayout()
        bottom_row.addStretch()
        bottom_row.addWidget(self.multi_view.sagittal_viewer)
        bottom_row.addStretch()

        main_layout.addLayout(top_row)
        main_layout.addLayout(bottom_row)

        # Track sliders for cleanup
        self.layer_slider_rows = {}

        # Setup UI components
        self.layer_list = QListWidget()
        self.layer_list.currentRowChanged.connect(self.on_layer_selected)

        self.load_btn = QPushButton("Load DICOM Folder")
        self.load_btn.clicked.connect(self.load_dicom)

        self.remove_button = QPushButton("Remove Current Layer")
        self.remove_button.clicked.connect(self.remove_current_layer)

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

        # Static sliders for opacity and slice offset
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setMinimum(0)
        self.opacity_slider.setMaximum(100)
        self.opacity_slider.setValue(100)
        self.opacity_slider.valueChanged.connect(self.on_opacity_changed)

        self.slice_offset_slider = QSlider(Qt.Horizontal)
        self.slice_offset_slider.setMinimum(-100)
        self.slice_offset_slider.setMaximum(100)
        self.slice_offset_slider.setValue(0)
        self.slice_offset_slider.valueChanged.connect(self.on_slice_offset_changed)

        self.rt_dose_layer = None

        self.setup_ui()

    def setup_ui(self):
        """
             This method creates and arranges all UI components, including sliders,
             panels, and control buttons, and connects them to the viewer controller
             and the main window layout.
        """

        # Create slider container for opacity and offset sliders
        self.slider_container = QVBoxLayout()
        self.axial_controller.set_slider_container(self.slider_container)
        self.coronal_controller.set_slider_container(self.slider_container)
        self.sagittal_controller.set_slider_container(self.slider_container)

        # Compose controls layout
        controls = QVBoxLayout()
        controls.addWidget(self.load_btn)
        controls.addWidget(self.remove_button)
        controls.addWidget(QLabel("Select Layer:"))
        controls.addWidget(self.layer_list)
        controls.addWidget(self.reset_sliders_button)

        # Static sliders for opacity and slice offset
        controls.addWidget(QLabel("Layer Opacity"))
        controls.addWidget(self.opacity_slider)
        controls.addWidget(QLabel("Layer Slice Offset"))
        controls.addWidget(self.slice_offset_slider)

        #Rotation sliders
        controls.addWidget(QLabel("Rotation Controls"))
        controls.addWidget(self.rotation_panel)

        #translation sliders
        controls.addWidget(QLabel("Translation Controls"))
        controls.addWidget(self.translation_panel)

        #Zoom in extra container
        controls.addWidget(QLabel("Zoom"))
        controls.addWidget(self.zoom_panel)

        # Compose main layout
        viewer_layout = QVBoxLayout()

        # Top row with Axial and Coronal views
        top_row = QHBoxLayout()
        top_row.addWidget(self.multi_view.axial_viewer)
        top_row.addWidget(self.multi_view.coronal_viewer)
        viewer_layout.addLayout(top_row)

        # Bottom row with Sagittal view centered
        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self.multi_view.sagittal_viewer)
        viewer_layout.addLayout(bottom_row)
        bottom_row.addStretch()

        main_layout = QHBoxLayout()
        main_layout.addLayout(controls, 2)
        main_layout.addLayout(viewer_layout, 5)

        # Add to the container
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def load_dicom(self):
        """
            Loads a DICOM folder and adds it as a new layer to the viewer.

            Prompts the user to select a DICOM folder, loads the volume using the viewer
            controller, and updates the layer list and controls if successful.

            Returns:
                None
            """
        # Open a dialog for the user to select a DICOM folder
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if not folder:
            return

        # Load the DICOM data once
        result_axial = self.axial_controller.load_dicom_folder(folder)
        if result_axial is None:
            return
        name, layer_axial, slider_rows = result_axial

        # Load the same DICOM data into coronal and sagittal controllers (no controls)
        result_coronal = self.coronal_controller.load_dicom_folder(folder, add_controls=False)
        result_sagittal = self.sagittal_controller.load_dicom_folder(folder, add_controls=False)

        # Reset zoom to 1.0 for all viewers
        for viewer in [self.multi_view.axial_viewer, self.multi_view.coronal_viewer, self.multi_view.sagittal_viewer]:
            viewer.view.resetTransform()
        self.current_zoom = 1.0
        self.zoom_panel.set_zoom(1.0)

        # Select the newly added layer in all views and update the UI
        self._extracted_from_on_layer_selected_27(0)
        self.layer_list.addItem(name)
        self.layer_list.setCurrentRow(self.layer_list.count() - 1)
        self.layer_slider_rows[name] = slider_rows
        self.update_layer_controls()

    def on_layer_selected(self, index):
        """
              Updates the selected layer in the viewer controller and refreshes the layer
              controls accordingly.

              Args:
                  index: The index of the newly selected layer.
              """
        # Getting all images and putting them in the selected layer
        self._extracted_from_on_layer_selected_27(index)
        # Adding the highlighted border for teh corresponding layer
        highlight_selected_layer(self.axial_controller.volume_layers, index)
        self.update_layer_controls()

    # TODO Rename this here and in `load_dicom` and `on_layer_selected`
    def _extracted_from_on_layer_selected_27(self, arg0):
        self.axial_controller.selected_layer_index = arg0
        self.coronal_controller.selected_layer_index = arg0
        self.sagittal_controller.selected_layer_index = arg0
        self.axial_controller.update_display()
        self.coronal_controller.update_display()
        self.sagittal_controller.update_display()

    def update_layer_controls(self):
        """
                Updates the rotation, translation, opacity, and slice offset controls to match the currently
                selected layer.

                If no layer is selected, resets the controls to default values.
                Otherwise, sets the controls to reflect the rotation and offset of the
                selected image layer.
        """
        if self.axial_controller.selected_layer_index is None:
            self.rotation_panel.set_rotations([0, 0, 0])
            self.translation_panel.set_offsets([0, 0])
            self.opacity_slider.setValue(100)
            self.slice_offset_slider.setValue(0)
        else:
            layer = self.axial_controller.volume_layers[self.axial_controller.selected_layer_index]
            self.rotation_panel.set_rotations(layer.rotation)
            self.translation_panel.set_offsets(layer.offset)
            self.opacity_slider.setValue(int(layer.opacity * 100))
            self.slice_offset_slider.setValue(layer.slice_offset)

    def on_rotation_changed(self, axis_index, value):
        """
                This method synchronizes the rotation for the specified axis across the
                axial, coronal, and sagittal controllers.

                Args:
                    axis_index: The index of the rotation axis (0 for LR, 1 for PA,
                    2 for IS).
                    value: The new rotation value in degrees.
                """
        index = self.axial_controller.selected_layer_index
        if index is None:
            return
        for controller in [self.axial_controller, self.coronal_controller, self.sagittal_controller]:
            if index < len(controller.volume_layers):
                controller.volume_layers[index].rotation[axis_index] = value
                controller.volume_layers[index].cached_rotated_volume = None
                controller.update_display()

    def remove_current_layer(self):
        """
        Removes the currently selected image layer from the viewer and updates
        the UI.

        This method ensures both the internal data and UI elements (like sliders)
        are cleaned up properly.
        """
        index = self.axial_controller.selected_layer_index
        if index is None:
            return

        item = self.layer_list.item(index)
        if item is None:
            return
        layer_name = item.text()

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

        # Remove the image layer from all views
        self.axial_controller.remove_current_layer()
        self.coronal_controller.remove_current_layer()
        self.sagittal_controller.remove_current_layer()

        # Remove the name from the list
        self.layer_list.takeItem(index)

        remaining = self.layer_list.count()

        # Update selected_layer_index safely for all views
        if remaining == 0:
            self.axial_controller.selected_layer_index = None
            self.coronal_controller.selected_layer_index = None
            self.sagittal_controller.selected_layer_index = None
        else:
            # If removed last item, select previous, else select same index
            new_index = min(index, remaining - 1)
            self.layer_list.setCurrentRow(new_index)
            self.axial_controller.selected_layer_index = new_index
            self.coronal_controller.selected_layer_index = new_index
            self.sagittal_controller.selected_layer_index = new_index

        # Refresh controls
        self.update_layer_controls()
        # Also update all views
        self.axial_controller.update_display()
        self.coronal_controller.update_display()
        self.sagittal_controller.update_display()


    def on_offset_changed(self, offset):
        """
            This method updates the translation of the currently selected image layer
            in the viewer controller when the translation panel's offset is changed.
        """
        for controller in [self.axial_controller, self.coronal_controller, self.sagittal_controller]:
            controller.update_translation(offset)

    def on_opacity_changed(self, value):
        """
            Updates the opacity of the currently selected layer in all views.
        """
        index = self.axial_controller.selected_layer_index
        if index is None:
            return
        for controller in [self.axial_controller, self.coronal_controller, self.sagittal_controller]:
            layer = controller.volume_layers[index]
            layer.opacity = value / 100.0
            controller.update_display()

    def on_slice_offset_changed(self, value):
        """
            Updates the slice offset of the currently selected layer in all views.
        """
        index = self.axial_controller.selected_layer_index
        if index is None:
            return
        for controller in [self.axial_controller, self.coronal_controller, self.sagittal_controller]:
            layer = controller.volume_layers[index]
            layer.slice_offset = value
            controller.update_global_slice_slider_range()
            controller.update_display()

    def reset_zoom(self):
        """
                This method resets the view's transformation, sets the internal zoom
                state to 1.0, and updates the zoom panel UI.
                """
        self.graphics_view.resetTransform()
        self.current_zoom = 1.0
        self.zoom_panel.set_zoom(1.0)

    def on_zoom_changed(self, new_zoom):
        """
            This method applies the new zoom factor relative to the current zoom,
            updating the internal zoom state.

            Args:
                new_zoom: The new zoom factor to apply to the graphics view.
        """
        scale_factor = new_zoom / self.current_zoom
        for viewer in [self.multi_view.axial_viewer, self.multi_view.coronal_viewer, self.multi_view.sagittal_viewer]:
            viewer.view.scale(scale_factor, scale_factor)
        self.current_zoom = new_zoom

    def reset_layer_controls(self):
        """
            This method restores the layer's rotation, translation, opacity,
            and slice offset, and updates the UI controls accordingly.
        """
        index = self.axial_controller.selected_layer_index
        if index is None:
            return

        for controller in [self.axial_controller, self.coronal_controller, self.sagittal_controller]:
            if controller.selected_layer_index is None:
                continue

            layer = controller.volume_layers[index]

            # Reset internal values
            layer.rotation = [0, 0, 0]
            layer.offset = (0, 0)
            layer.slice_offset = 0
            layer.opacity = 1.0
            layer.cached_rotated_volume = None

            controller.update_global_slice_slider_range()
            controller.update_display()

        # Reset UI controls
        self.rotation_panel.reset_rotation()
        self.translation_panel.reset_trans()
        self.opacity_slider.setValue(100)
        self.slice_offset_slider.setValue(0)

        self.zoom_panel.set_zoom(1.0)
        self.on_zoom_changed(1.0)

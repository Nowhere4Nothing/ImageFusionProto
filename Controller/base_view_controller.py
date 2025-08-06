import numpy as np
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QSlider
from utils.image_processing import process_layers
from utils.layer_loader import load_dicom_layer

class BaseViewerController:
    """
        Controls the logic and state for a single DICOM image viewer panel.

        This class manages loading DICOM volumes, handling image layers, updating display
        properties,
        and synchronizing UI controls with the underlying data for a specific view type
        (axial, coronal, or sagittal).
        """
    def __init__(self, view, scene, view_type):
        self.view = view
        self.scene = scene
        self.view_type = view_type

        self.volume_layers = []
        self.selected_layer_index = None
        self.slice_index = 0
        self.global_slice_offset = 0

        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self.update_display)

        self.slider_container = None
        self.slice_slider = None
        self.initial_slice_slider_value = 0

    def set_slider_container(self, layout):
        self.slider_container = layout

    def set_slice_slider(self, slider: QSlider):
        self.slice_slider = slider

    def load_dicom_folder(self, folder):
        """
                Loads a DICOM volume from the specified folder and adds it as a new layer.

                This method creates the layer, sets up its UI controls,
                updates the internal state, and refreshes the display for the current view type.

                Args:
                    folder: Path to the folder containing the DICOM files.

                Returns:
                    tuple: (name, layer, slider_rows) if successful, otherwise None.
                """
        # Load the DICOM volume and create a new layer and its UI controls
        layer, name, slider_rows = load_dicom_layer(
            folder,
            self.slider_container,
            self.update_opacity,
            self.update_slice_offset,
            update_display_cb=self.update_display,
        )

        #adding fail-safes
        if layer is None:
            return None

        # Add the layer's UI container to the slider container if present
        if self.slider_container and hasattr(layer, 'ui_container'):
            self.slider_container.addWidget(layer.ui_container)

        # Store slider rows and add the new layer to the internal list
        layer.slider_rows = slider_rows
        self.volume_layers.append(layer)
        self.selected_layer_index = len(self.volume_layers) - 1

        #store the initial value of the slice slider
        if self.slice_slider:
            self.initial_slice_slider_value = self.slice_slider.value()

        # Set the slice index to the middle of the volume for the current view type
        layer = self.volume_layers[self.selected_layer_index]
        depth = (
            layer.data.shape[0] if self.view_type == "axial" else
            layer.data.shape[1] if self.view_type == "coronal" else
            layer.data.shape[2]
        )
        self.slice_index = depth // 2
        if self.slice_slider:
            self.slice_slider.setValue(self.slice_index + self.global_slice_offset)

        #update the global slice range
        self.update_global_slice_slider_range()
        self.update_display()

        return name, layer, slider_rows

    def update_opacity(self, layer, value):
        print(f"Opacity value raw: {value}")
        layer.opacity = value /100
        self.update_display()

    def update_slice_offset(self, layer, value):
        layer.slice_offset = value
        self.update_global_slice_slider_range()
        self.update_display()

    def update_rotation(self, axis_index, value):
        if self.selected_layer_index is None:
            return
        layer = self.volume_layers[self.selected_layer_index]
        layer.rotation[axis_index] = value
        layer.cached_rotated_volume = None
        self._update_timer.start(150)

    def update_translation(self, offset):
        if self.selected_layer_index is None:
            return
        layer = self.volume_layers[self.selected_layer_index]
        layer.offset = offset
        self.update_display()

    def on_slice_change(self, value):
        self.slice_index = value - self.global_slice_offset
        self.update_display()

    def update_display(self):
        """
        Updates the display to show the current image slice for the selected layer and
        view type.

        This method clamps the slice index to valid bounds, processes the image layers
        to generate the current slice,
        normalizes the image for display, and updates the scene and view to show the
        resulting pixmap.
        """
        if not self.volume_layers:
            self.scene.clear()
            return

        layer = self.volume_layers[self.selected_layer_index]
        max_slice = (
            layer.data.shape[0] if self.view_type == "axial" else
            layer.data.shape[1] if self.view_type == "coronal" else
            layer.data.shape[2]
        )
        clamped_index = np.clip(self.slice_index, 0, max_slice - 1)

        if self.slice_index != clamped_index:
            print(f"Clamping {self.view_type} index from {self.slice_index} to {clamped_index}")
            self.slice_index = clamped_index

        img = process_layers(self.volume_layers, self.slice_index, view_type=self.view_type)

        # Normalize to 0-255 for grayscale QImage
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)

        height, width = img.shape
        qimage = QImage(img.data, width, height, width, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        self.scene.clear()
        pixmap_item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(pixmap_item.boundingRect())

            # Centered and scaled
        self.view.setAlignment(Qt.AlignCenter)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        # Print statements for debugging
        # print(f"{self.view_type.title()} pixmap size: {pixmap.width()}x{pixmap.height()}")
        # print(f"View viewport size: {self.view.viewport().size()}")
        # print(f"Scene rect: {self.scene.sceneRect()}")


    def update_global_slice_slider_range(self):
        if not self.volume_layers or not self.slice_slider:
            return

        min_index = float('inf')
        max_index = float('-inf')

        for layer in self.volume_layers:
            if self.view_type == "axial":
                dim = layer.data.shape[0]
            elif self.view_type == "coronal":
                dim = layer.data.shape[1]
            elif self.view_type == "sagittal":
                dim = layer.data.shape[2]
            else:
                continue

            offset = getattr(layer, 'slice_offset', 0)
            layer_min = offset
            layer_max = dim - 1 + offset

            min_index = min(min_index, layer_min)
            max_index = max(max_index, layer_max)

        # Compute offset so slider starts at slice 0
        self.global_slice_offset = -min_index
        slider_min = 0
        slider_max = max_index - min_index

        self.slice_slider.setMinimum(slider_min)
        self.slice_slider.setMaximum(slider_max)

        if not (slider_min <= self.slice_slider.value() <= slider_max):
            self.slice_slider.setValue(slider_min)

        self.slice_index = self.slice_slider.value() - self.global_slice_offset

    def select_layer(self, index):
        if 0 <= index < len(self.volume_layers):
            self.selected_layer_index = index
        else:
            self.selected_layer_index = None
        self.update_display()

    def remove_current_layer(self):
        index = self.selected_layer_index
        if index is None or not (0 <= index < len(self.volume_layers)):
            return

        self.volume_layers.pop(index)
        if len(self.volume_layers) == 0:
            self.selected_layer_index = None
        elif index >= len(self.volume_layers):
            self.selected_layer_index = len(self.volume_layers) - 1
        else:
            self.selected_layer_index = index

        self.update_global_slice_slider_range()
        self.update_display()

    def reset_zoom(self):
        self.scene.views()[0].resetTransform()

    def reset_global_slice_slider(self):
        """
        Resets the global slice slider to its initial position.
        """
        if self.slice_slider:
            self.slice_slider.setValue(self.initial_slice_slider_value)
            self.slice_index = self.slice_slider.value() - self.global_slice_offset
            self.update_display()


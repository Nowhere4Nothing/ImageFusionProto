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
        """
               This method allows the viewer controller to store a reference to the layout where UI elements
               (such as opacity and offset sliders) for each image layer will be added.
               """
        self.slider_container = layout

    def set_slice_slider(self, slider: QSlider):
        """
                This method stores a reference to the provided slider, allowing the controller
                to update and respond to slice changes.
                """
        self.slice_slider = slider

    def load_dicom_folder(self, folder):
        """
                Loads a DICOM volume from the specified folder and adds it as a new layer.

                This method creates the layer, sets up its UI controls,
                updates the internal state, and refreshes the display for the current view type.

                Args:
                    folder: Path to the folder containing the DICOM files.
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
        """
                This method is called when the opacity slider changes, setting the
                layer's opacity property and updating the view.
                """
        print(f"Opacity value raw: {value}")
        layer.opacity = value /100
        self.update_display()

    def update_slice_offset(self, layer, value):
        """
                This method is called when the slice offset slider changes,
                updating the layer's slice offset,
                recalculating the global slice slider range, and updating the view.
                """
        layer.slice_offset = value
        self.update_global_slice_slider_range()
        self.update_display()

    def update_rotation(self, axis_index, value):
        """
                This method sets the rotation for the given axis, invalidates any
                cached rotated volume, and starts a timer to trigger a delayed display
                update.
                """
        if self.selected_layer_index is None:
            return
        layer = self.volume_layers[self.selected_layer_index]
        layer.rotation[axis_index] = value
        layer.cached_rotated_volume = None
        self._update_timer.start(150)

    def update_translation(self, offset):
        """
                This method sets the offset property of the currently selected layer
                to the provided value and updates the view.
                """
        if self.selected_layer_index is None:
            return
        layer = self.volume_layers[self.selected_layer_index]
        layer.offset = offset
        self.update_display()

    def on_slice_change(self, value):
        """
                This method adjusts the internal slice index based on the global offset and
                refreshes the display to show the new slice.
                """
        self.slice_index = value - self.global_slice_offset
        self.update_display()

    def update_display(self):
        """
        This method clamps the slice index to valid bounds, processes the image layers
        to generate the current slice,
        normalizes the image for display, and updates the scene and view to show the
        resulting pixmap.
        """

        # If there are no layers, clear the scene and exit
        if not self.volume_layers:
            self.scene.clear()
            return

        # Get the currently selected layer
        layer = self.volume_layers[self.selected_layer_index]
        # Determine the maximum valid slice index for the current view type
        max_slice = (
            layer.data.shape[0] if self.view_type == "axial" else
            layer.data.shape[1] if self.view_type == "coronal" else
            layer.data.shape[2]
        )
        # Clamp the slice index to valid bounds
        clamped_index = np.clip(self.slice_index, 0, max_slice - 1)

        # If the slice index was out of bounds, update it and print a debug message
        if self.slice_index != clamped_index:
            print(f"Clamping {self.view_type} index from {self.slice_index} to {clamped_index}")
            self.slice_index = clamped_index

        # Process the image layers to generate the current 2D slice
        img = process_layers(self.volume_layers, self.slice_index, view_type=self.view_type)

        # Normalize to 0-255 for grayscale QImage
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        img = (img * 255).astype(np.uint8)

        # Convert the NumPy array to a QImage and then to a QPixmap
        height, width = img.shape
        qimage = QImage(img.data, width, height, width, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        # Clear the scene and add the new pixmap
        self.scene.clear()
        pixmap_item = self.scene.addPixmap(pixmap)
        self.scene.setSceneRect(pixmap_item.boundingRect())

        # Center and scale the view to fit the scene
        self.view.setAlignment(Qt.AlignCenter)
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)

        # Print statements for debugging
        # print(f"{self.view_type.title()} pixmap size: {pixmap.width()}x{pixmap.height()}")
        # print(f"View viewport size: {self.view.viewport().size()}")
        # print(f"Scene rect: {self.scene.sceneRect()}")

    def update_global_slice_slider_range(self):
        """
                Updates the range and offset of the global slice slider based on all
                loaded layers.

                This method calculates the minimum and maximum valid slice indices
                across all layers for the current view type,
                sets the slider's minimum and maximum accordingly, and adjusts the
                global slice offset so that the slider starts at zero.
                It also ensures the current slider value is within the valid range and
                updates the internal slice index.
                """
        if not self.volume_layers or not self.slice_slider:
            return

        min_index = float('inf')
        max_index = float('-inf')

        # Determine the minimum and maximum slice indices across all layers
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

        # Set the slider's minimum and maximum values
        self.slice_slider.setMinimum(slider_min)
        self.slice_slider.setMaximum(slider_max)

        # Ensure the current slider value is within the valid range
        if not (slider_min <= self.slice_slider.value() <= slider_max):
            self.slice_slider.setValue(slider_min)

        # Update the internal slice index based on the slider value and global offset
        self.slice_index = self.slice_slider.value() - self.global_slice_offset

    def select_layer(self, index):
        """
                This method sets the selected layer index to the given value if it is
                valid, or clears the selection otherwise, and refreshes the view.

                """
        if 0 <= index < len(self.volume_layers):
            self.selected_layer_index = index
        else:
            self.selected_layer_index = None
        self.update_display()

    def remove_current_layer(self):
        """
                This method deletes the selected layer from the internal list,
                updates the selected layer index,
                recalculates the global slice slider range, and refreshes
                the display to reflect the change.
                """

        # Get the index of the currently selected layer
        index = self.selected_layer_index

        # If no valid selection, do nothing
        if index is None or not (0 <= index < len(self.volume_layers)):
            return

        # Remove the layer from the list
        self.volume_layers.pop(index)

        # Update the selected layer index based on the new list length
        if len(self.volume_layers) == 0:
            self.selected_layer_index = None
        elif index >= len(self.volume_layers):
            self.selected_layer_index = len(self.volume_layers) - 1
        else:
            self.selected_layer_index = index

        # Update the global slice slider range and refresh the display
        self.update_global_slice_slider_range()
        self.update_display()

    def reset_zoom(self):
        """
                This method resets any transformations applied to the view,
                restoring the original zoom and pan.
                """
        self.scene.views()[0].resetTransform()

    def reset_global_slice_slider(self):
        """
        Resets the global slice slider to its initial position.
        """
        if self.slice_slider:
            self.slice_slider.setValue(self.initial_slice_slider_value)
            self.slice_index = self.slice_slider.value() - self.global_slice_offset
            self.update_display()


from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QSlider

from utils.image_processing import process_layers
from utils.layer_loader import load_dicom_layer

class ViewerController:
    """
        Manages the logic and state for the DICOM image viewer.

        This class handles loading DICOM volumes, managing image layers, updating display properties,
        and synchronizing UI controls with the underlying data.
        """
    def __init__(self, scene, view, view_type="axial"):
        self.view_type = view_type

        self.initial_slice_slider_value = None
        self.scene = scene
        self.view = view

        self.volume_layers = []
        self.selected_layer_index = None
        self.slice_index = 0
        self.global_slice_offset = 0

        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self.update_display)

        self.slider_container = None  # To be set externally (QVBoxLayout)
        self.slice_slider = None  # To be set externally (QSlider)

    def set_slider_container(self, layout):
        self.slider_container = layout

    def set_slice_slider(self, slider: QSlider):
        self.slice_slider = slider
        self.slice_slider.valueChanged.connect(self.on_slice_change)

    def load_dicom_folder(self, folder):
        """
        Loads a DICOM volume from the specified folder and adds it as a new layer.
        This method creates the layer, sets up its UI controls, updates the internal state,
        and refreshes the display.
        Args:
            folder: Path to the folder containing the DICOM files.
        Returns:
                uple: (name, layer, slider_rows) if successful, otherwise None.
        """

        layer, name, slider_rows = load_dicom_layer(
            folder,
            self.slider_container,
            self.update_opacity,
            self.update_slice_offset,
            update_display_cb = self.update_display,
        )

        if layer is None:
            return None

        if self.slider_container is not None and hasattr(layer, 'ui_container'):
            self.slider_container.addWidget(layer.ui_container)

        layer.slider_rows = slider_rows

        self.volume_layers.append(layer)
        self.selected_layer_index = len(self.volume_layers) - 1
        self.update_global_slice_slider_range()

        if self.slice_slider is not None:
            self.initial_slice_slider_value = self.slice_slider.value()
        else:
            self.initial_slice_slider_value = 0

        self.update_display()

        # Store references to sliders for cleanup later
        layer.slider_rows = slider_rows

        return name, layer, slider_rows

    def update_opacity(self, layer, value):
        layer.opacity = value / 100.0
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
        self.update_display()

    def select_layer(self, index):
        if 0 <= index < len(self.volume_layers):
            self.selected_layer_index = index
        else:
            self.selected_layer_index = None

        self.highlight_selected_layer()
        self.update_display()

    def update_display(self):
        # if not self.volume_layers:
        #     self.scene.clear()
        #     return
        #
        # img = process_layers(self.volume_layers, self.slice_index, view_type=self.view_type)
        # h, w = img.shape
        # qimage = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        # pixmap = QPixmap.fromImage(qimage)
        #
        # self.scene.clear()
        # pixmap_item = self.scene.addPixmap(pixmap)
        #
        # if not hasattr(self, "has_fit_once"):
        #     self.view.fitInView(pixmap_item, Qt.KeepAspectRatio)
        #     self.has_fit_once = True

        if not self.volume_layers:
            self.scene.clear()
            return

        img = process_layers(self.volume_layers, self.slice_index, view_type=self.view_type)
        h, w = img.shape
        qimage = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        self.scene.clear()
        pixmap_item = self.scene.addPixmap(pixmap)

        # Fit view once after loading image to show full image:
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)


    def update_global_slice_slider_range(self):
        if self.slice_slider is None:
            return

        if not self.volume_layers:
            self.slice_slider.setMinimum(0)
            self.slice_slider.setMaximum(0)
            self.global_slice_offset = 0
            return

        min_index = min(0 + l.slice_offset for l in self.volume_layers)
        max_index = max((l.data.shape[0] - 1) + l.slice_offset for l in self.volume_layers)

        self.global_slice_offset = -min_index
        slider_min = 0
        slider_max = max_index - min_index

        self.slice_slider.setMinimum(slider_min)
        self.slice_slider.setMaximum(slider_max)

        if not (slider_min <= self.slice_slider.value() <= slider_max):
            self.slice_slider.setValue(slider_min)

        self.slice_index = self.slice_slider.value() - self.global_slice_offset

    def on_slice_change(self, value):
        self.slice_index = value - self.global_slice_offset
        self.update_display()

    def remove_current_layer(self):
        index = self.selected_layer_index
        if index is None or not (0 <= index < len(self.volume_layers)):
            return

        self.volume_layers.pop(index)

        # Remove corresponding sliders from UI (assumes 2 widgets per layer)
        #TODO If adding more sliders / layers Update
        for _ in range(4):
            if self.slider_container and self.slider_container.count() > 0:
                item = self.slider_container.takeAt(self.slider_container.count() - 1)
                if widget := item.widget():
                    widget.deleteLater()

        if len(self.volume_layers) == 0:
            self.selected_layer_index = None
        elif index >= len(self.volume_layers):
            self.selected_layer_index = len(self.volume_layers) - 1
        else:
            self.selected_layer_index = index

        self.update_global_slice_slider_range()
        self.update_display()

    def update_translation(self, offset):
        if self.selected_layer_index is None:
            return
        layer = self.volume_layers[self.selected_layer_index]
        print(f"Setting layer.offset = {offset}")
        layer.offset = offset
        self.update_display()

    def reset_zoom(self):
        self.scene.views()[0].resetTransform()

    def highlight_selected_layer(self):
        for i, layer in enumerate(self.volume_layers):
            if hasattr(layer, 'ui_container'):
                if i == self.selected_layer_index:
                    layer.ui_container.setStyleSheet(
                        "border: 2px solid #0078d7; padding: 4px; border-radius: 5px;"
                    )  # blue-ish highlight
                else:
                    layer.ui_container.setStyleSheet(
                        "border: 1px solid gray; padding: 4px; border-radius: 5px;"
                    )

    def reset_global_slice_slider(self):
        """
        Resets the global slice slider to cover the full slice range based on current layers,
        sets the slider value to minimum (start),
        and triggers an update of the displayed image.
        """
        self.update_global_slice_slider_range()  # update min/max ranges

        # Set slider to minimum value safely
        if self.slice_slider:
            self.slice_slider.blockSignals(True)
            self.slice_slider.setValue(self.initial_slice_slider_value)
            self.slice_slider.blockSignals(False)

        # Update internal slice index accordingly
        self.slice_index = self.slice_slider.value() - self.global_slice_offset

        # Trigger a display update
        self.update_display()

    def set_view_type(self, view_type: str):
        if view_type != "axial":
            raise ValueError("This controller is only for axial view.")
        self.view_type = view_type
        self.update_display()
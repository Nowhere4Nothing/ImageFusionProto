from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap, QBrush, QColor
from PySide6.QtWidgets import QSlider, QLabel, QHBoxLayout

from utils.image_processing import process_layers
from utils.layer_loader import load_dicom_layer

class ViewerController:
    """
        Manages the logic and state for the DICOM image viewer.

        This class handles loading DICOM volumes, managing image layers, updating display properties,
        and synchronizing UI controls with the underlying data.
        """
    def __init__(self, scene):
        self.scene = scene

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
        layer, name, slider_rows = load_dicom_layer(
            folder,
            self.slider_container,
            self.update_opacity,
            self.update_slice_offset,
            update_display_cb = self.update_display,
        )
        if layer is None:
            return None

        layer.slider_rows = slider_rows

        self.volume_layers.append(layer)
        self.selected_layer_index = len(self.volume_layers) - 1
        self.update_global_slice_slider_range()
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
        if not self.volume_layers:
            self.scene.clear()
            return

        img = process_layers(self.volume_layers, self.slice_index)
        h, w = img.shape
        qimage = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(pixmap.width() * 2, pixmap.height() * 2, Qt.AspectRatioMode.KeepAspectRatio)

        self.scene.clear()
        self.scene.addPixmap(scaled_pixmap)

    def update_global_slice_slider_range(self):
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
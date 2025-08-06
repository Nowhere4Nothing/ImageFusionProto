from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QSlider

from utils.image_processing import process_layers
from utils.layer_loader import load_dicom_layer


def resample_volume_to_isotropic(volume, spacing, new_spacing=1.0):
    import SimpleITK as sitk
    sitk_image = sitk.GetImageFromArray(volume)
    original_spacing = spacing  # tuple (sx, sy, sz)
    original_size = sitk_image.GetSize()

    new_size = [
        int(round(original_size[0] * (original_spacing[0] / new_spacing))),
        int(round(original_size[1] * (original_spacing[1] / new_spacing))),
        int(round(original_size[2] * (original_spacing[2] / new_spacing))),
    ]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing((new_spacing, new_spacing, new_spacing))
    resampler.SetSize(new_size)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampled_image = resampler.Execute(sitk_image)
    return sitk.GetArrayFromImage(resampled_image)


class BaseViewerController:
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
        layer, name, slider_rows = load_dicom_layer(
            folder,
            self.slider_container,
            self.update_opacity,
            self.update_slice_offset,
            update_display_cb=self.update_display,
        )

        if layer is None:
            return None

        if self.slider_container and hasattr(layer, 'ui_container'):
            self.slider_container.addWidget(layer.ui_container)

        layer.slider_rows = slider_rows
        self.volume_layers.append(layer)
        self.selected_layer_index = len(self.volume_layers) - 1

        if self.slice_slider:
            self.initial_slice_slider_value = self.slice_slider.value()

        self.update_global_slice_slider_range()
        self.update_display()

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

    def update_translation(self, offset):
        if self.selected_layer_index is None:
            return
        layer = self.volume_layers[self.selected_layer_index]
        # Ensure offset is always [x, y, z]
        if len(offset) == 2:
            layer.offset = [offset[0], offset[1], 0]
        else:
            layer.offset = list(offset)
        self.update_display()

    def on_slice_change(self, value):
        self.slice_index = value - self.global_slice_offset
        self.update_display()

    def update_display(self):
        if not self.volume_layers:
            self.scene.clear()
            return

        img = process_layers(self.volume_layers, self.slice_index, view_type=self.view_type)
        h, w = img.shape
        qimage = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)

        self.scene.clear()
        pixmap_item = self.scene.addPixmap(pixmap)

        # Set scene rect exactly to pixmap size
        self.scene.setSceneRect(pixmap_item.boundingRect())

        print(f"Pixmap size: {pixmap.width()}x{pixmap.height()}")
        print(f"View viewport size: {self.view.viewport().size()}")
        print(f"Scene rect: {self.scene.sceneRect()}")
        # Fit the scene in the view to avoid clipping
        self.view.fitInView(self.scene.sceneRect(), Qt.KeepAspectRatio)


    def update_global_slice_slider_range(self):
        if not self.volume_layers or not self.slice_slider:
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



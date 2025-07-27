import os
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QListWidget, QGraphicsView, QGraphicsScene,
    QFrame, QSizePolicy
)
from PySide6.QtGui import QImage, QPixmap, QPainter, QBrush, QColor
from PySide6.QtCore import Qt, QEvent, QTimer

from volume_layer import VolumeLayer
from utils.dicom_loader import load_dicom_volume
from utils.image_processing import process_layers


class RotationControlPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sliders = []
        layout = QVBoxLayout(self)
        for i, axis in enumerate(["LR", "PA", "IS"]):
            layout.addWidget(QLabel(f"{axis} Rotation:"))
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-180, 180)

            slider.valueChanged.connect(lambda v, idx=i: parent.preview_rotation(idx, v))

            # apply final update only on release
            slider.sliderReleased.connect(lambda idx=i, s=slider: parent.update_rotation(idx, s.value()))

            layout.addWidget(slider)
            self.sliders.append(slider)

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("border:1px solid gray; padding:4px; border-radius:5px;")

class DicomViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("manual image fusion example")

        self.volume_layers = []
        self.selected_layer_index = None
        self.slice_index = 0
        self.global_slice_offset = 0

        self._update_timer = QTimer()
        self._update_timer.setSingleShot(True)
        self._update_timer.timeout.connect(self.update_display)

        # === UI and layout setup ===
        self.setup_ui()

        self.dragging = False
        self.last_mouse_pos = None
        self.graphics_view.viewport().installEventFilter(self)

    def _init_rotation_panel(self):
        self.rotation_panel = RotationControlPanel(self)
        self.rotation_sliders = self.rotation_panel.sliders

    def preview_rotation(self, axis_index, value):
        # For now just print, or do a lightweight update if you want
        print(f"Preview rotation axis {axis_index} = {value}")

    def setup_ui(self):
        self._init_graphics_view()
        self._init_layer_list()
        self._init_rotation_panel()
        self.slider_container = QVBoxLayout()
        self._init_global_slice_slider()

        controls = QVBoxLayout()
        for w in (
                self.load_btn,
                self.toggle_visibility_button,
                self.remove_button,
                QLabel("Select Layer:"), self.layer_list,
                QLabel("Rotation Controls"), self.rotation_panel,
                self.slider_container,
                QLabel("Global Slice"), self.slice_slider
        ):
            controls.addWidget(w) if isinstance(w, QWidget) else controls.addLayout(w)

        layout = QHBoxLayout()
        layout.addLayout(controls, 1)
        layout.addWidget(self.graphics_view, 4)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def _init_graphics_view(self):
        self.scene = QGraphicsScene()
        self.graphics_view = QGraphicsView(self.scene)
        self.graphics_view.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        self.graphics_view.setRenderHints(QPainter.Antialiasing |
                                          QPainter.SmoothPixmapTransform)

    def _init_layer_list(self):
        self.layer_list = QListWidget()
        self.layer_list.currentRowChanged.connect(self.select_layer)
        self.load_btn = QPushButton("Load DICOM Folder")
        self.load_btn.clicked.connect(self.load_dicom)
        self.remove_button = QPushButton("Remove Current Layer")
        # self.remove_button.clicked.connect(...)
        self.toggle_visibility_button = QPushButton("Hide Current Layer")
        self.remove_button.clicked.connect(self.remove_current_layer)
        # self.toggle_visibility_button.clicked.connect(self.toggle_current_layer_visibility)

    def _init_global_slice_slider(self):
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(100)
        self.slice_slider.setValue(50)
        self.slice_slider.valueChanged.connect(self.on_slice_change)

    # def update_rotation(self, axis_index, value):
    #     print(f"Rotation axis {axis_index} set to {value} degrees.")

    def load_dicom(self):
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if not folder:
            return

        volume = load_dicom_volume(folder)
        if volume is None:
            return

        layer = VolumeLayer(volume, os.path.basename(folder))
        self.volume_layers.append(layer)
        self.layer_list.addItem(layer.name)

        opacity_slider = self._extracted_from_load_dicom_15()
        self._extracted_from_load_dicom_18(opacity_slider, 100, 'Opacity: ', layer)
        slice_offset_slider = self._extracted_from_load_dicom_19(volume)
        self._extracted_from_load_dicom_18(
            slice_offset_slider, 0, 'Slice Offset: ', layer
        )
        self.selected_layer_index = len(self.volume_layers) - 1
        self.update_rotation_sliders()

        opacity_slider = self._extracted_from_load_dicom_15()
        opacity_slider.setValue(100)
        # opacity_slider.valueChanged.connect(lambda val, layer=layer: self.update_opacity(layer, val))
        self.slider_container.addWidget(QLabel(f"Opacity: {layer.name}"))
        self.slider_container.addWidget(opacity_slider)

        slice_offset_slider = self._extracted_from_load_dicom_19(volume)
        slice_offset_slider.setValue(0)
        slice_offset_slider.setSingleStep(1)
        # slice_offset_slider.valueChanged.connect(lambda val, layer=layer: self.update_slice_offset(layer, val))
        self.slider_container.addWidget(QLabel(f"Slice Offset: {layer.name}"))
        self.slider_container.addWidget(slice_offset_slider)

        self.update_global_slice_slider_range()
        self.update_display()

    # TODO Rename this here and in `load_dicom`
    def _extracted_from_load_dicom_15(self):
        # Setup opacity and slice offset sliders
        result = QSlider(Qt.Orientation.Horizontal)
        result.setMinimum(1)
        result.setMaximum(100)
        return result

    # TODO Rename this here and in `load_dicom`
    def _extracted_from_load_dicom_19(self, volume):
        result = QSlider(Qt.Orientation.Horizontal)
        result.setMinimum(-volume.shape[0] + 1)
        result.setMaximum(volume.shape[0] - 1)
        return result

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

    # TODO Rename this here and in `load_dicom`
    def _extracted_from_load_dicom_18(self, arg0, arg1, arg2, layer):
        arg0.setValue(arg1)
        # opacity_slider.valueChanged.connect(lambda val, layer=layer: self.update_opacity(layer, val))
        self.slider_container.addWidget(QLabel(f"{arg2}{layer.name}"))
        self.slider_container.addWidget(arg0)

    def select_layer(self, index):
        self.selected_layer_index = index
        self.update_display()

    def update_display(self):
        img = process_layers(self.volume_layers, self.slice_index)
        h, w = img.shape
        qimage = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
        pixmap = QPixmap.fromImage(qimage)
        scaled_pixmap = pixmap.scaled(pixmap.width() * 2, pixmap.height() * 2, Qt.AspectRatioMode.KeepAspectRatio)
        self.scene.clear()
        self.scene.addPixmap(scaled_pixmap)

    def update_rotation_sliders(self):
        if self.selected_layer_index is None:
            for slider in self.rotation_sliders:
                slider.blockSignals(True)
                slider.setValue(0)
                slider.blockSignals(False)
            return
        layer = self.volume_layers[self.selected_layer_index]
        for i in range(3):
            self.rotation_sliders[i].blockSignals(True)
            self.rotation_sliders[i].setValue(layer.rotation[i])
            self.rotation_sliders[i].blockSignals(False)

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

    def select_layer(self, index):
        if 0 <= index < len(self.volume_layers):
            self.selected_layer_index = index
        else:
            self.selected_layer_index = None
        self.update_rotation_sliders()
        self.update_display()

    def remove_current_layer(self):
        index = self.selected_layer_index
        if index is None or not (0 <= index < len(self.volume_layers)):
            return

        self.volume_layers.pop(index)
        self.layer_list.takeItem(index)

        # Remove corresponding opacity and slice sliders (assumes 2 widgets per layer)
        for _ in range(4):
            if self.slider_container.count() > 0:
                if widget := self.slider_container.takeAt(
                    self.slider_container.count() - 1
                ).widget():
                    widget.deleteLater()

        if len(self.volume_layers) == 0:
            self.selected_layer_index = None
        elif index >= len(self.volume_layers):
            self.selected_layer_index = len(self.volume_layers) - 1
        else:
            self.selected_layer_index = index

        self.layer_list.setCurrentRow(self.selected_layer_index if self.selected_layer_index is not None else -1)

        self.update_rotation_sliders()
        self.update_global_slice_slider_range()
        self.update_display()

    def on_slice_change(self, value):
        self.slice_index = value - self.global_slice_offset
        self.update_display()

    # def update_display(self):
        # if not self.volume_layers:
        #     self.scene.clear()
        #     return
        #
        # base_shape = self.volume_layers[0].data[0].shape
        # img = np.zeros(base_shape, dtype=np.float32)
        #
        # for layer in self.volume_layers:
        #     if not layer.visible:
        #         continue
        #
        #     volume = layer.data.copy()  # Work on a copy to avoid modifying original

        #     # Apply IS rotation (around PA axis) - axes (0,1)
        #     if layer.rotation[2] != 0:
        #         volume = rotate(volume, angle=layer.rotation[2], axes=(0, 1), reshape=False, mode='nearest')
        #
        #     # Apply PA rotation (around LR axis) - axes (0,2)
        #     if layer.rotation[1] != 0:
        #         volume = rotate(volume, angle=layer.rotation[1], axes=(0, 2), reshape=False, mode='nearest')
        #
        #     # Apply LR rotation (in-plane rotation) - axes (1,2)
        #     if layer.rotation[0] != 0:
        #         volume = rotate(volume, angle=layer.rotation[0], axes=(1, 2), reshape=False, mode='nearest')
        #
        #     # Determine slice index after rotation
        #     slice_idx = np.clip(self.slice_index + layer.slice_offset, 0, volume.shape[0] - 1)
        #     overlay = volume[slice_idx]
        #
        #     # Apply translation
        #     offset_x, offset_y = layer.offset
        #     shifted = np.roll(overlay, shift=offset_x, axis=1)
        #     shifted = np.roll(shifted, shift=offset_y, axis=0)
        #
        #     # Composite layer using opacity
        #     img = img * (1 - layer.opacity) + shifted * layer.opacity
        #
        # # Normalize and display
        # img = np.clip(img, 0, 1)
        # img_uint8 = (img * 255).astype(np.uint8)
        # h, w = img.shape
        # qimage = QImage(img_uint8.data, w, h, w, QImage.Format.Format_Grayscale8)
        # pixmap = QPixmap.fromImage(qimage)
        # scaled_pixmap = pixmap.scaled(pixmap.width() * 2, pixmap.height() * 2, Qt.AspectRatioMode.KeepAspectRatio)
        #
        # self.scene.clear()
        # self.scene.addPixmap(scaled_pixmap)




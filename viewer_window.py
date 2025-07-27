import os
import numpy as np
from PySide6.QtWidgets import (
    QMainWindow, QWidget, QFileDialog, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QSlider, QListWidget, QGraphicsView, QGraphicsScene,
    QFrame, QSizePolicy
)
from PySide6.QtGui import QImage, QPixmap, QPainter, QBrush, QColor
from PySide6.QtCore import Qt, QEvent

from volume_layer import VolumeLayer
from utils.dicom_loader import load_dicom_volume
from utils.image_processing import process_layers


class DicomViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("manual image fusion example")

        self.volume_layers = []
        self.selected_layer_index = None
        self.slice_index = 0
        self.global_slice_offset = 0

        # === UI and layout setup ===
        self.setup_ui()

        self.dragging = False
        self.last_mouse_pos = None
        self.graphics_view.viewport().installEventFilter(self)

    def setup_ui(self):
        # === Graphics View ===
        self.scene = QGraphicsScene()
        self.graphics_view = QGraphicsView(self.scene)
        self.graphics_view.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        self.graphics_view.setRenderHints(
            QPainter.RenderHint.Antialiasing | QPainter.RenderHint.SmoothPixmapTransform
        )

        self.layer_list = QListWidget()
        self.layer_list.currentRowChanged.connect(self.select_layer)

        load_button = QPushButton("Load DICOM Folder")
        load_button.clicked.connect(self.load_dicom)

        remove_button = QPushButton("Remove Current Layer")
        remove_button.clicked.connect(self.remove_current_layer)

        self.toggle_visibility_button = QPushButton("Hide Current Layer")
        # self.toggle_visibility_button.clicked.connect(self.toggle_current_layer_visibility)

        self.slider_container = QVBoxLayout()

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        self.slice_slider.valueChanged.connect(self.on_slice_change)

        # Rotation sliders
        self.rotation_sliders = []
        self.rotation_slider_container = QVBoxLayout()
        for i, axis in enumerate(["LR", "PA", "IS"]):
            label = QLabel(f"{axis} Rotation:")
            slider = QSlider(Qt.Orientation.Horizontal)
            slider.setMinimum(-180)
            slider.setMaximum(180)
            slider.setValue(0)
            slider.valueChanged.connect(lambda val, axis_index=i: self.update_rotation(axis_index, val))
            self.rotation_slider_container.addWidget(label)
            self.rotation_slider_container.addWidget(slider)
            self.rotation_sliders.append(slider)

        rotation_frame = QFrame()
        rotation_frame.setLayout(self.rotation_slider_container)
        rotation_frame.setFrameShape(QFrame.Shape.StyledPanel)
        rotation_frame.setStyleSheet("QFrame { border: 1px solid gray; border-radius: 5px; padding: 4px; }")

        controls = QVBoxLayout()
        controls.addWidget(load_button)
        controls.addWidget(self.toggle_visibility_button)
        controls.addWidget(remove_button)
        controls.addWidget(QLabel("Select Layer:"))
        controls.addWidget(self.layer_list)
        controls.addWidget(QLabel("Rotation Controls"))
        controls.addWidget(rotation_frame)
        controls.addLayout(self.slider_container)
        controls.addWidget(QLabel("Global Slice"))
        controls.addWidget(self.slice_slider)

        layout = QHBoxLayout()
        layout.addLayout(controls, 1)
        layout.addWidget(self.graphics_view, 4)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

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



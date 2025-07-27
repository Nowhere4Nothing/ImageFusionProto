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
        # remove_button.clicked.connect(self.remove_current_layer)

        self.toggle_visibility_button = QPushButton("Hide Current Layer")
        # self.toggle_visibility_button.clicked.connect(self.toggle_current_layer_visibility)

        self.slider_container = QVBoxLayout()

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(0)
        self.slice_slider.setValue(0)
        # self.slice_slider.valueChanged.connect(self.on_slice_change)

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

        # Setup opacity and slice offset sliders
        opacity_slider = QSlider(Qt.Orientation.Horizontal)
        opacity_slider.setMinimum(1)
        opacity_slider.setMaximum(100)
        opacity_slider.setValue(100)
        # opacity_slider.valueChanged.connect(lambda val, layer=layer: self.update_opacity(layer, val))
        self.slider_container.addWidget(QLabel(f"Opacity: {layer.name}"))
        self.slider_container.addWidget(opacity_slider)

        slice_offset_slider = QSlider(Qt.Orientation.Horizontal)
        slice_offset_slider.setMinimum(-volume.shape[0] + 1)
        slice_offset_slider.setMaximum(volume.shape[0] - 1)
        slice_offset_slider.setValue(0)
        # slice_offset_slider.valueChanged.connect(lambda val, layer=layer: self.update_slice_offset(layer, val))
        self.slider_container.addWidget(QLabel(f"Slice Offset: {layer.name}"))
        self.slider_container.addWidget(slice_offset_slider)

        self.selected_layer_index = len(self.volume_layers) - 1
        self.update_rotation_sliders()
        self.update_global_slice_slider_range()
        self.update_display()

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



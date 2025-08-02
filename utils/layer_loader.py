import os
from PySide6.QtWidgets import QSlider, QLabel, QHBoxLayout
from PySide6.QtCore import Qt

from volume_layer import VolumeLayer
from utils.dicom_loader import load_dicom_volume

def setup_slider_ui(slider, default_value, label_prefix, layer_name, update_callback, container_layout):
    slider.setValue(default_value)
    row = QHBoxLayout()

    label = QLabel(f"{label_prefix}{layer_name}")
    label.setFixedWidth(120)  # set a max width that works for your UI
    label.setToolTip(label.text())  # show full text on hover
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
    label.setStyleSheet("QLabel { text-overflow: ellipsis; }")  # ellipsis won't work here directly

    # Instead of stylesheet, use elide text with QLabel's setText method:
    metrics = label.fontMetrics()
    elided_text = metrics.elidedText(label.text(), Qt.TextElideMode.ElideRight, label.width())
    label.setText(elided_text)

    value_label = QLabel()
    def update_label(val):
        if "Opacity" in label_prefix:
            value_label.setText(f"{val}%")
        else:
            value_label.setText(str(val))

    update_label(default_value)
    slider.valueChanged.connect(update_label)
    slider.valueChanged.connect(update_callback)

    row.addWidget(label)
    row.addWidget(slider)
    row.addWidget(value_label)

    if container_layout:
        container_layout.addLayout(row)

    return row

def load_dicom_layer(folder, container_layout, update_opacity_cb, update_offset_cb):
    """
    Loads a DICOM folder, creates a VolumeLayer, and adds sliders for opacity and offset.
    """
    volume = load_dicom_volume(folder)
    if volume is None:
        return None, None, []

    layer = VolumeLayer(volume, os.path.basename(folder))
    # Setup sliders
    opacity_slider = create_opacity_slider()
    opacity_row = setup_slider_ui(
        opacity_slider,
        100,
        "Opacity: ",
        layer.name,
        lambda val: update_opacity_cb(layer, val),
        container_layout,
    )

    offset_slider = create_slice_offset_slider(volume)
    offset_row = setup_slider_ui(
        offset_slider,
        0,
        "Slice Offset: ",
        layer.name,
        lambda val: update_offset_cb(layer, val),
        container_layout,
    )
    slider_rows = [offset_row, opacity_row]
    return layer, layer.name, slider_rows

def create_opacity_slider():
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(1)
    slider.setMaximum(100)
    return slider

def create_slice_offset_slider(volume):
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(-volume.shape[0] + 1)
    slider.setMaximum(volume.shape[0] - 1)
    return slider
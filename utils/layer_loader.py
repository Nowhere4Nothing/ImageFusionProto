import os
from PySide6.QtWidgets import QSlider, QLabel, QHBoxLayout, QFrame, QVBoxLayout, QCheckBox
from PySide6.QtCore import Qt

from volume_layer import VolumeLayer
from utils.dicom_loader import load_dicom_volume


def setup_slider_ui(slider, default_value, label_prefix, layer_name, update_callback, container_layout):
    slider.setValue(default_value)
    row = QHBoxLayout()

    label = QLabel()
    full_text = f"{label_prefix}{layer_name}"
    label.setToolTip(full_text)
    label.setFixedWidth(120)
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    def update_label_text():
        metrics = label.fontMetrics()
        elided = metrics.elidedText(full_text, Qt.TextElideMode.ElideRight, label.width())
        label.setText(elided)

    label.resizeEvent = lambda event: update_label_text()

    update_label_text()

    value_label = QLabel()

    def update_val_label(val):
        if "Opacity" in label_prefix:
            value_label.setText(f"{val}%")
        else:
            value_label.setText(str(val))

    update_val_label(default_value)
    slider.valueChanged.connect(update_val_label)
    slider.valueChanged.connect(update_callback)

    row.addWidget(label)
    row.addWidget(slider)
    row.addWidget(value_label)

    if container_layout:
        container_layout.addLayout(row)

    return row, slider, value_label

def load_dicom_layer(folder, container_layout, update_opacity_cb, update_offset_cb, update_display_cb=None):
    """
    Loads a DICOM folder, creates a VolumeLayer, and adds sliders for opacity and offset.
    """
    volume = load_dicom_volume(folder)
    if volume is None:
        return None, None, []

    layer = VolumeLayer(volume, os.path.basename(folder))

    # Create a frame to contain this layer's controls
    frame = QFrame()
    frame.setFrameShape(QFrame.StyledPanel)
    frame.setStyleSheet("border: 1px solid gray; padding: 4px; border-radius: 5px;")
    layer.ui_container = frame

    layout = QVBoxLayout(frame)

    # Visibility checkbox
    checkbox = QCheckBox("Hide Layer")
    layer.visible = True
    checkbox.setChecked(layer.visible)

    def on_toggle(visible: bool):
        # Toggle this layer's visibility based on checkbox state
        print(f"Checkbox toggled: visible = {visible}")
        layer.visible = visible

        if visible:
            checkbox.setText("Hide Layer")
        else:
            checkbox.setText("Show Layer")

        if update_display_cb:
            print("Calling update_display_cb()")
            update_display_cb()

    checkbox.toggled.connect(on_toggle)
    layout.addWidget(checkbox)

    # Setup sliders
    opacity_slider = create_opacity_slider()
    setup_slider_ui(
        opacity_slider,
        100,
        "Opacity: ",
        layer.name,
        lambda val: update_opacity_cb(layer, val),
        layout,
    )

    offset_slider = create_slice_offset_slider(volume)
    setup_slider_ui(
        offset_slider,
        0,
        "Slice Offset: ",
        layer.name,
        lambda val: update_offset_cb(layer, val),
        layout,
    )

    layer.opacity_slider = opacity_slider
    layer.offset_slider = offset_slider

    container_layout.addWidget(frame)

    return layer, layer.name, [frame]

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

def reset_opacity_and_offset(layer, opacity_slider, offset_slider, update_display_cb=None):
    # Reset internal layer values
    layer.opacity = 1.0
    layer.slice_offset = 0

    _reset_slider_value(opacity_slider, 100)
    _reset_slider_value(offset_slider, 0)
    # Call display update callback if provided
    if update_display_cb:
        update_display_cb()

def _reset_slider_value(arg0, arg1):
    # Reset sliders (which will update UI labels due to connected signals)
    arg0.blockSignals(True)
    arg0.setValue(arg1)
    arg0.blockSignals(False)
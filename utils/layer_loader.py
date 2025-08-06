import os
from PySide6.QtWidgets import QSlider, QLabel, QHBoxLayout, QFrame, QVBoxLayout, QCheckBox
from PySide6.QtCore import Qt

from volume_layer import VolumeLayer
from utils.dicom_loader import load_dicom_volume

def setup_slider_ui(slider, default_value, label_prefix, layer_name, update_callback, container_layout):
    """
        Sets up a horizontal slider UI row with a label and value display for a layer control.

        This function creates a labeled slider row, connects value change callbacks,
        and adds it to the provided container layout.

        Args:
            slider: The QSlider widget to configure.
            default_value: The initial value to set for the slider.
            label_prefix: Prefix text for the label (e.g., "Opacity: ").
            layer_name: Name of the layer to display in the label.
            update_callback: Function to call when the slider value changes.
            container_layout: The layout to which the slider row will be added.

        Returns:
            tuple: (row layout, slider, value label) for further customization or reference.
        """
    slider.setValue(default_value)
    row = QHBoxLayout()

    label = QLabel()
    full_text = f"{label_prefix}{layer_name}"
    label.setToolTip(full_text)
    label.setFixedWidth(120)
    label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

    def update_label_text():
        """
                Updates the label text to fit within the available width,
                using ellipsis if necessary.

                This function ensures the label displays as much of the full text as
                possible without overflowing.
                """
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

    #connection the slider to the values
    update_val_label(default_value)
    slider.valueChanged.connect(update_val_label)
    slider.valueChanged.connect(update_callback)

    #adding it all to the container
    row.addWidget(label)
    row.addWidget(slider)
    row.addWidget(value_label)

    if container_layout:
        container_layout.addLayout(row)

    return row, slider, value_label

def load_dicom_layer( folder,
    container_layout,
    update_opacity_cb,
    update_offset_cb,
    update_display_cb=None,):
    """
        Loads a DICOM volume from the specified folder and creates a new image layer with
        UI controls.

        This function sets up the layer's opacity and slice offset sliders,
        a visibility checkbox, and adds the controls to the provided container layout.

        Args:
            folder: Path to the folder containing the DICOM files.
            container_layout: The layout to which the layer's UI controls will be added.
            update_opacity_cb: Callback function to handle opacity changes.
            update_offset_cb: Callback function to handle slice offset changes.
            update_display_cb: Optional callback function to update the display when layer properties change.

        Returns:
            tuple: (layer, layer name, list of control frames) if successful, otherwise
            (None, None, []).
        """
    #loading the info
    volume,spacing = load_dicom_volume(folder)

    #fallbacks to avoid bugs
    if volume is None:
        return None, None, []

    layer = VolumeLayer(volume, os.path.basename(folder))
    layer.spacing = spacing

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
        """
            Handles the toggling of a layer's visibility when the checkbox state changes.

            Updates the layer's visibility property, adjusts the checkbox label, and triggers
            a display update if a callback is provided.

            Args:
                visible: Boolean indicating whether the layer should be visible.
            """
        # Toggle this layer's visibility based on checkbox state
        #TODO Get working for the sagittal and coronal images
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

    #creating the opacity slider
    opacity_slider = create_opacity_slider()
    setup_slider_ui(
        opacity_slider,
        100,
        "Opacity: ",
        layer.name,
        lambda val: update_opacity_cb(layer, val),
        layout,
    )

    #creating the offset slider
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

    # adding the container to the GUI
    if container_layout is not None:
        container_layout.addWidget(frame)

    return layer, layer.name, [frame]

def create_opacity_slider():
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(0)
    slider.setMaximum(10)
    return slider

def create_slice_offset_slider(volume):
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(-volume.shape[0] + 1)
    slider.setMaximum(volume.shape[0] - 1)
    return slider

def reset_opacity_and_offset(layer, opacity_slider, offset_slider, update_display_cb=None):
    """
        Resets the opacity and slice offset of a layer to their default values.

        This function also updates the corresponding sliders and triggers
        a display update if a callback is provided.

        Args:
            layer: The layer object whose properties are to be reset.
            opacity_slider: The QSlider controlling the layer's opacity.
            offset_slider: The QSlider controlling the layer's slice offset.
            update_display_cb: Optional callback function to update the display after resetting.
        """
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

def highlight_selected_layer(volume_layers, selected_index):
    """
        Highlights the UI container of the selected layer in the layer list.

        This function updates the border style of each layer's UI container to visually
        indicate which layer is currently selected.

        Args:
            volume_layers: List of layer objects, each potentially containing a UI container.
            selected_index: Index of the currently selected layer.
        """
    for i, layer in enumerate(volume_layers):
        if hasattr(layer, 'ui_container') and layer.ui_container:
            if i == selected_index:
                layer.ui_container.setStyleSheet(
                    #only a blue border as fill made the checkbox hard to see
                    "border: 2px solid #0078d7; padding: 4px; border-radius: 5px;"
                )
            else:
                layer.ui_container.setStyleSheet(
                    "border: 1px solid gray; padding: 4px; border-radius: 5px;"
                )




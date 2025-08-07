from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSlider
from PySide6.QtCore import Qt

class RotationControlPanel(QFrame):
    """
        Provides a user interface panel for controlling 3D rotation of an image layer.

        This class creates sliders for each rotation axis (LR, PA, IS), displays their
        values, and emits callbacks when the rotation changes.
        """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sliders = []
        self.value_labels = []

        # Create a vertical layout for the panel
        layout = QVBoxLayout(self)
        # Create sliders and labels for each rotation axis
        for i, axis in enumerate(["LR", "PA", "IS"]):
            row = QHBoxLayout()
            label = QLabel(f"{axis} Rotation:")
            value_label = QLabel("0°")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-90, 90)
            slider.setValue(0)

            # Function to update the value label when the slider changes
            def make_value_updater(lbl):
                return lambda v: lbl.setText(f"{v}°")

            slider.valueChanged.connect(make_value_updater(value_label))

            # Connect slider changes to the rotation change handler
            slider.valueChanged.connect(lambda v, idx=i: self.on_rotation_change(idx, v))

            # Add widgets to the row layout
            row.addWidget(label)
            row.addWidget(slider)
            row.addWidget(value_label)
            layout.addLayout(row)

            # Store references to the slider and value label
            self.sliders.append(slider)
            self.value_labels.append(value_label)

        # Set the frame style and border for the panel
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("border:1px solid gray; padding:4px; border-radius:5px;")

        # Callback for when rotation changes
        self.rotation_changed_callback = None

    def on_rotation_change(self, axis_index, value):
        """
               This method is called when a rotation slider value changes and notifies
               any registered callback with the axis index and new value.

               Args:
                   axis_index: The index of the rotation axis (0 for LR, 1 for PA, 2 for IS).
                   value: The new rotation value in degrees.
               """
        if self.rotation_changed_callback:
            self.rotation_changed_callback(axis_index, value)

    def set_rotation_changed_callback(self, callback):
        """
                This method allows external code to register a callback that will be
                invoked with the axis index and new value when a rotation slider is
                adjusted.

                Args:
                    callback: A function that takes two arguments (axis_index, value).
                """
        self.rotation_changed_callback = callback

    def set_rotations(self, rotations):
        """
                This method updates each rotation slider to the specified value without
                emitting signals.

                Args:
                    rotations: A list or tuple of three values representing the rotation
                    for each axis (LR, PA, IS).
                """
        # rotations: list or tuple of 3 values
        for i, val in enumerate(rotations):
            self.sliders[i].blockSignals(True)
            self.sliders[i].setValue(val)
            self.sliders[i].blockSignals(False)

    def reset_rotation(self):
        """
                This method sets each rotation slider to zero and updates the
                corresponding label to "0°".
                """
        for slider, label in zip(self.sliders, self.value_labels):
            slider.blockSignals(True)
            slider.setValue(0)
            label.setText("0°")
            slider.blockSignals(False)
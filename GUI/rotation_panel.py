from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSlider
from PySide6.QtCore import Qt

class RotationControlPanel(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.sliders = []
        self.value_labels = []

        layout = QVBoxLayout(self)
        for i, axis in enumerate(["LR", "PA", "IS"]):
            row = QHBoxLayout()
            label = QLabel(f"{axis} Rotation:")
            value_label = QLabel("0°")
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-90, 90)
            slider.setValue(0)

            def make_value_updater(lbl):
                return lambda v: lbl.setText(f"{v}°")
            slider.valueChanged.connect(make_value_updater(value_label))

            # Forward rotation changes via signal or callback - assign externally
            # (We'll provide an update_rotation callback to set later)
            slider.valueChanged.connect(lambda v, idx=i: self.on_rotation_change(idx, v))

            row.addWidget(label)
            row.addWidget(slider)
            row.addWidget(value_label)
            layout.addLayout(row)

            self.sliders.append(slider)
            self.value_labels.append(value_label)

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("border:1px solid gray; padding:4px; border-radius:5px;")

        self.rotation_changed_callback = None

    def on_rotation_change(self, axis_index, value):
        if self.rotation_changed_callback:
            self.rotation_changed_callback(axis_index, value)

    def set_rotation_changed_callback(self, callback):
        self.rotation_changed_callback = callback

    def set_rotations(self, rotations):
        # rotations: list or tuple of 3 values
        for i, val in enumerate(rotations):
            self.sliders[i].blockSignals(True)
            self.sliders[i].setValue(val)
            self.sliders[i].blockSignals(False)

from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSlider
from PySide6.QtCore import Qt

class TranslationControlPanel(QFrame):
    def __init__(self):
        super().__init__()

        self.offset_changed_callback = None
        self.zoom_changed_callback = None

        layout = QVBoxLayout(self)

        self.sliders = []
        self.labels = []

        for i, axis in enumerate(['x', 'y', 'z']):
            row = QHBoxLayout()
            label = QLabel(f"Axis {axis} Offest:")
            value_label = QLabel("0 px")

            slider = QSlider(Qt.Horizontal)
            slider.setRange(-100, 100)
            slider.setValue(0)

            def make_value_updater(lbl):
                return lambda v: lbl.setText(f"{v} px")

            slider.valueChanged.connect(make_value_updater(value_label))
            slider.valueChanged.connect(lambda v, idx=i: self.on_offset_change(idx, v))

            row.addWidget(label)
            row.addWidget(slider)
            row.addWidget(value_label)

            layout.addLayout(row)
            self.sliders.append(slider)
            self.labels.append(value_label)

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("border:1px solid gray; padding:4px; border-radius:5px;")


    def on_offset_change(self, axis_index, value):
        if self.offset_changed_callback:
            # offset is (x, y)
            x = self.sliders[0].value()
            y = self.sliders[1].value()
            self.offset_changed_callback((x, y))

    def set_offset_changed_callback(self, callback):
        self.offset_changed_callback = callback

    def set_offsets(self, offsets):
        for i in range(2):
            self.sliders[i].blockSignals(True)
            self.sliders[i].setValue(offsets[i])
            self.sliders[i].blockSignals(False)

    
from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSlider
from PySide6.QtCore import Qt

class TranslationControlPanel(QFrame):
    """
        Initializes the translation control panel for adjusting image layer offsets.

        This constructor sets up sliders and labels for x, y, and z axis offsets,
        configures their ranges, and prepares the panel for integration with the main viewer interface.
    """
    def __init__(self):
        super().__init__()

        self.offset_changed_callback = None
        self.zoom_changed_callback = None

        layout = QVBoxLayout(self)

        self.sliders = []
        self.labels = []

        #TODO The Z axis does not work yet fix
        for i, axis in enumerate(['x', 'y']):
            row = QHBoxLayout()
            label = QLabel(f"Axis {axis} Offset:")
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
        """
            Handles changes to the translation offset sliders and notifies the callback.

            When a slider value changes, this method collects the current x and y offsets and
            calls the registered offset_changed_callback with the new values.

            Args:
                axis_index: The index of the axis that was changed (0 for x, 1 for y, 2 for z).
                value: The new value of the changed slider.
        """
        if self.offset_changed_callback:
            # offset is (x, y)
            x = self.sliders[0].value()
            y = self.sliders[1].value()
            self.offset_changed_callback((x, y))

    def set_offset_changed_callback(self, callback):
        self.offset_changed_callback = callback

    def set_offsets(self, offsets):
        """
            Sets the x and y offset slider values to the provided offsets.

            This method updates the slider positions for the x and y axes without
            emitting value changed signals.

            Args:
                offsets: A tuple or list containing the new x and y offset values.
        """
        for i in range(2):
            self.sliders[i].blockSignals(True)
            self.sliders[i].setValue(offsets[i])
            self.sliders[i].blockSignals(False)

    def reset_trans(self):
        for slider, label in zip(self.sliders, self.labels):
            slider.blockSignals(True)
            slider.setValue(0)
            label.setText("0 px")
            slider.blockSignals(False)

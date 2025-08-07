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

        # Callback for when the offset changes
        self.offset_changed_callback = None
        self.zoom_changed_callback = None

        # Set up the main vertical layout for the panel
        layout = QVBoxLayout(self)

        self.sliders = []
        self.labels = []

        # Create sliders and labels for x and y axis offsets
        for i, axis in enumerate(['x', 'y']):
            row = QHBoxLayout()
            label = QLabel(f"Axis {axis} Offset:")
            value_label = QLabel("0 px")

            # Create and configure the slider for this axis
            slider = QSlider(Qt.Horizontal)
            slider.setRange(-100, 100)
            slider.setValue(0)

            # Function to update the value label when the slider changes
            def make_value_updater(lbl):
                return lambda v: lbl.setText(f"{v} px")

            # Connect slider value changes to label updater and offset change handler
            slider.valueChanged.connect(make_value_updater(value_label))
            slider.valueChanged.connect(lambda v, idx=i: self.on_offset_change(idx, v))

            # Add widgets to the row layout
            row.addWidget(label)
            row.addWidget(slider)
            row.addWidget(value_label)

            # Add the row to the main layout and store references
            layout.addLayout(row)
            self.sliders.append(slider)
            self.labels.append(value_label)

        # Set the frame style and border for the panel
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("border:1px solid gray; padding:4px; border-radius:5px;")


    def on_offset_change(self, axis_index, value):
        """
            Handles changes to the translation offset sliders and notifies the callback.

            When a slider value changes, this method collects the current x and y offsets and
            calls the registered offset_changed_callback with the new values.

            Args:
                axis_index: The index of the axis that was changed (0 for x, 1 for y).
                value: The new value of the changed slider.
        """
        if self.offset_changed_callback:
            # offset is (x, y)
            x = self.sliders[0].value()
            y = self.sliders[1].value()
            self.offset_changed_callback((x, y))

    def set_offset_changed_callback(self, callback):
        """
                Sets the callback function to be called when the translation offset
                changes.

                This method allows external code to register a callback that will be
                invoked with the new (x, y) offset when a slider is adjusted.

                Args:
                    callback: A function that takes a tuple or list representing
                    the new (x, y) offset.
                """
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
        """
                Resets all translation sliders and value labels to their default state.

                This method sets each translation slider to zero and updates the
                corresponding label to "0 px".
                """
        for slider, label in zip(self.sliders, self.labels):
            slider.blockSignals(True)
            slider.setValue(0)
            label.setText("0 px")
            slider.blockSignals(False)

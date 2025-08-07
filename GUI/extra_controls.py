from PySide6.QtWidgets import QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSlider
from PySide6.QtCore import Qt

class ZoomControlPanel(QFrame):
    """
        Provides a user interface panel for controlling the zoom level of the viewer.

        This panel includes a slider and label for adjusting and displaying the zoom factor,
        and emits a callback when the zoom value changes.
    """
    def __init__(self, parent=None):
        super().__init__(parent)

        # Callback to notify when the zoom value changes
        self.zoom_changed_callback = None

        # Set up the main vertical layout for the panel
        layout = QVBoxLayout(self)

        # Create a horizontal row for the label, slider, and value display
        row = QHBoxLayout()
        label = QLabel("Extra:")
        self.value_label = QLabel("1.0×")

        # Create and configure the zoom slider
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(20, 300)  # Zoom from 0.5x to 3.0x
        self.zoom_slider.setValue(100)      # Default 1.0x

        # Update text and emit callback
        self.zoom_slider.valueChanged.connect(self.on_slider_value_changed)

        # Add widgets to the row layout
        row.addWidget(label)
        row.addWidget(self.zoom_slider)
        row.addWidget(self.value_label)

        layout.addLayout(row)

        # Set frame and border
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("border:1px solid gray; padding:4px; border-radius:5px;")

    def on_slider_value_changed(self, value):
        """
                This method updates the label to show the new zoom factor and calls the
                registered callback with the new value.

                Args:
                    value: The new value of the zoom slider.
                """
        # Update the label and emit the callback when the slider value changes
        zoom_factor = value / 100.0
        self.value_label.setText(f"{zoom_factor:.1f}×")
        if self.zoom_changed_callback:
            self.zoom_changed_callback(zoom_factor)

    def set_zoom_changed_callback(self, callback):
        """
                This method allows external code to register a callback that will be invoked
                with the new zoom factor when the slider is adjusted.

                Args:
                    callback: A function that takes a single float argument representing
                    the new zoom factor.
                """
        # Set the callback function to be called when the zoom value changes
        self.zoom_changed_callback = callback

    def set_zoom(self, zoom_factor):
        """
                This method updates the slider and label to reflect the given zoom value
                without emitting change signals.

                Args:
                    zoom_factor: The new zoom factor to set (e.g., 1.0 for 100%).
                """
        # Temporarily block signals to avoid triggering callbacks while updating the slider
        self.zoom_slider.blockSignals(True)
        # Set the slider value based on the zoom factor (e.g., 1.0x -> 100)
        self.zoom_slider.setValue(int(zoom_factor * 100))
        # Update the label to display the current zoom factor
        self.value_label.setText(f"{zoom_factor:.1f}×")
        self.zoom_slider.blockSignals(False)
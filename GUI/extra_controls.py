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

        self.zoom_changed_callback = None

        layout = QVBoxLayout(self)

        row = QHBoxLayout()
        label = QLabel("Extra:")
        self.value_label = QLabel("1.0×")

        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(50, 300)  # Zoom from 0.5x to 3.0x
        self.zoom_slider.setValue(100)      # Default 1.0x

        # Update text and emit callback
        self.zoom_slider.valueChanged.connect(self.on_slider_value_changed)

        row.addWidget(label)
        row.addWidget(self.zoom_slider)
        row.addWidget(self.value_label)

        layout.addLayout(row)

        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet("border:1px solid gray; padding:4px; border-radius:5px;")

    def on_slider_value_changed(self, value):
        zoom_factor = value / 100.0
        self.value_label.setText(f"{zoom_factor:.1f}×")
        if self.zoom_changed_callback:
            self.zoom_changed_callback(zoom_factor)

    def set_zoom_changed_callback(self, callback):
        self.zoom_changed_callback = callback

    def set_zoom(self, zoom_factor):
        self.zoom_slider.blockSignals(True)
        self.zoom_slider.setValue(int(zoom_factor * 100))
        self.value_label.setText(f"{zoom_factor:.1f}×")
        self.zoom_slider.blockSignals(False)
from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QLabel, QSlider
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainter

class BaseViewer(QWidget):
    """
    Base class for DICOM viewer widgets displaying a single orthogonal view (axial, coronal, or sagittal).

    This class sets up the graphics scene, view, slice slider, and provides a standard interface for loading DICOM folders and interacting with the view.
    """
    def __init__(self, controller_cls, title="Viewer", label_text="View", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)

        # Create the graphics scene and view for displaying images
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)

        # Set the background color and rendering hints for smoother display
        self.view.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        # Create and configure the slice slider (hidden by default)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(100)
        self.slice_slider.setValue(50)
        self.slice_slider.hide()

        # Set up the layout with a label, the view, and the slider
        layout = QVBoxLayout()
        layout.addWidget(QLabel(label_text))
        layout.addWidget(self.view)
        layout.addWidget(self.slice_slider)
        self.setLayout(layout)

        # Initialize the controller for managing view logic
        self.controller = controller_cls(self.scene, self.view)
        self.controller.set_slice_slider(self.slice_slider)

    def load_dicom_folder(self, folder):
        """
               Loads a DICOM volume from the specified folder and adds it as a new
               layer to the viewer.
               """
        return self.controller.load_dicom_folder(folder)

    def select_layer(self, index):
        """
                Selects the image layer at the specified index in the viewer.
                """
        self.controller.select_layer(index)

    def update_rotation(self, axis, value):
        """
                Updates the rotation value for the specified axis of the current image
                layer.
                """
        self.controller.update_rotation(axis, value)

    def update_translation(self, offset):
        """
                This method delegates the translation update to the associated controller.
                """
        self.controller.update_translation(offset)

    def remove_current_layer(self):
        """
                This method delegates the removal operation to the associated controller.
                """
        self.controller.remove_current_layer()

    def reset_view(self):
        """
                This method resets the QGraphicsView transformation and calls the
                controller's reset_zoom method.
                """
        self.view.resetTransform()
        self.controller.reset_zoom()
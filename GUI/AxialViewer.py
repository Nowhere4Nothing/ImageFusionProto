from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QLabel, QSizePolicy, QSlider
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainter

from Controller.viewer_controller_Axial import ViewerControllerAxial

#TODO Create a base class for AXIAL, CORONAL and SAGITTAL Viewers
class AxialViewer(QWidget):
    """
            Initializes the AxialViewer widget for displaying axial DICOM image slices.

            Sets up the graphics scene, view, slice slider, and controller for
            managing axial view interactions and rendering.
            """
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Axial Viewer")

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
        layout.addWidget(QLabel("Axial View"))
        layout.addWidget(self.view)
        layout.addWidget(self.slice_slider)
        self.setLayout(layout)

        # Initialize the controller for managing axial view logic
        self.controller = ViewerControllerAxial(self.scene, self.view)
        self.controller.set_slice_slider(self.slice_slider)

    def load_dicom_folder(self, folder):
        return self.controller.load_dicom_folder(folder)

    def select_layer(self, index):
        self.controller.select_layer(index)

    def update_rotation(self, axis, value):
        self.controller.update_rotation(axis, value)

    def update_translation(self, offset):
        self.controller.update_translation(offset)

    def remove_current_layer(self):
        self.controller.remove_current_layer()

    def reset_view(self):
        self.view.resetTransform()
        self.controller.reset_zoom()









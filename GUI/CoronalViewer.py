from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QSlider, QLabel
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainter

from Controller.viewer_controller_coronal import ViewerControllerCoronal

class CoronalViewer(QWidget):
    def __init__(self, volume = None):
        super().__init__()
        self.setWindowTitle("Coronal Viewer")

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)

        self.view.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(100)
        self.slice_slider.setValue(50)
        self.slice_slider.hide()

        layout = QVBoxLayout()
        layout.addWidget(QLabel("Coronal View"))
        layout.addWidget(self.view)
        layout.addWidget(self.slice_slider)
        self.setLayout(layout)

        # Create ViewerController with view_type="coronal"
        self.controller = ViewerControllerCoronal(self.scene, self.view)
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

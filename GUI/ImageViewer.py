from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QLabel, QSizePolicy, QSlider
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainter

from Controller.viewer_controller_Axial import ViewerController


class ImageViewer(QWidget):
    def __init__(self, title="Viewer"):
        super().__init__()

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
        layout.addWidget(QLabel("Axial View"))
        layout.addWidget(self.view)
        layout.addWidget(self.slice_slider)
        self.setLayout(layout)

        # Assign the controller
        self.controller = ViewerController(self.scene, self.view)
        self.controller.set_slice_slider(self.slice_slider)

    def display_image(self, qpixmap):
        # Scale pixmap to fit inside the view while keeping aspect ratio
        scaled_pixmap = qpixmap.scaled(
            self.view.viewport().size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.scene.clear()
        self.scene.addPixmap(scaled_pixmap)
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)





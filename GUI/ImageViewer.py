from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QLabel, QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainter

class ImageViewer(QWidget):
    def __init__(self, title="Viewer"):
        super().__init__()
        self.setWindowTitle(title)

        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)
        self.view.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        # Set fixed size or size policy
        self.view.setMinimumSize(300, 300)
        self.view.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        layout = QVBoxLayout()
        layout.addWidget(QLabel(title))
        layout.addWidget(self.view)
        self.setLayout(layout)

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





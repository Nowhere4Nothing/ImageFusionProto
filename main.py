import sys
import os
import numpy as np
import pydicom
from scipy.ndimage import rotate

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog,
    QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QSlider, QListWidget,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QFrame, QSizePolicy
)
from PySide6.QtGui import QImage, QPixmap, QPainter, QBrush, QColor
from PySide6.QtCore import Qt, QEvent

class DicomViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("manual image fusion example")



if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = DicomViewer()
    viewer.resize(1000, 800)
    viewer.show()
    sys.exit(app.exec())
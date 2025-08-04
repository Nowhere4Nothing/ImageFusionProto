from PySide6.QtWidgets import QWidget, QHBoxLayout

from GUI.ImageViewer import ImageViewer
from GUI.CoronalViewer import CoronalViewer
from GUI.SagittalViewer import SagittalViewer

class MultiViewWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.axial_viewer = ImageViewer("Axial View")
        self.coronal_viewer = CoronalViewer("Coronal View")
        self.sagittal_viewer = SagittalViewer("Sagittal View")

        layout = QHBoxLayout()
        layout.addWidget(self.axial_viewer)
        layout.addWidget(self.coronal_viewer)
        layout.addWidget(self.sagittal_viewer)

        self.setLayout(layout)
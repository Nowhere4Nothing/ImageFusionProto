from PySide6.QtWidgets import QWidget, QHBoxLayout

from GUI.CoronalViewer import CoronalViewer
from GUI.SagittalViewer import SagittalViewer
from GUI.AxialViewer import AxialViewer

class MultiViewWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.axial_viewer = AxialViewer()
        self.coronal_viewer = CoronalViewer()
        self.sagittal_viewer = SagittalViewer()

        layout = QHBoxLayout()
        layout.addWidget(self.axial_viewer)
        layout.addWidget(self.coronal_viewer)
        layout.addWidget(self.sagittal_viewer)

        self.setLayout(layout)
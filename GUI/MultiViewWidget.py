from PySide6.QtWidgets import QWidget, QHBoxLayout

from GUI.CoronalViewer import CoronalViewer
from GUI.SagittalViewer import SagittalViewer
from GUI.AxialViewer import AxialViewer

class MultiViewWidget(QWidget):
    """
        Provides a widget that displays axial, coronal,
        and sagittal DICOM viewers side by side.

        This class initializes and arranges the three orthogonal viewers in a horizontal
        layout for multi-planar visualization.
        """
    def __init__(self):
        super().__init__()

        #calling the individual viewers
        self.axial_viewer = AxialViewer()
        self.coronal_viewer = CoronalViewer()
        self.sagittal_viewer = SagittalViewer()

        #adding them to the layout
        layout = QHBoxLayout()
        layout.addWidget(self.axial_viewer)
        layout.addWidget(self.coronal_viewer)
        layout.addWidget(self.sagittal_viewer)

        self.setLayout(layout)
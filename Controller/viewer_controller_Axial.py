from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QSlider

from utils.image_processing import process_layers
from utils.layer_loader import load_dicom_layer

from Controller.base_view_controller import BaseViewerController


class ViewerController(BaseViewerController):
    """
        Manages the logic and state for the DICOM image viewer.

        This class handles loading DICOM volumes, managing image layers, updating display properties,
        and synchronizing UI controls with the underlying data.
        """
    def __init__(self, scene, view):
        super().__init__(view, scene, view_type="axial")

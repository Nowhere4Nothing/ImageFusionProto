from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QSlider

from utils.image_processing import process_layers
from utils.layer_loader import load_dicom_layer

from Controller.base_view_controller import BaseViewerController


class ViewerControllerSagittal(BaseViewerController):
    """
    Controller for managing coronal DICOM view logic.
    """
    def __init__(self, scene, view):
        super().__init__(view, scene, view_type="sagittal")

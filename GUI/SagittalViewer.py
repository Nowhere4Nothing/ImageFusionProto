from Controller.viewer_controller_sagittal import ViewerControllerSagittal
from GUI.base_viewer_GUI import BaseViewer

class SagittalViewer(BaseViewer):
    """
    Viewer widget for displaying sagittal DICOM image slices.

    Inherits from BaseViewer and uses the sagittal-specific controller.
    """
    def __init__(self):
        super().__init__(ViewerControllerSagittal, title="Sagittal Viewer", label_text="Sagittal View")
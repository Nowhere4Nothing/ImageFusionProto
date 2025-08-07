from Controller.viewer_controller_coronal import ViewerControllerCoronal
from GUI.base_viewer_GUI import BaseViewer

class CoronalViewer(BaseViewer):
    """
    Viewer widget for displaying coronal DICOM image slices.

    Inherits from BaseViewer and uses the coronal-specific controller.
    """
    def __init__(self):
        super().__init__(ViewerControllerCoronal, title="Coronal Viewer", label_text="Coronal View")
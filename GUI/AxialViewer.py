from Controller.viewer_controller_Axial import ViewerControllerAxial
from GUI.base_viewer_GUI import BaseViewer

class AxialViewer(BaseViewer):
    """
    Viewer widget for displaying axial DICOM image slices.

    Inherits from BaseViewer and uses the axial-specific controller.
    """
    def __init__(self):
        super().__init__(ViewerControllerAxial, title="Axial Viewer", label_text="Axial View")
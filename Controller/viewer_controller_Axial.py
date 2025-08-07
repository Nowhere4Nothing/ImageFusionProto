from Controller.base_view_controller import BaseViewerController


class ViewerControllerAxial(BaseViewerController):
    """
        Controller for managing the logic and state of the axial DICOM image viewer.

        Inherits from BaseViewerController and initializes the controller for the axial view type.
        """
    def __init__(self, scene, view):
        super().__init__(view, scene, view_type="axial")

from Controller.base_view_controller import BaseViewerController


class ViewerControllerCoronal(BaseViewerController):
    """
        Controller for managing the logic and state of the Coronal DICOM image viewer.

        Inherits from BaseViewerController and initializes the controller for the Coronal view type.
        """
    def __init__(self, scene, view):
        super().__init__(view, scene, view_type="coronal")

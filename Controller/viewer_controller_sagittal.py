from Controller.base_view_controller import BaseViewerController


class ViewerControllerSagittal(BaseViewerController):
    """
        Controller for managing the logic and state of the sagittal DICOM image viewer.

        Inherits from BaseViewerController and initializes the controller for the sagittal view type.
        """

    def __init__(self, scene, view):
        super().__init__(view, scene, view_type="sagittal")

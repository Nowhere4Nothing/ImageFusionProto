from PySide6.QtWidgets import QWidget, QVBoxLayout, QGraphicsView, QGraphicsScene, QLabel, QSlider
from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainter

class BaseViewer(QWidget):
    """
    Base class for DICOM viewer widgets displaying a single orthogonal view (axial, coronal, or sagittal).

    This class sets up the graphics scene, view, slice slider, and provides a standard interface for loading DICOM folders and interacting with the view.
    """
    def __init__(self, controller_cls, title="Viewer", label_text="View", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)

        # Create the graphics scene and view for displaying images
        self.scene = QGraphicsScene()
        self.view = QGraphicsView(self.scene)

        # Set the background color and rendering hints for smoother display
        self.view.setBackgroundBrush(QBrush(QColor(0, 0, 0)))
        self.view.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)

        # Create and configure the slice slider (hidden by default)
        self.slice_slider = QSlider(Qt.Horizontal)
        self.slice_slider.setMinimum(0)
        self.slice_slider.setMaximum(100)
        self.slice_slider.setValue(50)
        self.slice_slider.hide()

        # Set up the layout with a label, the view, and the slider
        layout = QVBoxLayout()
        layout.addWidget(QLabel(label_text))
        layout.addWidget(self.view)
        layout.addWidget(self.slice_slider)
        self.setLayout(layout)

        # Initialize the controller for managing view logic
        self.controller = controller_cls(self.scene, self.view)
        self.controller.set_slice_slider(self.slice_slider)

    def load_dicom_folder(self, folder):
        """
               Loads a DICOM volume from the specified folder and adds it as a new
               layer to the viewer.

               This method delegates the loading operation to the associated controller
               and returns the result.

               Args:
                   folder: Path to the folder containing the DICOM files.

               Returns:
                   The result of the controller's load_dicom_folder method,
                   typically a tuple with layer information.
               """
        return self.controller.load_dicom_folder(folder)

    def select_layer(self, index):
        """
                Selects the image layer at the specified index in the viewer.

                This method delegates the selection operation to the associated controller.

                Args:
                    index: The index of the layer to select.
                """
        self.controller.select_layer(index)

    def update_rotation(self, axis, value):
        """
                Updates the rotation value for the specified axis of the current image
                layer.

                This method delegates the rotation update to the associated controller.

                Args:
                    axis: The index of the rotation axis (e.g., 0 for LR, 1 for PA,
                    2 for IS).
                    value: The new rotation value in degrees.
                """
        self.controller.update_rotation(axis, value)

    def update_translation(self, offset):
        """
                Updates the translation (offset) of the current image layer in the viewer.

                This method delegates the translation update to the associated controller.

                Args:
                    offset: A tuple or list representing the new (x, y) translation values.
                """
        self.controller.update_translation(offset)

    def remove_current_layer(self):
        """
                Removes the currently selected image layer from the viewer.

                This method delegates the removal operation to the associated controller.
                """
        self.controller.remove_current_layer()

    def reset_view(self):
        """
                Resets the view to its default zoom and transformation state.

                This method resets the QGraphicsView transformation and calls the
                controller's reset_zoom method.
                """
        self.view.resetTransform()
        self.controller.reset_zoom()
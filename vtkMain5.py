from PySide6 import QtWidgets, QtGui, QtCore
import sys
import vtk_engine  # our C++ module

class FusionUI(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manual Co-registration")
        self.resize(1400,900)
        self.engine = vtk_engine.VTKEngine()
        self._build_ui()

    def _build_ui(self):
        layout = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(left)

        self.btn_fixed = QtWidgets.QPushButton("Load FIXED DICOM")
        self.btn_moving = QtWidgets.QPushButton("Load MOVING DICOM")
        self.btn_fixed.clicked.connect(self.load_fixed)
        self.btn_moving.clicked.connect(self.load_moving)
        form.addRow(self.btn_fixed)
        form.addRow(self.btn_moving)

        self.slices = {}
        for name in ["axial","coronal","sagittal"]:
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(0)
            slider.setValue(0)
            slider.valueChanged.connect(lambda val,n=name: self.update_slice(n,val))
            form.addRow(f"{name} slice", slider)
            self.slices[name] = slider

        self.viewers = {}
        grid = QtWidgets.QGridLayout()
        for i,(name,pos) in enumerate([("axial",(0,0)),("coronal",(0,1)),("sagittal",(1,0))]):
            viewer = QtWidgets.QGraphicsView()
            scene = QtWidgets.QGraphicsScene()
            viewer.setScene(scene)
            item = QtWidgets.QGraphicsPixmapItem()
            scene.addItem(item)
            self.viewers[name] = item
            grid.addWidget(viewer,*pos)
        layout.addWidget(left)
        right = QtWidgets.QWidget()
        right.setLayout(grid)
        layout.addWidget(right)

    def load_fixed(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self,"Select Fixed DICOM")
        if folder:
            self.engine.load_fixed(folder)

    def load_moving(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self,"Select Moving DICOM")
        if folder:
            self.engine.load_moving(folder)

    def update_slice(self, orientation, idx):
        arr = self.engine.get_slice(orientation, idx)
        h,w = arr.shape
        img = QtGui.QImage(arr.data, w,h,w,QtGui.QImage.Format_Grayscale8)
        pix = QtGui.QPixmap.fromImage(img)
        self.viewers[orientation].setPixmap(pix)

def main():
    app = QtWidgets.QApplication(sys.argv)
    w = FusionUI()
    w.show()
    sys.exit(app.exec())

if __name__=="__main__":
    main()

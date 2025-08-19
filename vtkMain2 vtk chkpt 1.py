"""
Manual Co-Registration Demo (VTK + PySide6, Qt display using VTK slice viewers)

• Loads a FIXED DICOM series and a MOVING DICOM series
• Applies real-time rigid transform (TX/TY/TZ in mm, RX/RY/RZ in degrees) to MOVING
• Reslices MOVING into FIXED space on the fly (vtkImageReslice)
• Blends FIXED + MOVING
• Displays axial/coronal/sagittal slices using vtkImageViewer2 embedded in Qt

Requirements:
    pip install PySide6 vtk numpy

Run:
    python app.py
"""
from __future__ import annotations
import sys
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from PySide6 import QtCore, QtWidgets
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

# ------------------------------ VTK Processing Engine ------------------------------

class VTKEngine:
    ORI_AXIAL = "axial"
    ORI_CORONAL = "coronal"
    ORI_SAGITTAL = "sagittal"

    def __init__(self):
        self.fixed_reader: Optional[vtk.vtkDICOMImageReader] = None
        self.moving_reader: Optional[vtk.vtkDICOMImageReader] = None

        # Transform
        self._tx = self._ty = self._tz = 0.0
        self._rx = self._ry = self._rz = 0.0
        self.transform = vtk.vtkTransform()
        self.transform.PostMultiply()

        # Reslice moving
        self.reslice3d = vtk.vtkImageReslice()
        self.reslice3d.SetInterpolationModeToLinear()
        self.reslice3d.SetBackgroundLevel(0.0)

        # Blend
        self.blend = vtk.vtkImageBlend()
        self.blend.SetOpacity(0, 1.0)
        self.blend.SetOpacity(1, 0.5)

    def load_fixed(self, dicom_dir: str) -> bool:
        """Load FIXED series; return True if successful"""
        files = list(Path(dicom_dir).glob("*"))
        if not any(f.is_file() for f in files):
            return False
        r = vtk.vtkDICOMImageReader()
        r.SetDirectoryName(str(Path(dicom_dir)))
        r.Update()
        self.fixed_reader = r
        self._wire_blend()
        self._sync_reslice_output_to_fixed()
        return True

    def load_moving(self, dicom_dir: str) -> bool:
        """Load MOVING series; return True if successful"""
        files = list(Path(dicom_dir).glob("*"))
        if not any(f.is_file() for f in files):
            return False
        r = vtk.vtkDICOMImageReader()
        r.SetDirectoryName(str(Path(dicom_dir)))
        r.Update()
        self.moving_reader = r
        self.reslice3d.SetInputConnection(r.GetOutputPort())
        self._apply_transform()
        self._wire_blend()
        self._sync_reslice_output_to_fixed()
        return True

    def set_opacity(self, alpha: float):
        self.blend.SetOpacity(1, float(np.clip(alpha, 0.0, 1.0)))

    def set_translation(self, tx: float, ty: float, tz: float):
        self._tx, self._ty, self._tz = float(tx), float(ty), float(tz)
        self._apply_transform()

    def set_rotation_deg(self, rx: float, ry: float, rz: float):
        self._rx, self._ry, self._rz = float(rx), float(ry), float(rz)
        self._apply_transform()

    def reset_transform(self):
        self._tx = self._ty = self._tz = 0.0
        self._rx = self._ry = self._rz = 0.0
        self.transform.Identity()
        self._apply_transform()

    def fixed_extent(self) -> Optional[Tuple[int, int, int, int, int, int]]:
        if not self.fixed_reader:
            return None
        return self.fixed_reader.GetOutput().GetExtent()

    def get_blended_image(self):
        """Return vtkImageData of blended FIXED+MOVING"""
        self.blend.Modified()
        self.blend.Update()
        return self.blend.GetOutput()

    # Internals
    def _apply_transform(self):
        t = vtk.vtkTransform()
        t.PostMultiply()
        t.Translate(self._tx, self._ty, self._tz)
        t.RotateX(self._rx)
        t.RotateY(self._ry)
        t.RotateZ(self._rz)
        self.transform.DeepCopy(t)
        self.reslice3d.SetResliceAxes(self.transform.GetMatrix())
        self.reslice3d.Modified()

    def _wire_blend(self):
        self.blend.RemoveAllInputs()
        if self.fixed_reader is not None:
            self.blend.AddInputConnection(self.fixed_reader.GetOutputPort())
        if self.moving_reader is not None:
            self.blend.AddInputConnection(self.reslice3d.GetOutputPort())
        self.blend.Modified()

    def _sync_reslice_output_to_fixed(self):
        if self.fixed_reader is None:
            return
        fixed = self.fixed_reader.GetOutput()
        self.reslice3d.SetOutputSpacing(fixed.GetSpacing())
        self.reslice3d.SetOutputOrigin(fixed.GetOrigin())
        self.reslice3d.SetOutputExtent(fixed.GetExtent())
        self.reslice3d.Modified()


# ------------------------------ VTK Slice Viewer ------------------------------

class VTKSliceViewer(QtWidgets.QFrame):
    """VTKImageViewer2 embedded in Qt using QVTKRenderWindowInteractor"""
    def __init__(self, orientation: str):
        super().__init__()
        self.orientation = orientation
        self.viewer: Optional[vtk.vtkImageViewer2] = None
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.vtk_widget)
        self.vtk_widget.Initialize()
        self.vtk_widget.Start()
        self.current_slice = 0
        self.connected = False

    def set_input_connection(self, port):
        if not port:
            return
        if self.viewer is None:
            # Lazy initialization
            self.viewer = vtk.vtkImageViewer2()
            self.viewer.SetRenderWindow(self.vtk_widget.GetRenderWindow())
            self.viewer.SetupInteractor(self.vtk_widget)
        self.viewer.SetInputConnection(port)
        self.connected = True
        self._update_orientation()
        self._reset_slice_range()
        self.viewer.Render()


    def _update_orientation(self):
        if not self.connected:
            return
        if self.orientation == "axial":
            self.viewer.SetSliceOrientationToXY()
        elif self.orientation == "coronal":
            self.viewer.SetSliceOrientationToXZ()
        elif self.orientation == "sagittal":
            self.viewer.SetSliceOrientationToYZ()
        self.viewer.Render()

    def _reset_slice_range(self):
        if not self.connected:
            return
        min_s, max_s = self.viewer.GetSliceMin(), self.viewer.GetSliceMax()
        self.current_slice = (min_s + max_s) // 2
        self.viewer.SetSlice(self.current_slice)
        self.viewer.Render()

    def set_slice(self, idx: int):
        if not self.connected:
            return
        min_s, max_s = self.viewer.GetSliceMin(), self.viewer.GetSliceMax()
        idx = max(min_s, min(max_s, idx))
        self.current_slice = idx
        self.viewer.SetSlice(idx)
        self.viewer.Render()


# ------------------------------ Qt UI ------------------------------

class FusionUI(QtWidgets.QWidget):
    loadFixed = QtCore.Signal(str)
    loadMoving = QtCore.Signal(str)
    txChanged = QtCore.Signal(float)
    tyChanged = QtCore.Signal(float)
    tzChanged = QtCore.Signal(float)
    rxChanged = QtCore.Signal(float)
    ryChanged = QtCore.Signal(float)
    rzChanged = QtCore.Signal(float)
    opacityChanged = QtCore.Signal(float)
    resetRequested = QtCore.Signal()
    axialSliceChanged = QtCore.Signal(int)
    coronalSliceChanged = QtCore.Signal(int)
    sagittalSliceChanged = QtCore.Signal(int)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Manual Co-Registration Demo (VTK slice viewers)")
        self.resize(1400, 900)
        self._build()

    def _build(self):
        root = QtWidgets.QHBoxLayout(self)

        # Controls
        left = QtWidgets.QWidget(self)
        form = QtWidgets.QFormLayout(left)
        btn_fixed = QtWidgets.QPushButton("Load FIXED DICOM")
        btn_moving = QtWidgets.QPushButton("Load MOVING DICOM")
        btn_fixed.clicked.connect(lambda: self._emit_folder(self.loadFixed))
        btn_moving.clicked.connect(lambda: self._emit_folder(self.loadMoving))
        form.addRow(btn_fixed)
        form.addRow(btn_moving)

        def slider(mini, maxi, init=0):
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setMinimum(mini); s.setMaximum(maxi); s.setValue(init)
            return s

        self.s_axial = slider(0, 0, 0)
        self.s_coronal = slider(0, 0, 0)
        self.s_sagittal = slider(0, 0, 0)
        self.s_axial.valueChanged.connect(self.axialSliceChanged.emit)
        self.s_coronal.valueChanged.connect(self.coronalSliceChanged.emit)
        self.s_sagittal.valueChanged.connect(self.sagittalSliceChanged.emit)
        form.addRow("Axial Slice", self.s_axial)
        form.addRow("Coronal Slice", self.s_coronal)
        form.addRow("Sagittal Slice", self.s_sagittal)

        self.s_tx = slider(-200, 200, 0)
        self.s_ty = slider(-200, 200, 0)
        self.s_tz = slider(-200, 200, 0)
        self.s_rx = slider(-180, 180, 0)
        self.s_ry = slider(-180, 180, 0)
        self.s_rz = slider(-180, 180, 0)
        self.s_op = slider(0, 100, 50)

        self.s_tx.valueChanged.connect(lambda v: self.txChanged.emit(float(v)))
        self.s_ty.valueChanged.connect(lambda v: self.tyChanged.emit(float(v)))
        self.s_tz.valueChanged.connect(lambda v: self.tzChanged.emit(float(v)))
        self.s_rx.valueChanged.connect(lambda v: self.rxChanged.emit(float(v)))
        self.s_ry.valueChanged.connect(lambda v: self.ryChanged.emit(float(v)))
        self.s_rz.valueChanged.connect(lambda v: self.rzChanged.emit(float(v)))
        self.s_op.valueChanged.connect(lambda v: self.opacityChanged.emit(float(v) / 100.0))

        form.addRow("TX (mm)", self.s_tx)
        form.addRow("TY (mm)", self.s_ty)
        form.addRow("TZ (mm)", self.s_tz)
        form.addRow("RX (°)", self.s_rx)
        form.addRow("RY (°)", self.s_ry)
        form.addRow("RZ (°)", self.s_rz)
        form.addRow("Overlay Opacity", self.s_op)

        btn_reset = QtWidgets.QPushButton("Reset Transform")
        btn_reset.clicked.connect(self.resetRequested.emit)
        form.addRow(btn_reset)

        # VTK slice viewers
        right = QtWidgets.QWidget(self)
        grid = QtWidgets.QGridLayout(right)
        self.viewer_ax = VTKSliceViewer("axial")
        self.viewer_co = VTKSliceViewer("coronal")
        self.viewer_sa = VTKSliceViewer("sagittal")
        grid.addWidget(self.viewer_ax, 0, 0)
        grid.addWidget(self.viewer_co, 0, 1)
        grid.addWidget(self.viewer_sa, 1, 0, 1, 2)

        root.addWidget(left, 0)
        root.addWidget(right, 1)

    def _emit_folder(self, signal: QtCore.SignalInstance):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select DICOM folder")
        if folder:
            signal.emit(folder)


# ------------------------------ Controller ------------------------------

class Controller(QtCore.QObject):
    def __init__(self, ui: FusionUI, engine: VTKEngine):
        super().__init__()
        self.ui = ui
        self.engine = engine
        self._wire()

    def _wire(self):
        self.ui.loadFixed.connect(self.on_load_fixed)
        self.ui.loadMoving.connect(self.on_load_moving)

        self.ui.txChanged.connect(lambda v: (self.engine.set_translation(v, self.engine._ty, self.engine._tz), self.refresh_all()))
        self.ui.tyChanged.connect(lambda v: (self.engine.set_translation(self.engine._tx, v, self.engine._tz), self.refresh_all()))
        self.ui.tzChanged.connect(lambda v: (self.engine.set_translation(self.engine._tx, self.engine._ty, v), self.refresh_all()))
        self.ui.rxChanged.connect(lambda v: (self.engine.set_rotation_deg(v, self.engine._ry, self.engine._rz), self.refresh_all()))
        self.ui.ryChanged.connect(lambda v: (self.engine.set_rotation_deg(self.engine._rx, v, self.engine._rz), self.refresh_all()))
        self.ui.rzChanged.connect(lambda v: (self.engine.set_rotation_deg(self.engine._rx, self.engine._ry, v), self.refresh_all()))
        self.ui.opacityChanged.connect(lambda a: (self.engine.set_opacity(a), self.refresh_all()))
        self.ui.resetRequested.connect(self.on_reset)

        self.ui.axialSliceChanged.connect(lambda i: self.refresh_slice("axial", i))
        self.ui.coronalSliceChanged.connect(lambda i: self.refresh_slice("coronal", i))
        self.ui.sagittalSliceChanged.connect(lambda i: self.refresh_slice("sagittal", i))

    def on_load_fixed(self, folder: str):
        if not self.engine.load_fixed(folder):
            QtWidgets.QMessageBox.warning(self.ui, "Error", "No DICOM files found in folder!")
            return
        self._sync_slice_ranges()
        self.refresh_all()

    def on_load_moving(self, folder: str):
        if not self.engine.load_moving(folder):
            QtWidgets.QMessageBox.warning(self.ui, "Error", "No DICOM files found in folder!")
            return
        self.refresh_all()

    def on_reset(self):
        self.engine.reset_transform()
        self.refresh_all()

    # Slice ranges
    def _sync_slice_ranges(self):
        ext = self.engine.fixed_extent()
        if not ext:
            return
        x0, x1, y0, y1, z0, z1 = ext
        self.ui.s_axial.setMinimum(z0); self.ui.s_axial.setMaximum(z1); self.ui.s_axial.setValue((z0+z1)//2)
        self.ui.s_coronal.setMinimum(y0); self.ui.s_coronal.setMaximum(y1); self.ui.s_coronal.setValue((y0+y1)//2)
        self.ui.s_sagittal.setMinimum(x0); self.ui.s_sagittal.setMaximum(x1); self.ui.s_sagittal.setValue((x0+x1)//2)

    def refresh_all(self):
        img = self.engine.get_blended_image()
        if img is None:
            return
        if self.engine.fixed_reader:
            self.ui.viewer_ax.set_input_connection(self.engine.blend.GetOutputPort())
            self.ui.viewer_co.set_input_connection(self.engine.blend.GetOutputPort())
            self.ui.viewer_sa.set_input_connection(self.engine.blend.GetOutputPort())
            self.refresh_slice("axial", self.ui.s_axial.value())
            self.refresh_slice("coronal", self.ui.s_coronal.value())
            self.refresh_slice("sagittal", self.ui.s_sagittal.value())

    def refresh_slice(self, orientation: str, idx: int):
        viewer_map = {"axial": self.ui.viewer_ax, "coronal": self.ui.viewer_co, "sagittal": self.ui.viewer_sa}
        viewer = viewer_map.get(orientation)
        if viewer:
            viewer.set_slice(idx)


# ------------------------------ Main ------------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    ui = FusionUI()
    engine = VTKEngine()
    Controller(ui, engine)
    ui.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()

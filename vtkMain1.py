"""
Manual Co-Registration Demo (VTK + PySide6, Qt display like OnkoDICOM)

• Loads a FIXED DICOM series and a MOVING DICOM series
• Applies real-time rigid transform (TX/TY/TZ in mm, RX/RY/RZ in degrees) to MOVING
• Reslices MOVING into FIXED space on the fly (vtkImageReslice)
• Blends MOVING over FIXED (vtkImageBlend) with adjustable opacity
• Displays axial/coronal/sagittal slices as QPixmaps in QGraphicsViews (no VTK rendering)

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
from PySide6 import QtCore, QtWidgets, QtGui
import vtk
from vtk.util.numpy_support import vtk_to_numpy


# ------------------------------ Core VTK engine (processing only) ------------------------------

class VTKEngine:
    """
    Minimal VTK processing engine:
    - Load FIXED and MOVING (vtkImageData).
    - MOVING is transformed by a vtkTransform, resliced into FIXED geometry.
    - Blend FIXED + MOVING'.
    - Extract a single 2D slice for a given orientation/index and return a NumPy array.
    """

    ORI_AXIAL = "axial"      # normal +Z, slice index in k
    ORI_CORONAL = "coronal"  # normal +Y, slice index in j
    ORI_SAGITTAL = "sagittal"# normal +X, slice index in i

    def __init__(self):
        # Volumes
        self.fixed_reader: Optional[vtk.vtkDICOMImageReader] = None
        self.moving_reader: Optional[vtk.vtkDICOMImageReader] = None

        # MOVING -> FIXED transform
        self._tx = self._ty = self._tz = 0.0
        self._rx = self._ry = self._rz = 0.0
        self.transform = vtk.vtkTransform()
        self.transform.PostMultiply()

        # Reslice moving into fixed space
        self.reslice3d = vtk.vtkImageReslice()
        self.reslice3d.SetInterpolationModeToLinear()
        self.reslice3d.SetBackgroundLevel(0.0)

        # Blend
        self.blend = vtk.vtkImageBlend()
        self.blend.SetOpacity(0, 1.0)
        self.blend.SetOpacity(1, 0.5)

        # Single-slice extractors per orientation
        self.slice_reslicers = {
            self.ORI_AXIAL: vtk.vtkImageReslice(),
            self.ORI_CORONAL: vtk.vtkImageReslice(),
            self.ORI_SAGITTAL: vtk.vtkImageReslice(),
        }
        for r in self.slice_reslicers.values():
            r.SetInterpolationModeToNearestNeighbor()
            r.SetOutputDimensionality(2)
            r.SetBackgroundLevel(0.0)
            r.SetInputConnection(self.blend.GetOutputPort())

    # -------- Public API --------
    def load_fixed(self, dicom_dir: str):
        r = vtk.vtkDICOMImageReader()
        r.SetDirectoryName(str(Path(dicom_dir)))
        r.Update()
        self.fixed_reader = r
        self._wire_blend()
        self._sync_reslice_output_to_fixed()

    def load_moving(self, dicom_dir: str):
        r = vtk.vtkDICOMImageReader()
        r.SetDirectoryName(str(Path(dicom_dir)))
        r.Update()
        self.moving_reader = r
        self.reslice3d.SetInputConnection(r.GetOutputPort())
        self._apply_transform()
        self._wire_blend()
        self._sync_reslice_output_to_fixed()

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

    def get_slice_uint8(self, orientation: str, index: int, wl: float = 40.0, ww: float = 400.0) -> Optional[np.ndarray]:
        """
        Returns a uint8 2D array (H x W) ready for QImage/QPixmap.
        WL/WW applied to the blended slice.
        """
        if self.fixed_reader is None:
            return None

        blend_out = self.blend.GetOutput()
        fixed = self.fixed_reader.GetOutput()
        extent = fixed.GetExtent()  # (x0,x1,y0,y1,z0,z1)
        spacing = fixed.GetSpacing()
        origin = fixed.GetOrigin()
        dir4 = fixed.GetDirectionMatrix()

        # Clamp index and build a per-slice reslice-axes (FIXED geometry)
        idx = index
        axes = vtk.vtkMatrix4x4()
        axes.DeepCopy(dir4)

        if orientation == self.ORI_AXIAL:
            idx = int(np.clip(idx, extent[4], extent[5]))
            oz = origin[2] + idx * spacing[2]
            axes.SetElement(0, 3, origin[0])
            axes.SetElement(1, 3, origin[1])
            axes.SetElement(2, 3, oz)
        elif orientation == self.ORI_CORONAL:
            idx = int(np.clip(idx, extent[2], extent[3]))
            oy = origin[1] + idx * spacing[1]
            axes.SetElement(0, 3, origin[0])
            axes.SetElement(1, 3, oy)
            axes.SetElement(2, 3, origin[2])
        elif orientation == self.ORI_SAGITTAL:
            idx = int(np.clip(idx, extent[0], extent[1]))
            ox = origin[0] + idx * spacing[0]
            axes.SetElement(0, 3, ox)
            axes.SetElement(1, 3, origin[1])
            axes.SetElement(2, 3, origin[2])
        else:
            return None

        r = self.slice_reslicers[orientation]
        r.SetResliceAxes(axes)
        r.SetInputData(blend_out)  # ensure it's wired to current blended volume
        r.Update()

        sl = r.GetOutput()
        dims = sl.GetDimensions()  # X, Y, 1
        if dims[0] == 0 or dims[1] == 0:
            return None

        arr = vtk_to_numpy(sl.GetPointData().GetScalars()).reshape(dims[1], dims[0])  # H, W

        # Apply WL/WW to uint8 (like CT default, tweak as needed)
        if ww <= 0:
            ww = 1.0
        lo = wl - ww / 2.0
        hi = wl + ww / 2.0
        arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
        return (arr * 255.0 + 0.5).astype(np.uint8)

    # -------- Internals --------
    def _apply_transform(self):
        # MOVING->FIXED transform
        t = vtk.vtkTransform()
        t.PostMultiply()
        t.Translate(self._tx, self._ty, self._tz)
        t.RotateX(self._rx)
        t.RotateY(self._ry)
        t.RotateZ(self._rz)
        self.transform.DeepCopy(t)
        # Put transform into reslice3d as axes
        self.reslice3d.SetResliceAxes(self.transform.GetMatrix())
        self.reslice3d.Modified()

    def _wire_blend(self):
        self.blend.RemoveAllInputs()
        if self.fixed_reader is not None:
            self.blend.AddInputConnection(self.fixed_reader.GetOutputPort())
        if self.moving_reader is not None:
            # reslice3d converts MOVING into FIXED grid before blending
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


# ------------------------------ Qt UI (three QGraphicsViews) ------------------------------

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
        self.setWindowTitle("Manual Co-Registration Demo (VTK processing, Qt display)")
        self.resize(1400, 900)
        self._build()

    def _build(self):
        root = QtWidgets.QHBoxLayout(self)

        # Left controls
        left = QtWidgets.QWidget(self)
        form = QtWidgets.QFormLayout(left)

        btn_fixed = QtWidgets.QPushButton("Load FIXED DICOM")
        btn_moving = QtWidgets.QPushButton("Load MOVING DICOM")
        btn_fixed.clicked.connect(lambda: self._emit_folder(self.loadFixed))
        btn_moving.clicked.connect(lambda: self._emit_folder(self.loadMoving))
        form.addRow(btn_fixed)
        form.addRow(btn_moving)

        # Slice sliders (ranges will be set after FIXED loads)
        def slice_slider():
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setMinimum(0); s.setMaximum(0); s.setValue(0)
            return s

        self.s_axial = slice_slider()
        self.s_coronal = slice_slider()
        self.s_sagittal = slice_slider()
        self.s_axial.valueChanged.connect(self.axialSliceChanged.emit)
        self.s_coronal.valueChanged.connect(self.coronalSliceChanged.emit)
        self.s_sagittal.valueChanged.connect(self.sagittalSliceChanged.emit)

        form.addRow("Axial Slice (Z)", self.s_axial)
        form.addRow("Coronal Slice (Y)", self.s_coronal)
        form.addRow("Sagittal Slice (X)", self.s_sagittal)

        def slider(mini, maxi, init=0):
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setMinimum(mini); s.setMaximum(maxi); s.setValue(init)
            return s

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

        # Right: three QGraphicsViews (axial / coronal / sagittal)
        right = QtWidgets.QWidget(self)
        grid = QtWidgets.QGridLayout(right)

        def make_view(title: str):
            group = QtWidgets.QGroupBox(title)
            v_layout = QtWidgets.QVBoxLayout(group)
            view = QtWidgets.QGraphicsView()
            view.setRenderHints(QtGui.QPainter.Antialiasing | QtGui.QPainter.SmoothPixmapTransform)
            scene = QtWidgets.QGraphicsScene()
            view.setScene(scene)
            v_layout.addWidget(view)
            return group, view, scene

        g_ax, self.view_ax, self.scene_ax = make_view("Axial")
        g_co, self.view_co, self.scene_co = make_view("Coronal")
        g_sa, self.view_sa, self.scene_sa = make_view("Sagittal")

        grid.addWidget(g_ax, 0, 0)
        grid.addWidget(g_co, 0, 1)
        grid.addWidget(g_sa, 1, 0, 1, 2)

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

        # Defaults
        self.window_level = 40.0
        self.window_width = 400.0

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

        self.ui.axialSliceChanged.connect(lambda i: self.refresh_one(VTKEngine.ORI_AXIAL, i))
        self.ui.coronalSliceChanged.connect(lambda i: self.refresh_one(VTKEngine.ORI_CORONAL, i))
        self.ui.sagittalSliceChanged.connect(lambda i: self.refresh_one(VTKEngine.ORI_SAGITTAL, i))

    # ---- Slots ----
    def on_load_fixed(self, folder: str):
        self.engine.load_fixed(folder)
        self._sync_slice_ranges()
        self._center_sliders()
        self.refresh_all()

    def on_load_moving(self, folder: str):
        self.engine.load_moving(folder)
        self.refresh_all()

    def on_reset(self):
        self.engine.reset_transform()
        self.refresh_all()

    # ---- Helpers ----
    def _sync_slice_ranges(self):
        ext = self.engine.fixed_extent()
        if not ext:
            return
        x0, x1, y0, y1, z0, z1 = ext
        self.ui.s_axial.blockSignals(True)
        self.ui.s_coronal.blockSignals(True)
        self.ui.s_sagittal.blockSignals(True)

        self.ui.s_axial.setMinimum(z0); self.ui.s_axial.setMaximum(z1)
        self.ui.s_coronal.setMinimum(y0); self.ui.s_coronal.setMaximum(y1)
        self.ui.s_sagittal.setMinimum(x0); self.ui.s_sagittal.setMaximum(x1)

        self.ui.s_axial.blockSignals(False)
        self.ui.s_coronal.blockSignals(False)
        self.ui.s_sagittal.blockSignals(False)

    def _center_sliders(self):
        ext = self.engine.fixed_extent()
        if not ext:
            return
        x0, x1, y0, y1, z0, z1 = ext
        self.ui.s_axial.setValue((z0 + z1) // 2)
        self.ui.s_coronal.setValue((y0 + y1) // 2)
        self.ui.s_sagittal.setValue((x0 + x1) // 2)

    def refresh_all(self):
        # Use current slider positions
        self.refresh_one(VTKEngine.ORI_AXIAL, self.ui.s_axial.value())
        self.refresh_one(VTKEngine.ORI_CORONAL, self.ui.s_coronal.value())
        self.refresh_one(VTKEngine.ORI_SAGITTAL, self.ui.s_sagittal.value())

    def refresh_one(self, orientation: str, index: int):
        img = self.engine.get_slice_uint8(orientation, index, self.window_level, self.window_width)
        if img is None:
            return
        h, w = img.shape
        # Make QImage that owns its own copy (avoid lifespan issues)
        qimg = QtGui.QImage(img.data, w, h, w, QtGui.QImage.Format_Grayscale8).copy()
        pix = QtGui.QPixmap.fromImage(qimg)

        if orientation == VTKEngine.ORI_AXIAL:
            self.ui.scene_ax.clear(); self.ui.scene_ax.addPixmap(pix)
            self.ui.view_ax.fitInView(self.ui.scene_ax.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
        elif orientation == VTKEngine.ORI_CORONAL:
            self.ui.scene_co.clear(); self.ui.scene_co.addPixmap(pix)
            self.ui.view_co.fitInView(self.ui.scene_co.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)
        elif orientation == VTKEngine.ORI_SAGITTAL:
            self.ui.scene_sa.clear(); self.ui.scene_sa.addPixmap(pix)
            self.ui.view_sa.fitInView(self.ui.scene_sa.itemsBoundingRect(), QtCore.Qt.KeepAspectRatio)


# ------------------------------ App entry ------------------------------

def main():
    app = QtWidgets.QApplication(sys.argv)
    ui = FusionUI()
    engine = VTKEngine()
    Controller(ui, engine)
    ui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

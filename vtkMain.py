"""
Manual Co‑Registration Demo (VTK + PySide6)

• Loads a FIXED DICOM series and a MOVING DICOM series
• Applies real‑time rigid transform (TX/TY/TZ in mm, RX/RY/RZ in degrees) to MOVING
• Reslices MOVING into FIXED space on the fly (vtkImageReslice)
• Blends MOVING over FIXED (vtkImageBlend) with adjustable opacity
• Renders a single axial slice that always faces the camera

This implements the same methodology 3D Slicer uses for manual registration: keep voxel data immutable,
update a 4×4 matrix live, reslice on display.

Requirements:
    pip install PySide6 vtk

Run:
    python app.py

Test data:
    Use any two DICOM folders (same patient or not) to see the overlay. If you only load FIXED,
    it will display just the base volume.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

from PySide6 import QtCore, QtWidgets
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

import vtk


class FusionPipeline:
    """VTK rendering pipeline for manual co‑registration with 3 orthogonal views."""

    def __init__(self, qvtk: QVTKRenderWindowInteractor, ui: FusionUI):
        self.renwin = qvtk.GetRenderWindow()
        self.iren = self.renwin.GetInteractor()
        self.ui = ui

        # --- Create three renderers (axial, coronal, sagittal) ---
        self.ren_axial = vtk.vtkRenderer()
        self.ren_coronal = vtk.vtkRenderer()
        self.ren_sagittal = vtk.vtkRenderer()
        self.renwin.AddRenderer(self.ren_axial)
        self.renwin.AddRenderer(self.ren_coronal)
        self.renwin.AddRenderer(self.ren_sagittal)

        # Background color
        for ren in [self.ren_axial, self.ren_coronal, self.ren_sagittal]:
            ren.SetBackground(0.08, 0.08, 0.08)

        # Layout renderers horizontally
        self.ren_axial.SetViewport(0.0, 0.0, 0.33, 1.0)
        self.ren_coronal.SetViewport(0.33, 0.0, 0.66, 1.0)
        self.ren_sagittal.SetViewport(0.66, 0.0, 1.0, 1.0)

        # --- Readers ---
        self.fixed_reader: Optional[vtk.vtkDICOMImageReader] = None
        self.moving_reader: Optional[vtk.vtkDICOMImageReader] = None

        # --- Transform ---
        self.transform = vtk.vtkTransform()
        self.transform.PostMultiply()

        # --- Reslice moving image ---
        self.reslice = vtk.vtkImageReslice()
        self.reslice.SetInterpolationModeToLinear()
        self.reslice.SetResliceAxes(self.transform.GetMatrix())

        # --- Blend fixed and moving ---
        self.blend = vtk.vtkImageBlend()
        self.blend.SetOpacity(0, 1.0)  # fixed
        self.blend.SetOpacity(1, 0.5)  # moving

         # --- Create slice mappers and actors for 3 views using vtkImageResliceMapper ---
        self.slice_mappers = {}
        self.slice_actors = {}
        renderers = {"axial": self.ren_axial, "coronal": self.ren_coronal, "sagittal": self.ren_sagittal}
        orientations = {"axial": 2, "coronal": 1, "sagittal": 0}

        # Create a dummy image for initial display
        dummy = vtk.vtkImageData()
        dummy.SetDimensions(1, 1, 1)
        dummy.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)

        for name in ["axial", "coronal", "sagittal"]:
            mapper = vtk.vtkImageSliceMapper()
            mapper.SetInputData(dummy)
            mapper.SetOrientation(orientations[name])
            mapper.SliceFacesCameraOn()
            mapper.SliceAtFocalPointOn()
            actor = vtk.vtkImageSlice()
            actor.SetMapper(mapper)
            self.slice_mappers[name] = mapper
            self.slice_actors[name] = actor
            renderers[name].AddViewProp(actor)

        # --- Parameters ---
        self._tx = self._ty = self._tz = 0.0
        self._rx = self._ry = self._rz = 0.0

        self._render()

    # -------------------- Loading --------------------
    def load_fixed(self, dicom_dir: str):
        reader = vtk.vtkDICOMImageReader()
        reader.SetDirectoryName(str(Path(dicom_dir)))
        reader.Update()
        self.fixed_reader = reader

        self._rebuild_blend_inputs()
        self._sync_output_grid_from_fixed()

        # Connect blend output to mappers for all views
        for name in ["axial", "coronal", "sagittal"]:
            self.slice_mappers[name].SetInputConnection(self.blend.GetOutputPort())

        self._reset_camera_to_volume(reader.GetOutput())
        self._render()
        self._update_slice_sliders()

    def load_moving(self, dicom_dir: str):
        reader = vtk.vtkDICOMImageReader()
        reader.SetDirectoryName(str(Path(dicom_dir)))
        reader.Update()
        self.moving_reader = reader

        self.reslice.SetInputConnection(reader.GetOutputPort())
        self._apply_reslice_axes()
        self._rebuild_blend_inputs()
        self._sync_output_grid_from_fixed()

        # Connect blend output to reslicers
        for reslice in [self.reslice_axial, self.reslice_coronal, self.reslice_sagittal]:
            reslice.SetInputConnection(self.blend.GetOutputPort())

        for name, reslice in zip(
            ["axial", "coronal", "sagittal"],
            [self.reslice_axial, self.reslice_coronal, self.reslice_sagittal]
        ):
            self.slice_mappers[name].SetInputConnection(reslice.GetOutputPort())

        if self.fixed_reader is None:
            self._reset_camera_to_volume(reader.GetOutput())

        self._render()
        self._update_slice_sliders()

    def _update_slice_sliders(self):
        """Update slider ranges and set initial positions to middle slice."""
        if self.fixed_reader is None:
            return
        extent = self.fixed_reader.GetOutput().GetExtent()

        # Axial (Z)
        self.ui.s_axial.setMinimum(extent[4])
        self.ui.s_axial.setMaximum(extent[5])
        self.ui.s_axial.blockSignals(True)
        self.ui.s_axial.setValue((extent[4] + extent[5]) // 2)
        self.ui.s_axial.blockSignals(False)

        # Coronal (Y) — flipped orientation
        self.ui.s_coronal.setMinimum(extent[2])
        self.ui.s_coronal.setMaximum(extent[3])
        self.ui.s_coronal.blockSignals(True)
        self.ui.s_coronal.setValue((extent[2] + extent[3]) // 2)
        self.ui.s_coronal.blockSignals(False)

        # Sagittal (X) — flipped orientation
        self.ui.s_sagittal.setMinimum(extent[0])
        self.ui.s_sagittal.setMaximum(extent[1])
        self.ui.s_sagittal.blockSignals(True)
        self.ui.s_sagittal.setValue((extent[0] + extent[1]) // 2)
        self.ui.s_sagittal.blockSignals(False)

        # Apply initial slices
        self.set_slice("axial", self.ui.s_axial.value())
        self.set_slice("coronal", self.ui.s_coronal.value())
        self.set_slice("sagittal", self.ui.s_sagittal.value())



    # -------------------- Transform updates --------------------
    def set_translation(self, tx: float, ty: float, tz: float):
        self._tx, self._ty, self._tz = float(tx), float(ty), float(tz)
        self._rebuild_transform()

    def set_rotation_deg(self, rx: float, ry: float, rz: float):
        self._rx, self._ry, self._rz = float(rx), float(ry), float(rz)
        self._rebuild_transform()

    def set_opacity(self, alpha: float):
        self.blend.SetOpacity(1, max(0.0, min(1.0, float(alpha))))
        self._render()

    def reset_transform(self):
        self._tx = self._ty = self._tz = 0.0
        self._rx = self._ry = self._rz = 0.0
        self.transform.Identity()
        self._apply_reslice_axes()
        self._render()

    # -------------------- Internals --------------------
    def _rebuild_transform(self):
        t = vtk.vtkTransform()
        t.PostMultiply()
        t.Translate(self._tx, self._ty, self._tz)
        t.RotateX(self._rx)
        t.RotateY(self._ry)
        t.RotateZ(self._rz)
        self.transform.SetMatrix(t.GetMatrix())
        self._apply_reslice_axes()
        self._render()

    def _apply_reslice_axes(self):
        self.reslice.SetResliceAxes(self.transform.GetMatrix())

    def _rebuild_blend_inputs(self):
        self.blend.RemoveAllInputs()
        if self.fixed_reader is not None:
            self.blend.AddInputConnection(self.fixed_reader.GetOutputPort())
        if self.moving_reader is not None:
            self.blend.AddInputConnection(self.reslice.GetOutputPort())

    def _connect_display(self):
        for mapper in self.slice_mappers.values():
            if self.blend.GetNumberOfInputConnections(0) > 0:
                mapper.SetInputConnection(self.blend.GetOutputPort())
            else:
                # placeholder to avoid errors
                placeholder = vtk.vtkImageData()
                placeholder.SetDimensions(1, 1, 1)
                placeholder.AllocateScalars(vtk.VTK_UNSIGNED_CHAR, 1)
                mapper.SetInputData(placeholder)

    def _sync_output_grid_from_fixed(self):
        if self.fixed_reader is None:
            return
        fixed = self.fixed_reader.GetOutput()
        self.reslice.SetOutputSpacing(fixed.GetSpacing())
        self.reslice.SetOutputOrigin(fixed.GetOrigin())
        self.reslice.SetOutputExtent(fixed.GetExtent())

    def _reset_camera_to_volume(self, image: vtk.vtkImageData):
        spacing = image.GetSpacing()
        origin = image.GetOrigin()
        extent = image.GetExtent()
        center = [
            origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0],
            origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1],
            origin[2] + 0.5 * (extent[4] + extent[5]) * spacing[2],
        ]

        # Axial: camera along +Z, up = +Y (normal)
        cam = self.ren_axial.GetActiveCamera()
        cam.SetFocalPoint(*center)
        cam.SetPosition(center[0], center[1], center[2] + 500)
        cam.SetViewUp(0, 1, 0)
        self.ren_axial.GetActiveCamera().ParallelProjectionOn()
        self.ren_axial.ResetCamera()
        self.ren_axial.ResetCameraClippingRange()

        # Coronal: camera along +Y, up = +Z (so the XZ plane is visible)
        cam = self.ren_coronal.GetActiveCamera()
        cam.SetFocalPoint(center[0], center[1], center[2])
        cam.SetPosition(center[0], center[1] + 500, center[2])
        cam.SetViewUp(0, 0, 1)
        self.ren_coronal.GetActiveCamera().ParallelProjectionOn()
        self.ren_coronal.ResetCamera()
        self.ren_coronal.ResetCameraClippingRange()

        # Sagittal: camera along +X, up = +Z (so the YZ plane is visible)
        cam = self.ren_sagittal.GetActiveCamera()
        cam.SetFocalPoint(center[0], center[1], center[2])
        cam.SetPosition(center[0] + 500, center[1], center[2])
        cam.SetViewUp(0, 0, 1)
        self.ren_sagittal.GetActiveCamera().ParallelProjectionOn()
        self.ren_sagittal.ResetCamera()
        self.ren_sagittal.ResetCameraClippingRange()



    def _render(self):
        for ren in [self.ren_axial, self.ren_coronal, self.ren_sagittal]:
            ren.ResetCameraClippingRange()
        self.renwin.Render()

    def set_slice(self, orientation: str, index: int):
        if self.fixed_reader is None:
            return

        image = self.fixed_reader.GetOutput()
        extent = image.GetExtent()

        # Clamp index to valid range for each orientation
        if orientation == "axial":
            index = max(extent[4], min(extent[5], index))
        elif orientation == "coronal":
            index = max(extent[2], min(extent[3], index))
        elif orientation == "sagittal":
            index = max(extent[0], min(extent[1], index))

        # Set the slice number for the mapper
        self.slice_mappers[orientation].SetSliceNumber(index)
        self._render()


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
        self.setWindowTitle("Manual Co‑Registration Demo (VTK)")
        self.resize(1280, 800)
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

        # --- New slice sliders ---
        def slice_slider():
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setMinimum(0)
            s.setMaximum(100)  # temporary, updated dynamically after loading
            s.setValue(0)
            return s

        self.s_axial = slice_slider()
        self.s_coronal = slice_slider()
        self.s_sagittal = slice_slider()

        self.s_axial.valueChanged.connect(lambda v: self.axialSliceChanged.emit(v))
        self.s_coronal.valueChanged.connect(lambda v: self.coronalSliceChanged.emit(v))
        self.s_sagittal.valueChanged.connect(lambda v: self.sagittalSliceChanged.emit(v))

        form.addRow("Axial Slice", self.s_axial)
        form.addRow("Coronal Slice", self.s_coronal)
        form.addRow("Sagittal Slice", self.s_sagittal)

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

        # Right: VTK view
        self.qvtk = QVTKRenderWindowInteractor(self)

        root.addWidget(left, 0)
        root.addWidget(self.qvtk, 1)

    def _emit_folder(self, signal: QtCore.SignalInstance):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select DICOM folder")
        if folder:
            signal.emit(folder)


class Controller(QtCore.QObject):
    def __init__(self, ui: FusionUI, pipeline: FusionPipeline):
        super().__init__()
        self.ui = ui
        self.pipeline = pipeline
        self._wire()

    def _wire(self):
        self.ui.loadFixed.connect(self.pipeline.load_fixed)
        self.ui.loadMoving.connect(self.pipeline.load_moving)
        self.ui.txChanged.connect(lambda v: self.pipeline.set_translation(v, self.pipeline._ty, self.pipeline._tz))
        self.ui.tyChanged.connect(lambda v: self.pipeline.set_translation(self.pipeline._tx, v, self.pipeline._tz))
        self.ui.tzChanged.connect(lambda v: self.pipeline.set_translation(self.pipeline._tx, self.pipeline._ty, v))
        self.ui.rxChanged.connect(lambda v: self.pipeline.set_rotation_deg(v, self.pipeline._ry, self.pipeline._rz))
        self.ui.ryChanged.connect(lambda v: self.pipeline.set_rotation_deg(self.pipeline._rx, v, self.pipeline._rz))
        self.ui.rzChanged.connect(lambda v: self.pipeline.set_rotation_deg(self.pipeline._rx, self.pipeline._ry, v))
        self.ui.opacityChanged.connect(self.pipeline.set_opacity)
        self.ui.resetRequested.connect(self.pipeline.reset_transform)

        self.ui.axialSliceChanged.connect(lambda v: self.pipeline.set_slice("axial", v))
        self.ui.coronalSliceChanged.connect(lambda v: self.pipeline.set_slice("coronal", v))
        self.ui.sagittalSliceChanged.connect(lambda v: self.pipeline.set_slice("sagittal", v))


def main():
    app = QtWidgets.QApplication(sys.argv)
    ui = FusionUI()
    pipeline = FusionPipeline(ui.qvtk, ui)
    Controller(ui, pipeline)

    # Start VTK interactor
    ui.qvtk.Initialize()
    ui.qvtk.Start()

    ui.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()

from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
import vtk
from vtkmodules.util import numpy_support

# ------------------------------ VTK Processing Engine ------------------------------

class VTKEngine:
    ORI_AXIAL = "axial"
    ORI_CORONAL = "coronal"
    ORI_SAGITTAL = "sagittal"

    def __init__(self):
        self.fixed_reader = None
        self.moving_reader = None
        self._blend_dirty = True

        # Transform parameters
        self._tx = self._ty = self._tz = 0.0
        self._rx = self._ry = self._rz = 0.0
        self.transform = vtk.vtkTransform()
        self.transform.PostMultiply()

        # Reslice moving image
        self.reslice3d = vtk.vtkImageReslice()
        self.reslice3d.SetInterpolationModeToLinear()
        self.reslice3d.SetBackgroundLevel(0.0)
        self.reslice3d.SetAutoCropOutput(1)

        # Blend
        self.blend = vtk.vtkImageBlend()
        self.blend.SetOpacity(0, 1.0)
        self.blend.SetOpacity(1, 0.5)

        # Offscreen renderer (unused for display but kept for pipeline completeness)
        self.renderer = vtk.vtkRenderer()
        self.render_window = vtk.vtkRenderWindow()
        self.render_window.SetOffScreenRendering(1)
        self.render_window.AddRenderer(self.renderer)
        self.vtk_image_actor = vtk.vtkImageActor()
        self.renderer.AddActor(self.vtk_image_actor)

    def load_fixed(self, dicom_dir: str) -> bool:
        files = list(Path(dicom_dir).glob("*"))
        if not any(f.is_file() for f in files):
            return False
        r = vtk.vtkDICOMImageReader()
        r.SetDirectoryName(str(Path(dicom_dir)))
        r.Update()
        self.fixed_reader = r

        # --- Set background level to lowest pixel value in fixed DICOM ---
        img = r.GetOutput()
        scalars = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())
        if scalars is not None and scalars.size > 0:
            min_val = float(scalars.min())
            self.reslice3d.SetBackgroundLevel(min_val)

        self._wire_blend()
        self._sync_reslice_output_to_fixed()
        return True

    def load_moving(self, dicom_dir: str) -> bool:
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
        self._blend_dirty = True

    def set_translation(self, tx: float, ty: float, tz: float):
        self._tx, self._ty, self._tz = float(tx), float(ty), float(tz)
        self._apply_transform()

    def set_rotation_deg(self, rx: float, ry: float, rz: float, orientation=None, slice_idx=None):
        self._rx, self._ry, self._rz = float(rx), float(ry), float(rz)
        self._apply_transform(orientation, slice_idx)

    def reset_transform(self):
        self._tx = self._ty = self._tz = 0.0
        self._rx = self._ry = self._rz = 0.0
        self.transform.Identity()
        self._blend_dirty = True
        self._apply_transform()

    def fixed_extent(self):
        if not self.fixed_reader:
            return None
        return self.fixed_reader.GetOutput().GetExtent()

    # ---------------- NEW FUNCTION ----------------
    def get_slice_numpy(self, orientation: str, slice_idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Returns (fixed_slice, moving_slice) as numpy arrays (uint8 2D), both aligned
        to the fixed volume’s geometry. Each can be None if missing.
        """
        if self.fixed_reader is None:
            return None, None

        fixed_img = self.fixed_reader.GetOutput()
        moving_img = self.reslice3d.GetOutput() if self.moving_reader else None

        # Update reslice if moving present
        if self.moving_reader:
            self.reslice3d.Update()

        def vtk_to_np_slice(img, orientation, slice_idx, window_center=40, window_width=400):
            if img is None or img.GetPointData() is None:
                return None
            extent = img.GetExtent()
            nx = extent[1] - extent[0] + 1
            ny = extent[3] - extent[2] + 1
            nz = extent[5] - extent[4] + 1
            scalars = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())
            if scalars is None:
                return None
            arr = scalars.reshape((nz, ny, nx))

            if orientation == VTKEngine.ORI_AXIAL:
                z = int(np.clip(slice_idx - extent[4], 0, nz - 1))
                arr2d = arr[z, :, :]
            elif orientation == VTKEngine.ORI_CORONAL:
                y = int(np.clip(slice_idx - extent[2], 0, ny - 1))
                arr2d = arr[:, y, :]
            elif orientation == VTKEngine.ORI_SAGITTAL:
                x = int(np.clip(slice_idx - extent[0], 0, nx - 1))
                arr2d = arr[:, :, x]
            else:
                return None

            # --- Apply CT windowing ---
            arr2d = arr2d.astype(np.float32)
            c = window_center
            w = window_width
            arr2d = np.clip((arr2d - (c - 0.5)) / (w - 1) + 0.5, 0, 1)
            arr2d = (arr2d * 255.0).astype(np.uint8)
            return np.ascontiguousarray(arr2d)

        fixed_slice = vtk_to_np_slice(fixed_img, orientation, slice_idx, window_center=40, window_width=400)
        moving_slice = vtk_to_np_slice(moving_img, orientation, slice_idx, window_center=40, window_width=400) if moving_img else None
        return fixed_slice, moving_slice

    # ---------------- REFACTORED OLD FUNCTION ----------------
    def get_slice_qimage(self, orientation: str, slice_idx: int, fixed_color="Purple", moving_color="Green", coloring_enabled=True) -> QtGui.QImage:
        fixed_slice, moving_slice = self.get_slice_numpy(orientation, slice_idx)
        if fixed_slice is None:
            return QtGui.QImage()

        h, w = fixed_slice.shape

        # Get current blend factor for moving image (0.0 = only fixed, 1.0 = only moving)
        blend = self.blend.GetOpacity(1) if self.moving_reader is not None else 0.0

        # Color mapping dictionary
        color_map = {
            "Grayscale":   lambda arr: arr,  # No coloring, just the original grayscale array
            "Green":       lambda arr: np.stack([np.zeros_like(arr), arr, np.zeros_like(arr)], axis=-1),
            "Purple":      lambda arr: np.stack([arr, np.zeros_like(arr), arr], axis=-1),
            "Blue":        lambda arr: np.stack([np.zeros_like(arr), np.zeros_like(arr), arr], axis=-1),
            "Yellow":      lambda arr: np.stack([arr, arr, np.zeros_like(arr)], axis=-1),
            "Red":         lambda arr: np.stack([arr, np.zeros_like(arr), np.zeros_like(arr)], axis=-1),
            "Cyan":        lambda arr: np.stack([np.zeros_like(arr), arr, arr], axis=-1),
        }

        # If coloring is disabled, always show both layers as standard grayscale and blend as uint8, no color mapping
        if not coloring_enabled:
            if moving_slice is None:
                arr2d = fixed_slice
            else:
                # Use the blend opacity as a true alpha for the moving image
                alpha = self.blend.GetOpacity(1) if self.moving_reader is not None else 0.5
                arr2d = (fixed_slice.astype(np.float32) * (1 - alpha) +
                         moving_slice.astype(np.float32) * alpha).astype(np.uint8)
            h, w = arr2d.shape
            qimg = QtGui.QImage(arr2d.data, w, h, w, QtGui.QImage.Format_Grayscale8)
            qimg = qimg.copy()
            # Aspect ratio correction (unchanged)
            if self.fixed_reader is not None:
                spacing = self.fixed_reader.GetOutput().GetSpacing()
                if orientation == VTKEngine.ORI_AXIAL:
                    spacing_y, spacing_x = spacing[1], spacing[0]
                elif orientation == VTKEngine.ORI_CORONAL:
                    spacing_y, spacing_x = spacing[2], spacing[0]
                elif orientation == VTKEngine.ORI_SAGITTAL:
                    spacing_y, spacing_x = spacing[2], spacing[1]
                else:
                    spacing_y, spacing_x = 1.0, 1.0
                phys_h = h * spacing_y
                phys_w = w * spacing_x
                aspect_ratio = phys_w / phys_h if phys_h != 0 else 1.0
                display_h = h
                display_w = int(round(h * aspect_ratio))
                qimg = qimg.scaled(display_w, display_h, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
            return qimg

        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        fixed_f = fixed_slice.astype(np.float32)
        if moving_slice is None:
            # Only fixed: use selected color
            if fixed_color == "Grayscale":
                # Show as single-channel grayscale
                qimg = QtGui.QImage(fixed_slice.data, w, h, w, QtGui.QImage.Format_Grayscale8)
                qimg = qimg.copy()
                # Aspect ratio correction (unchanged)
                if self.fixed_reader is not None:
                    spacing = self.fixed_reader.GetOutput().GetSpacing()
                    if orientation == VTKEngine.ORI_AXIAL:
                        spacing_y, spacing_x = spacing[1], spacing[0]
                    elif orientation == VTKEngine.ORI_CORONAL:
                        spacing_y, spacing_x = spacing[2], spacing[0]
                    elif orientation == VTKEngine.ORI_SAGITTAL:
                        spacing_y, spacing_x = spacing[2], spacing[1]
                    else:
                        spacing_y, spacing_x = 1.0, 1.0
                    phys_h = h * spacing_y
                    phys_w = w * spacing_x
                    aspect_ratio = phys_w / phys_h if phys_h != 0 else 1.0
                    display_h = h
                    display_w = int(round(h * aspect_ratio))
                    qimg = qimg.scaled(display_w, display_h, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
                return qimg
            else:
                rgb = np.clip(color_map.get(fixed_color, color_map["Purple"])(fixed_slice), 0, 255).astype(np.uint8)
        else:
            moving_f = moving_slice.astype(np.float32)
            if blend <= 0.5:
                fixed_opacity = 1.0
                moving_opacity = blend * 2.0
            else:
                fixed_opacity = 2.0 * (1.0 - blend)
                moving_opacity = 1.0
            if fixed_color == "Grayscale":
                fixed_rgb = np.stack([np.clip(fixed_opacity * fixed_f, 0, 255).astype(np.uint8)]*3, axis=-1)
            else:
                fixed_rgb = color_map.get(fixed_color, color_map["Purple"])(np.clip(fixed_opacity * fixed_f, 0, 255).astype(np.uint8))
            if moving_color == "Grayscale":
                moving_rgb = np.stack([np.clip(moving_opacity * moving_f, 0, 255).astype(np.uint8)]*3, axis=-1)
            else:
                moving_rgb = color_map.get(moving_color, color_map["Green"])(np.clip(moving_opacity * moving_f, 0, 255).astype(np.uint8))
            rgb = np.clip(fixed_rgb + moving_rgb, 0, 255).astype(np.uint8)

        qimg = QtGui.QImage(rgb.data, w, h, 3 * w, QtGui.QImage.Format_RGB888)
        qimg = qimg.copy()

        # --- Aspect ratio correction ---
        if self.fixed_reader is not None:
            spacing = self.fixed_reader.GetOutput().GetSpacing()
            # spacing: (sx, sy, sz)
            if orientation == VTKEngine.ORI_AXIAL:
                # arr2d shape: (y, x) → spacing: (sy, sx)
                spacing_y, spacing_x = spacing[1], spacing[0]
            elif orientation == VTKEngine.ORI_CORONAL:
                # arr2d shape: (z, x) → spacing: (sz, sx)
                spacing_y, spacing_x = spacing[2], spacing[0]
            elif orientation == VTKEngine.ORI_SAGITTAL:
                # arr2d shape: (z, y) → spacing: (sz, sy)
                spacing_y, spacing_x = spacing[2], spacing[1]
            else:
                spacing_y, spacing_x = 1.0, 1.0

            # Calculate the physical size of the image
            phys_h = h * spacing_y
            phys_w = w * spacing_x

            # Scale the image so that the displayed aspect ratio matches the physical aspect ratio
            aspect_ratio = phys_w / phys_h if phys_h != 0 else 1.0
            display_h = h
            display_w = int(round(h * aspect_ratio))
            qimg = qimg.scaled(display_w, display_h, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)

        return qimg

    # -------- Internals --------
    def _apply_transform(self, orientation=None, slice_idx=None):
        if not self.fixed_reader or not self.moving_reader:
            return
        img = self.fixed_reader.GetOutput()
        center = np.array(img.GetCenter())

        # If orientation and slice_idx are provided, compute the slice center
        if orientation is not None and slice_idx is not None:
            extent = img.GetExtent()
            spacing = img.GetSpacing()
            origin = img.GetOrigin()
            if orientation == VTKEngine.ORI_AXIAL:
                z = int(np.clip(slice_idx, extent[4], extent[5]))
                center = np.array([
                    origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0],
                    origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1],
                    origin[2] + z * spacing[2]
                ])
            elif orientation == VTKEngine.ORI_CORONAL:
                y = int(np.clip(slice_idx, extent[2], extent[3]))
                center = np.array([
                    origin[0] + 0.5 * (extent[0] + extent[1]) * spacing[0],
                    origin[1] + y * spacing[1],
                    origin[2] + 0.5 * (extent[4] + extent[5]) * spacing[2]
                ])
            elif orientation == VTKEngine.ORI_SAGITTAL:
                x = int(np.clip(slice_idx, extent[0], extent[1]))
                center = np.array([
                    origin[0] + x * spacing[0],
                    origin[1] + 0.5 * (extent[2] + extent[3]) * spacing[1],
                    origin[2] + 0.5 * (extent[4] + extent[5]) * spacing[2]
                ])
        t = vtk.vtkTransform()
        t.PostMultiply()
        t.Translate(-center)
        t.RotateX(self._rx)
        t.RotateY(self._ry)
        t.RotateZ(self._rz)
        t.Translate(center)
        t.Translate(self._tx, self._ty, self._tz)
        self.transform.DeepCopy(t)
        self.reslice3d.SetResliceAxes(self.transform.GetMatrix())
        self.reslice3d.Modified()
        self._blend_dirty = True

    def _wire_blend(self):
        self.blend.RemoveAllInputs()
        if self.fixed_reader is not None:
            self.blend.AddInputConnection(self.fixed_reader.GetOutputPort())
        if self.moving_reader is not None:
            self.blend.AddInputConnection(self.reslice3d.GetOutputPort())
        self._blend_dirty = True 

    def _sync_reslice_output_to_fixed(self):
        if self.fixed_reader is None:
            return
        fixed = self.fixed_reader.GetOutput()
        self.reslice3d.SetOutputSpacing(fixed.GetSpacing())
        self.reslice3d.SetOutputOrigin(fixed.GetOrigin())
        self.reslice3d.SetOutputExtent(fixed.GetExtent())
        self.reslice3d.Modified()

    def set_interpolation_linear(self, linear: bool = True):
        if linear:
            self.reslice3d.SetInterpolationModeToLinear()
        else:
            self.reslice3d.SetInterpolationModeToNearestNeighbor()

# ------------------------------ Transform Matrix Dialog ------------------------------

class TransformMatrixDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Transformation Matrix (4x4)")
        self.resize(400, 200)

        layout = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget(4, 4)
        self.table.horizontalHeader().hide()
        self.table.verticalHeader().hide()
        self.table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        layout.addWidget(self.table)

        # Initialize with identity matrix
        self._init_identity_matrix()

    def _init_identity_matrix(self):
        identity = [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
        for i in range(4):
            for j in range(4):
                item = QtWidgets.QTableWidgetItem(f"{identity[i][j]:.2f}")
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(i, j, item)

    def set_matrix(self, vtk_transform: vtk.vtkTransform):
        mat = vtk_transform.GetMatrix()
        for i in range(4):
            for j in range(4):
                item = QtWidgets.QTableWidgetItem(f"{mat.GetElement(i,j):.2f}")
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(i, j, item)

# ------------------------------ Qt Display ------------------------------

class SliceGraphicsView(QtWidgets.QGraphicsView):
    def __init__(self):
        super().__init__()
        self.scene = QtWidgets.QGraphicsScene()
        self.setScene(self.scene)
        self.pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)

    def set_slice_qimage(self, qimg: QtGui.QImage):
        pix = QtGui.QPixmap.fromImage(qimg)
        self.pixmap_item.setPixmap(pix)
        self.fitInView(self.pixmap_item, QtCore.Qt.KeepAspectRatio)

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
        self.setWindowTitle("Manual Co-Registration Demo (Offscreen VTK + QGraphicsView)")
        self.resize(1400, 900)
        self._matrix_dialog: TransformMatrixDialog | None = None
        self._build()

    def _build(self):
        root = QtWidgets.QHBoxLayout(self)
        left = QtWidgets.QWidget(self)
        form = QtWidgets.QFormLayout(left)

        btn_fixed = QtWidgets.QPushButton("Load FIXED DICOM")
        btn_moving = QtWidgets.QPushButton("Load MOVING DICOM")
        btn_fixed.clicked.connect(lambda: self._emit_folder(self.loadFixed))
        btn_moving.clicked.connect(lambda: self._emit_folder(self.loadMoving))
        form.addRow(btn_fixed)
        form.addRow(btn_moving)

        # --- Color selection dropdowns ---
        self.fixed_color_combo = QtWidgets.QComboBox()
        self.moving_color_combo = QtWidgets.QComboBox()
        color_options = ["Grayscale", "Green", "Purple", "Blue", "Yellow", "Red", "Cyan"]
        self.fixed_color_combo.addItems(color_options)
        self.moving_color_combo.addItems(color_options)
        self.fixed_color_combo.setCurrentText("Purple")
        self.moving_color_combo.setCurrentText("Green")
        form.addRow("Fixed Layer Color", self.fixed_color_combo)
        form.addRow("Moving Layer Color", self.moving_color_combo)

        # --- Add checkbox to enable/disable coloring ---
        self.coloring_checkbox = QtWidgets.QCheckBox("Enable Coloring")
        self.coloring_checkbox.setChecked(True)
        form.addRow(self.coloring_checkbox)

        def slider(mini, maxi, init=0):
            s = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            s.setMinimum(mini)
            s.setMaximum(maxi)
            s.setValue(init)
            s.setTickInterval(max(1, (maxi-mini)//10))
            s.setSingleStep(1)
            s.setPageStep(max(1, (maxi-mini)//10))
            return s

        self.s_axial = slider(0,0,0)
        self.s_coronal = slider(0,0,0)
        self.s_sagittal = slider(0,0,0)
        self.s_axial.valueChanged.connect(self.axialSliceChanged.emit)
        self.s_coronal.valueChanged.connect(self.coronalSliceChanged.emit)
        self.s_sagittal.valueChanged.connect(self.sagittalSliceChanged.emit)
        form.addRow("Axial Slice", self.s_axial)
        form.addRow("Coronal Slice", self.s_coronal)
        form.addRow("Sagittal Slice", self.s_sagittal)

        # Transforms
        self.s_tx = slider(-200,200,0)
        self.s_ty = slider(-200,200,0)
        self.s_tz = slider(-200,200,0)
        self.s_rx = slider(-1800,1800,0)
        self.s_ry = slider(-1800,1800,0)
        self.s_rz = slider(-1800,1800,0)
        self.s_op = slider(0,100,50)

        self.s_tx.valueChanged.connect(lambda v: self.txChanged.emit(float(v)))
        self.s_ty.valueChanged.connect(lambda v: self.tyChanged.emit(float(v)))
        self.s_tz.valueChanged.connect(lambda v: self.tzChanged.emit(float(v)))
        self.s_rx.valueChanged.connect(lambda v: self.rxChanged.emit(float(v/10.0)))
        self.s_ry.valueChanged.connect(lambda v: self.ryChanged.emit(float(v/10.0)))
        self.s_rz.valueChanged.connect(lambda v: self.rzChanged.emit(float(v/10.0)))
        self.s_op.valueChanged.connect(lambda v: self.opacityChanged.emit(float(v)/100.0))

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

        btn_show_matrix = QtWidgets.QPushButton("Show Transform Matrix")
        btn_show_matrix.clicked.connect(self._show_matrix_dialog)
        form.addRow(btn_show_matrix)

        # Graphics views for slices
        right = QtWidgets.QWidget(self)
        grid = QtWidgets.QGridLayout(right)
        self.viewer_ax = SliceGraphicsView()
        self.viewer_co = SliceGraphicsView()
        self.viewer_sa = SliceGraphicsView()
        grid.addWidget(self.viewer_ax,0,0)
        grid.addWidget(self.viewer_co,0,1)
        grid.addWidget(self.viewer_sa,1,0,1,2)

        root.addWidget(left,1)
        root.addWidget(right,2)

    def _emit_folder(self, signal: QtCore.SignalInstance):
        folder = QtWidgets.QFileDialog.getExistingDirectory(self, "Select DICOM folder")
        if folder:
            signal.emit(folder)

    def _show_matrix_dialog(self):
        if self._matrix_dialog is None:
            self._matrix_dialog = TransformMatrixDialog(self)
        # Always update the matrix to the current transform before showing
        # Find the controller and engine
        parent = self.parent()
        engine = None
        # Try to find the engine from the parent chain (robust for your structure)
        while parent is not None:
            if hasattr(parent, "engine"):
                engine = parent.engine
                break
            parent = getattr(parent, "parent", lambda: None)()
        # Fallback: try global variable if needed
        if engine is None and "engine" in globals():
            engine = globals()["engine"]
        if engine is not None:
            self._matrix_dialog.set_matrix(engine.transform)
        self._matrix_dialog.show()
        self._matrix_dialog.raise_()
        self._matrix_dialog.activateWindow()

# ------------------------------ Controller ------------------------------

class Controller(QtCore.QObject):
    DEBOUNCE_MS = 0
    def __init__(self, ui: FusionUI, engine: VTKEngine):
        super().__init__()
        self.ui = ui
        self.engine = engine
        self._debounce_timer = QtCore.QTimer(singleShot=True)
        self._debounce_timer.timeout.connect(self.refresh_all)
        self.fixed_color = "Purple"
        self.moving_color = "Green"
        self.coloring_enabled = True
        self._wire()

    def _wire(self):
        self.ui.loadFixed.connect(self.on_load_fixed)
        self.ui.loadMoving.connect(self.on_load_moving)
        self.ui.txChanged.connect(lambda v: self._update_transform())
        self.ui.tyChanged.connect(lambda v: self._update_transform())
        self.ui.tzChanged.connect(lambda v: self._update_transform())
        self.ui.rxChanged.connect(lambda v: self._update_transform())
        self.ui.ryChanged.connect(lambda v: self._update_transform())
        self.ui.rzChanged.connect(lambda v: self._update_transform())
        self.ui.opacityChanged.connect(lambda a: self._update_opacity(a))
        self.ui.resetRequested.connect(self.on_reset)
        self.ui.axialSliceChanged.connect(lambda i: self.refresh_slice("axial",i))
        self.ui.coronalSliceChanged.connect(lambda i: self.refresh_slice("coronal",i))
        self.ui.sagittalSliceChanged.connect(lambda i: self.refresh_slice("sagittal",i))
        self.ui.fixed_color_combo.currentTextChanged.connect(self._on_fixed_color_changed)
        self.ui.moving_color_combo.currentTextChanged.connect(self._on_moving_color_changed)
        self.ui.coloring_checkbox.stateChanged.connect(self._on_coloring_checkbox_changed)

    def _update_transform(self):
        # Determine which orientation and slice to use for rotation center
        # Priority: axial, then coronal, then sagittal (could be improved to use the last interacted)
        orientation = None
        slice_idx = None
        if self.ui.s_axial.hasFocus():
            orientation = VTKEngine.ORI_AXIAL
            slice_idx = self.ui.s_axial.value()
        elif self.ui.s_coronal.hasFocus():
            orientation = VTKEngine.ORI_CORONAL
            slice_idx = self.ui.s_coronal.value()
        elif self.ui.s_sagittal.hasFocus():
            orientation = VTKEngine.ORI_SAGITTAL
            slice_idx = self.ui.s_sagittal.value()
        else:
            # Default to axial if none focused
            orientation = VTKEngine.ORI_AXIAL
            slice_idx = self.ui.s_axial.value()

        self.engine.set_translation(
            self.ui.s_tx.value(), self.ui.s_ty.value(), self.ui.s_tz.value()
        )
        self.engine.set_rotation_deg(
            self.ui.s_rx.value()/10.0, self.ui.s_ry.value()/10.0, self.ui.s_rz.value()/10.0,
            orientation=orientation, slice_idx=slice_idx
        )
        self.engine.set_interpolation_linear(False)
        self._debounce_timer.start(self.DEBOUNCE_MS)

        # update matrix dialog if open
        if self.ui._matrix_dialog:
            self.ui._matrix_dialog.set_matrix(self.engine.transform)

    def _update_opacity(self, a: float):
        self.engine.set_opacity(a)
        self._debounce_timer.start(self.DEBOUNCE_MS)

    def on_load_fixed(self, folder:str):
        if not self.engine.load_fixed(folder):
            QtWidgets.QMessageBox.warning(self.ui,"Error","No DICOM files found in folder!")
            return
        self._sync_slice_ranges()
        self.refresh_all()

    def on_load_moving(self, folder:str):
        if not self.engine.load_moving(folder):
            QtWidgets.QMessageBox.warning(self.ui,"Error","No DICOM files found in folder!")
            return
        self.refresh_all()

    def on_reset(self):
        self.engine.reset_transform()
        self.refresh_all()
        self.ui.s_rx.setValue(0)
        self.ui.s_ry.setValue(0)
        self.ui.s_rz.setValue(0)
        self.ui.s_tx.setValue(0)
        self.ui.s_ty.setValue(0)
        self.ui.s_tz.setValue(0)

        if self.ui._matrix_dialog:
            self.ui._matrix_dialog.set_matrix(self.engine.transform)

    def _sync_slice_ranges(self):
        ext = self.engine.fixed_extent()
        if not ext:
            return
        x0,x1,y0,y1,z0,z1 = ext
        self.ui.s_axial.setMinimum(z0); self.ui.s_axial.setMaximum(z1); self.ui.s_axial.setValue((z0+z1)//2)
        self.ui.s_coronal.setMinimum(y0); self.ui.s_coronal.setMaximum(y1); self.ui.s_coronal.setValue((y0+y1)//2)
        self.ui.s_sagittal.setMinimum(x0); self.ui.s_sagittal.setMaximum(x1); self.ui.s_sagittal.setValue((x0+x1)//2)

    def refresh_all(self):
        self.engine.set_interpolation_linear(False)
        self.refresh_slice("axial",self.ui.s_axial.value())
        self.refresh_slice("coronal",self.ui.s_coronal.value())
        self.refresh_slice("sagittal",self.ui.s_sagittal.value())

    def refresh_slice(self, orientation:str, idx:int):
        qimg = self.engine.get_slice_qimage(
            orientation, idx,
            fixed_color=self.fixed_color,
            moving_color=self.moving_color,
            coloring_enabled=self.coloring_enabled
        )
        if orientation=="axial":
            self.ui.viewer_ax.set_slice_qimage(qimg)
        elif orientation=="coronal":
            self.ui.viewer_co.set_slice_qimage(qimg)
        elif orientation=="sagittal":
            self.ui.viewer_sa.set_slice_qimage(qimg)

    def _on_coloring_checkbox_changed(self, state):
        self.coloring_enabled = bool(state)
        self.refresh_all()

    def _on_fixed_color_changed(self, color):
        self.fixed_color = color
        self.refresh_all()

    def _on_moving_color_changed(self, color):
        self.moving_color = color
        self.refresh_all()

# ------------------------------ Main ------------------------------

if __name__=="__main__":
    app = QtWidgets.QApplication(sys.argv)
    ui = FusionUI()
    engine = VTKEngine()
    ctrl = Controller(ui,engine)
    ui.show()
    sys.exit(app.exec())

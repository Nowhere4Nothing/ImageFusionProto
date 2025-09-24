from __future__ import annotations
import sys, os
from pathlib import Path
import numpy as np
from PySide6 import QtCore, QtWidgets, QtGui
import vtk
from vtkmodules.util import numpy_support
import pydicom
import tempfile, shutil, atexit, gc, glob
import SimpleITK as sitk

# ------------------------------ DICOM Utilities ------------------------------

def get_first_slice_ipp(folder):
    """Return the ImagePositionPatient of the first slice in the folder."""
    # Get all DICOM files
    files = sorted([os.path.join(folder,f) for f in os.listdir(folder) if f.lower().endswith(".dcm")])
    if not files:
        return np.array([0.0,0.0,0.0])
    ds = pydicom.dcmread(files[0])
    return np.array(ds.ImagePositionPatient, dtype=float)

def compute_dicom_matrix(reader, origin_override=None):
    """Return a 4x4 voxel-to-world matrix for vtkDICOMImageReader."""
    image = reader.GetOutput()

    origin = np.array(image.GetOrigin())
    if origin_override is not None:
        origin = origin_override  # override with true DICOM IPP

    spacing = np.array(image.GetSpacing())

    # Direction cosines (IOP)
    direction_matrix = image.GetDirectionMatrix()
    direction = np.eye(3)
    if direction_matrix:  # VTK >=9
        for i in range(3):
            for j in range(3):
                direction[i, j] = direction_matrix.GetElement(i, j)

    M = np.eye(4)
    for i in range(3):
        M[0:3, i] = direction[0:3, i] * spacing[i]
    M[0:3, 3] = origin
    return M

def prepare_dicom_slice_dir(input_dir: str) -> str:
    """
    Copy only CT/MR slices into a temporary folder. Ignore RTDOSE, RTPLAN, RTSTRUCT.
    Returns the path to the temporary directory.
    """
    temp_dir = tempfile.mkdtemp(prefix="dicom_slices_")
    found = False

    IMAGE_MODALITIES = ["CT", "MR"]  # only volume-capable modalities

    for f in Path(input_dir).glob("*"):
        if not f.is_file():
            continue
        try:
            ds = pydicom.dcmread(str(f))  # read full DICOM
            modality = getattr(ds, "Modality", "").upper()

            if modality in IMAGE_MODALITIES and hasattr(ds, "PixelData"):
                shutil.copy(str(f), temp_dir)
                found = True

        except (pydicom.errors.InvalidDicomError, Exception):
            continue  # skip non-DICOM or unreadable files

    if not found:
        shutil.rmtree(temp_dir)
        raise ValueError(f"No valid CT/MR slices found in '{input_dir}'")

    return temp_dir

LPS_TO_RAS = np.diag([-1.0, -1.0, 1.0, 1.0])

def lps_matrix_to_ras(M: np.ndarray) -> np.ndarray:
    """Convert a voxel->LPS matrix into voxel->RAS."""
    return LPS_TO_RAS @ M

def lps_point_to_ras(pt: np.ndarray) -> np.ndarray:
    """Convert a 3- or 4-vector point from LPS to RAS (returns 3-vector)."""
    if pt.shape[0] == 3:
        v = np.array([pt[0], pt[1], pt[2], 1.0], dtype=float)
    else:
        v = pt.astype(float)
    vr = (LPS_TO_RAS @ v)
    return vr[0:3]

def cleanup_old_dicom_temp_dirs(temp_root=None):
    """
    Scan temp folder for old dicom slice dirs and delete them.
    Windows-safe: ignores folders in use.
    """
    if temp_root is None:
        temp_root = tempfile.gettempdir()

    pattern = os.path.join(temp_root, "dicom_slices_*")
    for folder in glob.glob(pattern):
        try:
            shutil.rmtree(folder)
            print(f"[CLEANUP] Removed old temp folder: {folder}")
        except Exception as e:
            print(f"[WARN] Could not remove {folder}: {e}")



# ------------------------------ VTK Processing Engine ------------------------------

class VTKEngine:
    ORI_AXIAL = "axial"
    ORI_CORONAL = "coronal"
    ORI_SAGITTAL = "sagittal"

    def __init__(self):
        self.fixed_reader = None
        self.moving_reader = None
        self.fixed_dir = None
        self.moving_dir = None
        self._blend_dirty = True
        self._pivot_actor = None


        # Cleanup old temp dirs on startup
        self.cleanup_old_temp_dirs()

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

        # Pre-registration transform
        self.pre_transform = np.eye(4)
        self.fixed_matrix = np.eye(4)
        self.moving_matrix = np.eye(4)
        
        # User transform (rotation + translation applied by user)
        self.user_transform = vtk.vtkTransform()
        self.user_transform.Identity()

        # Temporary directories created for DICOM slices
        self._temp_dirs = []
        atexit.register(self._cleanup_temp_dirs)

        # SimpleITK placeholder transforms
        self.sitk_transform = sitk.Euler3DTransform()
        self.sitk_matrix = np.eye(4, dtype=np.float64)  # Latest SITK matrix
        self.fixed_img = None


    def cleanup_old_temp_dirs(self):
        cleanup_old_dicom_temp_dirs()

    def _cleanup_temp_dirs(self):
        """
        Remove all temporary directories created for DICOM slices. 
        This method is called automatically at program exit to ensure cleanup of temporary resources.
        """
        for d in self._temp_dirs:
            try:
                shutil.rmtree(d, ignore_errors=True)
            except Exception as e:
                print(f"[WARN] Failed to clean temp dir {d}: {e}")
        self._temp_dirs.clear()


    # ---------------- Fixed Volume ----------------
    def load_fixed(self, dicom_dir: str) -> bool:
        try:
            slice_dir = prepare_dicom_slice_dir(dicom_dir)
            self._temp_dirs.append(slice_dir)
            self.fixed_dir = dicom_dir
        except ValueError as e:
            print(e)
            return False

        r = vtk.vtkDICOMImageReader()
        r.SetDirectoryName(str(slice_dir))
        r.Update()

        flip = vtk.vtkImageFlip()
        flip.SetInputConnection(r.GetOutputPort())
        flip.SetFilteredAxis(1)
        flip.Update()
        self.fixed_reader = flip

        # Compute voxel->LPS then LPS->RAS
        origin = get_first_slice_ipp(slice_dir)
        vox2lps = compute_dicom_matrix(r, origin_override=origin)
        self.fixed_matrix = lps_matrix_to_ras(vox2lps)

        print("Fixed voxel->RAS matrix (no flip):")
        print(self.fixed_matrix)

        # Debug: check RAS origin
        ras_origin = np.array([0.0, 0.0, 0.0, 1.0])
        voxel_at_ras0 = np.linalg.inv(self.fixed_matrix) @ ras_origin
        print("Voxel coords of RAS (0,0,0):", voxel_at_ras0)

        # Cleanup temp folder
        shutil.rmtree(slice_dir, ignore_errors=True)
        self._temp_dirs.remove(slice_dir)

        # Set background level
        img = r.GetOutput()
        scalars = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())
        if scalars is not None and scalars.size > 0:
            self.reslice3d.SetBackgroundLevel(float(scalars.min()))

        self._wire_blend()
        self._sync_reslice_output_to_fixed()

        # Load fixed as SITK image (with proper DICOM geometry)
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(self.fixed_dir)
        reader.SetFileNames(dicom_names)
        self.fixed_img = reader.Execute()
        return True



    def load_moving(self, dicom_dir: str) -> bool:
        try:
            slice_dir = prepare_dicom_slice_dir(dicom_dir)
            self._temp_dirs.append(slice_dir)
            self.moving_dir = dicom_dir
        except ValueError as e:
            print(e)
            return False

        r = vtk.vtkDICOMImageReader()
        r.SetDirectoryName(str(slice_dir))
        r.Update()

        flip = vtk.vtkImageFlip()
        flip.SetInputConnection(r.GetOutputPort())
        flip.SetFilteredAxis(1)
        flip.Update()
        self.moving_reader = flip

        # Compute voxel->LPS then LPS->RAS
        origin = get_first_slice_ipp(slice_dir)
        vox2lps = compute_dicom_matrix(r, origin_override=origin)
        self.moving_matrix = lps_matrix_to_ras(vox2lps)

        print("Moving voxel->RAS matrix (no flip):")
        print(self.moving_matrix)

        # Debug: check RAS origin
        ras_origin = np.array([0.0, 0.0, 0.0, 1.0])
        voxel_at_ras0 = np.linalg.inv(self.moving_matrix) @ ras_origin
        print("Voxel coords of RAS (0,0,0):", voxel_at_ras0)

        # Cleanup temp folder
        shutil.rmtree(slice_dir, ignore_errors=True)
        self._temp_dirs.remove(slice_dir)

        # --- Compute pre-registration transform ---
        R_fixed = self.fixed_matrix[0:3,0:3] / np.array([np.linalg.norm(self.fixed_matrix[0:3,i]) for i in range(3)])
        R_moving = self.moving_matrix[0:3,0:3] / np.array([np.linalg.norm(self.moving_matrix[0:3,i]) for i in range(3)])
        R = R_fixed.T @ R_moving

        t = self.moving_matrix[0:3,3] - self.fixed_matrix[0:3,3]

        pre_transform = np.eye(4)
        pre_transform[0:3,0:3] = R
        pre_transform[0:3,3] = t
        self.pre_transform = pre_transform

        print("--- Pre-registration transform ---")
        print(pre_transform)
        print("Pre-reg translation (mm):", t)

        # --- BAKE pre-transform into the moving image immediately ---
        # Create a reslice that applies pre_transform to the original moving image and
        # writes the result into the fixed image geometry (so VTK will then treat the
        # moving image as already aligned to the fixed geometry).
        vtk_pre_mat = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtk_pre_mat.SetElement(i,j, float(pre_transform[i,j]))

        bake_reslice = vtk.vtkImageReslice()
        bake_reslice.SetInputConnection(flip.GetOutputPort())
        bake_reslice.SetResliceAxes(vtk_pre_mat)
        bake_reslice.SetInterpolationModeToLinear()
        # Configure output geometry to match fixed
        if self.fixed_reader is not None:
            fixed = self.fixed_reader.GetOutput()
            fixed_origin_lps = np.array(fixed.GetOrigin(), dtype=float)
            fixed_origin_ras = lps_point_to_ras(fixed_origin_lps)
            bake_reslice.SetOutputSpacing(fixed.GetSpacing())
            bake_reslice.SetOutputOrigin(tuple(float(x) for x in fixed_origin_ras))
            bake_reslice.SetOutputExtent(fixed.GetExtent())
            bake_reslice.SetBackgroundLevel(self.reslice3d.GetBackgroundLevel())
        bake_reslice.Update()

        # Replace moving_reader with the baked reslice output; this means the moving image
        # now *contains* the pre-registration transform in its voxel data and no pre_transform
        # needs to be applied in subsequent transforms.
        self.moving_reader = bake_reslice
        # After baking, moving image geometry aligns with fixed, so update moving_matrix
        self.moving_matrix = np.array(self.fixed_matrix, copy=True)
        # Reset pre-transform since it's baked
        self.pre_transform = np.eye(4)

        # Now connect reslice3d input to the baked moving image and set identity axes
        identity_mat = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                identity_mat.SetElement(i, j, float(np.eye(4)[i, j]))
        self.reslice3d.SetInputConnection(self.moving_reader.GetOutputPort())
        self.reslice3d.SetResliceAxes(identity_mat)
        self._sync_reslice_output_to_fixed()
        self._wire_blend()

        return True



    # ---------------- Transformation Utilities ----------------
    def set_translation(self, tx: float, ty: float, tz: float):
        self._tx, self._ty, self._tz = float(tx), float(ty), float(tz)
        self._apply_transform()

    def set_rotation_deg(self, rx: float, ry: float, rz: float, orientation=None, slice_idx=None):
        self._rx, self._ry, self._rz = float(rx), float(ry), float(rz)
        self._apply_transform(orientation, slice_idx, pivot_mode="current_slice" if orientation and slice_idx is not None else "dataset_center")

    def reset_transform(self):
        self._tx = self._ty = self._tz = 0.0
        self._rx = self._ry = self._rz = 0.0
        self.transform.Identity()
        self._blend_dirty = True
        self._apply_transform()

    def set_opacity(self, alpha: float):
        self.blend.SetOpacity(1, float(np.clip(alpha, 0.0, 1.0)))
        self._blend_dirty = True

    def fixed_extent(self):
        if not self.fixed_reader:
            return None
        return self.fixed_reader.GetOutput().GetExtent()


    # ---------------- Slice Extraction ----------------
    def get_slice_numpy(self, orientation: str, slice_idx: int) -> tuple[np.ndarray | None, np.ndarray | None]:
        if self.fixed_reader is None:
            return None, None
        fixed_img = self.fixed_reader.GetOutput()
        moving_img = self.reslice3d.GetOutput() if self.moving_reader else None
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

            arr2d = arr2d.astype(np.float32)
            c = window_center
            w = window_width
            arr2d = np.clip((arr2d - (c - 0.5)) / (w - 1) + 0.5, 0, 1)
            arr2d = (arr2d * 255.0).astype(np.uint8)
            return np.ascontiguousarray(arr2d)

        fixed_slice = vtk_to_np_slice(fixed_img, orientation, slice_idx)
        moving_slice = vtk_to_np_slice(moving_img, orientation, slice_idx) if moving_img else None
        return fixed_slice, moving_slice

    def get_slice_qimage(self, orientation: str, slice_idx: int, fixed_color="Purple", moving_color="Green", coloring_enabled=True) -> QtGui.QImage:
        fixed_slice, moving_slice = self.get_slice_numpy(orientation, slice_idx)
        if fixed_slice is None:
            return QtGui.QImage()
        h, w = fixed_slice.shape

        blend = self.blend.GetOpacity(1) if self.moving_reader is not None else 0.0
        color_map = {
            "Grayscale":   lambda arr: arr,
            "Green":       lambda arr: np.stack([np.zeros_like(arr), arr, np.zeros_like(arr)], axis=-1),
            "Purple":      lambda arr: np.stack([arr, np.zeros_like(arr), arr], axis=-1),
            "Blue":        lambda arr: np.stack([np.zeros_like(arr), np.zeros_like(arr), arr], axis=-1),
            "Yellow":      lambda arr: np.stack([arr, arr, np.zeros_like(arr)], axis=-1),
            "Red":         lambda arr: np.stack([arr, np.zeros_like(arr), np.zeros_like(arr)], axis=-1),
            "Cyan":        lambda arr: np.stack([np.zeros_like(arr), arr, arr], axis=-1),
        }

        def aspect_ratio_correct(qimg, h, w, orientation):
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
                return qimg.scaled(display_w, display_h, QtCore.Qt.IgnoreAspectRatio, QtCore.Qt.SmoothTransformation)
            return qimg

        def grayscale_qimage(arr2d, h, w, orientation):
            qimg = QtGui.QImage(arr2d.data, w, h, w, QtGui.QImage.Format_Grayscale8)
            qimg = qimg.copy()
            return aspect_ratio_correct(qimg, h, w, orientation)

        if not coloring_enabled:
            if moving_slice is None:
                return grayscale_qimage(fixed_slice, h, w, orientation)
            else:
                alpha = self.blend.GetOpacity(1)
                arr2d = (fixed_slice.astype(np.float32) * (1 - alpha) +
                         moving_slice.astype(np.float32) * alpha).astype(np.uint8)
                return grayscale_qimage(arr2d, h, w, orientation)

        fixed_f = fixed_slice.astype(np.float32)
        if moving_slice is None:
            if fixed_color == "Grayscale":
                return grayscale_qimage(fixed_slice, h, w, orientation)
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
        return aspect_ratio_correct(qimg, h, w, orientation)


    def _update_sitk_transform(self, orientation=None, slice_idx=None):
        if not self.fixed_reader:
            return

        # Step 0: Get DICOM first slice IPP
        dicom_ipp0 = get_first_slice_ipp(self.fixed_dir)
        sitk_origin = np.array(self.fixed_img.GetOrigin())
        origin_shift = dicom_ipp0 - sitk_origin  # shift to align SITK origin with DICOM IPP

        # Step 1: Default pivot = volume center
        voxel = np.array(self.fixed_img.GetSize()) / 2.0

        # Step 2: Override pivot if slice info provided
        if orientation == "multi" and isinstance(slice_idx, tuple) and len(slice_idx) == 3:
            axial_idx, coronal_idx, sagittal_idx = slice_idx
            # Use all three indices to compute the center voxel
            voxel = np.array([
                sagittal_idx,
                coronal_idx,
                axial_idx
            ])
        elif orientation is not None and slice_idx is not None:
            if orientation == VTKEngine.ORI_AXIAL:
                voxel = np.array([
                    self.fixed_img.GetWidth() // 2,
                    self.fixed_img.GetHeight() // 2,
                    slice_idx
                ])
            elif orientation == VTKEngine.ORI_CORONAL:
                voxel = np.array([
                    self.fixed_img.GetWidth() // 2,
                    slice_idx,
                    self.fixed_img.GetDepth() // 2
                ])
            elif orientation == VTKEngine.ORI_SAGITTAL:
                voxel = np.array([
                    slice_idx,
                    self.fixed_img.GetHeight() // 2,
                    self.fixed_img.GetDepth() // 2
                ])

        # Step 3: Convert voxel → physical point in DICOM space
        center = np.array(self.fixed_img.TransformIndexToPhysicalPoint([int(v) for v in voxel]))

        # Apply DICOM origin shift
        center += origin_shift

        # Step 5: Build rotation matrix (Rx, Ry, Rz in radians)
        rx, ry, rz = np.deg2rad([self._rx, self._ry, -self._rz])

        Rx = np.array([[1, 0, 0],
                    [0, np.cos(rx), -np.sin(rx)],
                    [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                    [0, 1, 0],
                    [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                    [np.sin(rz), np.cos(rz), 0],
                    [0, 0, 1]])

        R = Rz @ Ry @ Rx  # ITK order: Rz * Ry * Rx

        # Step 6: Bake pivot into translation (match VTK _apply_transform)
        c = np.array(center)                    # pivot in physical space
        user_t = np.array([-self._tx, -self._ty, self._tz])  # world XYZ translation

        baked_t = R @ (-c) + c + user_t

        # Step 7: Build final 4x4
        mat = np.eye(4, dtype=np.float64)
        mat[:3, :3] = R
        mat[:3, 3] = baked_t
        self.sitk_matrix = mat

        # Step 8: Convert LPS → RAS (for VTK/Slicer comparison)
        LPS_to_RAS = np.diag([-1, -1, 1, 1])
        ras_matrix = LPS_to_RAS @ self.sitk_matrix @ LPS_to_RAS
        self.sitk_matrix = ras_matrix


    # ---------------- Internal Transform Application ----------------
    def _vtk_matrix_from_numpy(self, mat: np.ndarray) -> vtk.vtkMatrix4x4:
        vtkmat = vtk.vtkMatrix4x4()
        for i in range(4):
            for j in range(4):
                vtkmat.SetElement(i, j, float(mat[i, j]))
        return vtkmat

    def bake_user_transform_into_moving(self):
        """Bake the current pre_transform + user_transform into the moving image data.

        After this call:
        - moving_reader will be replaced by a resampled image whose voxel data contains the
          effect of the pre-registration and user transforms.
        - pre_transform will be reset to identity.
        - user transform state (internal) will be reset to identity (tx/ty/tz, rx/ry/rz -> 0)
        - moving_matrix will be set to fixed_matrix so future transforms are relative to the
          new baked image geometry.
        """
        if self.moving_reader is None or self.fixed_reader is None:
            return

        # Build numpy version of user transform
        user_vtk_mat = self.user_transform.GetMatrix()
        user_np = np.eye(4)
        for i in range(4):
            for j in range(4):
                user_np[i, j] = user_vtk_mat.GetElement(i, j)

        total = self.pre_transform @ user_np

        vtk_total = self._vtk_matrix_from_numpy(total)

        bake_reslice = vtk.vtkImageReslice()
        bake_reslice.SetInputConnection(self.moving_reader.GetOutputPort())
        bake_reslice.SetResliceAxes(vtk_total)
        bake_reslice.SetInterpolationModeToLinear()

        # Allow rotated volume to expand so it doesn’t get cut
        bake_reslice.SetAutoCropOutput(True)
        bake_reslice.Update()

        # Configure output geometry to match fixed
        fixed = self.fixed_reader.GetOutput()
        fixed_origin_lps = np.array(fixed.GetOrigin(), dtype=float)
        fixed_origin_ras = lps_point_to_ras(fixed_origin_lps)
        #bake_reslice.SetOutputSpacing(fixed.GetSpacing())
        #bake_reslice.SetOutputOrigin(tuple(float(x) for x in fixed_origin_ras))
        #bake_reslice.SetOutputExtent(fixed.GetExtent())

        bake_reslice.Update()

        # Reset background level based on baked data
        img = bake_reslice.GetOutput()
        scalars = numpy_support.vtk_to_numpy(img.GetPointData().GetScalars())
        if scalars is not None and scalars.size > 0:
            bake_reslice.SetBackgroundLevel(float(scalars.min()))

        # Replace pipeline
        self.moving_reader = bake_reslice
        # now moving voxel data is in fixed space; update matrices
        self.moving_matrix = np.array(self.fixed_matrix, copy=True)
        self.pre_transform = np.eye(4)

        # Reset user transform state
        self.user_transform.Identity()
        self._tx = self._ty = self._tz = 0.0
        self._rx = self._ry = self._rz = 0.0
        self.sitk_matrix = np.eye(4, dtype=np.float64)

        # Re-attach reslice3d to the baked image and use identity reslice axes (user transforms will now
        # be applied on top of the baked image only)
        identity_mat = self._vtk_matrix_from_numpy(np.eye(4))
        self.reslice3d.SetInputConnection(self.moving_reader.GetOutputPort())
        self.reslice3d.SetResliceAxes(identity_mat)

        self._sync_reslice_output_to_fixed()
        self._wire_blend()
        self._blend_dirty = True





    def _apply_transform(self, orientation=None, slice_idx=None): 
        if not self.fixed_reader or not self.moving_reader: 
            return
        img = self.fixed_reader.GetOutput()
        center = np.array(img.GetCenter())

        extent = img.GetExtent()
        spacing = img.GetSpacing()
        origin = img.GetOrigin()

        # If orientation is "multi" and slice_idx is a tuple, use all three indices to compute a 3D center
        if orientation == "multi" and isinstance(slice_idx, tuple) and len(slice_idx) == 3:
            axial_idx, coronal_idx, sagittal_idx = slice_idx
            # Compute the center in world coordinates using all three indices
            x = int(np.clip(sagittal_idx, extent[0], extent[1]))
            y = int(np.clip(coronal_idx, extent[2], extent[3]))
            z = int(np.clip(axial_idx, extent[4], extent[5]))
            center = np.array([
                origin[0] + x * spacing[0],
                origin[1] + y * spacing[1],
                origin[2] + z * spacing[2]
            ])
            # Note: pre_transform was baked into the moving image in load_moving, so no offset is applied here
        elif orientation is not None and slice_idx is not None:
            # Fallback to old behavior for single orientation
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

        ras_center = -self.moving_matrix[0:3,3]

        # ---------------- User transform only ---------------- 
        user_t = vtk.vtkTransform() 
        user_t.PostMultiply() 

        # Apply user translation 
        user_t.Translate(self._tx, self._ty, self._tz) 

        # Move volume to origin for rotation 
        user_t.Translate(-center) 
        # Apply user rotations 
        user_t.RotateX(self._rx) 
        user_t.RotateY(self._ry) 
        user_t.RotateZ(self._rz) 
        
        # Move volume back 
        user_t.Translate(center) 

        # ---------------- Combined transform for reslice (ONLY user transform)
        final_t = vtk.vtkTransform() 
        final_t.PostMultiply() 

        # Since pre-registration is baked into the moving image, do NOT concatenate pre_transform here.
        final_t.Concatenate(user_t) # user transform only
        self.transform.DeepCopy(final_t) 
        self.user_transform.DeepCopy(user_t) 
        self.reslice3d.SetResliceAxes(self.transform.GetMatrix()) 
        self.reslice3d.Modified() 
        self._update_sitk_transform(orientation, slice_idx)
        self._blend_dirty = True 



    # ---------------- Pipeline Utilities ----------------
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
        fixed_origin_lps = np.array(fixed.GetOrigin(), dtype=float)
        fixed_origin_ras = lps_point_to_ras(fixed_origin_lps)

        self.reslice3d.SetOutputSpacing(fixed.GetSpacing())
        self.reslice3d.SetOutputOrigin(tuple(float(x) for x in fixed_origin_ras))
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

        # Add copy button
        self.copy_btn = QtWidgets.QPushButton("Copy to Clipboard")
        self.copy_btn.clicked.connect(self.copy_matrix_to_clipboard)
        layout.addWidget(self.copy_btn)

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

    def set_matrix(self, mat):
        self._current_matrix = mat
        for i in range(4):
            for j in range(4):
                value = float(mat[i, j]) if hasattr(mat, "__getitem__") else mat.GetElement(i, j)
                item = QtWidgets.QTableWidgetItem(f"{value:.5f}")
                item.setTextAlignment(QtCore.Qt.AlignCenter)
                self.table.setItem(i, j, item)

    def copy_matrix_to_clipboard(self):
        # Format matrix as requested: 5 decimals, space-separated, one row per line
        mat = self._current_matrix
        if hasattr(mat, "GetElement"):
            # vtkMatrix4x4
            rows = [
                " ".join(f"{mat.GetElement(i, j):.5f}" for j in range(4))
                for i in range(4)
            ]
        else:
            # numpy or similar
            rows = [
                " ".join(f"{float(mat[i, j]):.5f}" for j in range(4))
                for i in range(4)
            ]
        text = "\n".join(rows)
        QtWidgets.QApplication.clipboard().setText(text)

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
        self.s_tx = slider(-500,500,0)
        self.s_ty = slider(-500,500,0)
        self.s_tz = slider(-500,500,0)
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
            self._matrix_dialog.set_matrix(engine.sitk_matrix)
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
        self._last_active_slider = None

    def _wire(self):
        self.ui.loadFixed.connect(self.on_load_fixed)
        self.ui.loadMoving.connect(self.on_load_moving)
        self.ui.txChanged.connect(lambda v: self._handle_slider_change("tx"))
        self.ui.tyChanged.connect(lambda v: self._handle_slider_change("ty"))
        self.ui.tzChanged.connect(lambda v: self._handle_slider_change("tz"))
        self.ui.rxChanged.connect(lambda v: self._handle_slider_change("rx"))
        self.ui.ryChanged.connect(lambda v: self._handle_slider_change("ry"))
        self.ui.rzChanged.connect(lambda v: self._handle_slider_change("rz"))
        self.ui.opacityChanged.connect(lambda a: self._update_opacity(a))
        self.ui.resetRequested.connect(self.on_reset)
        self.ui.axialSliceChanged.connect(lambda i: self.refresh_slice("axial",i))
        self.ui.coronalSliceChanged.connect(lambda i: self.refresh_slice("coronal",i))
        self.ui.sagittalSliceChanged.connect(lambda i: self.refresh_slice("sagittal",i))
        self.ui.fixed_color_combo.currentTextChanged.connect(self._on_fixed_color_changed)
        self.ui.moving_color_combo.currentTextChanged.connect(self._on_moving_color_changed)
        self.ui.coloring_checkbox.stateChanged.connect(self._on_coloring_checkbox_changed)

    def _handle_slider_change(self, slider_name):
        # If the active slider changes, bake the current transform into the moving image.
        if self._last_active_slider is not None and self._last_active_slider != slider_name:
            # Bake current transforms into the moving image so VTK treats them as part of the voxel data.
            self.engine.bake_user_transform_into_moving()
            # Reset the on-screen transform sliders because their effect is now baked.
            self.reset_transform_sliders()

        self._last_active_slider = slider_name
        self._update_transform()

    def reset_transform_sliders(self):
        for slider in [
            self.ui.s_tx, self.ui.s_ty, self.ui.s_tz,
            self.ui.s_rx, self.ui.s_ry, self.ui.s_rz
        ]:
            slider.blockSignals(True)
            slider.setValue(0)
            slider.blockSignals(False)

    def _update_transform(self):
        # Use the current values of all three slice sliders to compute a 3D center for rotation
        axial_idx = self.ui.s_axial.value()
        coronal_idx = self.ui.s_coronal.value()
        sagittal_idx = self.ui.s_sagittal.value()

        # Set translation and rotation, then apply the transform ONCE with all current values
        self.engine._tx = float(self.ui.s_tx.value())
        self.engine._ty = float(self.ui.s_ty.value())
        self.engine._tz = float(self.ui.s_tz.value())
        self.engine._rx = float(self.ui.s_rx.value() / 10.0)
        self.engine._ry = float(self.ui.s_ry.value() / 10.0)
        self.engine._rz = float(self.ui.s_rz.value() / 10.0)

        self.engine._apply_transform(
            orientation="multi",
            slice_idx=(axial_idx, coronal_idx, sagittal_idx)
        )
        engine.set_interpolation_linear(False)
        self._debounce_timer.start(self.DEBOUNCE_MS)

        # update matrix dialog if open
        if self.ui._matrix_dialog:
            self.ui._matrix_dialog.set_matrix(self.engine.sitk_matrix)

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
        print("[DEBUG] Moving image loaded and processed.")
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
            self.ui._matrix_dialog.set_matrix(self.engine.sitk_matrix)

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

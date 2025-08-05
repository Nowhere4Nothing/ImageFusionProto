import os
import numpy as np
import pydicom


def load_dicom_volume(folder):
    slices = []
    for file in sorted(os.listdir(folder)):
        path = os.path.join(folder, file)
        try:
            ds = pydicom.dcmread(path)
            # Only keep CT slices with image data and position info
            if getattr(ds, "Modality", "") != "CT":
                continue
            if not hasattr(ds, "ImagePositionPatient"):
                print(f"Skipping {file}: Missing ImagePositionPatient")
                continue
            if not hasattr(ds, "pixel_array"):
                print(f"Skipping {file}: No image data")
                continue

            slices.append(ds)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

    if not slices:
        print("No valid DICOM slices.")
        return None, None

    try:
        slices = sorted(slices, key=lambda s: float(s.ImagePositionPatient[2]))
    except AttributeError:
        print("Some slices are missing ImagePositionPatient metadata.")
        return None, None

    # Stack images into volume
    volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    volume -= volume.min()
    if volume.max() != 0:
        volume /= volume.max()

    # === Extract voxel spacing (z, y, x) ===
    ds0 = slices[0]
    try:
        pixel_spacing = [float(x) for x in ds0.PixelSpacing]  # [row, col] -> (y, x)
        slice_thickness = float(ds0.SliceThickness)  # z
        spacing = (slice_thickness, pixel_spacing[0], pixel_spacing[1])  # (z, y, x)
    except AttributeError:
        print("Missing spacing metadata. Using default spacing (1.0, 1.0, 1.0)")
        spacing = (1.0, 1.0, 1.0)

    return volume, spacing

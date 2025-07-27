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

            try:
                _ = ds.pixel_array
            except Exception:
                print(f"Skipping {file}: Unable to read pixel data")
                continue

            slices.append(ds)
        except Exception as e:
            print(f"Failed to read {path}: {e}")
            continue

    if not slices:
        print("No valid DICOM slices.")
        return None

    try:
        slices = sorted(slices, key=lambda s: float(s.ImagePositionPatient[2]))
    except AttributeError:
        print("Some slices are missing ImagePositionPatient metadata.")
        return None
    except Exception as e:
        print(f"Sorting slices failed due to missing or invalid ImagePositionPatient: {e}")
        return None

    volume = np.stack([s.pixel_array for s in slices]).astype(np.float32)
    volume -= volume.min()
    max_val = volume.max()
    if volume.max() != 0:
        volume /= max_val

    return volume
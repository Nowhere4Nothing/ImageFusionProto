import numpy as np
import SimpleITK as sitk

def sitk_rotate_volume(volume, rotation_angles_deg):
    # volume: numpy array (z, y, x)
    # rotation_angles_deg: [LR, PA, IS] in degrees
    # Map to SimpleITK Euler3DTransform: (alpha, beta, gamma) = (x, y, z)
    angles_rad = [np.deg2rad(a) for a in rotation_angles_deg]
    sitk_image = sitk.GetImageFromArray(volume)
    size = sitk_image.GetSize()  # (x, y, z)
    center = [s / 2.0 for s in size]  # (x, y, z) center in index space
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center)
    # Set rotation: (alpha, beta, gamma) = (LR, PA, IS)
    transform.SetRotation(angles_rad[0], angles_rad[1], angles_rad[2])
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(0)
    rotated = resampler.Execute(sitk_image)
    return sitk.GetArrayFromImage(rotated)

def process_layers(volume_layers, slice_index):
    base_shape = volume_layers[0].data[0].shape
    img = np.zeros(base_shape, dtype=np.float32)

    for layer in volume_layers:
        if not layer.visible:
            continue

        volume = layer.data.copy()

        # Use SimpleITK for 3D rotation
        if any(r != 0 for r in layer.rotation):
            volume = sitk_rotate_volume(volume, layer.rotation)

        slice_idx = np.clip(slice_index + layer.slice_offset, 0, volume.shape[0] - 1)
        overlay = volume[slice_idx]

        shifted = np.roll(overlay, shift=layer.offset[0], axis=1)
        shifted = np.roll(shifted, shift=layer.offset[1], axis=0)

        img = img * (1 - layer.opacity) + shifted * layer.opacity

    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

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
    """
      Combines multiple image layers into a single 2D image slice with transformations
      and blending. Each layer can be rotated, translated, and blended according to its properties.

      Args:
          volume_layers: List of layer objects, each containing 3D data and transformation
          attributes. slice_index: Integer index specifying which slice to extract and process.

      Returns:
          np.ndarray: The resulting 2D image as an 8-bit unsigned integer array.
      """
    base_shape = volume_layers[0].data[0].shape
    img = np.zeros(base_shape, dtype=np.float32)

    for layer in volume_layers:
        if not layer.visible:
            continue

        volume = layer.data.copy()

        # Apply 3D rotation with SimpleITK
        if any(r != 0 for r in layer.rotation):
            volume = sitk_rotate_volume(volume, layer.rotation)

        slice_idx = np.clip(slice_index + layer.slice_offset, 0, volume.shape[0] - 1)
        overlay = volume[slice_idx]

        # Apply translation without wraparound
        x_offset, y_offset = layer.offset
        shifted = translate_image(overlay, x_offset, y_offset)

        # Blend into final image using layer opacity
        img = img * (1 - layer.opacity) + shifted * layer.opacity

    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def translate_image(img, x_offset, y_offset):
    """
    Shift a 2D image without wraparound, filling empty space with 0s.
    """
    h, w = img.shape
    result = np.zeros_like(img)

    # Determine target coordinates in destination
    x_start = max(0, x_offset)
    y_start = max(0, y_offset)
    x_end = min(w, w + x_offset) if x_offset >= 0 else w + x_offset
    y_end = min(h, h + y_offset) if y_offset >= 0 else h + y_offset

    # Determine source coordinates from original image
    src_x_start = max(0, -x_offset)
    src_y_start = max(0, -y_offset)
    src_x_end = src_x_start + (x_end - x_start)
    src_y_end = src_y_start + (y_end - y_start)

    # Apply the shift
    result[y_start:y_end, x_start:x_end] = img[src_y_start:src_y_end, src_x_start:src_x_end]

    return result

import numpy as np
import cv2
import SimpleITK as sitk

def resize_to_match(base, target_shape):
    return cv2.resize(base, (target_shape[1], target_shape[0]))

def sitk_rotate_volume(volume, rotation_angles_deg):
    angles_rad = [np.deg2rad(a) for a in rotation_angles_deg]
    sitk_image = sitk.GetImageFromArray(volume)
    size = sitk_image.GetSize()  # (x, y, z)
    spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()
    # Compute center in physical coordinates
    center_phys = [origin[i] + spacing[i] * size[i] / 2.0 for i in range(3)]

    transform = sitk.Euler3DTransform()
    transform.SetCenter(center_phys)
    transform.SetRotation(angles_rad[0], angles_rad[1], angles_rad[2])
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(0)
    rotated = resampler.Execute(sitk_image)
    return sitk.GetArrayFromImage(rotated)

def process_layers(volume_layers, slice_index, view_type):
    """
      Combines multiple image layers into a single 2D image slice with transformations
      and blending. Each layer can be rotated, translated, and blended according to its properties.

      Args:
          volume_layers: List of layer objects, each containing 3D data and transformation
          attributes. slice_index: Integer index specifying which slice to extract and process.

      Returns:
          np.ndarray: The resulting 2D image as an 8-bit unsigned integer array.
      """
    #default
    global overlay
    if not volume_layers:
        return np.zeros((512, 512), dtype=np.uint8)

    print("ViewType:", view_type)
    base_vol = volume_layers[0].data
    # Determine base_shape to match the actual extracted slice for each view
    if view_type == "axial":
        base_shape = (base_vol.shape[1], base_vol.shape[2])
    elif view_type == "coronal":
        base_shape = (base_vol.shape[1], base_vol.shape[0])
    elif view_type == "sagittal":
        base_shape = (base_vol.shape[2], base_vol.shape[0])
    else:
        raise ValueError(f"Unknown view type: {view_type}")

    img = np.zeros(base_shape, dtype=np.float32)

    for layer in volume_layers:
        if not getattr(layer, 'visible', True):
            continue

        volume = layer.data.copy()

        # print(f"Volume shape: {volume.shape}")

        # Apply 3D rotation with SimpleITK if rotation present
        if any(r != 0 for r in getattr(layer, 'rotation', [0, 0, 0])):
            volume = sitk_rotate_volume(volume, layer.rotation)

        # Calculate adjusted slice index with offset, clipped to valid range
        if view_type == "axial":
            max_slice_index = volume.shape[0] - 1
            slice_index = np.clip(slice_index, 0, max_slice_index)
            overlay = volume[slice_index, :, :]
        elif view_type == "coronal":
            max_slice_index = volume.shape[1] - 1
            slice_index = np.clip(slice_index, 0, max_slice_index)
            overlay = volume[:, slice_index, :].T
        elif view_type == "sagittal":
            max_slice_index = volume.shape[2] - 1
            slice_index = np.clip(slice_index, 0, max_slice_index)
            overlay = volume[:, :, slice_index].T
            overlay = np.rot90(overlay, k=2)

        # print(f"Overlay shape before padding: {overlay.shape}")

        # # Apply translation (XY plane)
        overlay = overlay.astype(np.float32)
        if overlay.max() > 1.0:
            overlay /= 255.0

        x_offset, y_offset = getattr(layer, 'offset', (0, 0))
        overlay = translate_image(overlay, x_offset, y_offset)

        # Apply opacity blend
        opacity = np.clip(getattr(layer, 'opacity', 1.0), 0.0, 1.0)
        print(f"Layer: {layer.name}, opacity: {opacity}, overlay max: {overlay.max():.3f}, min: {overlay.min():.3f}")

        if img.shape != overlay.shape:
            overlay = resize_to_match(overlay, img.shape)
            print(f"Resized overlay from {overlay.shape} to {img.shape}")

        img = img * (1 - opacity) + overlay * opacity
    # print(f"View: {view_type}, overlay shape: {overlay.shape}")
    # print(f"img shape: {img.shape}, expected {base_shape[0]}x{base_shape[1]}")

    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def translate_image(img, x_offset, y_offset):
    """
    Shift a 2D image without wraparound, filling empty space with 0s.
    """
    h, w = img.shape
    result = np.zeros_like(img)

    src_x_start, src_x_end, dst_x_start, dst_x_end = calculate_shift_coords(x_offset, w)
    src_y_start, src_y_end, dst_y_start, dst_y_end = calculate_shift_coords(y_offset, h)

    if src_x_end > src_x_start and src_y_end > src_y_start:
        result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = img[src_y_start:src_y_end,
                                                               src_x_start:src_x_end]

    print(f"Applying offset: x={x_offset}, y={y_offset}")

    return result

def calculate_shift_coords(offset, length):
    """
    Calculate source and destination slice indices for 1D shift.

    Args:
        offset (int): shift amount (positive or negative)
        length (int): length of the dimension

    Returns:
        (src_start, src_end, dst_start, dst_end): tuple of indices for slicing
    """
    if offset >= 0:
        src_start = 0
        src_end = max(0, length - offset)
        dst_start = offset
        dst_end = offset + src_end
    else:
        src_start = -offset
        src_end = length
        dst_start = 0
        dst_end = src_end - src_start

    return src_start, src_end, dst_start, dst_end



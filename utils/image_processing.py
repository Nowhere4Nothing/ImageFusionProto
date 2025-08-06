import numpy as np
import cv2
import SimpleITK as sitk

def resize_to_match(base, target_shape):
    return cv2.resize(base, (target_shape[1], target_shape[0]))

def sitk_transform_volume(volume, rotation_angles_deg, translation_px, spacing=(1.0, 1.0, 1.0)):
    """
    Apply rotation and translation to a 3D numpy volume using SimpleITK.
    rotation_angles_deg: [LR, PA, IS] in degrees
    translation_px: [x, y, z] in pixels (x=LR, y=PA, z=IS)
    spacing: voxel spacing (z, y, x)
    """
    angles_rad = [np.deg2rad(a) for a in rotation_angles_deg]
    sitk_image = sitk.GetImageFromArray(volume)
    size = sitk_image.GetSize()  # (x, y, z)
    spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()
    # Compute center in physical coordinates
    center_phys = [origin[i] + spacing[i] * size[i] / 2.0 for i in range(3)]

    # Convert translation from pixels to physical units (x, y, z)
    # spacing: (x, y, z) in SimpleITK, but from DICOM loader it's (z, y, x)
    # So: spacing_sitk = (spacing[2], spacing[1], spacing[0])
    spacing_sitk = (spacing[2], spacing[1], spacing[0])
    translation_phys = [
        translation_px[0] * spacing_sitk[0],
        translation_px[1] * spacing_sitk[1],
        translation_px[2] * spacing_sitk[2],
    ]

    transform = sitk.Euler3DTransform()
    transform.SetCenter(center_phys)
    transform.SetRotation(angles_rad[0], angles_rad[1], angles_rad[2])
    transform.SetTranslation(translation_phys)

    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(0)
    transformed = resampler.Execute(sitk_image)
    return sitk.GetArrayFromImage(transformed)

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
    # We'll set base_shape after extracting the overlay for each view
    img = None

    for layer in volume_layers:
        if not getattr(layer, 'visible', True):
            continue

        volume = layer.data.copy()

        # Compose translation vector: [x, y, z] in pixels
        offset = getattr(layer, 'offset', (0, 0))
        slice_offset = getattr(layer, 'slice_offset', 0)
        translation_px = [offset[0], offset[1], slice_offset]

        # Apply 3D rotation and translation with SimpleITK
        if any(r != 0 for r in getattr(layer, 'rotation', [0, 0, 0])) or any(translation_px):
            spacing = getattr(layer, 'spacing', (1.0, 1.0, 1.0))
            volume = sitk_transform_volume(volume, layer.rotation, translation_px, spacing)

        # Calculate adjusted slice index with offset, clipped to valid range
        if view_type == "axial":
            max_slice_index = volume.shape[0] - 1
            slice_idx = np.clip(slice_index, 0, max_slice_index)
            overlay = volume[slice_idx, :, :]  # (y, x)
        elif view_type == "coronal":
            max_slice_index = volume.shape[1] - 1
            slice_idx = np.clip(slice_index, 0, max_slice_index)
            overlay = volume[:, slice_idx, :]  # (z, x)
            overlay = np.rot90(overlay, k=2)  # 90 deg CW
        elif view_type == "sagittal":
            max_slice_index = volume.shape[2] - 1
            slice_idx = np.clip(slice_index, 0, max_slice_index)
            overlay = volume[:, :, slice_idx]  # (z, y)
            overlay = np.rot90(overlay, k=2)  # 90 deg CW

        # Set base_shape on first valid overlay
        if img is None:
            img = np.zeros(overlay.shape, dtype=np.float32)

        # No need for 2D translation; already handled in 3D
        shifted = overlay

        # Blend into final image using layer opacity
        opacity = getattr(layer, 'opacity', 1.0)

        if img.shape != shifted.shape:
            shifted = resize_to_match(shifted, img.shape)

        img = img * (1 - opacity) + shifted * opacity

    if img is None:
        # No visible layers, return a default square
        img = np.zeros((512, 512), dtype=np.float32)

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

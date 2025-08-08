import numpy as np
import cv2
import SimpleITK as sitk

def resize_to_match(base, target_shape):
    """
       This function uses OpenCV to resize the input image to the given (height, width),
        ensuring compatibility for blending or display.

       Args:
           base: The input 2D NumPy array to be resized.
           target_shape: A tuple (height, width) specifying the desired output shape.

       Returns:
           np.ndarray: The resized image as a 2D NumPy array.
       """
    return cv2.resize(base, (target_shape[1], target_shape[0]))

def sitk_rotate_volume(volume, rotation_angles_deg):
    """

       This function applies a 3D Euler rotation to the input volume around its physical
       center and returns the rotated volume as a NumPy array.

       Args:
           volume: The input 3D NumPy array representing the image volume.
           rotation_angles_deg: A list or tuple of three rotation angles (in degrees) for the
           x, y, and z axes.

       Returns:
           np.ndarray: The rotated 3D image volume as a NumPy array.
       """

    # Convert rotation angles from degrees to radians
    angles_rad = [np.deg2rad(a) for a in rotation_angles_deg]

    # Convert the numpy array to a SimpleITK image
    sitk_image = sitk.GetImageFromArray(volume)
    size = sitk_image.GetSize()  # (x, y, z)
    spacing = sitk_image.GetSpacing()
    origin = sitk_image.GetOrigin()

    # Compute the center of rotation in physical coordinates
    center_phys = [origin[i] + spacing[i] * size[i] / 2.0 for i in range(3)]

    # Create a 3D Euler transform and set its center and rotation
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center_phys)
    transform.SetRotation(angles_rad[0], angles_rad[1], angles_rad[2])

    # Set up the resampler to apply the transform
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(sitk_image)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetTransform(transform)
    resampler.SetDefaultPixelValue(0)

    # Apply the transform and convert back to numpy array
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

    base_vol = volume_layers[0].data

    # Always use (height, width) for display (Y, X) = (base_vol.shape[1], base_vol.shape[2])
    base_shape = (base_vol.shape[1], base_vol.shape[2])  # (Y, X) for all views

    # Initialize the output image as a float32 array for blending
    img = np.zeros(base_shape, dtype=np.float32)

    # Process each layer in the list
    for layer in volume_layers:
        # Skip invisible layers
        if not getattr(layer, 'visible', True):
            continue

        # Copy the layer's volume data
        volume = layer.data.copy()

        # Apply 3D rotation with SimpleITK if rotation present
        if any(r != 0 for r in getattr(layer, 'rotation', [0, 0, 0])):
            volume = sitk_rotate_volume(volume, layer.rotation)

        if view_type == "axial":
            max_slice_index = volume.shape[0] - 1
            slice_index = np.clip(slice_index, 0, max_slice_index)
            overlay = volume[slice_index, :, :]  # (Y, X)
        elif view_type == "coronal":
            max_slice_index = volume.shape[1] - 1
            slice_index = np.clip(slice_index, 0, max_slice_index)
            overlay = volume[:, slice_index, :]  # (Z, X)
            # Resize to (Y, X) for display
            overlay = resize_to_match(overlay, base_shape)
        elif view_type == "sagittal":
            max_slice_index = volume.shape[2] - 1
            slice_index = np.clip(slice_index, 0, max_slice_index)
            overlay = volume[:, :, slice_index]  # (Z, Y)
            # Resize to (Y, X) for display
            overlay = resize_to_match(overlay, base_shape)

        # Convert overlay to float32 and normalize if needed
        overlay = overlay.astype(np.float32)
        if overlay.max() > 1.0:
            overlay /= 255.0

        # Apply translation in the XY plane
        x_offset, y_offset = getattr(layer, 'offset', (0, 0))
        overlay = translate_image(overlay, x_offset, y_offset)

        # Blend the overlay into the output image using the layer's opacity
        opacity = np.clip(getattr(layer, 'opacity', 1.0), 0.0, 1.0)

        # Perform alpha blending
        img = img * (1 - opacity) + overlay * opacity

    # Convert the final image to 8-bit unsigned integer format for display
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

def translate_image(img, x_offset, y_offset):
    """
    Shift a 2D image without wraparound, filling empty space with 0s.
    """
    h, w = img.shape

    # Create an output image filled with zeros
    result = np.zeros_like(img)

    # Calculate source and destination slice indices for x and y axes
    src_x_start, src_x_end, dst_x_start, dst_x_end = calculate_shift_coords(x_offset, w)
    src_y_start, src_y_end, dst_y_start, dst_y_end = calculate_shift_coords(y_offset, h)

    # Copy the valid region from the source image to the destination
    if src_x_end > src_x_start and src_y_end > src_y_start:
        result[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = img[src_y_start:src_y_end,
                                                               src_x_start:src_x_end]

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
    # Positive offset: source starts at 0, destination starts at offset
    if offset >= 0:
        src_start = 0
        src_end = max(0, length - offset)
        dst_start = offset
        dst_end = offset + src_end
    else:
        # Negative offset: source starts at -offset, destination starts at 0
        src_start = -offset
        src_end = length
        dst_start = 0
        dst_end = src_end - src_start

    return src_start, src_end, dst_start, dst_end



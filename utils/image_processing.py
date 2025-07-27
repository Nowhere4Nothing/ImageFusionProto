import numpy as np
from scipy.ndimage import rotate

def process_layers(volume_layers, slice_index):
    if not volume_layers:
        return None

    base_shape = volume_layers[0].data[0].shape
    composite_img = np.zeros(base_shape, dtype=np.float32)
    composite_alpha = np.zeros(base_shape, dtype=np.float32)

    for layer in volume_layers:
        if not layer.visible:
            continue

        volume = layer.data.copy()

        # Apply rotations
        if layer.rotation[2] != 0:
            volume = rotate(volume, angle=layer.rotation[2], axes=(0, 1), reshape=False, mode='nearest')
        if layer.rotation[1] != 0:
            volume = rotate(volume, angle=layer.rotation[1], axes=(0, 2), reshape=False, mode='nearest')
        if layer.rotation[0] != 0:
            volume = rotate(volume, angle=layer.rotation[0], axes=(1, 2), reshape=False, mode='nearest')

        slice_idx = np.clip(slice_index + layer.slice_offset, 0, volume.shape[0] - 1)
        overlay = volume[slice_idx]

        # Offest shift
        shifted = np.roll(overlay, shift=layer.offset[0], axis=1)
        shifted = np.roll(shifted, shift=layer.offset[1], axis=0)

        # Create alpha mask
        alpha = layer.opacity * (shifted > 0).astype(np.float32)  # mask out background (zero pixels)

        # Sequential alpha compositing
        composite_img = composite_img * (1 - alpha) + shifted * alpha
        composite_alpha = composite_alpha + alpha * (1 - composite_alpha)

    final_img = np.clip(composite_img, 0, 1) * 255
    return final_img.astype(np.uint8)

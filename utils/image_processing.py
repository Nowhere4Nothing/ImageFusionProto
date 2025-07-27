import numpy as np
from scipy.ndimage import rotate

def process_layers(volume_layers, slice_index):
    base_shape = volume_layers[0].data[0].shape
    img = np.zeros(base_shape, dtype=np.float32)

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

        shifted = np.roll(overlay, shift=layer.offset[0], axis=1)
        shifted = np.roll(shifted, shift=layer.offset[1], axis=0)

        img = img * (1 - layer.opacity) + shifted * layer.opacity

    return (np.clip(img, 0, 1) * 255).astype(np.uint8)

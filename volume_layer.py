class VolumeLayer:
    def __init__(self, volume_data, name,  spacing=None):
        self.data = volume_data
        self.name = name
        self.opacity = 1.0
        self.offset = [0, 0, 0]  # [x, y, z] for pixelwise translation in all axes
        self.slice_offset = 0  # (deprecated, use offset[2])
        self.rotation = [0, 0, 0]  #[LR, PA, IS]
        self.visible = True
        self.spacing = spacing
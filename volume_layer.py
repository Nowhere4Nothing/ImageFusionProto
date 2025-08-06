class VolumeLayer:
    def __init__(self, volume_data, name,  spacing=None):
        self.data = volume_data
        self.name = name
        self.opacity = 1.0
        self.offset = [0, 0]
        self.slice_offset = 0
        self.rotation = [0, 0, 0]  #[LR, PA, IS]
        self.visible = True
        self.spacing = spacing



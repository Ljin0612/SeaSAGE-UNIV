from torch import nn

class SeaSAGEYOLOv8Head(nn.Module):
    """Experimental YOLOv8-style decoupled head placeholder for later ablations."""
    def __init__(self, num_classes: int, in_channels: int = 256):
        super().__init__(); self.num_classes=num_classes
    def forward(self, features):
        raise NotImplementedError("YOLOv8-style head is experimental; use --head frcnn for stable smoke tests.")

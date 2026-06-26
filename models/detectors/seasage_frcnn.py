from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models.detection import fasterrcnn_resnet50_fpn

class SeaSAGEFasterRCNN(nn.Module):
    """Stable first-stage detector wrapper around torchvision Faster R-CNN."""
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None, num_classes=num_classes)
    def forward(self, images, targets=None):
        return self.model(images, targets)

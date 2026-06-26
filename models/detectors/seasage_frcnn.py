from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator

from models.backbones.univ_backbone import UNIVBackbone


class SeaSAGEFRCNNBackbone(nn.Module):
    """Adapt UNIV RGB features to the feature-map contract Faster R-CNN expects."""

    def __init__(self, univ_weights: str | None = None, out_channels: int = 256):
        super().__init__()
        print("Using SeaSAGEUNIV RGB-only backbone")
        print(f"UNIV weights path: {univ_weights}")
        self.univ = UNIVBackbone(weights=univ_weights)
        self.proj = nn.Conv2d(self.univ.embed_dim, out_channels, kernel_size=1)
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        features = self.univ(x, output_format="bchw")
        features = self.proj(features)
        print(f"[SeaSAGEFRCNNBackbone] feature shape: {tuple(features.shape)}")
        return OrderedDict([("0", features)])


class SeaSAGEFasterRCNN(nn.Module):
    """Faster R-CNN detector using the real UNIV RGB backbone."""

    def __init__(self, num_classes: int, univ_weights: str | None = None):
        super().__init__()
        backbone = SeaSAGEFRCNNBackbone(univ_weights=univ_weights)
        anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),))
        self.model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            min_size=224,
            max_size=640,
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)

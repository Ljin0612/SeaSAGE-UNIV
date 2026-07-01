from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.ops import MultiScaleRoIAlign

from models.backbones.univ_backbone import UNIVBackbone


class SeaSAGEFRCNNBackbone(nn.Module):
    """Adapt UNIV RGB features to the feature-map contract Faster R-CNN expects."""

    def __init__(self, univ_weights: str | None = None, out_channels: int = 256):
        super().__init__()
        print("Using SeaSAGEUNIV RGB-only single-scale backbone")
        print(f"UNIV weights path: {univ_weights}")
        self.univ = UNIVBackbone(weights=univ_weights)
        self.proj = nn.Conv2d(self.univ.out_channels, out_channels, kernel_size=1)
        self.out_channels = out_channels
        self._printed_shape = False

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        features = self.univ(x, output_format="bchw")
        features = self.proj(features)
        if not self._printed_shape:
            print(f"[SeaSAGEFRCNNBackbone] feature shape: {tuple(features.shape)}")
            self._printed_shape = True
        return OrderedDict([("0", features)])


class SeaSAGEFPNBackbone(nn.Module):
    """Build a lightweight four-level FPN adapter from the UNIV 14x14 feature map."""

    def __init__(self, univ_weights: str | None = None, out_channels: int = 256):
        super().__init__()
        print("Using SeaSAGEUNIV RGB-only multi-scale FPN adapter")
        print(f"UNIV weights path: {univ_weights}")
        self.univ = UNIVBackbone(weights=univ_weights)
        self.p5 = nn.Conv2d(self.univ.out_channels, out_channels, kernel_size=1)
        self.p4 = nn.Sequential(nn.Upsample(size=(28, 28), mode="bilinear", align_corners=False), nn.Conv2d(self.univ.out_channels, out_channels, kernel_size=1), nn.ReLU(inplace=True))
        self.p3 = nn.Sequential(nn.Upsample(size=(56, 56), mode="bilinear", align_corners=False), nn.Conv2d(self.univ.out_channels, out_channels, kernel_size=1), nn.ReLU(inplace=True))
        self.p2 = nn.Sequential(nn.Upsample(size=(112, 112), mode="bilinear", align_corners=False), nn.Conv2d(self.univ.out_channels, out_channels, kernel_size=1), nn.ReLU(inplace=True))
        self.out_channels = out_channels
        self._printed_shapes = False

    def forward(self, x: torch.Tensor) -> OrderedDict[str, torch.Tensor]:
        features = self.univ(x, output_format="bchw")
        outputs = OrderedDict((
            ("P2", self.p2(features)),
            ("P3", self.p3(features)),
            ("P4", self.p4(features)),
            ("P5", self.p5(features)),
        ))
        if not self._printed_shapes:
            for name, feat in outputs.items():
                print(f"[SeaSAGEFPNBackbone] {name} shape: {tuple(feat.shape)}")
            self._printed_shapes = True
        return outputs


class SeaSAGEFasterRCNN(nn.Module):
    """Faster R-CNN detector using the real UNIV RGB backbone."""

    def __init__(
        self,
        num_classes: int,
        univ_weights: str | None = None,
        imgsz: int = 640,
        mode: str = "rgb_only",
        backbone_type: str = "seasage_univ_single",
    ):
        super().__init__()
        if mode != "rgb_only":
            raise ValueError(f"SeaSAGEFasterRCNN only supports mode='rgb_only' for this detector, got {mode!r}")
        if backbone_type in {"seasage_univ", "seasage_univ_single"}:
            backbone = SeaSAGEFRCNNBackbone(univ_weights=univ_weights)
            anchor_generator = AnchorGenerator(sizes=((16, 32, 64, 128, 256),), aspect_ratios=((0.5, 1.0, 2.0),))
            roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
        elif backbone_type == "seasage_univ_fpn":
            backbone = SeaSAGEFPNBackbone(univ_weights=univ_weights)
            anchor_generator = AnchorGenerator(
                sizes=((16,), (32,), (64,), (128,)),
                aspect_ratios=((0.5, 1.0, 2.0),) * 4,
            )
            roi_pooler = MultiScaleRoIAlign(featmap_names=["P2", "P3", "P4", "P5"], output_size=7, sampling_ratio=2)
        else:
            raise ValueError("backbone_type must be 'seasage_univ_single' or 'seasage_univ_fpn'")
        self.backbone_type = "seasage_univ_single" if backbone_type == "seasage_univ" else backbone_type
        self.model = FasterRCNN(
            backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=224,
            max_size=imgsz,
        )

    def forward(self, images, targets=None):
        return self.model(images, targets)

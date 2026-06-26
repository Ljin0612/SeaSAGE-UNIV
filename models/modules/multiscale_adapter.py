from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F


class MultiScaleAdapter(nn.Module):
    """Convert BxNxD unified patches into P3/P4/P5 detection features."""

    def __init__(self, in_dim: int = 384, out_channels: int = 256, debug: bool = True, use_p2: bool = False):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_channels, 1)
        self.debug = debug
        self.use_p2 = use_p2
        self._printed_shapes = False

    def forward(self, patches: torch.Tensor, imgsz: int = 640):
        if patches.ndim != 3:
            raise ValueError(f"patches must be BxNxD, got {tuple(patches.shape)}")
        b, n, d = patches.shape
        h = w = int(math.sqrt(n))
        if h * w != n:
            raise ValueError(f"Patch count N={n} is not a square grid; cannot reshape to feature map.")
        if imgsz % 32 != 0:
            raise ValueError(f"imgsz={imgsz} must be divisible by 32 for P3/P4/P5 generation.")
        x = self.proj(patches.transpose(1, 2).reshape(b, d, h, w))
        expected = {"p3": (imgsz // 8, imgsz // 8), "p4": (imgsz // 16, imgsz // 16), "p5": (imgsz // 32, imgsz // 32)}
        feats = {name: F.interpolate(x, size=size, mode="bilinear", align_corners=False) for name, size in expected.items()}
        if self.use_p2:
            feats["p2"] = F.interpolate(x, size=(imgsz // 4, imgsz // 4), mode="bilinear", align_corners=False)
        for name, size in expected.items():
            if feats[name].shape[-2:] != size:
                raise RuntimeError(f"{name.upper()} shape mismatch: expected {size}, got {tuple(feats[name].shape[-2:])}")
        if self.debug and not self._printed_shapes:
            print("[MultiScaleAdapter] " + ", ".join(f"{k}:{tuple(v.shape)}" for k, v in feats.items()))
            self._printed_shapes = True
        return feats

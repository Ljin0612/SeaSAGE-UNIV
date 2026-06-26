from __future__ import annotations

import torch
from torch import nn


class CrossModalFusion(nn.Module):
    """RGB-IR patch feature fusion: add, concat projection, or gated attention."""

    def __init__(self, dim: int = 384, mode: str = "add"):
        super().__init__()
        self.mode = mode
        self.proj = nn.Linear(dim * 2, dim)
        self.gate = nn.Linear(dim * 2, 1)
        self.last_attention_map: torch.Tensor | None = None

    def forward(self, rgb: torch.Tensor, ir: torch.Tensor | None = None, return_attention: bool = False):
        if ir is None or self.mode == "rgb_only":
            self.last_attention_map = None
            return (rgb, None) if return_attention else rgb
        if rgb.shape != ir.shape:
            raise ValueError(f"RGB/IR feature shapes must match, got {tuple(rgb.shape)} and {tuple(ir.shape)}")
        if self.mode == "add":
            fused, attn = rgb + ir, None
        else:
            cat = torch.cat([rgb, ir], dim=-1)
            if self.mode == "concat":
                fused, attn = self.proj(cat), None
            elif self.mode == "attention":
                attn = torch.sigmoid(self.gate(cat))
                fused = attn * rgb + (1.0 - attn) * ir
            else:
                raise ValueError(f"Unsupported fusion mode: {self.mode}")
        self.last_attention_map = attn
        return (fused, attn) if return_attention else fused

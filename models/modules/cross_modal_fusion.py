import torch
from torch import nn

class CrossModalFusion(nn.Module):
    """RGB-IR patch feature fusion: add, concat projection, or gated attention."""
    def __init__(self, dim: int = 384, mode: str = "add"):
        super().__init__(); self.mode=mode; self.proj=nn.Linear(dim*2, dim); self.gate=nn.Linear(dim*2, dim)
    def forward(self, rgb: torch.Tensor, ir: torch.Tensor | None = None) -> torch.Tensor:
        if ir is None: return rgb
        if self.mode == "add": return rgb + ir
        cat = torch.cat([rgb, ir], dim=-1)
        if self.mode == "concat": return self.proj(cat)
        if self.mode == "attention":
            g = torch.sigmoid(self.gate(cat)); return g * rgb + (1 - g) * ir
        raise ValueError(f"Unsupported fusion mode: {self.mode}")

"""UNIV/ViT backbone wrapper for SeaSAGE-UNIV."""
from __future__ import annotations
from pathlib import Path
from typing import Optional
import numpy as np
import torch
from torch import nn

# NumPy 1.24 removed aliases used by some older ViT/UNIV utilities.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]


class SimplePatchViT(nn.Module):
    """Lightweight patch encoder used until the original UNIV code is integrated."""
    def __init__(self, in_chans: int = 3, embed_dim: int = 384, patch_size: int = 16):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # B,D,H',W'
        b, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)  # B,N,D
        return self.norm(x)


class UNIVBackbone(nn.Module):
    """Wrapper that returns patch features as BxNxD and can load local checkpoints."""
    def __init__(self, weights: Optional[str] = None, embed_dim: int = 384, patch_size: int = 16, strict: bool = False):
        super().__init__()
        self.encoder = SimplePatchViT(embed_dim=embed_dim, patch_size=patch_size)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        if weights:
            self.load_pretrained(weights, strict=strict)

    def load_pretrained(self, weights: str, strict: bool = False) -> None:
        path = Path(weights)
        if not path.exists():
            print(f"[UNIVBackbone] checkpoint not found: {path}. Training from scratch.")
            return
        try:
            ckpt = torch.load(path, map_location="cpu")
        except Exception:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        cleaned = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state.items() if torch.is_tensor(v)}
        result = self.encoder.load_state_dict(cleaned, strict=strict)
        print(f"[UNIVBackbone] loaded keys: {len(cleaned)}")
        print(f"[UNIVBackbone] missing keys: {result.missing_keys}")
        print(f"[UNIVBackbone] unexpected keys: {result.unexpected_keys}")

    def forward(self, x: torch.Tensor, output_format: str = "bnd") -> torch.Tensor:
        patches = self.encoder(x)
        if output_format == "bchw":
            b, n, d = patches.shape
            h = w = int(n ** 0.5)
            return patches.transpose(1, 2).reshape(b, d, h, w)
        return patches

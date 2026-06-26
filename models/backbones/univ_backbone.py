"""UNIV/ConvMAE backbone wrapper for SeaSAGE-UNIV."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

# NumPy 1.24 removed aliases used by some older ViT/UNIV utilities.
if not hasattr(np, "float"):
    np.float = float  # type: ignore[attr-defined]
if not hasattr(np, "int"):
    np.int = int  # type: ignore[attr-defined]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_UNIV_ROOT = _REPO_ROOT / "UNIV-main"
_MCMAE_DIR = _UNIV_ROOT / "models" / "backbone" / "mcmae"

try:
    import importlib.util
    import types

    package_name = "_seasage_univ_mcmae"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(_MCMAE_DIR)]  # type: ignore[attr-defined]
        sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(
        f"{package_name}.models_convmae",
        _MCMAE_DIR / "models_convmae.py",
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load UNIV ConvMAE module from {_MCMAE_DIR}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    convmae_convvit_base_patch16 = module.convmae_convvit_base_patch16
except Exception as exc:  # pragma: no cover - surfaced when UNIV deps are missing.
    convmae_convvit_base_patch16 = None  # type: ignore[assignment]
    _UNIV_IMPORT_ERROR = exc
else:
    _UNIV_IMPORT_ERROR = None


class UNIVBackbone(nn.Module):
    """Thin wrapper around the real UNIV ConvMAE encoder.

    UNIV's released ConvMAE model is configured for 224x224 inputs and returns
    a 14x14 sequence with 768 channels. For detection we resize RGB tensors to
    that native size before calling ``forward(..., mask_ratio=0.0)`` so no patch
    tokens are dropped.
    """

    def __init__(
        self,
        weights: Optional[str] = None,
        projection_dim: int | None = None,
        patch_size: int = 16,
        strict: bool = False,
        input_size: int = 224,
    ):
        super().__init__()
        if convmae_convvit_base_patch16 is None:
            raise ImportError(
                "Failed to import UNIV-main ConvMAE backbone. Install the UNIV "
                f"dependencies (for example timm). Original error: {_UNIV_IMPORT_ERROR}"
            )
        self.encoder = convmae_convvit_base_patch16()
        # convmae_convvit_base_patch16 emits 768-channel patch features.
        # Report the real encoder dimension instead of any downstream model width.
        self.raw_dim = 768
        self.out_channels = 768
        self.feature_dim = 768
        self.embed_dim = 768
        self.projection_dim = projection_dim
        self.patch_size = patch_size
        self.input_size = input_size
        if weights:
            self.load_pretrained(weights, strict=strict)

    def _select_state_dict(self, ckpt):
        if not isinstance(ckpt, dict):
            return ckpt
        for key in ("model", "state_dict", "student", "teacher", "backbone"):
            value = ckpt.get(key)
            if isinstance(value, dict) and any(torch.is_tensor(v) for v in value.values()):
                return value
        if any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
        return {}

    def load_pretrained(self, weights: str, strict: bool = False) -> None:
        path = Path(weights)
        print(f"[UNIVBackbone] UNIV weights path: {path}")
        if not path.exists():
            print(f"[UNIVBackbone] checkpoint not found: {path}. Training from scratch.")
            return
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        state = self._select_state_dict(ckpt)
        cleaned = {}
        for key, value in state.items():
            if not torch.is_tensor(value):
                continue
            name = key
            for prefix in ("module.backbone.", "backbone.", "module.encoder.", "encoder.", "module."):
                if name.startswith(prefix):
                    name = name[len(prefix):]
                    break
            cleaned[name] = value
        result = self.encoder.load_state_dict(cleaned, strict=strict)
        loaded = sorted(set(cleaned) - set(result.unexpected_keys))
        print(f"[UNIVBackbone] Loaded UNIV checkpoint: {path}")
        print(f"[UNIVBackbone] loaded keys ({len(loaded)}): {loaded[:20]}{' ...' if len(loaded) > 20 else ''}")
        print(f"[UNIVBackbone] missing keys ({len(result.missing_keys)}): {result.missing_keys}")
        print(f"[UNIVBackbone] unexpected keys ({len(result.unexpected_keys)}): {result.unexpected_keys}")

    def forward(self, x: torch.Tensor, output_format: str = "bnd") -> torch.Tensor:
        if x.shape[-2:] != (self.input_size, self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        patches, _ = self.encoder(x, mask_ratio=0.0, return_last_attention=False)
        if output_format == "bchw":
            b, n, d = patches.shape
            h = w = int(n ** 0.5)
            features = patches.transpose(1, 2).reshape(b, d, h, w).contiguous()
            assert features.shape[1] == self.out_channels
            print(f"[UNIVBackbone] feature shape: {tuple(features.shape)}")
            return features
        assert patches.shape[-1] == self.out_channels
        print(f"[UNIVBackbone] feature shape: {tuple(patches.shape)}")
        return patches

from __future__ import annotations

import math

from torch import nn

from models.backbones.univ_backbone import UNIVBackbone
from models.modules.semantic_label_assignment import SemanticLabelAssignment
from models.modules.semantic_aware_pccl import SemanticAwarePCCL
from models.modules.cross_modal_fusion import CrossModalFusion
from models.modules.multiscale_adapter import MultiScaleAdapter


ABLATION_MODES = {
    "rgb_only",
    "ir_only",
    "rgb_ir_no_pccl",
    "rgb_ir_pccl",
    "rgb_ir_semantic_pccl",
    "rgb_ir_semantic_pccl_attention_fusion",
}


class SeaSAGEUNIV(nn.Module):
    """Semantic-aware RGB-IR maritime detection framework core."""

    def __init__(
        self,
        univ_weights: str | None = None,
        mode: str = "rgb_only",
        fusion_mode: str = "add",
        model_dim: int = 384,
        num_semantic_groups: int = 8,
        temperature: float = 0.07,
        background_weight: float = 0.25,
        pccl_normalization: str = "valid_count",
    ):
        super().__init__()
        if mode == "rgb":
            mode = "rgb_only"
        if mode == "rgbir":
            mode = "rgb_ir_semantic_pccl"
        if mode not in ABLATION_MODES:
            raise ValueError(f"Unsupported ablation mode: {mode}. Choose from {sorted(ABLATION_MODES)}")
        if mode == "rgb_ir_semantic_pccl_attention_fusion":
            fusion_mode = "attention"
        self.mode = mode
        self.rgb_encoder = UNIVBackbone(univ_weights)
        self.ir_encoder = UNIVBackbone(univ_weights)
        encoder_dim = self.rgb_encoder.out_channels
        self.encoder_dim = encoder_dim
        self.model_dim = model_dim
        self.projection_enabled = encoder_dim != model_dim
        self.rgb_proj = nn.Conv2d(encoder_dim, model_dim, kernel_size=1)
        self.ir_proj = nn.Conv2d(encoder_dim, model_dim, kernel_size=1)
        print(f"UNIV encoder_dim: {encoder_dim}")
        print(f"SeaSAGE model_dim: {model_dim}")
        print(f"projection enabled: {self.projection_enabled}")
        self.assign = SemanticLabelAssignment(num_semantic_groups=num_semantic_groups)
        self.pccl = SemanticAwarePCCL(
            temperature=temperature,
            background_weight=background_weight,
            normalization=pccl_normalization,
        )
        self.fusion = CrossModalFusion(model_dim, fusion_mode)
        self.adapter = MultiScaleAdapter(model_dim)

    @property
    def needs_ir(self) -> bool:
        return self.mode != "rgb_only"

    def _project_patches(self, patches, proj):
        if patches.ndim != 3:
            raise ValueError(f"patches must be BxNxD, got {tuple(patches.shape)}")
        b, n, d = patches.shape
        h = w = int(math.sqrt(n))
        if h * w != n:
            raise ValueError(f"Patch count N={n} is not a square grid; cannot project feature map.")
        x = patches.transpose(1, 2).reshape(b, d, h, w)
        x = proj(x)
        return x.flatten(2).transpose(1, 2).contiguous()

    def forward(self, rgb, ir=None, imgsz: int = 640, return_loss: bool = True):
        if self.mode == "ir_only":
            if ir is None:
                raise ValueError("IR input is required when mode='ir_only'.")
            patch_features = self._project_patches(self.ir_encoder(ir), self.ir_proj)
            return {"features": self.adapter(patch_features, imgsz=imgsz), "loss_sem_pccl": None, "attention_map": None}

        rgb_f = self._project_patches(self.rgb_encoder(rgb), self.rgb_proj)
        loss_sem_pccl = None
        attention_map = None
        fused = rgb_f

        if self.needs_ir:
            if ir is None:
                raise ValueError(f"IR input is required when mode='{self.mode}'.")
            ir_f = self._project_patches(self.ir_encoder(ir), self.ir_proj)
            if self.mode in {"rgb_ir_semantic_pccl", "rgb_ir_semantic_pccl_attention_fusion"}:
                assignment = self.assign(rgb_f, ir_f)
                if return_loss:
                    loss_sem_pccl = self.pccl(
                        rgb_f,
                        ir_f,
                        assignment.semantic_labels,
                        assignment.semantic_confidence,
                        assignment.foreground_mask,
                    )
            elif self.mode == "rgb_ir_pccl" and return_loss:
                # Original non-semantic patch-level contrastive baseline: each aligned patch is its own class.
                import torch

                labels = torch.arange(rgb_f.shape[1], device=rgb_f.device).repeat(rgb_f.shape[0], 1)
                loss_sem_pccl = self.pccl(rgb_f, ir_f, labels)
            fused, attention_map = self.fusion(rgb_f, ir_f, return_attention=True)

        return {"features": self.adapter(fused, imgsz=imgsz), "loss_sem_pccl": loss_sem_pccl, "attention_map": attention_map}

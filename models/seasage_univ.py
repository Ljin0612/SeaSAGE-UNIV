from __future__ import annotations
from torch import nn
from models.backbones.univ_backbone import UNIVBackbone
from models.modules.semantic_label_assignment import SemanticLabelAssignment
from models.modules.semantic_aware_pccl import SemanticAwarePCCL
from models.modules.cross_modal_fusion import CrossModalFusion
from models.modules.multiscale_adapter import MultiScaleAdapter

class SeaSAGEUNIV(nn.Module):
    """SeaSAGE-UNIV core: encoders, semantic assignment/PCCL, fusion, multiscale adapter."""
    def __init__(self, univ_weights: str | None = None, mode: str = "rgb", fusion_mode: str = "add", embed_dim: int = 384):
        super().__init__(); self.mode=mode
        self.rgb_encoder = UNIVBackbone(univ_weights, embed_dim=embed_dim)
        self.ir_encoder = UNIVBackbone(univ_weights, embed_dim=embed_dim) if mode == "rgbir" else None
        self.assign = SemanticLabelAssignment(); self.pccl = SemanticAwarePCCL()
        self.fusion = CrossModalFusion(embed_dim, fusion_mode); self.adapter = MultiScaleAdapter(embed_dim)
    def forward(self, rgb, ir=None, imgsz: int = 640, return_loss: bool = True):
        rgb_f = self.rgb_encoder(rgb)
        loss_pccl = None; fused = rgb_f
        if self.mode == "rgbir":
            if ir is None: raise ValueError("IR input is required when mode='rgbir'.")
            ir_f = self.ir_encoder(ir)
            labels = self.assign(rgb_f, ir_f)
            loss_pccl = self.pccl(rgb_f, ir_f, labels) if return_loss else None
            fused = self.fusion(rgb_f, ir_f)
        return {"features": self.adapter(fused, imgsz=imgsz), "loss_pccl": loss_pccl}

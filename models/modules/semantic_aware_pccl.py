from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SemanticAwarePCCL(nn.Module):
    """Semantic-aware RGB-IR patch contrastive loss."""

    def __init__(self, temperature: float = 0.07, background_weight: float = 0.25, eps: float = 1e-8):
        super().__init__()
        self.temperature = temperature
        self.background_weight = background_weight
        self.eps = eps

    def forward(
        self,
        rgb_features: torch.Tensor,
        ir_features: torch.Tensor,
        semantic_labels: torch.Tensor,
        semantic_confidence: torch.Tensor | None = None,
        foreground_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if rgb_features.shape != ir_features.shape:
            raise ValueError("rgb_features and ir_features must have identical BxNxD shape.")
        if semantic_labels.shape != rgb_features.shape[:2]:
            raise ValueError("semantic_labels must have shape BxN.")

        rgb = F.normalize(rgb_features, dim=-1)
        ir = F.normalize(ir_features, dim=-1)
        logits = torch.matmul(rgb, ir.transpose(1, 2)) / self.temperature
        positive = semantic_labels.unsqueeze(2).eq(semantic_labels.unsqueeze(1))
        if foreground_mask is not None:
            positive = positive & (foreground_mask.unsqueeze(2) | foreground_mask.unsqueeze(1))
        positive_count = positive.sum(dim=-1)

        logp_rgb_ir = F.log_softmax(logits, dim=-1)
        logp_ir_rgb = F.log_softmax(logits.transpose(1, 2), dim=-1)
        safe_pos = positive_count.clamp_min(1)
        loss_rgb_ir = -(positive.float() * logp_rgb_ir).sum(dim=-1) / safe_pos
        loss_ir_rgb = -(positive.transpose(1, 2).float() * logp_ir_rgb).sum(dim=-1) / safe_pos
        loss = 0.5 * (loss_rgb_ir + loss_ir_rgb)

        valid = positive_count > 0
        weights = torch.ones_like(loss)
        if semantic_confidence is not None:
            weights = weights * semantic_confidence.clamp(0, 1)
        weights = weights * torch.where(semantic_labels == 0, self.background_weight, 1.0)
        weights = weights * valid.float()
        denom = weights.sum().clamp_min(self.eps)
        return (loss * weights).sum() / denom

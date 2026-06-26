from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class SemanticAwarePCCL(nn.Module):
    """Semantic-aware RGB-IR patch contrastive loss."""

    def __init__(
        self,
        temperature: float = 0.07,
        background_weight: float = 0.25,
        eps: float = 1e-8,
        normalization: str = "valid_count",
    ):
        super().__init__()
        if normalization not in {"valid_count", "weight_sum"}:
            raise ValueError("normalization must be either 'valid_count' or 'weight_sum'.")
        self.temperature = temperature
        self.background_weight = background_weight
        self.eps = eps
        self.normalization = normalization

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
        same_label = semantic_labels.unsqueeze(2).eq(semantic_labels.unsqueeze(1))
        if foreground_mask is None:
            positive = same_label
            anchor_weight = torch.ones_like(semantic_labels, dtype=logits.dtype)
        else:
            foreground = foreground_mask.bool()
            fg_i = foreground.unsqueeze(2)
            fg_j = foreground.unsqueeze(1)
            fg_fg = fg_i & fg_j
            bg_bg = (~fg_i) & (~fg_j)
            # Foreground-background mixed pairs are not positives even when their
            # semantic labels match. Background-background positives are kept
            # because sea/sky context is informative. The background_weight is
            # applied once at the anchor level below, not inside positive
            # averaging, so it cannot be canceled by the per-anchor denominator.
            positive = same_label & (fg_fg | bg_bg)
            anchor_weight = torch.where(
                foreground,
                logits.new_ones(()),
                logits.new_full((), self.background_weight),
            )
        positive_count = positive.float().sum(dim=-1)

        logp_rgb_ir = F.log_softmax(logits, dim=-1)
        logp_ir_rgb = F.log_softmax(logits.transpose(1, 2), dim=-1)
        safe_pos = positive_count.clamp_min(1)
        loss_rgb_ir = -(positive.float() * logp_rgb_ir).sum(dim=-1) / safe_pos
        loss_ir_rgb = -(positive.transpose(1, 2).float() * logp_ir_rgb).sum(dim=-1) / safe_pos
        loss = 0.5 * (loss_rgb_ir + loss_ir_rgb)

        valid = positive_count > 0
        weights = anchor_weight
        if semantic_confidence is not None:
            weights = weights * semantic_confidence.clamp(0, 1)
        weights = weights * valid.float()

        # normalization='valid_count' divides by the number of valid anchors, so
        # lower background_weight strongly suppresses background contributions.
        # normalization='weight_sum' divides by the sum of anchor weights, which
        # preserves the overall loss scale while still changing anchor balance.
        if self.normalization == "valid_count":
            denom = valid.float().sum().clamp_min(self.eps)
        else:
            denom = weights.sum().clamp_min(self.eps)
        return (loss * weights).sum() / denom

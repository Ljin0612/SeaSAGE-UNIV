from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SemanticAssignmentOutput:
    semantic_labels: torch.Tensor
    semantic_confidence: torch.Tensor
    foreground_mask: torch.Tensor
    background_mask: torch.Tensor

    def __iter__(self):
        yield self.semantic_labels
        yield self.semantic_confidence



class SemanticLabelAssignment(nn.Module):
    """Generate stable patch-level semantic pseudo labels for RGB/IR features.

    The first implementation uses normalized RGB/IR feature similarity followed by
    per-image k-means.  Label ``0`` is reserved for background-like patches; labels
    ``1..num_semantic_groups-1`` are foreground-like semantic groups.
    """

    def __init__(
        self,
        num_semantic_groups: int = 8,
        iterations: int = 10,
        mode: str = "kmeans",
        foreground_ratio: float = 0.35,
        debug: bool = True,
        num_clusters: int | None = None,
    ):
        super().__init__()
        if num_clusters is not None:
            num_semantic_groups = num_clusters
        if num_semantic_groups < 2:
            raise ValueError("num_semantic_groups must be >= 2 so foreground/background can be separated.")
        self.num_semantic_groups = num_semantic_groups
        self.iterations = iterations
        self.mode = mode
        self.foreground_ratio = foreground_ratio
        self.debug = debug

    def attention_guided_assignment(self, *args, **kwargs):
        raise NotImplementedError("attention-guided semantic assignment is reserved for the next version.")

    def saliency_guided_assignment(self, *args, **kwargs):
        raise NotImplementedError("saliency-guided semantic assignment is reserved for the next version.")

    @torch.no_grad()
    def forward(self, rgb_features: torch.Tensor, ir_features: torch.Tensor | None = None) -> SemanticAssignmentOutput:
        if rgb_features.ndim != 3:
            raise ValueError(f"rgb_features must be BxNxD, got {tuple(rgb_features.shape)}")
        if ir_features is not None and ir_features.shape != rgb_features.shape:
            raise ValueError(f"ir_features shape {tuple(ir_features.shape)} must match rgb_features {tuple(rgb_features.shape)}")
        if self.mode != "kmeans":
            raise ValueError(f"Unsupported semantic assignment mode: {self.mode}")

        rgb = F.normalize(rgb_features, dim=-1)
        feats = rgb if ir_features is None else F.normalize(0.5 * (rgb_features + ir_features), dim=-1)
        cross_sim = torch.ones(feats.shape[:2], device=feats.device, dtype=feats.dtype)
        if ir_features is not None:
            cross_sim = (rgb * F.normalize(ir_features, dim=-1)).sum(dim=-1).clamp(-1, 1)

        batch_labels, batch_conf, batch_fg = [], [], []
        for bi, x in enumerate(feats):
            n = x.shape[0]
            k = min(self.num_semantic_groups, n)
            # Deterministic farthest-point initialization improves stability over random centers.
            centers = [x[0]]
            min_dist = torch.cdist(x, centers[0].view(1, -1)).squeeze(1)
            for _ in range(1, k):
                idx = min_dist.argmax()
                centers.append(x[idx])
                min_dist = torch.minimum(min_dist, torch.cdist(x, centers[-1].view(1, -1)).squeeze(1))
            centers = torch.stack(centers, dim=0)

            labels = torch.zeros(n, dtype=torch.long, device=x.device)
            for _ in range(self.iterations):
                distances = torch.cdist(x, centers)
                labels = distances.argmin(dim=1)
                for ci in range(k):
                    mask = labels == ci
                    if mask.any():
                        centers[ci] = F.normalize(x[mask].mean(dim=0), dim=0)

            distances = torch.cdist(x, centers)
            nearest = distances.argmin(dim=1)
            sorted_dist, _ = distances.sort(dim=1)
            margin = sorted_dist[:, 1] - sorted_dist[:, 0] if k > 1 else torch.ones(n, device=x.device)
            confidence = torch.sigmoid(5.0 * margin) * ((cross_sim[bi] + 1.0) * 0.5)

            cluster_scores = torch.stack([cross_sim[bi][nearest == ci].mean() if (nearest == ci).any() else x.new_tensor(-1) for ci in range(k)])
            bg_cluster = int(cluster_scores.argmin().item())
            labels = nearest + 1
            labels[nearest == bg_cluster] = 0

            # Also mark low-confidence/low-cross-modal-similarity patches as background-like.
            fg_budget = max(1, int(round(n * self.foreground_ratio)))
            sim_threshold = torch.topk(cross_sim[bi], k=fg_budget).values.min()
            foreground_mask = (labels != 0) & (cross_sim[bi] >= sim_threshold)
            labels[~foreground_mask] = 0

            if self.debug:
                counts = torch.bincount(labels, minlength=self.num_semantic_groups).detach().cpu().tolist()
                print(f"[SemanticLabelAssignment] batch={bi} semantic_group_counts={counts}")

            batch_labels.append(labels)
            batch_conf.append(confidence.clamp(0, 1))
            batch_fg.append(foreground_mask)

        semantic_labels = torch.stack(batch_labels, dim=0)
        foreground_mask = torch.stack(batch_fg, dim=0)
        return SemanticAssignmentOutput(
            semantic_labels=semantic_labels,
            semantic_confidence=torch.stack(batch_conf, dim=0),
            foreground_mask=foreground_mask,
            background_mask=~foreground_mask,
        )

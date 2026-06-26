import torch
from torch import nn
import torch.nn.functional as F

class SemanticLabelAssignment(nn.Module):
    """Patch-level pseudo semantic labels via simplified k-means over RGB/IR features."""
    def __init__(self, num_clusters: int = 8, iterations: int = 5, mode: str = "kmeans"):
        super().__init__(); self.num_clusters=num_clusters; self.iterations=iterations; self.mode=mode
    def forward(self, rgb_features: torch.Tensor, ir_features: torch.Tensor | None = None) -> torch.Tensor:
        feats = rgb_features if ir_features is None else 0.5 * (rgb_features + ir_features)
        b,n,d = feats.shape; labels=[]
        for x in F.normalize(feats, dim=-1):
            k = min(self.num_clusters, n); centers = x[:k].clone()
            lab = torch.zeros(n, dtype=torch.long, device=x.device)
            for _ in range(self.iterations):
                lab = torch.cdist(x, centers).argmin(dim=1)
                for c in range(k):
                    if (lab == c).any(): centers[c] = x[lab == c].mean(0)
            labels.append(lab)
        return torch.stack(labels, 0)

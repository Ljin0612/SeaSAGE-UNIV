import torch
from torch import nn
import torch.nn.functional as F

class SemanticAwarePCCL(nn.Module):
    """Contrastive loss: pull same semantic RGB/IR patches together, push others apart."""
    def __init__(self, temperature: float = 0.07):
        super().__init__(); self.temperature = temperature
    def forward(self, rgb_features: torch.Tensor, ir_features: torch.Tensor, semantic_labels: torch.Tensor) -> torch.Tensor:
        rgb = F.normalize(rgb_features, dim=-1); ir = F.normalize(ir_features, dim=-1)
        logits = torch.matmul(rgb, ir.transpose(1,2)) / self.temperature
        same = semantic_labels.unsqueeze(2).eq(semantic_labels.unsqueeze(1)).float()
        logp = F.log_softmax(logits, dim=-1)
        loss = -(same * logp).sum(-1) / same.sum(-1).clamp_min(1.0)
        return loss.mean()

import torch
from torch import nn
import torch.nn.functional as F

class MultiScaleAdapter(nn.Module):
    """Convert BxNxD UNIV patches into P3/P4/P5 detection features."""
    def __init__(self, in_dim: int = 384, out_channels: int = 256, debug: bool = True, use_p2: bool = False):
        super().__init__(); self.proj=nn.Conv2d(in_dim,out_channels,1); self.debug=debug; self.use_p2=use_p2
    def forward(self, patches: torch.Tensor, imgsz: int = 640):
        b,n,d = patches.shape; h=w=int(n**0.5)
        x = patches.transpose(1,2).reshape(b,d,h,w); x = self.proj(x)
        feats = {
            "p3": F.interpolate(x, size=(imgsz//8, imgsz//8), mode="bilinear", align_corners=False),
            "p4": F.interpolate(x, size=(imgsz//16, imgsz//16), mode="bilinear", align_corners=False),
            "p5": F.interpolate(x, size=(imgsz//32, imgsz//32), mode="bilinear", align_corners=False),
        }
        if self.use_p2: feats["p2"] = F.interpolate(x, size=(imgsz//4, imgsz//4), mode="bilinear", align_corners=False)
        if self.debug:
            print("[MultiScaleAdapter] " + ", ".join(f"{k}:{tuple(v.shape)}" for k,v in feats.items()))
        return feats

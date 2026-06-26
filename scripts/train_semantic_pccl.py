#!/usr/bin/env python
import argparse, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.modules.semantic_label_assignment import SemanticLabelAssignment
from models.modules.semantic_aware_pccl import SemanticAwarePCCL

def main():
    p=argparse.ArgumentParser(); p.add_argument('--batch',type=int,default=2); p.add_argument('--patches',type=int,default=64); p.add_argument('--dim',type=int,default=384); p.add_argument('--temperature',type=float,default=0.07); a=p.parse_args()
    rgb=torch.randn(a.batch,a.patches,a.dim); ir=torch.randn(a.batch,a.patches,a.dim)
    labels=SemanticLabelAssignment()(rgb,ir); loss=SemanticAwarePCCL(a.temperature)(rgb,ir,labels); print(f"loss_pccl={loss.item():.6f}")
if __name__=='__main__': main()

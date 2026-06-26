#!/usr/bin/env python
import argparse, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.rgbir_paired_dataset import RGBIRPairedDataset

def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--epochs',type=int,default=1); p.add_argument('--batch',type=int,default=1); p.add_argument('--imgsz',type=int,default=640); p.add_argument('--device',default='cpu'); p.add_argument('--project',default='runs/detect'); p.add_argument('--name',default='exp'); p.add_argument('--univ-weights'); p.add_argument('--fusion-mode',default='add'); p.add_argument('--pccl-weight',type=float,default=1.0); p.add_argument('--num-workers',type=int,default=0); a=p.parse_args()
    try: RGBIRPairedDataset(a.data,'train',a.imgsz)
    except FileNotFoundError as e: raise SystemExit(str(e))
    raise SystemExit('RGB-IR training loop is reserved until paired data parsing is implemented.')
if __name__=='__main__': main()

#!/usr/bin/env python
import argparse, os, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.rgb_dataset import SeaShipsRGBDataset, detection_collate
from models.detectors.seasage_frcnn import SeaSAGEFasterRCNN

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--epochs',type=int,default=1); p.add_argument('--batch',type=int,default=1); p.add_argument('--imgsz',type=int,default=640); p.add_argument('--device',default='cpu'); p.add_argument('--project',default='runs/detect'); p.add_argument('--name',default='exp'); p.add_argument('--univ-weights',default=None); p.add_argument('--head',choices=['frcnn','yolov8'],default='frcnn'); p.add_argument('--num-workers',type=int,default=0); return p.parse_args()
def main():
    a=parse_args(); device=torch.device(f"cuda:{a.device}" if str(a.device).isdigit() and torch.cuda.is_available() else a.device)
    ds=SeaShipsRGBDataset(a.data,'train',a.imgsz); dl=DataLoader(ds,batch_size=a.batch,shuffle=True,num_workers=a.num_workers,collate_fn=detection_collate)
    num_classes=len(ds.names)+1
    if a.head!='frcnn': raise NotImplementedError('YOLOv8 head is experimental; use --head frcnn.')
    model=SeaSAGEFasterRCNN(num_classes).to(device); opt=torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.9)
    out=Path(a.project)/a.name; out.mkdir(parents=True,exist_ok=True)
    print(f"[train] UNIV weights argument reserved for SeaSAGE backbone: {a.univ_weights}")
    for e in range(a.epochs):
        model.train()
        for images,targets,_ in dl:
            images=[x.to(device) for x in images]; targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
            losses=model(images,targets); loss=sum(losses.values()); opt.zero_grad(); loss.backward(); opt.step(); print(f"epoch={e+1} loss={loss.item():.4f}"); break
    torch.save({'model':model.state_dict(),'args':vars(a)}, out/'last.pt'); print(f"saved {out/'last.pt'}")
if __name__=='__main__': main()

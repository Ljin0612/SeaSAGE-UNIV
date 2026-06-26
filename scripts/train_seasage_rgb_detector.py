#!/usr/bin/env python
import argparse, os, sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.rgb_dataset import SeaShipsRGBDataset, detection_collate
from models.detectors.seasage_frcnn import SeaSAGEFasterRCNN

def parse_args():
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--epochs',type=int,default=1); p.add_argument('--batch',type=int,default=1); p.add_argument('--imgsz',type=int,default=640); p.add_argument('--device',default='cpu'); p.add_argument('--project',default='runs/detect'); p.add_argument('--name',default='exp'); p.add_argument('--univ-weights',default=None); p.add_argument('--head',choices=['frcnn','yolov8'],default='frcnn'); p.add_argument('--num-workers',type=int,default=0); p.add_argument('--smoke-batches',type=int,default=0,help='Number of batches to train per epoch for smoke tests; 0 means full dataloader.'); return p.parse_args()
def main():
    a=parse_args(); device=torch.device(f"cuda:{a.device}" if str(a.device).isdigit() and torch.cuda.is_available() else a.device)
    ds=SeaShipsRGBDataset(a.data,'train',a.imgsz); dl=DataLoader(ds,batch_size=a.batch,shuffle=True,num_workers=a.num_workers,collate_fn=detection_collate)
    num_classes=len(ds.names)+1
    if a.head!='frcnn': raise NotImplementedError('YOLOv8 head is experimental; use --head frcnn.')
    model=SeaSAGEFasterRCNN(num_classes).to(device); opt=torch.optim.SGD(model.parameters(),lr=1e-4,momentum=0.9)
    out=Path(a.project)/a.name; out.mkdir(parents=True,exist_ok=True)
    print(f"[train] UNIV weights argument reserved for SeaSAGE backbone: {a.univ_weights}")
    if a.smoke_batches < 0:
        raise ValueError('--smoke-batches must be >= 0.')
    if a.smoke_batches > 0:
        print(f"Smoke mode enabled: training only {a.smoke_batches} batches per epoch.")
    for e in range(a.epochs):
        model.train()
        num_batches_trained = 0
        loss_total = 0.0
        for batch_idx,(images,targets,_) in enumerate(dl, start=1):
            images=[x.to(device) for x in images]; targets=[{k:v.to(device) for k,v in t.items()} for t in targets]
            losses=model(images,targets); loss=sum(losses.values()); opt.zero_grad(); loss.backward(); opt.step()
            num_batches_trained += 1
            loss_total += float(loss.item())
            print(f"epoch={e+1} batch={batch_idx} loss={loss.item():.4f}")
            if a.smoke_batches > 0 and num_batches_trained >= a.smoke_batches:
                break
        print(f"epoch={e+1} num_batches_trained={num_batches_trained} loss_total={loss_total:.4f}")
    torch.save({'model':model.state_dict(),'args':vars(a)}, out/'last.pt'); print(f"saved {out/'last.pt'}")
if __name__=='__main__': main()

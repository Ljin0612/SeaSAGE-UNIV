#!/usr/bin/env python
from __future__ import annotations
import argparse, sys
from pathlib import Path
from PIL import Image, ImageDraw
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.rgb_dataset import SeaShipsRGBDataset
from models.detectors.seasage_frcnn import SeaSAGEFasterRCNN
from scripts.eval_seasage_detector import select_state_dict


def parse_args():
    p=argparse.ArgumentParser(description='Visualize SeaSAGE Faster R-CNN predictions with GT boxes.')
    p.add_argument('--weights', required=True); p.add_argument('--data', required=True); p.add_argument('--split', default='test', choices=['train','val','test'])
    p.add_argument('--imgsz', type=int, default=640); p.add_argument('--device', default='cpu'); p.add_argument('--univ-weights', default=None)
    p.add_argument('--score-thres', type=float, default=0.25); p.add_argument('--max-images', type=int, default=20); p.add_argument('--save-dir', default='runs/predict/seasage_rgb')
    p.add_argument('--backbone-type', choices=['seasage_univ_single','seasage_univ_fpn'], default=None)
    return p.parse_args()

def names_map(names):
    return {int(k)+1:str(v) for k,v in names.items()} if isinstance(names,dict) else {i+1:str(v) for i,v in enumerate(names)}

def draw_box(draw, box, text, color, width=3, dashed=False):
    x1,y1,x2,y2=[float(v) for v in box]
    if dashed:
        step=8
        for x in range(int(x1), int(x2), step*2): draw.line([(x,y1),(min(x+step,x2),y1)], fill=color, width=width); draw.line([(x,y2),(min(x+step,x2),y2)], fill=color, width=width)
        for y in range(int(y1), int(y2), step*2): draw.line([(x1,y),(x1,min(y+step,y2))], fill=color, width=width); draw.line([(x2,y),(x2,min(y+step,y2))], fill=color, width=width)
    else:
        draw.rectangle([x1,y1,x2,y2], outline=color, width=width)
    draw.text((x1+2, max(0,y1-12)), text, fill=color)

def main():
    a=parse_args(); device=torch.device(f'cuda:{a.device}' if str(a.device).isdigit() and torch.cuda.is_available() else a.device)
    ckpt=torch.load(a.weights, map_location='cpu', weights_only=False); ckpt_args=ckpt.get('args',{}) if isinstance(ckpt,dict) else {}
    univ=a.univ_weights or ckpt_args.get('univ_weights'); backbone=a.backbone_type or ckpt_args.get('backbone_type','seasage_univ_single')
    if not univ: raise ValueError('--univ-weights or checkpoint args univ_weights is required')
    ds=SeaShipsRGBDataset(a.data,a.split,a.imgsz); names=names_map(ds.names); model=SeaSAGEFasterRCNN(len(names)+1, univ_weights=univ, imgsz=a.imgsz, backbone_type=backbone)
    model.load_state_dict(select_state_dict(ckpt), strict=True); model.to(device).eval(); out=Path(a.save_dir); out.mkdir(parents=True, exist_ok=True)
    n=min(len(ds), a.max_images if a.max_images>0 else len(ds))
    with torch.no_grad():
        for i in range(n):
            item=ds[i]; output=model([item['image'].to(device)])[0]; keep=output['scores'].detach().cpu()>=a.score_thres
            boxes=output['boxes'].detach().cpu()[keep]; labels=output['labels'].detach().cpu()[keep]; scores=output['scores'].detach().cpu()[keep]
            img=Image.open(item['path']).convert('RGB').resize((a.imgsz,a.imgsz)); draw=ImageDraw.Draw(img)
            for b,l in zip(item['boxes'], item['labels']): draw_box(draw,b,f"GT {names.get(int(l), int(l))}",'lime',width=3,dashed=False)
            for b,l,s in zip(boxes,labels,scores): draw_box(draw,b,f"P {names.get(int(l), int(l))} {float(s):.2f}",'red',width=2,dashed=True)
            avg=float(scores.mean().item()) if scores.numel() else 0.0
            draw.rectangle([0,0,360,22], fill='black'); draw.text((4,4), f"GT={len(item['labels'])} Pred={len(labels)} AvgScore={avg:.3f}", fill='white')
            save=out/(Path(item['path']).stem + '_pred.jpg'); img.save(save)
            print(f'{save}: gt={len(item["labels"])} pred={len(labels)} avg_score={avg:.6f}')
if __name__=='__main__': main()

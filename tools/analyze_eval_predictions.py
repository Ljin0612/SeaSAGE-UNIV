#!/usr/bin/env python
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.rgb_dataset import SeaShipsRGBDataset
from scripts.eval_seasage_detector import evaluate_class


def parse_args():
    p=argparse.ArgumentParser(description='Analyze SeaSAGE Faster R-CNN predictions against YOLO-format GT.')
    p.add_argument('--predictions', required=True, help='Path to predictions.json from eval_seasage_detector.py')
    p.add_argument('--data', required=True)
    p.add_argument('--split', choices=['train','val','test'], default='test')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--iou-thres', type=float, default=0.5)
    p.add_argument('--save-dir', default=None)
    return p.parse_args()


def class_name_map(names):
    return {int(k)+1: str(v) for k,v in names.items()} if isinstance(names, dict) else {i+1: str(v) for i,v in enumerate(names)}


def main():
    a=parse_args(); pred_path=Path(a.predictions)
    rows=json.loads(pred_path.read_text())
    ds=SeaShipsRGBDataset(a.data, a.split, a.imgsz)
    names=class_name_map(ds.names); class_ids=sorted(names)
    gt_by_path={}
    gts=[]
    for i in range(len(ds)):
        item=ds[i]; tgt={'boxes': item['boxes'], 'labels': item['labels']}
        gt_by_path[str(item['path'])]=tgt; gts.append(tgt)
    preds=[]; pred_by_path={r['image_path']: r for r in rows}
    for i in range(len(ds)):
        path=str(ds.images[i])
        r=pred_by_path.get(path, {'boxes': [], 'labels': [], 'scores': []})
        preds.append({'boxes': torch.tensor(r.get('boxes', []), dtype=torch.float32).reshape(-1,4), 'labels': torch.tensor(r.get('labels', []), dtype=torch.int64), 'scores': torch.tensor(r.get('scores', []), dtype=torch.float32)})
    gt_counts={cid:0 for cid in class_ids}; pred_counts={cid:0 for cid in class_ids}; score_sums=defaultdict(float)
    for t in gts:
        for cid in class_ids: gt_counts[cid]+=int((t['labels']==cid).sum().item())
    for p in preds:
        for cid in class_ids:
            m=p['labels']==cid; pred_counts[cid]+=int(m.sum().item()); score_sums[cid]+=float(p['scores'][m].sum().item()) if m.any() else 0.0
    metrics={cid:evaluate_class(preds,gts,cid,a.iou_thres) for cid in class_ids}
    out=Path(a.save_dir) if a.save_dir else pred_path.parent; out.mkdir(parents=True, exist_ok=True)
    with (out/'analyze_metrics.csv').open('w', newline='') as f:
        w=csv.writer(f); w.writerow(['class_id_frcnn','class_id_yolo','class_name','gt_count','prediction_count','average_score','tp','fp','fn','iou_threshold'])
        for cid in class_ids:
            tp=metrics[cid]['tp']; fp=metrics[cid]['fp']; fn=gt_counts[cid]-tp; avg=score_sums[cid]/pred_counts[cid] if pred_counts[cid] else 0.0
            w.writerow([cid,cid-1,names[cid],gt_counts[cid],pred_counts[cid],f'{avg:.6f}',tp,fp,fn,f'{a.iou_thres:.2f}'])
    lines=['# Prediction Analysis','',f'* Predictions: `{pred_path}`',f'* Data config: `{a.data}`',f'* Split: `{a.split}`',f'* IoU threshold: {a.iou_thres:.2f}','','## Label Mapping Checks','', '* YOLO labels are expected to be raw class IDs `0..5` in label txt files.', '* Faster R-CNN target labels are expected to be `1..6` because `0` is background.', '* Faster R-CNN prediction labels are expected to be `1..6`.', '* Evaluation/AP display maps labels back to class names via `label - 1` / config `names`.', '', '## Per-class Diagnostics','', '| class | YOLO id | FRCNN id | GT | Pred | Avg score | TP | FP | FN |', '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for cid in class_ids:
        tp=metrics[cid]['tp']; fp=metrics[cid]['fp']; fn=gt_counts[cid]-tp; avg=score_sums[cid]/pred_counts[cid] if pred_counts[cid] else 0.0
        lines.append(f'| {names[cid]} | {cid-1} | {cid} | {gt_counts[cid]} | {pred_counts[cid]} | {avg:.6f} | {tp} | {fp} | {fn} |')
    (out/'analyze_summary.md').write_text('\n'.join(lines)+'\n')
    print(f'Saved analysis to {out}')
if __name__=='__main__': main()

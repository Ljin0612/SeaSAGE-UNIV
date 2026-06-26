#!/usr/bin/env python
import argparse, csv
from pathlib import Path

def main():
    p=argparse.ArgumentParser(); p.add_argument('--weights',required=True); p.add_argument('--data',required=True); p.add_argument('--split',default='test'); p.add_argument('--imgsz',type=int,default=640); p.add_argument('--batch',type=int,default=1); p.add_argument('--device',default='cpu'); p.add_argument('--project',default='runs/eval'); p.add_argument('--name',default='exp'); a=p.parse_args()
    out=Path(a.project)/a.name; out.mkdir(parents=True,exist_ok=True)
    metrics={'Precision':0.0,'Recall':0.0,'mAP50':0.0,'mAP50:95':0.0}
    (out/'eval_summary.md').write_text('\n'.join([f"# Evaluation Summary", *[f"- {k}: {v}" for k,v in metrics.items()], "- per-class AP50: TODO integrate evaluator"]))
    with (out/'eval_metrics.csv').open('w',newline='') as f: w=csv.writer(f); w.writerow(metrics.keys()); w.writerow(metrics.values())
    print(metrics)
if __name__=='__main__': main()

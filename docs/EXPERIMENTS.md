# Experiments

This document records the current RGB-only UNIV-FRCNN experiment workflow for SeaSAGE-UNIV. It does not change the model structure or introduce new metrics.

## Backbones under test

### Single-scale UNIV adapter

Use the single-scale adapter with:

```bash
--backbone-type seasage_univ_single
```

### Multi-scale FPN UNIV adapter

Use the FPN adapter with:

```bash
--backbone-type seasage_univ_fpn
```

## Recorded single-scale e10 result

The existing single-scale epoch-10 result is:

| Metric | Value |
| --- | ---: |
| Precision | 0.029851 |
| Recall | 0.308589 |
| mAP50 | 0.116880 |
| mAP50:95 | 0.044544 |

Per-class AP50:

| Class | AP50 |
| --- | ---: |
| cargo ship | 0.050733 |
| container ship | 0.000000 |
| fishing boat | 0.326192 |
| passenger ship | 0.001230 |
| island | 0.000000 |
| floatage | 0.323126 |

## Single-scale e10 commands

Training:

```bash
PYTHONPATH=. python scripts/train_seasage_rgb_detector.py \
  --data configs/seaships24790.local.yaml \
  --epochs 10 \
  --batch 2 \
  --imgsz 640 \
  --device cuda:0 \
  --project runs/detect/seasage_rgb \
  --name univ_single_e10 \
  --univ-weights pretrained/checkpoint0400.pth \
  --head frcnn \
  --num-workers 4 \
  --backbone-type seasage_univ_single
```

Evaluation:

```bash
PYTHONPATH=. python scripts/eval_seasage_detector.py \
  --weights runs/detect/seasage_rgb/univ_single_e10/last.pt \
  --data configs/seaships24790.local.yaml \
  --split test \
  --imgsz 640 \
  --batch 2 \
  --device cuda:0 \
  --project runs/eval/seasage_rgb \
  --name univ_single_e10_test \
  --univ-weights pretrained/checkpoint0400.pth \
  --backbone-type seasage_univ_single
```

## FPN smoke test command

```bash
PYTHONPATH=. python scripts/train_seasage_rgb_detector.py \
  --data configs/seaships24790.local.yaml \
  --epochs 1 \
  --batch 1 \
  --imgsz 640 \
  --device cuda:0 \
  --project runs/detect/seasage_rgb \
  --name univ_fpn_smoke \
  --univ-weights pretrained/checkpoint0400.pth \
  --head frcnn \
  --num-workers 0 \
  --smoke-batches 2 \
  --backbone-type seasage_univ_fpn
```

## FPN e10 commands

Training:

```bash
PYTHONPATH=. python scripts/train_seasage_rgb_detector.py \
  --data configs/seaships24790.local.yaml \
  --epochs 10 \
  --batch 2 \
  --imgsz 640 \
  --device cuda:0 \
  --project runs/detect/seasage_rgb \
  --name univ_fpn_e10 \
  --univ-weights pretrained/checkpoint0400.pth \
  --head frcnn \
  --num-workers 4 \
  --backbone-type seasage_univ_fpn
```

Evaluation:

```bash
PYTHONPATH=. python scripts/eval_seasage_detector.py \
  --weights runs/detect/seasage_rgb/univ_fpn_e10/last.pt \
  --data configs/seaships24790.local.yaml \
  --split test \
  --imgsz 640 \
  --batch 2 \
  --device cuda:0 \
  --project runs/eval/seasage_rgb \
  --name univ_fpn_e10_test \
  --univ-weights pretrained/checkpoint0400.pth \
  --backbone-type seasage_univ_fpn
```

## Current conclusion

The current single-scale adapter exposes a 14×14 feature map, which is not detection-friendly enough for this task. The next experiment should verify whether `seasage_univ_fpn` improves detection performance by providing multi-scale features to Faster R-CNN.

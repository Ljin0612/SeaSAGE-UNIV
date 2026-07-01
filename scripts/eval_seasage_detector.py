#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.rgb_dataset import SeaShipsRGBDataset, detection_collate
from models.detectors.seasage_frcnn import SeaSAGEFasterRCNN
from scripts.device_utils import resolve_device


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate SeaSAGE-UNIV RGB-only Faster R-CNN with real detection metrics.")
    p.add_argument("--weights", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--device", default="cpu")
    p.add_argument("--project", default="runs/eval/seasage_rgb")
    p.add_argument("--name", default="exp")
    p.add_argument("--univ-weights", default=None)
    p.add_argument("--score-thres", type=float, default=0.001)
    p.add_argument("--iou-thres", type=float, default=0.50, help="IoU threshold used for Precision/Recall matching.")
    p.add_argument("--backbone-type", choices=["seasage_univ_single", "seasage_univ_fpn"], default=None)
    return p.parse_args()


def select_state_dict(ckpt: Any) -> dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        for key in ("model", "state_dict"):
            value = ckpt.get(key)
            if isinstance(value, dict):
                return value
        if any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt
    raise TypeError("Unsupported checkpoint format. Expected {'model': state_dict}, {'state_dict': state_dict}, or a raw state_dict.")


def iou_matrix(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    area1 = ((boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (boxes1[:, 3] - boxes1[:, 1]).clamp(min=0))[:, None]
    area2 = ((boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (boxes2[:, 3] - boxes2[:, 1]).clamp(min=0))[None, :]
    return inter / (area1 + area2 - inter).clamp(min=1e-9)


def ap_from_pr(recall: torch.Tensor, precision: torch.Tensor) -> float:
    mrec = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
    mpre = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
    for i in range(mpre.numel() - 2, -1, -1):
        mpre[i] = torch.maximum(mpre[i], mpre[i + 1])
    idx = torch.where(mrec[1:] != mrec[:-1])[0]
    return float(torch.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]).item())


def evaluate_class(preds, gts, class_id: int, iou_thr: float):
    gt_by_image: dict[int, dict[str, Any]] = {}
    n_gt = 0
    for img_id, target in enumerate(gts):
        mask = target["labels"] == class_id
        boxes = target["boxes"][mask].cpu()
        gt_by_image[img_id] = {"boxes": boxes, "matched": torch.zeros((boxes.shape[0],), dtype=torch.bool)}
        n_gt += boxes.shape[0]

    pred_rows = []
    for img_id, pred in enumerate(preds):
        mask = pred["labels"] == class_id
        for box, score in zip(pred["boxes"][mask].cpu(), pred["scores"][mask].cpu()):
            pred_rows.append((float(score), img_id, box))
    pred_rows.sort(key=lambda x: x[0], reverse=True)

    tp = torch.zeros((len(pred_rows),), dtype=torch.float32)
    fp = torch.zeros((len(pred_rows),), dtype=torch.float32)
    for i, (_, img_id, box) in enumerate(pred_rows):
        info = gt_by_image[img_id]
        gt_boxes = info["boxes"]
        if gt_boxes.numel() == 0:
            fp[i] = 1
            continue
        ious = iou_matrix(box.view(1, 4), gt_boxes).squeeze(0)
        best_iou, best_idx = torch.max(ious, dim=0)
        if best_iou >= iou_thr and not bool(info["matched"][best_idx]):
            tp[i] = 1
            info["matched"][best_idx] = True
        else:
            fp[i] = 1

    if len(pred_rows) == 0:
        return {"ap": 0.0, "tp": 0, "fp": 0, "n_gt": int(n_gt)}
    tp_cum = torch.cumsum(tp, dim=0)
    fp_cum = torch.cumsum(fp, dim=0)
    recall = tp_cum / max(float(n_gt), 1.0)
    precision = tp_cum / torch.clamp(tp_cum + fp_cum, min=1e-9)
    return {"ap": ap_from_pr(recall, precision) if n_gt > 0 else 0.0, "tp": int(tp.sum().item()), "fp": int(fp.sum().item()), "n_gt": int(n_gt)}


def compute_metrics(preds, gts, class_ids, pr_iou_thr: float):
    ap50_per_class = {cid: evaluate_class(preds, gts, cid, 0.50)["ap"] for cid in class_ids}
    map50 = sum(ap50_per_class.values()) / max(len(class_ids), 1)
    thresholds = [round(x / 100, 2) for x in range(50, 100, 5)]
    aps = []
    for thr in thresholds:
        for cid in class_ids:
            aps.append(evaluate_class(preds, gts, cid, thr)["ap"])
    pr = [evaluate_class(preds, gts, cid, pr_iou_thr) for cid in class_ids]
    tp = sum(x["tp"] for x in pr)
    fp = sum(x["fp"] for x in pr)
    n_gt = sum(x["n_gt"] for x in pr)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / n_gt if n_gt else 0.0
    return {"Precision": precision, "Recall": recall, "mAP50": map50, "mAP50:95": sum(aps) / max(len(aps), 1), "AP50_per_class": ap50_per_class}


def class_name_map(names):
    if isinstance(names, dict):
        return {int(k) + 1: str(v) for k, v in names.items()}
    return {i + 1: str(v) for i, v in enumerate(names)}


def main():
    a = parse_args()
    weights = Path(a.weights)
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")
    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    univ_weights = a.univ_weights or ckpt_args.get("univ_weights")
    if not univ_weights:
        raise ValueError("--univ-weights was not provided and checkpoint['args']['univ_weights'] is missing. Refusing to evaluate with the wrong backbone weights.")

    device = resolve_device(a.device)
    ds = SeaShipsRGBDataset(a.data, a.split, a.imgsz)
    dl = DataLoader(ds, batch_size=a.batch, shuffle=False, num_workers=0, collate_fn=detection_collate)
    names = class_name_map(ds.names)
    num_classes = len(names) + 1
    print(f"weights path: {weights}")
    print(f"data config: {a.data}")
    print(f"split: {a.split}")
    print(f"num_classes: {num_classes}")
    backbone_type = a.backbone_type or ckpt_args.get("backbone_type", "seasage_univ_single")
    print(f"model architecture: SeaSAGEFasterRCNN(num_classes=num_classes, univ_weights=..., imgsz=..., mode='rgb_only', backbone_type='{backbone_type}')")

    model = SeaSAGEFasterRCNN(num_classes=num_classes, univ_weights=univ_weights, imgsz=a.imgsz, mode="rgb_only", backbone_type=backbone_type)
    state = select_state_dict(ckpt)
    print(f"loaded checkpoint keys: {list(state.keys())[:20]}{' ...' if len(state) > 20 else ''}")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()

    preds, gts, pred_json = [], [], []
    with torch.no_grad():
        for images, targets, batch in dl:
            images_dev = [x.to(device) for x in images]
            outputs = model(images_dev)
            for output, target, meta in zip(outputs, targets, batch):
                keep = output["scores"].detach().cpu() >= a.score_thres
                pred = {"boxes": output["boxes"].detach().cpu()[keep], "labels": output["labels"].detach().cpu()[keep], "scores": output["scores"].detach().cpu()[keep]}
                tgt = {"boxes": target["boxes"].detach().cpu(), "labels": target["labels"].detach().cpu()}
                preds.append(pred); gts.append(tgt)
                pred_json.append({"image_path": meta["path"], "boxes": pred["boxes"].tolist(), "labels": pred["labels"].tolist(), "scores": pred["scores"].tolist()})

    metrics = compute_metrics(preds, gts, sorted(names), a.iou_thres)
    out = Path(a.project) / a.name
    out.mkdir(parents=True, exist_ok=True)
    with (out / "eval_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f); writer.writerow(["Metric", "Value"])
        for k in ("Precision", "Recall", "mAP50", "mAP50:95"):
            writer.writerow([k, f"{metrics[k]:.6f}"])
        for cid, ap in metrics["AP50_per_class"].items():
            writer.writerow([f"AP50/{names[cid]}", f"{ap:.6f}"])
    with (out / "predictions.json").open("w") as f:
        json.dump(pred_json, f)
    lines = ["# Evaluation Summary", "", "* Model: SeaSAGE-UNIV RGB-only UNIV-FRCNN", f"* Dataset: {a.data}", f"* Split: {a.split}", f"* Weights path: {weights}", f"* UNIV weights path: {univ_weights}", f"* imgsz: {a.imgsz}", f"* batch: {a.batch}", f"* Precision: {metrics['Precision']:.6f}", f"* Recall: {metrics['Recall']:.6f}", f"* mAP50: {metrics['mAP50']:.6f}", f"* mAP50:95: {metrics['mAP50:95']:.6f}", "", "## AP50 per class"]
    lines += [f"* {names[cid]}: {ap:.6f}" for cid, ap in metrics["AP50_per_class"].items()]
    (out / "eval_summary.md").write_text("\n".join(lines) + "\n")
    print(f"Saved evaluation to {out}")
    for k in ("Precision", "Recall", "mAP50", "mAP50:95"):
        print(f"{k}: {metrics[k]:.6f}")


if __name__ == "__main__":
    main()

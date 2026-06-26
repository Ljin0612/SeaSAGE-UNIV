#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets.rgbir_paired_dataset import RGBIRPairedDataset
from models.seasage_univ import ABLATION_MODES


def main():
    p = argparse.ArgumentParser(description="Train SeaSAGE semantic-aware RGB-IR detector.")
    p.add_argument("--data", required=True, help="Path to local RGB-IR dataset yaml; no dataset paths are hard-coded.")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="cpu")
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default="exp")
    p.add_argument("--univ-weights")
    p.add_argument("--fusion-mode", choices=["add", "concat", "attention"], default="add")
    p.add_argument("--pccl-weight", type=float, default=1.0, help="lambda_pccl in loss = loss_det + lambda_pccl * loss_sem_pccl")
    p.add_argument("--num-semantic-groups", type=int, default=8)
    p.add_argument("--ablation-mode", choices=sorted(ABLATION_MODES), default="rgb_ir_semantic_pccl")
    p.add_argument("--num-workers", type=int, default=0)
    args = p.parse_args()

    try:
        RGBIRPairedDataset(args.data, "train", args.imgsz)
    except (FileNotFoundError, ValueError, NotImplementedError) as exc:
        raise SystemExit(f"RGB-IR dataset unavailable: {exc}")

    # The real detector training loop is intentionally not faked.  When the paired
    # dataset parser is completed, it must log these fields from genuine training/eval.
    run_dir = Path(args.project) / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "loss_det", "loss_sem_pccl", "loss_total", "mAP50", "mAP50:95"])
        writer.writeheader()
    raise SystemExit(
        "RGB-IR training loop is not implemented until real paired data parsing/detection head integration is available; "
        f"created metrics schema at {metrics_path}. Evaluation must use detection outputs only."
    )


if __name__ == "__main__":
    main()

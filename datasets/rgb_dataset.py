from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import yaml
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as TF

class SeaShipsRGBDataset(Dataset):
    """SeaShips24790 YOLO-format RGB dataset."""
    def __init__(self, data: str, split: str = "train", imgsz: int = 640):
        self.cfg_path = Path(data)
        if not self.cfg_path.exists():
            raise FileNotFoundError(f"Dataset config not found: {data}. Create configs/seaships24790.local.yaml from the template.")
        self.cfg: Dict = yaml.safe_load(self.cfg_path.read_text())
        self.root = Path(self.cfg.get("path", "."))
        self.split = split
        self.imgsz = imgsz
        split_entry = self.cfg.get(split)
        if split_entry is None:
            raise KeyError(f"Split '{split}' missing in {data}")
        self.images = self._collect_images(split_entry)
        self.names = self.cfg.get("names", {})

    def _collect_images(self, entry) -> List[Path]:
        p = Path(entry)
        if not p.is_absolute():
            p = self.root / p
        if p.is_file() and p.suffix == ".txt":
            return [Path(x.strip()) if Path(x.strip()).is_absolute() else self.root / x.strip() for x in p.read_text().splitlines() if x.strip()]
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted([x for x in p.rglob("*") if x.suffix.lower() in exts])

    def __len__(self): return len(self.images)

    def _label_path(self, img: Path) -> Path:
        parts = list(img.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            return Path(*parts).with_suffix(".txt")
        return img.with_suffix(".txt")

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        image = Image.open(img_path).convert("RGB")
        ow, oh = image.size
        image = TF.resize(image, [self.imgsz, self.imgsz])
        tensor = TF.to_tensor(image)
        boxes, labels = [], []
        lp = self._label_path(img_path)
        if lp.exists():
            for line in lp.read_text().splitlines():
                c, x, y, w, h = map(float, line.split()[:5])
                x1 = (x - w / 2) * self.imgsz; y1 = (y - h / 2) * self.imgsz
                x2 = (x + w / 2) * self.imgsz; y2 = (y + h / 2) * self.imgsz
                boxes.append([x1, y1, x2, y2]); labels.append(int(c) + 1)  # Faster R-CNN reserves 0 for background
        return {"image": tensor, "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1,4), "labels": torch.tensor(labels, dtype=torch.int64), "path": str(img_path), "orig_size": (oh, ow)}

def detection_collate(batch):
    images = [b["image"] for b in batch]
    targets = [{"boxes": b["boxes"], "labels": b["labels"]} for b in batch]
    return images, targets, batch

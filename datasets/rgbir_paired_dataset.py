from __future__ import annotations
from pathlib import Path
try:
    from torch.utils.data import Dataset
except ModuleNotFoundError:
    class Dataset:  # type: ignore[no-redef]
        pass

class RGBIRPairedDataset(Dataset):
    """Interface for paired RGB-IR images and YOLO detection labels.

    Expected local config keys: rgb_root, ir_root, label_root, and split files
    (train/val/test). Real data are not fabricated; missing configs raise a
    clear error for first-stage engineering.
    """
    def __init__(self, data: str, split: str = "train", imgsz: int = 640):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError("RGB-IR paired dataset is not configured. Please create configs/seasage_rgbir.local.yaml.")
        import yaml
        self.cfg = yaml.safe_load(path.read_text())
        required = ["rgb_root", "ir_root", "label_root", split]
        missing = [k for k in required if not self.cfg.get(k)]
        if missing:
            raise ValueError(f"RGB-IR paired dataset config is incomplete: missing {missing}")
        self.items = []
        # TODO: parse split file and return synchronized rgb, ir, boxes, labels.
        raise NotImplementedError("RGB-IR paired dataset parsing is reserved until real paired data paths are configured.")

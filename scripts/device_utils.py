from __future__ import annotations

import torch


def resolve_device(device_arg: str) -> torch.device:
    """Resolve CLI device arguments, falling back numeric GPU IDs to CPU when CUDA is unavailable."""
    device_str = str(device_arg)
    if device_str.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{device_str}")
        return torch.device("cpu")
    return torch.device(device_str)

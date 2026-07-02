# Installation

This project intentionally keeps PyTorch out of `requirements.txt` because the correct wheel depends on the CUDA version available on each server.

## 1. Create a conda environment

```bash
conda create -n seasage_univ python=3.10 -y
conda activate seasage_univ
```

## 2. Install PyTorch

For a CUDA 12.1 server, install PyTorch with:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If your server uses a different CUDA version, choose the matching command from the official PyTorch installation selector instead of hard-coding the CUDA 12.1 wheel.

## 3. Install project dependencies

```bash
pip install -r requirements.txt
```

## 4. Check CUDA availability

```bash
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
PY
```

## 5. Prepare UNIV weights

Place the UNIV pretrained checkpoint at:

```text
pretrained/checkpoint0400.pth
```

The `pretrained/` directory is local-only and should not be committed.

## 6. Prepare local data configuration

Create a local SeaShips config at:

```text
configs/seaships24790.local.yaml
```

Use the tracked template configs as references, and fill in paths for your local machine or server. Do not write absolute local paths into tracked documentation or shared config files.

## 7. Keep local artifacts out of GitHub

The following paths are local artifacts and should not be committed:

- `pretrained/`
- `runs/`
- `results/`
- `configs/*.local.yaml`

These entries are already covered by `.gitignore`.

## 8. Run tests

```bash
PYTHONPATH=. python -m pytest tests -q
```

## 9. Run py_compile checks

```bash
python -m py_compile \
  scripts/train_seasage_rgb_detector.py \
  scripts/eval_seasage_detector.py \
  scripts/predict_seasage_detector.py
```

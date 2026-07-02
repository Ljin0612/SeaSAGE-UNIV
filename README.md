# SeaSAGE-UNIV

Semantic-Aware Guidance and Alignment for RGB-Infrared Maritime Detection.

This repository contains the first-stage engineering scaffold for a modular RGB/IR maritime detector. Copy template configs in `configs/*.yaml` to `configs/*.local.yaml` and fill local dataset paths. Local configs, pretrained weights, and experiment outputs are intentionally ignored by git.

## Project status

SeaSAGE-UNIV currently supports RGB-only UNIV-FRCNN with single-scale and multi-scale FPN adapters.

## Installation

See [docs/INSTALL.md](docs/INSTALL.md) for environment setup, PyTorch installation guidance, CUDA checks, local artifact preparation, and validation commands.

## Experiments

See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the current single-scale and FPN experiment workflow, recorded single-scale e10 metrics, and reproduction commands.

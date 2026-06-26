# SeaSAGE-UNIV Experiment Plan

## Baselines

1. **RGB-only baseline**: UNIV/ViT feature learning with a stable detection head on SeaShips24790 RGB images.
2. **IR-only baseline**: train the same pipeline on infrared images when IR labels are available.
3. **Early fusion**: concatenate RGB and IR inputs before feature extraction as a simple reference.
4. **Feature fusion**: compare add, concat, and attention fusion at patch-feature level.
5. **UNIV-Faster R-CNN**: reproduce the stable detector setup and use it as the main engineering baseline.
6. **UNIV-YOLOv8-style**: evaluate the experimental anchor-free decoupled head after the stable pipeline is complete.
7. **SeaSAGE-UNIV fusion**: train RGB/IR encoders with semantic assignment, PCCL, fusion, and detector supervision.

## Ablation Studies

- Fusion mode: add vs concat vs attention.
- PCCL weight and temperature.
- Number of semantic pseudo-label clusters.
- RGB-only vs IR-only vs RGB-IR.
- P3/P4/P5 vs optional P2 small-object branch.
- Frozen vs fine-tuned UNIV backbone.

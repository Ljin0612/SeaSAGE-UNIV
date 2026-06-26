# SeaSAGE-UNIV Method Design

SeaSAGE-UNIV (Semantic-Aware Guidance and Alignment for RGB-Infrared Maritime Detection) targets RGB-infrared maritime object detection with a unified representation inspired by UNIV.

## Overall Architecture

The framework contains RGB/IR encoders, semantic pseudo-label assignment, semantic-aware patch contrastive alignment, cross-modal fusion, a multiscale adapter, and a detection head.

## RGB Encoder / IR Encoder

Both encoders wrap a UNIV/ViT-style patch backbone. The first engineering stage provides a lightweight patch encoder and a checkpoint-loading interface for later integration with the full UNIV implementation.

## Semantic Label Assignment

`SemanticLabelAssignment` assigns patch-level pseudo semantic labels. The initial implementation uses simple k-means over normalized patch embeddings while preserving interfaces for attention- or saliency-based assignment.

## Semantic-aware PCCL

`SemanticAwarePCCL` aligns RGB and IR patches. Patches with identical pseudo semantic labels are pulled together, while different-label patches are separated through temperature-scaled contrastive logits.

## Unified Cross-modal Representation

`CrossModalFusion` supports add, concat-projection, and attention-gated fusion to produce a unified RGB-IR representation for downstream detection.

## Detection Head

The stable first-stage head is Faster R-CNN. A YOLOv8-style anchor-free decoupled head interface is reserved as experimental so it does not affect stable Faster R-CNN experiments.

import math

import pytest
import torch

from models.modules.semantic_aware_pccl import SemanticAwarePCCL


def test_background_anchor_contribution_is_downweighted_with_valid_count_normalization():
    rgb = torch.zeros(1, 2, 4)
    ir = torch.zeros(1, 2, 4)
    labels = torch.tensor([[0, 1]])
    foreground = torch.tensor([[True, False]])

    loss_fn = SemanticAwarePCCL(temperature=1.0, background_weight=0.1)

    foreground_only = loss_fn(
        rgb,
        ir,
        labels,
        semantic_confidence=torch.tensor([[1.0, 0.0]]),
        foreground_mask=foreground,
    )
    background_only = loss_fn(
        rgb,
        ir,
        labels,
        semantic_confidence=torch.tensor([[0.0, 1.0]]),
        foreground_mask=foreground,
    )

    assert torch.isclose(foreground_only, torch.tensor(math.log(2) / 2), atol=1e-6)
    assert torch.isclose(background_only, torch.tensor(0.1 * math.log(2) / 2), atol=1e-6)
    assert background_only < foreground_only


def test_weight_sum_normalization_keeps_scale_for_uniform_anchor_losses():
    rgb = torch.zeros(1, 2, 4)
    ir = torch.zeros(1, 2, 4)
    labels = torch.tensor([[0, 1]])
    foreground = torch.tensor([[True, False]])

    loss_fn = SemanticAwarePCCL(temperature=1.0, background_weight=0.1, normalization="weight_sum")

    loss = loss_fn(rgb, ir, labels, foreground_mask=foreground)

    assert torch.isclose(loss, torch.tensor(math.log(2)), atol=1e-6)


def test_invalid_normalization_raises_value_error():
    with pytest.raises(ValueError, match="normalization"):
        SemanticAwarePCCL(normalization="bad")


class _FakeConvMAE(torch.nn.Module):
    def forward(self, x, mask_ratio=0.0, return_last_attention=False):
        patches = torch.zeros(x.shape[0], 14 * 14, 768, device=x.device, dtype=x.dtype)
        return patches, None


def _patch_fake_univ(monkeypatch):
    import models.backbones.univ_backbone as univ_backbone

    monkeypatch.setattr(univ_backbone, "convmae_convvit_base_patch16", lambda: _FakeConvMAE())
    monkeypatch.setattr(univ_backbone, "_UNIV_IMPORT_ERROR", None)


def test_univ_backbone_reports_real_convmae_dimension(monkeypatch):
    _patch_fake_univ(monkeypatch)
    from models.backbones.univ_backbone import UNIVBackbone

    backbone = UNIVBackbone()
    assert backbone.out_channels == 768
    assert backbone.feature_dim == 768
    assert backbone.embed_dim == 768

    features = backbone(torch.zeros(1, 3, 224, 224))
    assert features.shape[-1] == 768

    feature_map = backbone(torch.zeros(1, 3, 224, 224), output_format="bchw")
    assert feature_map.shape[1] == backbone.out_channels


def test_seasage_univ_projects_encoder_features_to_model_dim(monkeypatch):
    _patch_fake_univ(monkeypatch)
    from models.seasage_univ import SeaSAGEUNIV

    model = SeaSAGEUNIV(mode="rgb_ir_no_pccl", model_dim=384)
    assert model.encoder_dim == 768
    assert model.model_dim == 384
    assert model.projection_enabled is True
    assert model.rgb_proj.in_channels == model.rgb_encoder.out_channels
    assert model.rgb_proj.out_channels == 384

    projected = model._project_patches(torch.zeros(1, 14 * 14, 768), model.rgb_proj)
    assert projected.shape[-1] == 384

    output = model(
        torch.zeros(1, 3, 224, 224),
        torch.zeros(1, 3, 224, 224),
        imgsz=224,
        return_loss=False,
    )
    assert set(output["features"]) == {"p3", "p4", "p5"}


def test_seasage_frcnn_projection_uses_univ_out_channels(monkeypatch):
    _patch_fake_univ(monkeypatch)
    from models.detectors.seasage_frcnn import SeaSAGEFRCNNBackbone

    backbone = SeaSAGEFRCNNBackbone(out_channels=256)
    assert backbone.univ.out_channels == 768
    assert backbone.proj.in_channels == backbone.univ.out_channels
    assert backbone.proj.out_channels == 256

    output = backbone(torch.zeros(1, 3, 224, 224))
    assert output["0"].shape[1] == 256

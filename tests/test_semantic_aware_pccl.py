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

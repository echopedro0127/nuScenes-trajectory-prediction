import inspect

import pytest
import torch

from trajectory_prediction.online.corrector import ResidualCorrector
from trajectory_prediction.online.interaction_corrector import (
    InteractionCorrector,
    matched_width,
    predict_interaction,
)


def inputs():
    return (torch.randn(2, 12, 8), torch.arange(12)[None].expand(2, -1) >= 3,
            torch.randn(2, 2, 3, 4), torch.ones(2, 2, 3, dtype=torch.bool),
            torch.randn(2, 2, 5, 8), torch.ones(2, 2, 5, dtype=torch.bool))


@pytest.mark.parametrize("mode", ["map", "neighbors", "fusion"])
def test_missing_context_is_exact_fallback_and_past_mask_is_zero(mode):
    args = list(inputs())
    model = InteractionCorrector(ResidualCorrector(hidden_dim=8), mode, 8)
    args[3].zero_()
    args[5].zero_()
    torch.nn.init.constant_(model.head[-1].weight, .1)
    actual = model(*args)
    torch.testing.assert_close(actual, model.fallback(*args[:2]), atol=0, rtol=0)
    assert torch.count_nonzero(actual[:, :3]) == 0
    assert all(not p.requires_grad for p in model.fallback.parameters())


@pytest.mark.parametrize("mode", ["map", "neighbors", "fusion"])
def test_padding_invariance_with_nonzero_map_neighbor_contribution_and_gradients(mode):
    args = list(inputs())
    model = InteractionCorrector(ResidualCorrector(hidden_dim=8), mode, 8)
    torch.nn.init.constant_(model.head[-1].weight, .1)
    args[3][:, 1] = False
    args[5][:, 1] = False
    first = model(*args)
    changed = [arg.clone() for arg in args]
    changed[2][~changed[3]] = 99999
    changed[4][~changed[5]] = -99999
    torch.testing.assert_close(first, model(*changed), rtol=0, atol=0)
    first.sum().backward()
    for encoder in (model.map_encoder, model.neighbor_encoder):
        if encoder is not None:
            assert any(p.grad is not None and torch.count_nonzero(p.grad) for p in encoder.parameters())
    assert all(p.grad is None for p in model.fallback.parameters())


def test_fusion_keeps_other_modality_when_one_missing():
    args = list(inputs())
    model = InteractionCorrector(ResidualCorrector(hidden_dim=8), "fusion", 8)
    torch.nn.init.constant_(model.head[-1].weight, .1)
    for absent in (3, 5):
        changed = [arg.clone() for arg in args]
        changed[absent].zero_()
        assert not torch.equal(model(*changed), model.fallback(*args[:2]))


def test_parameter_matched_control_and_prediction_boundary():
    for mode in ("map", "neighbors", "fusion"):
        model = InteractionCorrector(ResidualCorrector(), mode)
        count = sum(p.numel() for p in model.parameters())
        width = matched_width(count)
        actual = sum(p.numel() for p in ResidualCorrector(hidden_dim=width).parameters())
        assert abs(actual - count) / count < .02
    assert set(inspect.signature(predict_interaction).parameters) == {
        "model", "observation", "base", "map_view", "neighbor_view", "device"}

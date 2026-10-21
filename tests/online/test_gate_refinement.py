import numpy as np
import pytest
import torch

from trajectory_prediction.online.gate_refinement import (
    RefinementDataset,
    paired_revision,
    shrink_weights,
)
from trajectory_prediction.online.quality_gate import blend_forecasts


def arrays():
    return {
        "instance_token": np.array(["i0", "i1"]),
        "sample_token": np.array(["s0", "s1"]),
        "scene_name": np.array(["a", "b"]),
        "base": np.zeros((2, 12, 2), np.float32),
        "target": np.ones((2, 12, 2), np.float32),
        "cv": np.zeros((2, 11, 12, 2), np.float32),
        "map": np.full((6, 2, 11, 12, 2), 2, np.float32),
        "original": np.ones((2, 11, 12, 2), np.float32),
        "quality": np.zeros((6, 2, 11, 8), np.float32),
        "valid": np.ones((6, 2, 11), bool),
        "age_s": np.zeros((6, 2, 11), np.float32),
    }


def test_original_adapter_rebuilds_disagreement_without_mutating_cache():
    a = arrays()
    data = RefinementDataset(a, candidate="original")
    before = data.fingerprint()
    inputs = data.inputs(np.array([0, 10, 11, 22]), torch.device("cpu"))
    assert set(inputs) == {"cv", "map", "quality", "valid", "mask", "age_s"}
    torch.testing.assert_close(inputs["map"], torch.ones((4, 12, 2)))
    torch.testing.assert_close(inputs["quality"][:, 7], torch.full((4,), 2**0.5 / 20))
    assert data.fingerprint() == before and (a["quality"] == 0).all() and (a["map"] == 2).all()
    np.testing.assert_array_equal(
        data.previous_indices(np.array([0, 1, 10, 11, 12, 22])), [0, 0, 9, 11, 11, 22]
    )
    with pytest.raises(ValueError):
        RefinementDataset(a, candidate="future")


def test_short_cap_identity_endpoint_and_exact_hard_fallback():
    g = torch.tensor([0.4, 0.4, 0.4], requires_grad=True)
    h = torch.tensor([0.5, 1.0, 2.0])
    torch.testing.assert_close(shrink_weights(g, h, 0), g)
    w = shrink_weights(g, h, 1.0)
    torch.testing.assert_close(w, torch.tensor([0.0, 0.2, 0.4]))
    cv = torch.randn(3, 12, 2)
    candidate = torch.randn(3, 12, 2)
    torch.testing.assert_close(blend_forecasts(cv, candidate, w)[0], cv[0], atol=0, rtol=0)
    with pytest.raises(ValueError):
        shrink_weights(g, h, -1)


def test_revision_uses_same_absolute_future_and_gradients_both_branches():
    now = torch.zeros(3, 12, 2, requires_grad=True)
    previous = torch.ones(3, 12, 2, requires_grad=True)
    mask = torch.arange(12)[None] >= torch.tensor([1, 2, 11])[:, None]
    has_previous = torch.tensor([False, True, True])
    loss = paired_revision(now, previous, mask, has_previous)
    torch.testing.assert_close(loss, torch.tensor(2**0.5))
    loss.backward()
    assert now.grad[0].count_nonzero() == 0 and previous.grad[0].count_nonzero() == 0
    assert now.grad[1, :2].count_nonzero() == 0 and previous.grad[1, :2].count_nonzero() == 0
    assert now.grad.count_nonzero() > 0 and previous.grad.count_nonzero() > 0
    empty = paired_revision(now, previous, mask, torch.zeros(3, dtype=torch.bool))
    assert empty == 0 and empty.requires_grad


def test_invalid_packet_stays_exact_cv_with_nonzero_time_cap():
    from trajectory_prediction.online.quality_gate import QualityGate

    a = arrays()
    a["valid"][4] = False
    a["valid"][5, :, 5:] = False
    dataset = RefinementDataset(a)
    selected = np.array([4 * 22, 5 * 22 + 5, 5 * 22 + 10])
    inputs = dataset.inputs(selected, torch.device("cpu"))
    g = QualityGate()(inputs["quality"], inputs["valid"])
    weight = shrink_weights(g, torch.full((3,), 3.0), 1.0)
    prediction = blend_forecasts(inputs["cv"], inputs["map"], weight)
    torch.testing.assert_close(weight, torch.zeros(3), atol=0, rtol=0)
    torch.testing.assert_close(prediction, inputs["cv"], atol=0, rtol=0)

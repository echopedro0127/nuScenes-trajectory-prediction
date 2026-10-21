"""Causal prefixes, observed-error feedback and initialized horizon gate contracts."""
import copy

import numpy as np
import pytest
import torch

from trajectory_prediction.online.horizon_features import HorizonDataset
from trajectory_prediction.online.horizon_gate import (
    VARIANTS,
    HorizonGate,
    blend_horizon,
    measure_horizon,
)
from trajectory_prediction.online.quality_gate import CONDITIONS, QualityGate, packet_schedule


def example_arrays(n=3):
    rng = np.random.default_rng(10)
    history = rng.normal(size=(n, 5, 2)).astype(np.float32)
    position = rng.normal(size=(n, 11, 2)).cumsum(1).astype(np.float32)
    velocity = (position-np.concatenate([history[:, -1:], position[:, :-1]], 1))/.5
    previous_velocity = np.concatenate([(history[:, -1:]-history[:, -2:-1])/.5, velocity[:, :-1]], 1)
    lead = (np.arange(12)[None]-np.arange(11)[:, None])*.5
    cv = (position[:, :, None]+velocity[:, :, None]*lead[None, :, :, None]).astype(np.float32)
    maps = np.broadcast_to(cv[None]+.2, (6, *cv.shape)).copy()
    valid = np.stack([np.broadcast_to(packet_schedule(c)["age_valid"], (n, 11)) for c in CONDITIONS]).copy()
    age = np.stack([np.broadcast_to(packet_schedule(c)["age_s"], (n, 11)) for c in CONDITIONS]).copy()
    return {"history": history, "position": position, "velocity": velocity,
        "previous_velocity": previous_velocity, "cv": cv, "map": maps,
        "original": maps[0].copy(), "quality": rng.random((6, n, 11, 8), dtype=np.float32),
        "valid": valid, "age_s": age, "target": rng.normal(size=(n, 12, 2)).astype(np.float32),
        "base": rng.normal(size=(n, 12, 2)).astype(np.float32),
        "scene_name": np.array([f"scene{i}" for i in range(n)]),
        "instance_token": np.array([f"i{i}" for i in range(n)]),
        "sample_token": np.array([f"s{i}" for i in range(n)])}


def test_future_observations_and_labels_cannot_change_prefix_features():
    arrays = example_arrays()
    original = copy.deepcopy(arrays)
    data = HorizonDataset(arrays)
    indices = np.array([c*33+i*11+k for c in range(6) for i in range(3) for k in range(5)])
    before = data.inputs(indices, torch.device("cpu"))["features"]
    changed = copy.deepcopy(arrays)
    changed["position"][:, 5:] += 900
    changed["velocity"][:, 5:] *= -20
    changed["previous_velocity"][:, 5:] *= 30
    changed["map"][:, :, 5:] += 100
    changed["cv"][:, 5:] -= 300
    changed["target"][:] = 1e7
    after = HorizonDataset(changed).inputs(indices, torch.device("cpu"))["features"]
    torch.testing.assert_close(before, after, rtol=0, atol=0)
    for name in arrays:
        np.testing.assert_array_equal(arrays[name], original[name])


def test_feedback_uses_previous_forecast_at_the_newly_arrived_absolute_time():
    arrays = example_arrays()
    arrays["cv"][0, 0, 1] = arrays["position"][0, 1]+[3, 4]
    arrays["map"][0, 0, 0, 1] = arrays["position"][0, 1]+[0, 2]
    arrays["cv"][0, 1, 2] = arrays["position"][0, 2]+[0, 10]
    data = HorizonDataset(arrays)
    np.testing.assert_array_equal(data.feedback[0, 0, 0], 0)
    np.testing.assert_allclose(data.feedback[0, 0, 1], [np.log(6), np.log(3), 1, 1])
    assert data.feedback[0, 0, 2, 0] == pytest.approx(np.log(7))
    invalid = copy.deepcopy(arrays)
    invalid["valid"][0, 0, 1] = False
    values = HorizonDataset(invalid).feedback
    np.testing.assert_array_equal(values[0, 0, 1:3, [1, 3]], 0)
    assert values[0, 0, 3, 3] == 1


def test_feature_state_is_invariant_to_sample_order_and_stationary_input_is_finite():
    arrays = example_arrays()
    data = HorizonDataset(arrays)
    order = np.array([2, 0, 1])
    swapped = {name: value[:, order] if name in ("map", "valid", "quality", "age_s") else value[order] for name, value in arrays.items()}
    actual = HorizonDataset(swapped)
    np.testing.assert_array_equal(actual.feedback, data.feedback[:, order])
    np.testing.assert_array_equal(actual.direction, data.direction[:, order])
    for name in ("velocity", "previous_velocity", "history"):
        arrays[name][:] = 0
    still = HorizonDataset(arrays).inputs(np.arange(33), torch.device("cpu"))["features"]
    assert torch.isfinite(still).all()
    torch.testing.assert_close(still[..., 8:12], torch.zeros_like(still[..., 8:12]))


def test_all_eight_variants_have_337_parameters_and_match_p6_at_initialization():
    data = HorizonDataset(example_arrays())
    inputs = data.inputs(np.arange(data.size), torch.device("cpu"))
    parent = QualityGate().eval()
    with torch.no_grad():
        parent.network[-1].weight.fill_(.03)
        expected = parent(inputs["quality"], inputs["valid"])
    assert len(VARIANTS) == 8
    for variant in VARIANTS:
        model = HorizonGate(variant, parent.state_dict()).eval()
        assert sum(p.numel() for p in model.parameters()) == 337
        weights = model(inputs["features"], inputs["valid"], inputs["mask"])
        torch.testing.assert_close(weights[inputs["mask"]], expected[:, None].expand_as(weights)[inputs["mask"]], atol=1e-7, rtol=1e-6)
        assert not weights[~inputs["mask"]].any()
        forecast = blend_horizon(inputs["cv"], inputs["map"], weights)
        assert torch.equal(forecast[~inputs["valid"]], inputs["cv"][~inputs["valid"]])
    with pytest.raises(ValueError):
        HorizonGate("RDH", parent.state_dict())


def test_disabled_horizon_is_scalar_and_enabled_horizon_can_vary_by_lead_time():
    data = HorizonDataset(example_arrays())
    inputs = data.inputs(np.array([0, 1, 2]), torch.device("cpu"))
    for variant in ("B", "D", "R", "DR"):
        model = HorizonGate(variant, QualityGate().state_dict())
        with torch.no_grad():
            model.network[0].weight[:, 8:].fill_(.2)
            model.network[-1].weight.fill_(.2)
        weight = model(inputs["features"], inputs["valid"], inputs["mask"])
        for row in range(3):
            values = weight[row, inputs["mask"][row]]
            torch.testing.assert_close(values, values[0].expand_as(values))
    model = HorizonGate("H", QualityGate().state_dict())
    with torch.no_grad():
        model.network[0].weight[:, 16] = .2
        model.network[-1].weight.fill_(.2)
    weight = model(inputs["features"], inputs["valid"], inputs["mask"])
    assert weight[0, 1] != weight[0, -1]


def test_complete_sequence_metrics_preserve_exact_missing_cv():
    data = HorizonDataset(example_arrays())
    model = HorizonGate("DRH", QualityGate().state_dict())
    metrics, weights = measure_horizon(data, model, device=torch.device("cpu"), batch_size=17)
    assert metrics.shape == (6, 3, 11, 3) and weights.shape == (6, 3, 11, 12)
    assert np.isfinite(metrics).all() and not weights[4].any()
    from trajectory_prediction.online.batched_evaluation import forecast_metrics
    np.testing.assert_array_equal(metrics[4], forecast_metrics(data.arrays["cv"], data.arrays["target"]))

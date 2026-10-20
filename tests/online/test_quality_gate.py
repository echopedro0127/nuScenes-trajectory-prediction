import numpy as np
import pytest
import torch

from trajectory_prediction.online.quality_gate import (
    QualityGate,
    blend_forecasts,
    fixed_weights,
    packet_schedule,
    quality_features,
)


def test_packet_availability_drop_and_stale_hard_expiry():
    schedule = packet_schedule("delay_1000ms")
    np.testing.assert_array_equal(schedule["packet_step"][:3], [0, 0, 1])
    np.testing.assert_array_equal(schedule["available_step"][:3], [0, 0, 3])
    dropped = packet_schedule("dropout")
    np.testing.assert_array_equal(dropped["packet_step"][3:6], [3, 3, 3])
    assert dropped["packet_step"][6] == 7
    stale = packet_schedule("stale")
    assert stale["age_s"][4] == 2 and stale["age_valid"][4]
    assert stale["age_s"][5] == 2.5 and not stale["age_valid"][5]
    assert not packet_schedule("missing")["age_valid"].any()
    for name in ("clean", "dropout", "stale", "delay_500ms"):
        prefix = packet_schedule(name, count=4)
        for field, value in prefix.items():
            np.testing.assert_array_equal(value, packet_schedule(name)[field][:4])
    with pytest.raises(ValueError):
        packet_schedule("future")


def test_zero_gate_is_exact_cv_and_predictions_do_not_take_labels():
    torch.set_num_threads(1)
    gate = QualityGate(16)
    quality = torch.ones(4, 8)
    valid = torch.tensor([True, False, True, False])
    weights = gate(quality, valid)
    assert weights.shape == (4,) and (weights >= 0).all() and (weights <= 1).all()
    assert torch.equal(weights[~valid], torch.zeros(2))
    cv, candidate = torch.randn(4, 12, 2), torch.randn(4, 12, 2)
    predicted = blend_forecasts(cv, candidate, weights)
    assert torch.equal(predicted[~valid], cv[~valid])
    assert torch.equal(blend_forecasts(cv, candidate, torch.ones(4)), candidate)
    fixed = fixed_weights(torch.tensor([0., .5, 2., 3.]), valid)
    assert fixed[0] == .5 and fixed[1] == fixed[3] == 0


def test_quality_uses_observation_and_forecast_only():
    base = np.zeros((2, 12, 2), np.float32)
    position, velocity = np.ones((2, 2), np.float32), np.ones((2, 2), np.float32) * 2
    quality = quality_features(position, velocity, velocity, base, np.array([1, 11]), np.array([1., 0.]), np.array([0., 3.]), base, base + 2)
    assert quality.shape == (2, 8) and quality.dtype == np.float32
    assert quality[1, 1] == 1
    np.testing.assert_allclose(quality[:, 7], np.sqrt(8) / 20, atol=1e-7)
    np.testing.assert_allclose(quality[:, 5], 0)

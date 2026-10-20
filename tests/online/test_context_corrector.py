import numpy as np
import pytest
import torch

from trajectory_prediction.context.maps import MapView
from trajectory_prediction.online.context_corrector import MapCorrector, predict_offsets
from trajectory_prediction.online.context_features import residual_features
from trajectory_prediction.online.corrector import ResidualCorrector
from trajectory_prediction.online.dataset import CorrectionDataset
from trajectory_prediction.online.observations import replay_observations


def inputs():
    history = np.arange(10, dtype=np.float32).reshape(5, 2) - 8
    future = np.arange(24, dtype=np.float32).reshape(12, 2)
    return history, future, future + 1


def test_observation_features_match_old_dataset_at_all_steps():
    history, future, base = inputs()
    for obs, _ in replay_observations(history, future, anchor_timestamp_us=0):
        features, mask = residual_features(obs, base)
        old = CorrectionDataset(base[None], future[None], history[None], obs.update_step)[0]
        np.testing.assert_allclose(features, old["features"].numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(mask, old["mask"].numpy())


def test_no_map_exactly_equals_frozen_original_corrector():
    torch.manual_seed(1)
    model = MapCorrector(ResidualCorrector(hidden_dim=8), hidden_dim=8)
    features, mask = torch.randn(2, 12, 8), torch.ones(2, 12, dtype=torch.bool)
    points, valid = torch.randn(2, 2, 3, 4), torch.zeros(2, 2, 3, dtype=torch.bool)
    actual = model(features, mask, points, valid)
    torch.testing.assert_close(actual, model.fallback(features, mask), rtol=0, atol=0)
    assert all(not p.requires_grad for p in model.fallback.parameters())


def test_masked_map_padding_and_past_horizon_do_not_change_predictions():
    model = MapCorrector(ResidualCorrector(hidden_dim=8), hidden_dim=8)
    features, mask = torch.randn(1, 12, 8), torch.arange(12)[None] >= 3
    points, valid = torch.randn(1, 2, 3, 4), torch.zeros(1, 2, 3, dtype=torch.bool)
    valid[0, 0, 0] = True
    first = model(features, mask, points, valid)
    changed = points.clone()
    changed[~valid] = 99999
    torch.testing.assert_close(first, model(features, mask, changed, valid))
    assert torch.count_nonzero(first[:, :3]) == 0
    first.sum().backward()
    assert any(p.grad is not None for p in model.encoder.parameters())
    assert all(p.grad is None for p in model.fallback.parameters())


def test_real_prediction_api_accepts_only_observation_base_and_map():
    import inspect
    history, future, base = inputs()
    obs, _ = next(replay_observations(history, future, anchor_timestamp_us=0))
    view = MapView(np.zeros((1, 2, 4)), np.zeros((1, 2), bool), ("",))
    model = MapCorrector(ResidualCorrector(hidden_dim=8), hidden_dim=8)
    assert set(inspect.signature(predict_offsets).parameters) == {"model", "observation", "base", "map_view", "device"}
    result = predict_offsets(model, obs, base, view, torch.device("cpu"))
    assert result.shape == (12, 2)


@pytest.mark.parametrize("base", [np.zeros((11, 2)), np.full((12, 2), np.nan)])
def test_invalid_base_rejected(base):
    history, future, _ = inputs()
    obs, _ = next(replay_observations(history, future, anchor_timestamp_us=0))
    with pytest.raises(ValueError):
        residual_features(obs, base)

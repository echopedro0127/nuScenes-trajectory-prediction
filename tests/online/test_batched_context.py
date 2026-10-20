import numpy as np
import torch

from trajectory_prediction.online.batched_context import (
    BatchedContext,
    forward_inputs,
    residual_batch,
)
from trajectory_prediction.online.context_features import residual_features
from trajectory_prediction.online.corrector import ResidualCorrector
from trajectory_prediction.online.interaction_corrector import InteractionCorrector
from trajectory_prediction.online.observations import replay_observations


def fixture():
    n = 2
    history = np.tile(np.arange(10, dtype=np.float32).reshape(1, 5, 2), (n, 1, 1)) - 8
    future = np.tile(np.arange(24, dtype=np.float32).reshape(1, 12, 2), (n, 1, 1))
    arrays = {"history_xy": history, "future_xy": future}
    maps = {"features": np.zeros((n, 11, 1, 2, 4), np.float32), "mask": np.zeros((n, 11, 1, 2), bool)}
    neighbors = {"features": np.zeros((n, 11, 8, 5, 6), np.float32),
                 "mask": np.zeros((n, 11, 8, 5), bool), "velocity_mask": np.zeros((n, 11, 8, 5), bool),
                 "source_timestamp_us": np.zeros((n, 11, 8, 5), np.int64),
                 "timestamp_us": np.tile(np.arange(1, 12, dtype=np.int64) * 500_000, (n, 1))}
    return arrays, maps, neighbors, future + 1


def test_vectorized_residual_matches_observation_api_at_all_steps():
    arrays, _maps, _neighbors, base = fixture()
    for k in range(1, 12):
        current = arrays["future_xy"][:, k - 1]
        previous = arrays["history_xy"][:, -1] if k == 1 else arrays["future_xy"][:, k - 2]
        features, mask = residual_batch(current, previous, base, np.full(2, k))
        for i in range(2):
            observed, _ = list(replay_observations(arrays["history_xy"][i], arrays["future_xy"][i], anchor_timestamp_us=0))[k - 1]
            expected, horizon = residual_features(observed, base[i])
            np.testing.assert_allclose(features[i], expected, atol=1e-5, rtol=1e-5)
            np.testing.assert_array_equal(mask[i], horizon)


def test_batch_input_api_excludes_labels_and_future_perturbation():
    bundle = fixture()
    dataset = BatchedContext(*bundle)
    inputs = dataset.inputs(np.array([0, 1]), torch.device("cpu"))
    assert "target" not in inputs and "future" not in inputs
    changed = {name: value.copy() for name, value in bundle[0].items()}
    changed["future_xy"][:, 1:] += 10000
    second = BatchedContext(changed, *bundle[1:]).inputs(np.array([0]), torch.device("cpu"))
    torch.testing.assert_close(inputs["features"][:1], second["features"], atol=0, rtol=0)
    model = InteractionCorrector(ResidualCorrector(hidden_dim=8), "fusion", 8)
    assert forward_inputs(model, inputs).shape == (2, 12, 2)
    assert dataset.labels(np.array([0, 1]), torch.device("cpu")).shape == (2, 12, 2)

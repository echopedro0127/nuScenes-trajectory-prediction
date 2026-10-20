from copy import deepcopy

import numpy as np
import torch

from trajectory_prediction.online.corrector import ResidualCorrector
from trajectory_prediction.online.radar_corrector import RadarCorrector, RadarDataset, forward_model


def data_fixture():
    n = 3
    future = np.tile(np.arange(24, dtype=np.float32).reshape(1, 12, 2) / 10, (n, 1, 1))
    arrays = {"instance_token": np.array(["i0", "i1", "i2"]), "sample_token": np.array(["s0", "s1", "s2"]),
        "scene_name": np.array(["a", "a", "b"]), "history_xy": np.zeros((n, 5, 2), np.float32), "future_xy": future}
    cache = {"position": np.zeros((5, n, 11, 2), np.float32), "velocity": np.ones((5, n, 11, 2), np.float32),
        "radar_features": np.ones((5, n, 11, 13), np.float32), "present": np.ones((5, n, 11), bool)}
    return arrays, cache, future + 1


def test_radar_model_falls_back_exactly_and_dataset_keeps_labels_separate():
    torch.set_num_threads(1)
    data = RadarDataset(*data_fixture())
    inputs = data.inputs(np.array([0, 1, 33, 55]), torch.device("cpu"))
    assert set(inputs) == {"features", "mask", "radar", "present"}
    control = ResidualCorrector(hidden_dim=8)
    model = RadarCorrector(deepcopy(control), 8)
    assert all(not p.requires_grad for p in model.fallback.parameters())
    torch.nn.init.constant_(model.head[-1].weight, .2)
    inputs["present"][:] = False
    torch.testing.assert_close(forward_model(model, inputs), control(inputs["features"], inputs["mask"]), atol=0, rtol=0)
    assert data.labels(np.array([0, 1]), torch.device("cpu")).shape == (2, 12, 2)
    before = inputs["features"].clone()
    data.arrays["future_xy"][:] += 10000
    torch.testing.assert_close(before, data.inputs(np.array([0, 1, 33, 55]), torch.device("cpu"))["features"], atol=0, rtol=0)

from copy import deepcopy

import numpy as np
import pytest
import torch

from trajectory_prediction.context.gate_preparation import generate_candidates
from trajectory_prediction.online.corrector import ResidualCorrector
from trajectory_prediction.online.gate_dataset import GateDataset
from trajectory_prediction.online.gate_training import train_gate
from trajectory_prediction.online.interaction_corrector import InteractionCorrector


def datasets():
    n = 2
    future = np.tile(np.arange(24, dtype=np.float32).reshape(1, 12, 2) / 10, (n, 1, 1))
    arrays = {"history_xy": np.zeros((n, 5, 2), np.float32), "future_xy": future,
        "instance_token": np.array(["i0", "i1"]), "sample_token": np.array(["s0", "s1"]), "scene_name": np.array(["a", "b"])}
    maps = {"anchor_timestamp_us": np.full(n, 2000000, np.int64), "timestamp_us": np.tile(2000000 + np.arange(1, 12, dtype=np.int64) * 500000, (n, 1)),
        "features": np.ones((n, 11, 1, 2, 4), np.float32), "mask": np.ones((n, 11, 1, 2), bool)}
    original = ResidualCorrector(hidden_dim=8)
    model = InteractionCorrector(deepcopy(original), "map", 8)
    values = generate_candidates(arrays, maps, future + 1, original, model)
    conditional = {"map", "quality", "valid", "packet_step", "source_us", "available_us", "age_s", "coverage"}
    return [GateDataset({name: value[:, [i]] if name in conditional else value[[i]] for name, value in values.items()}) for i in range(2)]


def test_gate_training_scene_selection_best_state_and_actual_data_identity(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    train, selected = datasets()
    model = train_gate(train, selected, {}, tmp_path, epochs=2, batch_size=32, seed=2027, patience=1)
    payload = torch.load(tmp_path / "best.pt", weights_only=True)
    assert payload["metadata"]["train_scenes"] == ["a"]
    assert 1 <= payload["metadata"]["best_epoch"] <= payload["metadata"]["completed_epochs"] <= 2
    assert sum(p.numel() for p in model.parameters()) == 161
    import trajectory_prediction.online.gate_training as engine
    monkeypatch.setattr(engine, "run_epoch", lambda *a, **kw: pytest.fail("completed checkpoint must be reused"))
    again = train_gate(train, selected, {}, tmp_path, epochs=2, batch_size=32, seed=2027, patience=1)
    for a, b in zip(model.parameters(), again.parameters(), strict=True): torch.testing.assert_close(a, b)
    train.arrays["quality"][0, 0, 0, 0] += 100
    with pytest.raises(ValueError, match="identity"):
        train_gate(train, selected, {}, tmp_path, epochs=2, batch_size=32, seed=2027, patience=1)
    with pytest.raises(ValueError, match="overlap"):
        train_gate(train, train, {}, tmp_path / "bad")

from copy import deepcopy

import pytest
import torch
from test_interaction_training import fixture

from trajectory_prediction.online.batched_context import BatchedContext
from trajectory_prediction.online.scaled_training import train_scaled_suite


def subset(bundle, indices):
    return tuple({name: value[indices] for name, value in part.items()} if isinstance(part, dict) else part[indices]
                 for part in bundle)


def test_scaled_training_selects_restores_and_reuses_only_matching_stages(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    bundle, metadata, _map_contract, _neighbor_contract = fixture()
    train = BatchedContext(*subset(bundle, [0, 1]))
    selection = BatchedContext(*subset(bundle, [2]))
    models = train_scaled_suite(train, selection, metadata, tmp_path, epochs=1, batch_size=16,
                                hidden_dim=8, seed=7, patience=2, device=torch.device("cpu"))
    assert set(models) == {"original", "map", "neighbors", "fusion", "matched_map", "matched_neighbors", "matched_fusion"}
    import trajectory_prediction.online.scaled_training as engine
    monkeypatch.setattr(engine, "_epoch", lambda *args, **kwargs: pytest.fail("completed stages must be reused"))
    again = train_scaled_suite(train, selection, metadata, tmp_path, epochs=1, batch_size=16,
                               hidden_dim=8, seed=7, patience=2, device=torch.device("cpu"))
    for name in models:
        for a, b in zip(models[name].parameters(), again[name].parameters(), strict=True):
            torch.testing.assert_close(a, b)
    train.base = train.base.copy() + 12345
    with pytest.raises(ValueError, match="identity"):
        train_scaled_suite(train, selection, metadata, tmp_path, epochs=1, batch_size=16,
                            hidden_dim=8, seed=7, patience=2, device=torch.device("cpu"))
    with pytest.raises(ValueError, match="identity"):
        train_scaled_suite(train, selection, metadata, tmp_path, epochs=1, batch_size=16,
                            hidden_dim=8, seed=8, patience=2, device=torch.device("cpu"))
    bad = deepcopy(metadata)
    bad["selection_scenes"] = ["a"]
    with pytest.raises(ValueError):
        train_scaled_suite(train, selection, bad, tmp_path / "bad", epochs=1, batch_size=16,
                            hidden_dim=8, seed=7, patience=2, device=torch.device("cpu"))

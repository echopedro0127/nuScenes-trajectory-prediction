import json
from copy import deepcopy

import pytest
import torch
from test_gate_training import datasets
from test_independent_gate_training import metadata_fixture


def test_rejects_old_or_wrong_initializer_before_training(tmp_path):
    from trajectory_prediction.online.horizon_features import HorizonDataset
    from trajectory_prediction.online.independent_horizon_training import train_independent_horizon
    train, selection = datasets()
    with pytest.raises(ValueError, match="initializer"):
        train_independent_horizon(HorizonDataset(train.arrays), HorizonDataset(selection.arrays),
                                  metadata_fixture(), tmp_path / "missing.pt", tmp_path / "run", epochs=1)
    assert not (tmp_path / "run").exists()


def test_new_gate_initializes_rh_and_checkpoint_replays(tmp_path):
    from trajectory_prediction.online.horizon_features import HorizonDataset
    from trajectory_prediction.online.independent_gate_training import train_independent_gate
    from trajectory_prediction.online.independent_horizon_training import (
        load_independent_horizon,
        train_independent_horizon,
    )
    torch.set_num_threads(1)
    train, selection = datasets()
    metadata = metadata_fixture()
    gate_dir = tmp_path / "gate"
    train_independent_gate(train, selection, metadata, gate_dir, epochs=1, batch_size=32)
    output = tmp_path / "rh"
    train, selection = HorizonDataset(train.arrays), HorizonDataset(selection.arrays)
    model = train_independent_horizon(train, selection, metadata, gate_dir / "best.pt", output, epochs=1, batch_size=32)
    payload = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    expected = payload["metadata"]["independent_inputs"]
    restored, _ = load_independent_horizon(output / "best.pt", expected)
    assert sum(p.numel() for p in restored.parameters()) == 337
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
    report = json.loads((output / "report.json").read_text())
    assert report["initialization_max_error"] < 1e-5
    assert report["missing_weights_zero"]
    with pytest.raises(FileExistsError):
        train_independent_horizon(train, selection, metadata, gate_dir / "best.pt", output, epochs=1)
    bad = deepcopy(expected)
    bad["gate_checkpoint"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity"):
        load_independent_horizon(output / "best.pt", bad)
    payload["metadata"]["variant"] = "H"
    torch.save(payload, tmp_path / "bad.pt")
    with pytest.raises(ValueError):
        load_independent_horizon(tmp_path / "bad.pt", expected)

from copy import deepcopy

import pytest
import torch
from test_gate_training import datasets
from test_independent_gate_training import metadata_fixture


def test_independent_consistency_reproduces_zero_and_reloads(tmp_path):
    from trajectory_prediction.online.independent_consistency_training import (
        load_independent_consistency,
        train_independent_consistency,
    )
    from trajectory_prediction.online.independent_gate_training import train_independent_gate
    from trajectory_prediction.online.independent_horizon_training import train_independent_horizon
    from trajectory_prediction.online.innovation_consistency import InnovationDataset
    torch.set_num_threads(1)
    train, selection = (InnovationDataset(d.arrays) for d in datasets())
    metadata = metadata_fixture()
    train_independent_gate(train, selection, metadata, tmp_path / "gate", epochs=1, batch_size=32)
    rh_path = tmp_path / "rh/best.pt"
    train_independent_horizon(train, selection, metadata, tmp_path / "gate/best.pt", rh_path.parent, epochs=1, batch_size=32)
    for arm in ("revision_0", "revision_adaptive"):
        output = tmp_path / arm
        model = train_independent_consistency(train, selection, rh_path, output, arm=arm, epochs=1, batch_size=32)
        payload = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
        inputs = payload["metadata"]["independent_inputs"]
        restored, _ = load_independent_consistency(output / "best.pt", inputs, arm=arm)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, restored.state_dict()[name], atol=0, rtol=0)
        assert (output / "completed.json").exists()
        if arm == "revision_0":
            rh = torch.load(rh_path, map_location="cpu", weights_only=True)["model"]
            assert all(torch.equal(value, rh[name]) for name, value in restored.state_dict().items())
        with pytest.raises(FileExistsError):
            train_independent_consistency(train, selection, rh_path, output, arm=arm)
        bad = deepcopy(inputs)
        bad["rh_checkpoint"]["sha256"] = "0" * 64
        with pytest.raises(ValueError, match="identity"):
            load_independent_consistency(output / "best.pt", bad, arm=arm)
    with pytest.raises(ValueError, match="arm"):
        train_independent_consistency(train, selection, rh_path, tmp_path / "bad", arm="consistent_010")

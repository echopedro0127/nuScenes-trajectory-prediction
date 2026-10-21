import copy
import json

import numpy as np
import pytest
import torch
from test_horizon_gate import example_arrays

from trajectory_prediction.online.horizon_features import HorizonDataset
from trajectory_prediction.online.horizon_training import train_horizon, validate_role
from trajectory_prediction.online.quality_gate import QualityGate
from trajectory_prediction.online.refinement_training import completed_payload


def datasets():
    a, b = example_arrays(2), example_arrays(2)
    a["scene_name"][:] = ["train0", "train1"]
    b["scene_name"][:] = ["select0", "select1"]
    return HorizonDataset(a), HorizonDataset(b)


def test_horizon_training_reuses_completed_identity_and_rejects_input_change(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    train, selection = datasets()
    initial = copy.deepcopy(QualityGate().state_dict())
    model = train_horizon(train, selection, {}, tmp_path, initial=initial, variant="DRH", epochs=2, batch_size=32)
    payload = completed_payload(tmp_path)
    assert payload["metadata"]["parameters"] == 337
    assert payload["metadata"]["variant"] == "DRH"
    assert payload["metadata"]["completed_epochs"] <= 2
    import trajectory_prediction.online.horizon_training as engine
    monkeypatch.setattr(engine, "run_horizon_epoch", lambda *a, **k: pytest.fail("completed stage must reuse"))
    again = train_horizon(train, selection, {}, tmp_path, initial=initial, variant="DRH", epochs=2, batch_size=32)
    for x, y in zip(model.parameters(), again.parameters(), strict=True):
        torch.testing.assert_close(x, y)
    train.arrays["map"][0, 0, 0, -1, 0] += 1
    with pytest.raises(ValueError, match="identity"):
        train_horizon(train, selection, {}, tmp_path, initial=initial, variant="DRH", epochs=2, batch_size=32)
    with pytest.raises(ValueError, match="overlap"):
        train_horizon(train, train, {}, tmp_path/"bad", initial=initial, variant="B")


def test_horizon_completion_marker_tampering_rejected_and_unfinished_stage_restarts(tmp_path):
    torch.set_num_threads(1)
    train, selection = datasets()
    initial = QualityGate().state_dict()
    train_horizon(train, selection, {}, tmp_path, initial=initial, variant="B", epochs=1, batch_size=64)
    marker = tmp_path/"completed.json"
    marker.unlink()
    train_horizon(train, selection, {}, tmp_path, initial=initial, variant="B", epochs=1, batch_size=64)
    value = json.loads(marker.read_text())
    value["checkpoint_sha256"] = "bad"
    marker.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="artifact"):
        train_horizon(train, selection, {}, tmp_path, initial=initial, variant="B", epochs=1, batch_size=64)


def test_role_rejects_permuted_variant_mask_or_settings():
    from trajectory_prediction.online.horizon_gate import HorizonGate
    from trajectory_prediction.online.horizon_training import DEFAULT_SETTINGS
    model = HorizonGate("D", QualityGate().state_dict())
    meta = {**DEFAULT_SETTINGS, "variant": "D", "parameters": 337, "best_epoch": 1, "completed_epochs": 1}
    validate_role(model, meta, "D", DEFAULT_SETTINGS)
    with pytest.raises(ValueError, match="role"):
        validate_role(model, meta, "H", DEFAULT_SETTINGS)
    model.feature_mask[8:12] = 0
    with pytest.raises(ValueError, match="role"):
        validate_role(model, meta, "D", DEFAULT_SETTINGS)
    with pytest.raises(ValueError, match="role"):
        validate_role(HorizonGate("D"), {**meta, "epochs": 99}, "D", DEFAULT_SETTINGS)
    assert np.isfinite(model.network[0].weight.detach().numpy()).all()


def test_completed_reuse_rejects_changed_feature_mask_even_with_new_checksum(tmp_path):
    from trajectory_prediction.utils.provenance import sha256_file
    train, selection = datasets()
    initial = QualityGate().state_dict()
    train_horizon(train, selection, {}, tmp_path, initial=initial, variant="D", epochs=1, batch_size=64)
    payload = torch.load(tmp_path/"best.pt", weights_only=True)
    payload["model"]["feature_mask"][8:12] = 0
    torch.save(payload, tmp_path/"best.pt")
    marker = json.loads((tmp_path/"completed.json").read_text())
    marker["checkpoint_sha256"] = sha256_file(tmp_path/"best.pt")
    (tmp_path/"completed.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="role"):
        train_horizon(train, selection, {}, tmp_path, initial=initial, variant="D", epochs=1, batch_size=64)


def test_full_initial_metrics_parity_for_all_variants():
    from trajectory_prediction.online import horizon_training
    assert hasattr(horizon_training, "check_initialization")
    _, selection = datasets()
    result = horizon_training.check_initialization(selection, QualityGate().state_dict(), torch.device("cpu"))
    assert set(result) == {"B", "D", "R", "H", "DR", "DH", "RH", "DRH"}
    assert all(row["update_count"] == selection.size for row in result.values())
    assert all(row["max_metric_difference"] <= 1e-6 for row in result.values())

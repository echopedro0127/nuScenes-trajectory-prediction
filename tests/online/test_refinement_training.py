from copy import deepcopy

import numpy as np
import pytest
import torch

from trajectory_prediction.online.gate_refinement import RefinementDataset
from trajectory_prediction.online.quality_gate import QualityGate
from trajectory_prediction.online.refinement_training import fit_calibration, train_refinement


def data(scene):
    n = 2
    cv = np.zeros((n, 11, 12, 2), np.float32)
    a = {
        "instance_token": np.array([scene + "i0", scene + "i1"]),
        "sample_token": np.array(["s0", "s1"]),
        "scene_name": np.array([scene] * n),
        "base": np.zeros((n, 12, 2), np.float32),
        "target": np.ones((n, 12, 2), np.float32),
        "cv": cv,
        "map": np.full((6, n, 11, 12, 2), 2, np.float32),
        "original": np.ones((n, 11, 12, 2), np.float32),
        "quality": np.zeros((6, n, 11, 8), np.float32),
        "valid": np.ones((6, n, 11), bool),
        "age_s": np.zeros((6, n, 11), np.float32),
    }
    return RefinementDataset(a)


def test_calibration_recovers_known_mixture_and_empty_cells():
    torch.set_num_threads(1)
    dataset = data("train")
    result = fit_calibration(dataset, batch_size=17)
    assert result["constant"] == 0.5
    lookup = np.array(result["lookup"])
    assert lookup.shape == (11, 5) and np.all(lookup[:, 0] == 0.5) and np.all(lookup[:, 1:] == 0)
    dataset.arrays["valid"][:] = False
    empty = fit_calibration(dataset, batch_size=17)
    assert empty["constant"] == 0 and np.all(np.array(empty["lookup"]) == 0)


def test_pair_training_best_state_reuse_and_input_identity(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    train, selection = data("train"), data("selection")
    seed_model = QualityGate()
    model = train_refinement(
        train,
        selection,
        {},
        tmp_path,
        initial=deepcopy(seed_model.state_dict()),
        epochs=2,
        batch_size=32,
        patience=1,
        revision_weight=0.1,
    )
    payload = torch.load(tmp_path / "best.pt", weights_only=True)
    assert payload["metadata"]["best_epoch"] <= payload["metadata"]["completed_epochs"] <= 2
    assert payload["metadata"]["train_scenes"] == ["train"]
    assert (tmp_path / "completed.json").is_file()
    import trajectory_prediction.online.refinement_training as engine

    monkeypatch.setattr(
        engine, "run_epoch", lambda *a, **k: pytest.fail("should reuse completed checkpoint")
    )
    again = train_refinement(
        train,
        selection,
        {},
        tmp_path,
        initial=deepcopy(seed_model.state_dict()),
        epochs=2,
        batch_size=32,
        patience=1,
        revision_weight=0.1,
    )
    for a, b in zip(model.parameters(), again.parameters(), strict=True):
        torch.testing.assert_close(a, b)
    train.arrays["original"][0, 0, 0, 0] += 1
    with pytest.raises(ValueError, match="identity"):
        train_refinement(
            train,
            selection,
            {},
            tmp_path,
            initial=deepcopy(seed_model.state_dict()),
            epochs=2,
            batch_size=32,
            patience=1,
            revision_weight=0.1,
        )
    with pytest.raises(ValueError, match="overlap"):
        train_refinement(train, train, {}, tmp_path / "bad")


def test_missing_completion_marker_restarts_unfinished_publication(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    train, selection = data("train"), data("selection")
    initial = QualityGate().state_dict()
    train_refinement(train, selection, {}, tmp_path, initial=initial, epochs=1, batch_size=64)
    marker = tmp_path / "completed.json"
    if marker.exists():
        marker.unlink()
    import trajectory_prediction.online.refinement_training as engine

    original = engine.run_epoch
    called = []

    def epoch(*a, **kw):
        called.append(True)
        return original(*a, **kw)

    monkeypatch.setattr(engine, "run_epoch", epoch)
    train_refinement(train, selection, {}, tmp_path, initial=initial, epochs=1, batch_size=64)
    assert called and marker.exists()

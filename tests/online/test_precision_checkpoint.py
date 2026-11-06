import json

import numpy as np
import pytest
import torch
from test_consistency_training import datasets

from trajectory_prediction.online import precision_checkpoint as experiment
from trajectory_prediction.online.horizon_gate import HorizonGate
from trajectory_prediction.online.quality_gate import QualityGate
from trajectory_prediction.online.refinement_training import state_fingerprint
from trajectory_prediction.utils.provenance import sha256_file


def reference():
    return dict(zip(experiment.FIELDS, (1., 2., 1., 1., 5.), strict=True))


def row(epoch, ade=1., jitter=.9, **changes):
    return {**reference(), "epoch": epoch, "ade": ade, "jitter": jitter, **changes}


def test_two_rules_select_from_same_trajectory_and_fallback_to_rh():
    logs = [row(1, ade=.999, jitter=.99), row(2, ade=1.0004, jitter=.95),
            row(3, ade=1.001, jitter=.8)]
    result = experiment.select_epochs(logs, reference())
    assert result == {"ade_best": 1, "constrained_best": 2}
    assert experiment.select_epochs([row(1, ade=1.001)], reference())["constrained_best"] == 0


@pytest.mark.parametrize("field,value", [("clean_ade", 1.001), ("fde", 2.006), ("clean_fde_p95", 5.03)])
def test_clean_and_tail_constraints_are_enforced(field, value):
    assert experiment.select_epochs([row(1, **{field: value})], reference())["constrained_best"] == 0


def test_equal_scores_prefer_earliest_epoch_and_reference():
    assert experiment.select_epochs([row(2), row(1)], reference())["constrained_best"] == 1
    assert experiment.select_epochs([row(1, jitter=1.)], reference())["constrained_best"] == 0


@pytest.mark.parametrize("logs", [[], [row(1), row(1)], [row(0)], [row(1, ade=np.nan)], [row(1, jitter=-1.)]])
def test_rejects_invalid_or_duplicate_epoch_records(logs):
    with pytest.raises(ValueError):
        experiment.select_epochs(logs, reference())


def test_summarize_checks_shape_and_preserves_clean_quantile():
    metrics = np.ones((6, 2, 11, 3), np.float64)
    metrics[0, ..., 1] = 3.
    values = experiment.summarize(metrics)
    assert values["clean_fde_p95"] == 3. and values["ade"] == 1.
    with pytest.raises(ValueError):
        experiment.summarize(metrics[:5])


def test_training_saves_all_epochs_and_replays_selected_checkpoints(tmp_path):
    torch.set_num_threads(1)
    train, selection = datasets()
    initial = QualityGate().state_dict()
    root = tmp_path / "new-run"
    report = experiment.train_path(train, selection, initial, HorizonGate("RH", initial), root,
                                   seed=2027, epochs=2, batch_size=64)
    assert report["completed_epochs"] == 2
    assert len(list(root.glob("epoch-*.pt"))) == 2
    marker = json.loads((root / "completed.json").read_text())
    assert marker["metrics_sha256"] == sha256_file(root / "selected_metrics.npz")
    verified = experiment.verify_run(root, selection, initial, torch.device("cpu"))
    assert verified["selected"] == report["selected"]
    assert not report["val_run"]
    checkpoint = root / "epoch-001.pt"
    original = torch.load(checkpoint, weights_only=True)
    original["model"]["network.0.bias"][0] += 1
    torch.save(original, checkpoint)
    with pytest.raises(ValueError, match="checksum"):
        experiment.verify_run(root, selection, initial, torch.device("cpu"))
    report["checkpoints"]["1"]["sha256"] = sha256_file(checkpoint)
    (root / "report.json").write_text(json.dumps(report))
    marker = json.loads((root / "completed.json").read_text())
    marker["report_sha256"] = sha256_file(root / "report.json")
    (root / "completed.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError, match="fingerprint"):
        experiment.verify_run(root, selection, initial, torch.device("cpu"))


def test_overlap_and_existing_output_are_rejected(tmp_path):
    train, selection = datasets()
    initial = QualityGate().state_dict()
    with pytest.raises(ValueError, match="overlap"):
        experiment.train_path(train, train, initial, HorizonGate("RH", initial), tmp_path / "bad")
    with pytest.raises(FileExistsError):
        experiment.train_path(train, selection, initial, HorizonGate("RH", initial), tmp_path)


def test_identical_epoch_loop_matches_frozen_strength_training(tmp_path):
    from trajectory_prediction.online.strength_training import train_strength

    torch.set_num_threads(1)
    train, selection = datasets()
    initial = QualityGate().state_dict()
    old = train_strength(train, selection, {}, tmp_path / "parent", initial=initial,
                         arm="adaptive_matched_005", epochs=2, batch_size=64)
    root = tmp_path / "new"
    report = experiment.train_path(train, selection, initial, HorizonGate("RH", initial), root,
                                   epochs=2, batch_size=64)
    epoch = str(report["selected"]["ade_best"])
    assert report["checkpoints"][epoch]["fingerprint"] == state_fingerprint(old.state_dict())

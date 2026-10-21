from copy import deepcopy

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.context.gate_preparation import generate_candidates
from trajectory_prediction.online.corrector import ResidualCorrector
from trajectory_prediction.online.gate_dataset import GateDataset
from trajectory_prediction.online.gate_evaluation import (
    benchmark_gate,
    evaluate_gate,
    recovery_table,
    write_gate_outputs,
)
from trajectory_prediction.online.interaction_corrector import InteractionCorrector
from trajectory_prediction.online.quality_gate import QualityGate


def fixture():
    torch.set_num_threads(1)
    n = 2
    future = np.tile(np.arange(24, dtype=np.float32).reshape(1, 12, 2) / 10, (n, 1, 1))
    arrays = {
        "history_xy": np.zeros((n, 5, 2), np.float32),
        "future_xy": future,
        "instance_token": np.array(["i0", "i1"]),
        "sample_token": np.array(["s0", "s1"]),
        "scene_name": np.array(["a", "b"]),
    }
    maps = {
        "anchor_timestamp_us": np.full(n, 2000000, np.int64),
        "timestamp_us": np.tile(2000000 + np.arange(1, 12, dtype=np.int64) * 500000, (n, 1)),
        "features": np.ones((n, 11, 1, 2, 4), np.float32),
        "mask": np.ones((n, 11, 1, 2), bool),
    }
    original = ResidualCorrector(hidden_dim=8)
    model = InteractionCorrector(deepcopy(original), "map", 8)
    return GateDataset(generate_candidates(arrays, maps, future + 1, original, model))


def test_all_conditions_strategies_exact_missing_cv_and_complete_outputs(tmp_path):
    data = fixture()
    results, timings, weights = evaluate_gate(data, QualityGate(), batch_size=7)
    assert len(results) == 7
    for strategy in ("fixed_gate", "learned_gate"):
        np.testing.assert_array_equal(results[strategy][4], results["sliding"][4])
        np.testing.assert_array_equal(weights[strategy][4], 0)
        np.testing.assert_array_equal(weights[strategy][5, :, 5:], 0)
    for values in results.values():
        assert (values[:, :, 0, 2] == 0).all()
    summary = write_gate_outputs(data, results, timings, weights, tmp_path, {"seed": 2027})
    frame = pd.read_csv(tmp_path / "per_update.csv.gz")
    assert len(frame) == 2 * 11 * 6 * 7 == summary["row_count"]
    assert not frame.duplicated(["strategy", "condition", "instance_token", "update_step"]).any()
    benchmark_gate(data, QualityGate(), tmp_path, torch.device("cpu"))


def test_recovery_requires_two_consecutive_after_packet_seven():
    values = np.zeros((6, 2, 11, 3))
    values[3, :, :, 0] = 1
    values[3, 0, 6:8, 0] = 0.2
    values[3, 1, 10, 0] = 0
    rows = recovery_table(
        {"x": values},
        {"instance_token": ["a", "b"], "sample_token": ["c", "d"], "scene_name": ["s", "s"]},
        2027,
    )
    assert rows.recovered.tolist() == [True, False]
    assert rows.recovery_seconds.iloc[0] == 0
    assert np.isnan(rows.recovery_seconds.iloc[1])


def test_report_stream_rejects_edited_metrics_even_with_updated_artifact_hash(
    tmp_path, monkeypatch
):
    import json
    from pathlib import Path

    import pytest

    import trajectory_prediction.analysis.gate_report as report
    import trajectory_prediction.online.gate_evaluation as evaluation
    from trajectory_prediction.online.gate_dataset import save_gate_cache
    from trajectory_prediction.utils.provenance import sha256_file

    data = fixture()
    model = QualityGate()
    contract = {
        "seed": 2027,
        "split": "val",
        "gate_config": {"max_age_s": 2.0, "fixed_scale": 0.5, "decay_s": 1.0},
        "preparation_sources": {},
    }
    cache = tmp_path / "cache"
    save_gate_cache(cache, data.arrays, contract)
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"fixture")
    from trajectory_prediction.online.gate_dataset import load_gate_cache

    monkeypatch.setattr(
        report, "load_verified_gate_cache", lambda root, *a, **kw: load_gate_cache(root, contract)
    )
    monkeypatch.setattr(report, "preparation_sources", dict)
    monkeypatch.setattr(
        report, "load_gate_checkpoint", lambda *a: (model, {"run_identity": "fixture"})
    )
    provenance = {
        "seed": 2027,
        "candidate_contract": contract,
        "candidate_manifest": {
            "path": str(cache / "manifest.json"),
            "sha256": sha256_file(cache / "manifest.json"),
        },
        "checkpoint": {"path": str(checkpoint), "sha256": sha256_file(checkpoint)},
        "training_run_identity": "fixture",
        "evaluation_source_sha256": sha256_file(Path(evaluation.__file__)),
    }
    results, timings, weights = evaluate_gate(data, model, batch_size=7)
    root = tmp_path / "evaluation"
    write_gate_outputs(data, results, timings, weights, root, provenance)
    benchmark_gate(data, model, root, torch.device("cpu"))
    replay = report.evaluate_gate

    def same_execution_device(data, model, *, device):
        assert device == torch.device("cuda")
        return replay(data, model, device=torch.device("cpu"))

    monkeypatch.setattr(report, "evaluate_gate", same_execution_device)
    report.audit_run(root, device=torch.device("cuda"))
    frame = pd.read_csv(root / "per_update.csv.gz")
    frame.loc[0, "ade"] += 1
    frame.to_csv(root / "per_update.csv.gz", index=False, compression="gzip")
    summary = json.loads((root / "summary.json").read_text())
    summary["sha256"]["per_update.csv.gz"] = sha256_file(root / "per_update.csv.gz")
    (root / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="metric reconstruction"):
        report.audit_run(root, device=torch.device("cuda"))

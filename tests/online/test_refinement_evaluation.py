import numpy as np
import pandas as pd
import pytest
import torch

from trajectory_prediction.online.gate_refinement import RefinementDataset
from trajectory_prediction.online.quality_gate import QualityGate
from trajectory_prediction.online.refinement_evaluation import (
    STRATEGIES,
    benchmark_suite,
    choose_champions,
    evaluate_suite,
    write_suite_outputs,
)


def fixture():
    torch.set_num_threads(1)
    n = 2
    a = {
        "instance_token": np.array(["i0", "i1"]),
        "sample_token": np.array(["s0", "s1"]),
        "scene_name": np.array(["a", "b"]),
        "base": np.zeros((n, 12, 2), np.float32),
        "target": np.ones((n, 12, 2), np.float32),
        "cv": np.zeros((n, 11, 12, 2), np.float32),
        "map": np.full((6, n, 11, 12, 2), 2, np.float32),
        "original": np.ones((n, 11, 12, 2), np.float32),
        "quality": np.zeros((6, n, 11, 8), np.float32),
        "valid": np.ones((6, n, 11), bool),
        "age_s": np.zeros((6, n, 11), np.float32),
        "coverage": np.ones((6, n, 11), np.float32),
        "packet_step": np.tile(np.arange(1, 12, dtype=np.int64), (6, n, 1)),
        "timestamp_us": np.tile(2000000 + np.arange(1, 12, dtype=np.int64) * 500000, (n, 1)),
        "source_us": np.tile(2000000 + np.arange(1, 12, dtype=np.int64) * 500000, (6, n, 1)),
        "available_us": np.tile(2000000 + np.arange(1, 12, dtype=np.int64) * 500000, (6, n, 1)),
    }
    a["valid"][4] = False
    a["valid"][5, :, 5:] = False
    suite = {
        "models": {
            name: QualityGate()
            for name in (
                "p6_learned",
                "original_gate",
                "finetune_0",
                "consistent_002",
                "consistent_010",
            )
        },
        "calibration": {"constant": 0.5, "lookup": np.full((11, 5), 0.2).tolist()},
        "references": {},
    }
    policy = {"short_tau": 1.0, "consistent_model": "consistent_002", "consistent_tau": 1.0}
    return RefinementDataset(a), suite, policy


def test_complete_suite_parity_invalid_cv_and_numeric_outputs(tmp_path):
    data, suite, policy = fixture()
    results, weights, timings = evaluate_suite(data, suite, policy, batch_size=7)
    assert tuple(results) == STRATEGIES and len(results) == 12
    for strategy, values in results.items():
        assert values.shape == (6, 2, 11, 3) and (values[:, :, 0, 2] == 0).all()
        if strategy != "ungated_map":
            np.testing.assert_array_equal(values[4], results["sliding"][4])
        if strategy not in ("ungated_map", "sliding"):
            assert (weights[strategy][4] == 0).all() and (weights[strategy][5, :, 5:] == 0).all()
    policy["short_tau"] = 0
    no_shrink, _, _ = evaluate_suite(data, suite, policy, batch_size=7)
    np.testing.assert_array_equal(no_shrink["short_gate"], no_shrink["p6_learned"])
    summary = write_suite_outputs(data, results, weights, timings, tmp_path, {"seed": 2027})
    assert summary["row_count"] == 132 * 12
    assert (tmp_path / "metrics.npz").is_file() and (tmp_path / "weights.npz").is_file()
    benchmark_suite(data, suite, policy, tmp_path, torch.device("cpu"))
    assert (tmp_path / "pair_forward.json").is_file()


def test_accuracy_and_stability_use_fixed_seed_average_and_one_percent_constraint():
    rows = []
    for seed in (2027, 2028, 2029):
        for strategy in STRATEGIES:
            ade, jitter = {
                "p6_learned": (1.0, 1.0),
                "original_gate": (0.9, 1.2),
                "consistent_002": (1.005, 0.8),
                "constant_gate": (1.02, 0.1),
            }.get(strategy, (1.1, 1.5))
            for condition in (
                "clean",
                "delay_500ms",
                "delay_1000ms",
                "dropout",
                "missing",
                "stale",
            ):
                rows.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "condition": condition,
                        "ade": ade,
                        "jitter": jitter,
                    }
                )
    frame = pd.DataFrame(rows)
    chosen = choose_champions(frame)
    assert chosen["accuracy"] == "original_gate" and chosen["stable"] == "consistent_002"
    with pytest.raises(ValueError, match="seed"):
        choose_champions(frame.query("seed != 2029"))
    with pytest.raises(ValueError, match="complete"):
        choose_champions(frame.query("condition == 'clean'"))
    frame.loc[frame.strategy == "constant_gate", ["ade", "jitter"]] = [0.9, 0.1]
    assert choose_champions(frame)["accuracy"] == "constant_gate"
    frame.loc[frame.strategy == "time_age_gate", ["ade", "jitter"]] = [0.9, 0.0]
    assert choose_champions(frame)["accuracy"] == "constant_gate"


def test_wrong_model_role_and_initial_state_are_rejected():
    from trajectory_prediction.online.refinement_evaluation import validate_model_role

    meta = {
        "candidate": "map",
        "revision_weight": 0.1,
        "epochs": 10,
        "patience": 3,
        "batch_size": 4096,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "smoothness_weight": 0.05,
        "hidden_dim": 16,
        "initial_state_sha256": "p6",
        "best_epoch": 1,
        "completed_epochs": 2,
    }
    validate_model_role("consistent_010", meta, "p6", "random")
    for key, value in (
        ("candidate", "original"),
        ("revision_weight", 0.0),
        ("initial_state_sha256", "random"),
    ):
        with pytest.raises(ValueError, match="role"):
            validate_model_role("consistent_010", {**meta, key: value}, "p6", "random")


def test_report_recomputes_numeric_values_even_if_artifact_hash_is_updated(tmp_path, monkeypatch):
    import json

    from trajectory_prediction.analysis import refinement_report as report
    from trajectory_prediction.utils.provenance import sha256_file

    data, suite, policy = fixture()
    checkpoint = tmp_path / "p6" / "training" / "seed-2027" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"fixture")
    suite["references"] = {
        "p6_learned": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)}
    }
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "manifest.json").write_text("{}")
    contract = {"seed": 2027, "split": "val"}
    selected = {"policy": policy, "references": {"2027": {"suite": suite["references"]}}}
    selectionroot = tmp_path / "selection"
    selectionroot.mkdir()
    (selectionroot / "selection.json").write_text(json.dumps(selected))
    provenance = {
        "seed": 2027,
        "split": "val",
        "candidate_contract": contract,
        "candidate_manifest": {
            "path": str((cache / "manifest.json").resolve()),
            "sha256": sha256_file(cache / "manifest.json"),
        },
        "selection": {
            "path": str((selectionroot / "selection.json").resolve()),
            "sha256": sha256_file(selectionroot / "selection.json"),
        },
        "training_root": str(tmp_path / "training"),
        "p6_training_root": str(checkpoint.parents[1]),
        "suite": suite["references"],
    }
    values, weights, timings = evaluate_suite(data, suite, policy, batch_size=7)
    root = tmp_path / "evaluation"
    write_suite_outputs(data, values, weights, timings, root, provenance)
    benchmark_suite(data, suite, policy, root, torch.device("cpu"))
    old = checkpoint.parents[2] / "evaluation" / "seed-2027"
    old.mkdir(parents=True)
    frame = pd.read_csv(root / "by_condition.csv")
    frame = frame[frame.strategy.isin(["sliding", "ungated_map", "p6_fixed", "p6_learned"])].copy()
    frame["strategy"] = frame.strategy.replace(
        {"p6_fixed": "fixed_gate", "p6_learned": "learned_gate"}
    )
    frame.to_csv(old / "by_condition.csv", index=False)
    (old / "summary.json").write_text(
        json.dumps(
            {
                "provenance": {
                    "candidate_manifest": provenance["candidate_manifest"],
                    "checkpoint": suite["references"]["p6_learned"],
                },
                "sha256": {"by_condition.csv": sha256_file(old / "by_condition.csv")},
            }
        )
    )
    monkeypatch.setattr(report, "load_selection", lambda _: selected)
    monkeypatch.setattr(
        report, "load_verified_gate_cache", lambda *a, **k: (data.arrays, {"contract": contract})
    )
    monkeypatch.setattr(report, "load_suite", lambda *a, **k: suite)
    report.audit_run(root, selected, torch.device("cpu"))
    values["original_gate"][0, 0, 0, 0] += 1
    np.savez_compressed(root / "metrics.npz", **values)
    summary = json.loads((root / "summary.json").read_text())
    summary["sha256"]["metrics.npz"] = sha256_file(root / "metrics.npz")
    (root / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="candidate reconstruction"):
        report.audit_run(root, selected, torch.device("cpu"))

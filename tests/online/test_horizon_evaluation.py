import json

import numpy as np
import pandas as pd
import pytest
import torch
from test_horizon_gate import example_arrays

from trajectory_prediction.online.horizon_artifacts import publish_run, verify_run
from trajectory_prediction.online.horizon_evaluation import (
    CONDITIONS,
    SEEDS,
    STRATEGIES,
    choose_horizon,
)
from trajectory_prediction.online.horizon_features import HorizonDataset
from trajectory_prediction.online.horizon_gate import VARIANTS
from trajectory_prediction.online.horizon_streaming import new_state, stream_inputs
from trajectory_prediction.utils.provenance import sha256_file


def selection_frame():
    values = {"consistent_short": (1., 2., 1., 6.), "consistent_010": (1.002, 2.002, .9, 6.001),
        "B": (.999, 2., 1.01, 6.), "D": (.99, 2., 1., 6.), "R": (.998, 2., .88, 6.),
        "DRH": (.98, 2.02, .8, 6.)}
    return pd.DataFrame([{"seed": s, "condition": c, "strategy": name,
        **dict(zip(("ade", "fde", "jitter", "fde_p95"), values.get(name, (1.1, 2., 1.5, 6.)), strict=True))}
        for s in SEEDS for c in CONDITIONS for name in STRATEGIES])


def test_selection_enforces_fde_tail_constraints_and_both_advancement_routes():
    frame = selection_frame()
    result = choose_horizon(frame)
    assert result["accuracy"] == "D" and result["stable"] == "R"
    assert result["advance"] and set(result["advance_roles"]) == {"accuracy", "stable"}
    with pytest.raises(ValueError, match="complete"):
        choose_horizon(frame.query("condition != 'missing'"))
    with pytest.raises(ValueError, match="complete"):
        choose_horizon(pd.concat((frame, frame.iloc[:1])))
    frame.loc[frame.strategy == "H", ["ade", "fde", "jitter", "fde_p95"]] = [.99, 2., 1., 6.]
    assert choose_horizon(frame)["accuracy"] == "D"


def test_baseline_winner_does_not_advance_or_claim_new_success():
    frame = selection_frame()
    frame.loc[frame.strategy.isin(VARIANTS), ["ade", "jitter"]] = [1.1, 1.5]
    result = choose_horizon(frame)
    assert result["accuracy"] == "consistent_short"
    assert result["stable"] == "consistent_010"
    assert not result["advance"] and not result["advance_roles"]


def test_online_state_updates_equal_batched_prefix_features_without_future_reads():
    data = HorizonDataset(example_arrays())
    for c in range(6):
        for i in range(data.n):
            state = new_state(data, c, i)
            for k in range(11):
                actual = stream_inputs(data, c, i, k, state, torch.device("cpu"))
                expected = data.inputs(np.array([c*data.n*11+i*11+k]), torch.device("cpu"))
                torch.testing.assert_close(actual["features"], expected["features"], atol=1e-6, rtol=1e-6)
                assert state["next_step"] == k+1
            with pytest.raises(ValueError, match="state"):
                stream_inputs(data, c, i, 10, state, torch.device("cpu"))


def test_horizon_replay_rejects_updated_hash_numeric_tampering_and_bad_shape(tmp_path):
    results = {name: np.zeros((6, 2, 11, 3), np.float64) for name in STRATEGIES}
    weights = {name: np.zeros((6, 2, 11, 12), np.float32) for name in STRATEGIES}
    tables = {"by_condition.csv": pd.DataFrame({"name": ["B"], "ade": [0.]})}
    meta = {"sample_count": 2, "seed": 2027}
    publish_run(tmp_path, results, weights, tables, meta)
    verify_run(tmp_path, results, weights, tables, meta)
    modified = {k: v.copy() for k, v in results.items()}
    modified["DRH"][0, 0, 0, 0] = 1
    np.savez_compressed(tmp_path/"metrics.npz", **modified)
    manifest = json.loads((tmp_path/"summary.json").read_text())
    manifest["sha256"]["metrics.npz"] = sha256_file(tmp_path/"metrics.npz")
    (tmp_path/"summary.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="replay"):
        verify_run(tmp_path, results, weights, tables, meta)
    weights["B"] = weights["B"][..., 0]
    with pytest.raises(ValueError, match="shape"):
        publish_run(tmp_path, results, weights, tables, meta)


def test_complete_suite_tables_numeric_replay_and_real_online_profile(tmp_path, monkeypatch):
    from test_refinement_evaluation import fixture

    from trajectory_prediction.analysis.horizon_report import audit_latency
    from trajectory_prediction.online.horizon_evaluation import (
        benchmark_horizon,
        derive_tables,
        evaluate_models,
    )
    from trajectory_prediction.online.horizon_gate import HorizonGate
    old, suite, policy = fixture()
    observed = example_arrays(2)
    arrays = {**old.arrays, **{name: observed[name] for name in ("history", "position", "velocity", "previous_velocity")}}
    data = HorizonDataset(arrays)
    initial = suite["models"]["p6_learned"].state_dict()
    models = {v: HorizonGate(v, initial) for v in VARIANTS}
    result, weights = evaluate_models(data, suite, policy, models, torch.device("cpu"))
    tables = derive_tables(data, result, weights, 2027)
    assert len(tables["by_condition.csv"]) == 6*len(STRATEGIES)
    assert set(tables["quantiles.csv"].strategy) == set(STRATEGIES)
    assert (tables["gate_distribution.csv"]["count"] == data.n*66).all()
    monkeypatch.setattr(data, "inputs", lambda *args, **kwargs: pytest.fail("baseline must not build D/R/H features"))
    benchmark_horizon(data, suite, policy, models, tmp_path, torch.device("cpu"))
    monkeypatch.undo()
    profile = audit_latency(tmp_path, data, torch.device("cpu"))
    assert profile["key_count"] == min(256, data.size)
    meta = {"sample_count": data.n, "seed": 2027, "latency_profile": True}
    publish_run(tmp_path, result, weights, tables, meta)
    verify_run(tmp_path, result, weights, tables, meta)


def test_main_does_not_access_any_val_cache_when_frozen_selection_fails(tmp_path, monkeypatch, capsys):
    import trajectory_prediction.online.horizon_evaluation as engine
    (tmp_path/"selection.json").write_text("{}")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(engine, "load_horizon_selection", lambda root: {"decision": {"advance": False}})
    monkeypatch.setattr(engine, "run_split", lambda *args, **kwargs: pytest.fail("must not access val"))
    argv = [item for name in ("cache-root", "training-root", "p6-training-root", "reference-training-root", "reference-selection-root", "selection-root", "output-root") for item in ("--"+name, str(tmp_path))]
    engine.main([*argv, "--device", "cpu"])
    assert "val_not_run" in capsys.readouterr().out

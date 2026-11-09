import pytest


def test_started_record_is_idempotent_only_for_identical_binding(tmp_path):
    from trajectory_prediction.evaluation.independent_holdout import begin_run
    binding = {"protocol_identity": "a" * 64, "sources": {"new.py": "b" * 64}}
    first = begin_run(tmp_path, binding)
    assert first == begin_run(tmp_path, binding)
    with pytest.raises(ValueError, match="binding"):
        begin_run(tmp_path, {**binding, "sources": {"new.py": "c" * 64}})


def test_started_record_rejects_unbound_existing_outputs(tmp_path):
    from trajectory_prediction.evaluation.independent_holdout import begin_run
    (tmp_path / "results.json").write_text("{}")
    with pytest.raises(ValueError, match="unbound"):
        begin_run(tmp_path, {"protocol_identity": "a" * 64})


def test_six_real_inference_paths_preserve_missing_fallback():
    from copy import deepcopy

    import numpy as np
    import torch

    from trajectory_prediction.context.gate_preparation import generate_candidates
    from trajectory_prediction.evaluation.independent_holdout import _measure, summarize
    from trajectory_prediction.online.corrector import ResidualCorrector
    from trajectory_prediction.online.horizon_gate import HorizonGate
    from trajectory_prediction.online.interaction_corrector import InteractionCorrector
    from trajectory_prediction.online.quality_gate import QualityGate

    torch.set_num_threads(1)
    n = 2
    future = np.tile(np.arange(24, dtype=np.float32).reshape(1, 12, 2) / 10, (n, 1, 1))
    arrays = {"history_xy": np.zeros((n, 5, 2), np.float32), "future_xy": future,
              "instance_token": np.array(["i0", "i1"]), "sample_token": np.array(["s0", "s1"]),
              "scene_name": np.array(["a", "b"])}
    maps = {"anchor_timestamp_us": np.full(n, 2000000, np.int64),
            "timestamp_us": np.tile(2000000 + np.arange(1, 12) * 500000, (n, 1)),
            "features": np.ones((n, 11, 1, 2, 4), np.float32), "mask": np.ones((n, 11, 1, 2), bool)}
    original = ResidualCorrector(hidden_dim=8)
    map_model = InteractionCorrector(deepcopy(original), "map", 8)
    data = generate_candidates(arrays, maps, future + 1, original, map_model)
    gate = QualityGate().eval()
    rh = HorizonGate("RH", gate.state_dict()).eval()
    metrics, weights = _measure(data, ({"original": original, "map": map_model}, gate, rh, deepcopy(rh)), torch.device("cpu"))
    assert len(metrics) == 6
    for values in metrics.values():
        assert values.shape == (6, n, 11, 3) and np.isfinite(values).all()
    for name, value in weights.items():
        assert not value[4].any()
        np.testing.assert_array_equal(metrics[name][4], metrics["cv"][4])
    policy = {"strategies": list(metrics), "conditions": list(range(6)), "holdout_counts": {"log_count": 2},
              "paired_bootstrap": {"replicates": 100, "seed": 11}}
    report, draws = summarize(metrics, np.array(["l0", "l1"]), policy)
    assert len(draws) == 100 and len(report["aggregate"]) == 6
    with pytest.raises(ValueError, match="all frozen strategies"):
        summarize({"cv": metrics["cv"]}, np.array(["l0", "l1"]), policy)

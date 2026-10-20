import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from trajectory_prediction.analysis.gate_report import audit_latency, combine_scene_intervals
from trajectory_prediction.online.gate_evaluation import LATENCY_SCOPE
from trajectory_prediction.online.scaled_evaluation import benchmark_keys


def test_scene_bootstrap_averages_fixed_seeds_before_resampling():
    rows = []
    for seed in (2027, 2028, 2029):
        for strategy in ("learned_gate", "sliding", "filtered", "fixed_gate", "ungated_map"):
            for scene, count in (("a", 2), ("b", 6)):
                rows.append(
                    {
                        "seed": seed,
                        "condition": "clean",
                        "strategy": strategy,
                        "scene_name": scene,
                        "update_count": count,
                        **{
                            m + "_sum": count * (1 if strategy == "learned_gate" else 2)
                            for m in ("ade", "fde", "jitter")
                        },
                    }
                )
    result = combine_scene_intervals(pd.DataFrame(rows))
    assert len(result) == 12
    assert all(row["delta"] == -1 and row["ci95"] == [-1, -1] for row in result)
    with pytest.raises(ValueError, match="seed"):
        combine_scene_intervals(pd.DataFrame(rows).query("seed != 2029"))


def test_latency_requires_fixed_keys_scoped_measurements_and_complete_fields(tmp_path):
    expected = pd.DataFrame(
        {
            "flat_index": benchmark_keys(300),
            "condition": "clean",
            "instance_token": "i",
            "sample_token": "s",
            "scene_name": "scene",
            "update_step": 1,
            "timestamp_us": 2500000,
        }
    )
    expected.to_csv(tmp_path / "latency_keys.csv", index=False)
    rows = pd.DataFrame(
        [
            {"strategy": s, "flat_index": k, "total_ms": 1.0}
            for s in ("fixed_gate", "learned_gate")
            for k in expected.flat_index
        ]
    )
    rows.to_csv(tmp_path / "latency_measurements.csv", index=False)
    digest = lambda name: hashlib.sha256((tmp_path / name).read_bytes()).hexdigest()
    latency = {
        "key_count": 256,
        "warmup": 16,
        "inference_batch_size": 1,
        "key_selection_seed": 2027,
        "scope": LATENCY_SCOPE,
        "excluded": ["map_query", "P4_candidate_forward", "cold_cache_load"],
        "device": "cpu",
        "device_name": "fixture",
        "torch_version": "fixture",
        "keys_sha256": digest("latency_keys.csv"),
        "measurements_sha256": digest("latency_measurements.csv"),
        "strategies": {s: {"p50": 1.0, "p95": 1.0} for s in ("fixed_gate", "learned_gate")},
    }
    (tmp_path / "latency.json").write_text(json.dumps(latency))
    audit_latency(tmp_path, 300, expected)
    del latency["scope"]
    (tmp_path / "latency.json").write_text(json.dumps(latency))
    with pytest.raises(ValueError, match="latency"):
        audit_latency(tmp_path, 300, expected)
    assert np.array_equal(expected.flat_index, benchmark_keys(300))


def test_missing_group_columns_are_rejected():
    from trajectory_prediction.analysis.gate_report import _equal_frame

    with pytest.raises(ValueError, match="columns"):
        _equal_frame(pd.DataFrame({"key": [1]}), pd.DataFrame({"key": [1], "ade": [2.0]}), ["key"])

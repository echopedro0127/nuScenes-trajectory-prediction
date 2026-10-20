import json
from types import SimpleNamespace

import numpy as np
import pytest

from trajectory_prediction.analysis.context_ablation import paired_scene_interval, report_ablation
from trajectory_prediction.online.batched_evaluation import write_batch_outputs
from trajectory_prediction.online.interaction_report import STRATEGIES


def test_scene_bootstrap_preserves_clusters_and_weighted_constant_delta():
    sums = np.array([2., 200.])
    counts = np.array([1, 100])
    result = paired_scene_interval(sums, counts, replicates=2000)
    assert result["delta"] == 2
    np.testing.assert_allclose(result["ci95"], [2, 2])
    result = paired_scene_interval(np.array([0., 100.]), counts, replicates=2000)
    assert result["delta"] == pytest.approx(100 / 101)
    np.testing.assert_allclose(result["ci95"], [0, 1])


def artifacts(root):
    n = 3
    data = SimpleNamespace(base=np.zeros((n, 12, 2), np.float32), size=n * 11,
        arrays={"instance_token": np.array(["i0", "i1", "i2"]), "sample_token": np.array(["s0", "s1", "s2"]),
                "scene_name": np.array(["a", "b", "b"]), "future_xy": np.zeros((n, 12, 2), np.float32)},
        maps={"mask": np.zeros((n, 11, 1, 1), bool), "timestamp_us": np.tile(np.arange(1, 12) * 500000, (n, 1))},
        neighbors={"mask": np.zeros((n, 11, 1, 5), bool), "velocity_mask": np.zeros((n, 11, 1, 5), bool),
                   "source_timestamp_us": np.zeros((n, 11, 1, 5), np.int64), "timestamp_us": np.tile(np.arange(1, 12) * 500000, (n, 1))})
    paths = []
    for seed in (2027, 2028, 2029):
        path = root / str(seed)
        values = {name: np.full((n, 11, 3), 1. if name in ("frozen", "sliding") else 2. + (seed - 2027) * .1)
                  for name in STRATEGIES}
        write_batch_outputs(data, values, dict.fromkeys(STRATEGIES, 1.), path,
                            {"seed": seed, "evaluation_identity": "same", "training_config_sha256": "same"})
        paths.append(path)
    return paths


def test_report_requires_three_unique_aligned_seeds_and_complete_rows(tmp_path):
    paths = artifacts(tmp_path)
    result = report_ablation(paths, tmp_path / "report")
    assert result["scene_count"] == 2
    assert result["metrics"]["original"]["ade"]["seed_std"] == pytest.approx(.1)
    assert result["contrasts"]["map-original"]["ade"]["delta"] == 0
    with pytest.raises(ValueError, match="seed"):
        report_ablation(paths[:2], tmp_path / "missing")
    with pytest.raises(ValueError, match="seed"):
        report_ablation([paths[0]] * 3, tmp_path / "duplicate")
    summary = paths[1] / "summary.json"
    value = json.loads(summary.read_text())
    value["ordered_key_sha256"] = "different"
    summary.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="key|identity"):
        report_ablation(paths, tmp_path / "mismatch")


def test_report_rejects_modified_rows(tmp_path):
    paths = artifacts(tmp_path)
    with (paths[0] / "per_update.csv.gz").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="checksum"):
        report_ablation(paths, tmp_path / "report")


def test_report_rejects_missing_rows_even_with_updated_artifact_checksum(tmp_path):
    import pandas as pd

    from trajectory_prediction.utils.provenance import sha256_file
    paths = artifacts(tmp_path)
    path = paths[0] / "per_update.csv.gz"
    frame = pd.read_csv(path).iloc[:-1]
    frame.to_csv(path, index=False, compression="gzip")
    summary = paths[0] / "summary.json"
    value = json.loads(summary.read_text())
    value["per_update_sha256"] = sha256_file(path)
    summary.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="missing rows|key"):
        report_ablation(paths, tmp_path / "report")


def test_report_rejects_metrics_inconsistent_with_scene_stats(tmp_path):
    import pandas as pd

    from trajectory_prediction.utils.provenance import sha256_file
    paths = artifacts(tmp_path)
    path = paths[0] / "per_update.csv.gz"
    frame = pd.read_csv(path)
    frame.loc[0, "ade"] += 100
    frame.to_csv(path, index=False, compression="gzip")
    summary = paths[0] / "summary.json"
    value = json.loads(summary.read_text())
    value["per_update_sha256"] = sha256_file(path)
    summary.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="statistics differ"):
        report_ablation(paths, tmp_path / "report")

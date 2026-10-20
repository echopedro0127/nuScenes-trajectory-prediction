import json

import numpy as np
import pandas as pd
import pytest

from trajectory_prediction.analysis.radar_report import report_radar
from trajectory_prediction.online.radar_corrector import RadarDataset
from trajectory_prediction.online.radar_evaluation import STRATEGIES, write_radar_outputs
from trajectory_prediction.online.scaled_evaluation import benchmark_keys
from trajectory_prediction.utils.provenance import sha256_file


def fixtures(root):
    n = 3
    times = np.tile(2000000 + np.arange(1, 12) * 500000, (n, 1))
    arrays = {"instance_token": np.array(["i0", "i1", "i2"]), "sample_token": np.array(["s0", "s1", "s2"]),
              "scene_name": np.array(["a", "b", "b"]), "future_xy": np.zeros((n, 12, 2), np.float32)}
    cache = {"timestamp_us": times, "source_us": np.broadcast_to(times, (5, n, 11)), "fresh": np.ones((5, n, 11), bool),
             "present": np.ones((5, n, 11), bool), "point_count": np.ones((5, n, 11), np.int32), "radar_features": np.zeros((5, n, 11, 13), np.float32)}
    data = RadarDataset(arrays, cache, np.zeros((n, 12, 2), np.float32))
    paths = []
    for seed in (2027, 2028, 2029):
        path = root / str(seed)
        values = {}
        for strategy in STRATEGIES:
            value = 1. if strategy in ("frozen", "sliding", "filtered") else 2. + .1 * (seed - 2027) - (.1 if strategy == "radar" else 0)
            values[strategy] = np.full((5, n, 11, 3), value)
            values[strategy][..., 2] = 0
        write_radar_outputs(data, values, dict.fromkeys(STRATEGIES, 1.), path,
            {"seed": seed, "evaluation_identity": "same", "training_config_sha256": "same", "evaluation_source_sha256": "same"})
        frame = pd.read_csv(path / "per_update.csv.gz").iloc[:data.size]
        selected = benchmark_keys(data.size)
        keys = frame.iloc[selected][["condition", "instance_token", "sample_token", "scene_name", "update_step", "timestamp_us"]].copy()
        keys.insert(0, "flat_index", selected)
        keys.to_csv(path / "latency_keys.csv", index=False)
        fields = ("total_ms", "read_decode_ms", "transform_ms", "aggregate_ms", "radar_total_ms", "feature_transfer_ms", "head_return_ms")
        measurements = pd.DataFrame([{"mode": mode, "flat_index": int(index), **{field: value if field == "total_ms" else 0. for field in fields}}
            for mode, value in (("cached_radar", 1.), ("raw_radar", 10.)) for index in selected])
        measurements.to_csv(path / "latency_measurements.csv", index=False)
        latency = {"key_count": len(keys), "warmup": 16, "inference_batch_size": 1, "decoded_frame_cache": False,
            "keys_sha256": sha256_file(path / "latency_keys.csv"), "measurements_sha256": sha256_file(path / "latency_measurements.csv"),
            "device": "cpu", "device_name": "cpu", "torch_version": "test", "raw_scope": "test",
            "modes": {mode: {field: {"p50": value if field == "total_ms" else 0., "p95": value if field == "total_ms" else 0.} for field in fields} for mode, value in (("cached_radar", 1.), ("raw_radar", 10.))}}
        (path / "latency.json").write_text(json.dumps(latency))
        paths.append(path)
    return paths


def test_radar_report_scene_pairing_seed_std_and_recovery(tmp_path):
    paths = fixtures(tmp_path)
    result = report_radar(paths, tmp_path / "report")
    assert result["scene_count"] == 2
    assert result["metrics"]["noise"]["original"]["ade"]["seed_std"] == pytest.approx(.1)
    assert result["contrasts"]["noise"]["radar-original"]["ade"]["delta"] == pytest.approx(-.1)
    np.testing.assert_allclose(result["contrasts"]["noise"]["radar-original"]["ade"]["ci95"], [-.1, -.1])
    assert result["recovery"]["radar"]["unrecovered"] == 0
    with pytest.raises(ValueError, match="seed"):
        report_radar(paths[:2], tmp_path / "missing")
    with pytest.raises(ValueError, match="seed"):
        report_radar([paths[0]] * 3, tmp_path / "duplicate")


def test_radar_report_rejects_missing_rows_and_latency_mismatch(tmp_path):
    paths = fixtures(tmp_path)
    latency = paths[1] / "latency.json"
    values = json.loads(latency.read_text())
    values["warmup"] = 0
    latency.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="latency"):
        report_radar(paths, tmp_path / "bad-latency")
    values["warmup"] = 16
    latency.write_text(json.dumps(values))
    csv = paths[0] / "per_update.csv.gz"
    frame = pd.read_csv(csv).iloc[:-1]
    frame.to_csv(csv, index=False, compression="gzip")
    summary = paths[0] / "summary.json"
    values = json.loads(summary.read_text())
    values["sha256"][csv.name] = sha256_file(csv)
    summary.write_text(json.dumps(values))
    with pytest.raises(ValueError, match="row|key"):
        report_radar(paths, tmp_path / "missing-rows")


def test_report_requires_latency_total_and_stage_fields(tmp_path):
    paths = fixtures(tmp_path)
    for path in paths:
        filename = path / "latency.json"
        value = json.loads(filename.read_text())
        value["modes"] = {"cached_radar": {}, "raw_radar": {}}
        filename.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="latency"):
        report_radar(paths, tmp_path / "empty-latency")


def test_report_rejects_cross_seed_sensor_diagnostic_changes(tmp_path):
    paths = fixtures(tmp_path)
    root = paths[1]
    path = root / "per_update.csv.gz"
    frame = pd.read_csv(path)
    frame["radar_points"] = 999
    frame.to_csv(path, index=False, compression="gzip")
    path = root / "by_condition.csv"
    condition = pd.read_csv(path)
    condition["mean_points"] = 999
    condition.to_csv(path, index=False)
    path = root / "summary.json"
    value = json.loads(path.read_text())
    for name in ("per_update.csv.gz", "by_condition.csv"):
        value["sha256"][name] = sha256_file(root / name)
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="diagnostic"):
        report_radar(paths, tmp_path / "wrong-points")

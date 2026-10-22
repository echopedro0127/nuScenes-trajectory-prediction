"""Strict P6.4 numeric artifact publication and prediction replay comparison."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from trajectory_prediction.online.horizon_gate import VARIANTS
from trajectory_prediction.online.innovation_consistency import ARMS
from trajectory_prediction.utils.provenance import sha256_file

BASELINES = ("sliding", "p6_learned", "finetune_0", "time_age_gate", "consistent_short", "consistent_010")
HISTORICAL = (*BASELINES, *VARIANTS)
STRATEGIES = (*HISTORICAL, *ARMS)
LATENCY_FILES = ("latency.json", "latency_keys.csv", "latency_measurements.csv")


def validate_numeric(results, weights, n):
    if tuple(results) != STRATEGIES or tuple(weights) != STRATEGIES:
        raise ValueError("complete ordered horizon strategy set required")
    mask = np.arange(12)[None] >= np.arange(1, 12)[:, None]
    for name in STRATEGIES:
        a, w = results[name], weights[name]
        if (a.shape != (6, n, 11, 3) or a.dtype != np.float64 or w.shape != (6, n, 11, 12)
            or w.dtype != np.float32 or not np.isfinite(a).all() or not np.isfinite(w).all()
            or np.any(a < 0) or np.any((w < 0) | (w > 1)) or np.any(w[..., ~mask] != 0)):
            raise ValueError("horizon artifact shape/dtype/range mismatch: "+name)


def publish_run(root, results, weights, tables, metadata):
    validate_numeric(results, weights, metadata["sample_count"])
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root/"summary.json").unlink(missing_ok=True)
    np.savez_compressed(root/"metrics.npz", **results)
    np.savez_compressed(root/"weights.npz", **weights)
    for filename, frame in tables.items():
        frame.to_csv(root/filename, index=False)
    names = ("metrics.npz", "weights.npz", *tables, *(LATENCY_FILES if metadata.get("latency_profile") else ()))
    summary = {"schema_version": 1, "metadata": metadata, "strategies": list(STRATEGIES),
        "metric_shape": [6, metadata["sample_count"], 11, 3], "weight_shape": [6, metadata["sample_count"], 11, 12],
        "row_count": 6*metadata["sample_count"]*11*len(STRATEGIES),
        "sha256": {name: sha256_file(root/name) for name in names}}
    temporary = root/".summary.tmp"
    temporary.write_text(json.dumps(summary, indent=2)+"\n")
    temporary.replace(root/"summary.json")
    return summary


def verify_run(root, results, weights, tables, metadata):
    validate_numeric(results, weights, metadata["sample_count"])
    root = Path(root)
    saved = json.loads((root/"summary.json").read_text())
    names = {"metrics.npz", "weights.npz", *tables, *(LATENCY_FILES if metadata.get("latency_profile") else ())}
    if (set(saved) != {"schema_version", "metadata", "strategies", "metric_shape", "weight_shape", "row_count", "sha256"}
        or saved["schema_version"] != 1 or saved["metadata"] != metadata or saved["strategies"] != list(STRATEGIES)
        or saved["metric_shape"] != [6, metadata["sample_count"], 11, 3]
        or saved["weight_shape"] != [6, metadata["sample_count"], 11, 12]
        or saved["row_count"] != 6*metadata["sample_count"]*11*len(STRATEGIES)
        or set(saved["sha256"]) != names):
        raise ValueError("horizon replay schema/metadata mismatch")
    for filename, checksum in saved["sha256"].items():
        if sha256_file(root/filename) != checksum:
            raise ValueError("horizon replay artifact checksum mismatch")
    for filename, arrays in (("metrics.npz", results), ("weights.npz", weights)):
        with np.load(root/filename, allow_pickle=False) as artifact:
            if set(artifact.files) != set(arrays):
                raise ValueError("horizon replay strategy set mismatch")
            for name, value in arrays.items():
                actual = artifact[name]
                if (actual.shape != value.shape or actual.dtype != value.dtype
                    or not np.isfinite(actual).all() or not np.allclose(actual, value, atol=1e-8, rtol=1e-7)):
                    raise ValueError("horizon replay prediction mismatch: "+name)
    for filename, frame in tables.items():
        try:
            pd.testing.assert_frame_equal(pd.read_csv(root/filename, float_precision="round_trip"), frame,
                check_dtype=False, atol=1e-10, rtol=1e-9)
        except AssertionError as error:
            raise ValueError("horizon replay table mismatch: "+filename) from error
    return saved

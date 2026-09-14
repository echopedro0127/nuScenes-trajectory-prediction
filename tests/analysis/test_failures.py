from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from trajectory_prediction.analysis.failures import analyze_arrays, select_worst_cases


def test_worst_case_selection_deduplicates_rankings() -> None:
    frame = pd.DataFrame(
        {
            "instance_token": ["a", "b", "c"],
            "sample_token": ["1", "2", "3"],
            "ade": [9.0, 8.0, 1.0],
            "fde": [2.0, 10.0, 1.0],
        }
    )

    selected = select_worst_cases(frame, top_k=1)

    assert set(selected["instance_token"]) == {"a", "b"}
    assert list(selected["rank_source"]) == ["ade", "fde"]


def test_analyze_arrays_explodes_overlapping_maneuver_tags(tmp_path: Path) -> None:
    future = np.zeros((2, 12, 2), dtype=np.float32)
    future[1, :, 1] = np.arange(1, 13, dtype=np.float32)
    prediction = future.copy()
    prediction[0, :-1, 0] = 3.0
    prediction[1, -1, 0] = 3.0
    arrays = {
        "future_xy": future,
        "instance_token": np.asarray(["i0", "i1"]),
        "sample_token": np.asarray(["s0", "s1"]),
        "scene_name": np.asarray(["scene0", "scene1"]),
    }
    thresholds = {
        "low_speed_mps": 2.0,
        "turn_yaw_deg": 15.0,
        "high_curvature_yaw_deg": 45.0,
        "acceleration_mps2": 1.0,
    }

    per_sample, maneuver, worst = analyze_arrays(
        arrays, prediction, thresholds=thresholds, top_k=1
    )

    assert len(per_sample) >= 2
    assert set(maneuver["maneuver"]) >= {"low_speed", "straight"}
    assert set(worst["rank_source"]) == {"ade", "fde"}
    per_sample.to_csv(tmp_path / "per_sample_analysis.csv", index=False)
    maneuver.to_csv(tmp_path / "maneuver_metrics.csv", index=False)
    json.dumps(worst.to_dict(orient="records"), allow_nan=False)


def test_analyze_arrays_rejects_identifier_mismatch() -> None:
    arrays = {
        "future_xy": np.zeros((1, 12, 2), dtype=np.float32),
        "instance_token": np.asarray(["i0"]),
        "sample_token": np.asarray([]),
        "scene_name": np.asarray(["scene0"]),
    }
    with pytest.raises(ValueError, match="identifier"):
        analyze_arrays(arrays, np.zeros((1, 12, 2), dtype=np.float32), {})

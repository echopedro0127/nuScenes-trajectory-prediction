from __future__ import annotations

import numpy as np
import pytest

from trajectory_prediction.online.evaluator import aggregate_metrics, update_metrics


def test_identical_update_has_zero_error_and_jitter() -> None:
    target = np.array([[0, 1], [0, 2]], np.float32)

    values = update_metrics(target.copy(), target, target.copy(), 1.5)

    assert values == {"ade": 0.0, "fde": 0.0, "jitter": 0.0, "latency_ms": 1.5}


def test_aggregate_metrics_reports_percentiles() -> None:
    rows = [
        {"strategy": "frozen", "ade": 1.0, "fde": 2.0, "jitter": 0.5, "latency_ms": 1.0},
        {"strategy": "frozen", "ade": 3.0, "fde": 4.0, "jitter": 1.5, "latency_ms": 3.0},
    ]

    summary = aggregate_metrics(rows)

    assert summary["sample_count"] == 2
    assert summary["ade"] == 2.0
    assert summary["fde"] == 3.0
    assert summary["latency_p50_ms"] == 2.0
    assert summary["latency_p95_ms"] == 2.9


@pytest.mark.parametrize(
    ("current", "target", "previous"),
    [
        (np.zeros((2, 2), np.float32), np.zeros((3, 2), np.float32), np.zeros((2, 2), np.float32)),
        (np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float32), np.zeros((2, 3), np.float32)),
        (np.full((2, 2), np.nan, np.float32), np.zeros((2, 2), np.float32), np.zeros((2, 2), np.float32)),
    ],
)
def test_update_metrics_rejects_invalid_arrays(
    current: np.ndarray, target: np.ndarray, previous: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        update_metrics(current, target, previous, 1.0)

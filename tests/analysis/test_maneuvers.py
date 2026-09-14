from __future__ import annotations

import numpy as np
import pytest

from trajectory_prediction.analysis.maneuvers import describe_motion, tag_maneuvers

THRESHOLDS = {
    "low_speed_mps": 2.0,
    "turn_yaw_deg": 15.0,
    "high_curvature_yaw_deg": 45.0,
    "acceleration_mps2": 1.0,
}


def test_stationary_trajectory_is_low_speed_and_straight() -> None:
    future = np.zeros((12, 2), dtype=np.float32)

    stats = describe_motion(future, dt=0.5)

    assert {"low_speed", "straight"} <= set(tag_maneuvers(stats, THRESHOLDS))
    assert stats["mean_speed_mps"] == 0.0
    assert stats["yaw_change_deg"] == 0.0


def test_quarter_turn_is_high_curvature() -> None:
    angles = np.linspace(0.0, np.pi / 2.0, 12)
    future = np.stack([10.0 * np.sin(angles), 10.0 * (1.0 - np.cos(angles))], axis=1)

    stats = describe_motion(future.astype(np.float32), dt=0.5)

    assert abs(stats["yaw_change_deg"]) >= 45.0
    assert "high_curvature" in tag_maneuvers(stats, THRESHOLDS)


def test_acceleration_and_deceleration_tags_are_exclusive() -> None:
    accelerating = np.cumsum(np.arange(1.0, 13.0, dtype=np.float32)[:, None], axis=0)
    accelerating = np.concatenate([accelerating, np.zeros((12, 1), dtype=np.float32)], axis=1)
    decelerating = accelerating[::-1].copy()

    assert "acceleration" in tag_maneuvers(describe_motion(accelerating, 0.5), THRESHOLDS)
    assert "deceleration" in tag_maneuvers(describe_motion(decelerating, 0.5), THRESHOLDS)


@pytest.mark.parametrize(
    ("future_xy", "dt"),
    [
        (np.zeros((0, 2), dtype=np.float32), 0.5),
        (np.zeros((12, 3), dtype=np.float32), 0.5),
        (np.full((12, 2), np.nan, dtype=np.float32), 0.5),
        (np.zeros((12, 2), dtype=np.float32), 0.0),
        (np.zeros((12, 2), dtype=np.float32), np.nan),
    ],
)
def test_describe_motion_rejects_invalid_inputs(future_xy: np.ndarray, dt: float) -> None:
    with pytest.raises(ValueError):
        describe_motion(future_xy, dt)


def test_tag_maneuvers_requires_all_thresholds() -> None:
    with pytest.raises(ValueError, match="threshold"):
        tag_maneuvers(
            {
                "mean_speed_mps": 0.0,
                "max_speed_mps": 0.0,
                "mean_acceleration_mps2": 0.0,
                "yaw_change_deg": 0.0,
                "displacement_m": 0.0,
            },
            {},
        )

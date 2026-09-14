from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Real

import numpy as np

_STAT_KEYS = ("mean_speed_mps", "max_speed_mps", "mean_acceleration_mps2", "yaw_change_deg", "displacement_m")
_THRESHOLD_KEYS = ("low_speed_mps", "turn_yaw_deg", "high_curvature_yaw_deg", "acceleration_mps2")


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi), preserving a vectorized NumPy contract."""
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def _validate_dt(dt: object) -> float:
    if isinstance(dt, bool) or not isinstance(dt, Real):
        raise ValueError("dt must be a finite positive number")  # noqa: TRY004
    value = float(dt)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError("dt must be a finite positive number")
    return value


def describe_motion(future_xy: np.ndarray, dt: float) -> dict[str, float]:
    """Describe a local-frame future trajectory starting at the current origin.

    The first displacement is measured from the current agent at ``(0, 0)`` to the
    first future point. Very small segments are excluded from yaw-change estimates
    to keep stopped agents from producing arbitrary heading changes.
    """
    step_seconds = _validate_dt(dt)
    points = np.asarray(future_xy)
    if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] != 2:
        raise ValueError("future_xy must have shape [T >= 1, 2]")
    if not np.issubdtype(points.dtype, np.floating):
        raise ValueError("future_xy must be a floating-point array")
    if not np.isfinite(points).all():
        raise ValueError("future_xy must contain only finite values")

    points64 = points.astype(np.float64, copy=False)
    with_origin = np.concatenate((np.zeros((1, 2), dtype=np.float64), points64), axis=0)
    delta = np.diff(with_origin, axis=0)
    speed = np.linalg.norm(delta, axis=1) / step_seconds
    valid = speed > 0.2
    if speed.size >= 2:
        headings = np.arctan2(delta[:, 1], delta[:, 0])
        heading_delta = _wrap(np.diff(headings))
        valid_turn = valid[1:] & valid[:-1]
        yaw_change = float(heading_delta[valid_turn].sum()) if valid_turn.any() else 0.0
        acceleration = np.diff(speed) / step_seconds
        mean_acceleration = float(acceleration.mean())
    else:
        yaw_change = 0.0
        mean_acceleration = 0.0

    return {
        "mean_speed_mps": float(speed.mean()),
        "max_speed_mps": float(speed.max()),
        "mean_acceleration_mps2": mean_acceleration,
        "yaw_change_deg": float(np.degrees(yaw_change)),
        "displacement_m": float(np.linalg.norm(points64[-1])),
    }


def _finite_stat(stats: Mapping[str, object], key: str) -> float:
    value = stats.get(key)
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
        raise ValueError(f"motion stats must contain finite numeric {key}")
    return float(value)


def _thresholds(thresholds: Mapping[str, object]) -> dict[str, float]:
    values: dict[str, float] = {}
    for key in _THRESHOLD_KEYS:
        value = thresholds.get(key)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(float(value)):
            raise ValueError(f"threshold {key} must be finite numeric")
        numeric = float(value)
        if numeric < 0.0:
            raise ValueError(f"threshold {key} must be nonnegative")
        values[key] = numeric
    if values["high_curvature_yaw_deg"] < values["turn_yaw_deg"]:
        raise ValueError("high_curvature_yaw_deg must be >= turn_yaw_deg")
    return values


def tag_maneuvers(stats: Mapping[str, object], thresholds: Mapping[str, object]) -> list[str]:
    """Assign overlapping maneuver tags using explicit, validated thresholds."""
    if not isinstance(stats, Mapping):
        raise ValueError("motion stats must be a mapping")  # noqa: TRY004
    if not isinstance(thresholds, Mapping):
        raise ValueError("thresholds must be a mapping")  # noqa: TRY004
    values = {key: _finite_stat(stats, key) for key in _STAT_KEYS}
    limits = _thresholds(thresholds)

    tags: list[str] = []
    if values["mean_speed_mps"] < limits["low_speed_mps"]:
        tags.append("low_speed")
    yaw = abs(values["yaw_change_deg"])
    if yaw >= limits["high_curvature_yaw_deg"]:
        tags.append("high_curvature")
    elif yaw >= limits["turn_yaw_deg"]:
        tags.append("turn")
    else:
        tags.append("straight")
    acceleration = values["mean_acceleration_mps2"]
    if acceleration >= limits["acceleration_mps2"]:
        tags.append("acceleration")
    elif acceleration <= -limits["acceleration_mps2"]:
        tags.append("deceleration")
    return tags

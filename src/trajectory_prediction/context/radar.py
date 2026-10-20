"""Radar coordinate transforms and position-gated robust statistics, without labels."""
import math
from dataclasses import dataclass

import numpy as np
from pyquaternion import Quaternion


@dataclass(frozen=True)
class RadarConfig:
    radius_m: float = 4.
    window_us: int = 2000000
    sweeps_per_channel: int = 3

    def __post_init__(self):
        if isinstance(self.radius_m, bool) or not isinstance(self.radius_m, (int, float)) or not math.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("positive finite radar radius required")
        if type(self.window_us) is not int or not 0 < self.window_us <= 2000000 or type(self.sweeps_per_channel) is not int or self.sweeps_per_channel < 1:
            raise ValueError("invalid radar window/sweep limit")


@dataclass(frozen=True)
class RadarView:
    features: np.ndarray
    point_count: int
    frame_tokens: tuple = ()
    frame_source_us: tuple = ()
    frame_available_us: tuple = ()
    pdh0_median: float = 0.

    def __post_init__(self):
        features = np.array(self.features, np.float32, copy=True)
        if features.shape != (13,) or not np.isfinite(features).all() or type(self.point_count) is not int or self.point_count < 0 or (not self.point_count and features.any()):
            raise ValueError("invalid radar statistics/mask")
        if len(self.frame_tokens) != len(self.frame_source_us) or len(self.frame_tokens) != len(self.frame_available_us):
            raise ValueError("radar frame timeline shape mismatch")
        features.setflags(write=False)
        object.__setattr__(self, "features", features)

    @property
    def present(self):
        return self.point_count > 0


def transform_returns(points, calibration, pose, anchor_xy, anchor_yaw):
    points = np.asarray(points, float)
    anchor_xy = np.asarray(anchor_xy, float)
    if points.ndim != 2 or points.shape[0] != 18 or anchor_xy.shape != (2,) or not np.isfinite(points).all() or not np.isfinite(anchor_xy).all() or not math.isfinite(anchor_yaw):
        raise ValueError("invalid radar coordinates")
    sensor_rotation = Quaternion(calibration["rotation"]).rotation_matrix
    ego_rotation = Quaternion(pose["rotation"]).rotation_matrix
    global_xyz = (points[:3].T @ sensor_rotation.T + np.asarray(calibration["translation"])) @ ego_rotation.T + np.asarray(pose["translation"])
    velocity = np.column_stack((points[8:10].T, np.zeros(points.shape[1]))) @ (ego_rotation @ sensor_rotation).T
    angle = np.pi / 2 - anchor_yaw
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    xy = (global_xyz[:, :2] - anchor_xy) @ rotation.T
    vxy = velocity[:, :2] @ rotation.T
    return np.column_stack((xy, vxy, points[5], points[10], points[15])).astype(np.float32)


def aggregate_returns(points, source_us, channels, center, timestamp_us, config, frames=()):
    points, source_us, channels, center = np.asarray(points), np.asarray(source_us), np.asarray(channels), np.asarray(center)
    if points.ndim != 2 or points.shape[1] != 7 or source_us.shape != (len(points),) or channels.shape != source_us.shape or center.shape != (2,) or not np.isfinite(points).all() or not np.isfinite(center).all():
        raise ValueError("invalid radar aggregation inputs")
    if not np.issubdtype(source_us.dtype, np.integer) or np.any(source_us > timestamp_us) or np.any(timestamp_us - source_us > config.window_us) or np.any(source_us < 0):
        raise ValueError("radar points outside causal window")
    valid = np.linalg.norm(points[:, :2] - center, axis=1) <= config.radius_m
    selected = points[valid]
    features = np.zeros(13, np.float32)
    count = len(selected)
    if count:
        velocity = np.median(selected[:, 2:4], axis=0)
        age = (timestamp_us - source_us[valid]) / 1e6
        features[0] = np.log1p(count) / np.log(129)
        features[1:3] = np.median(selected[:, :2] - center, axis=0) / config.radius_m
        features[3:5] = velocity / 10.
        features[5:7] = np.median(np.abs(selected[:, 2:4] - velocity), axis=0) / 10.
        features[7] = np.median(selected[:, 4]) / 20.
        features[8] = (selected[:, 5] == 1).mean()
        features[9:12] = np.array([age.min(), np.median(age), age.max()]) / 2.
        features[12] = len(np.unique(channels[valid])) / 5.
    return RadarView(features, count, tuple(frame.token for frame in frames),
        tuple(frame.source_timestamp_us for frame in frames), tuple(frame.available_timestamp_us for frame in frames),
        float(np.median(selected[:, 6])) if count else 0.)

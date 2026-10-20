"""Static lane geometry selected only from the current causal observation."""
import math
from dataclasses import dataclass

import numpy as np

from trajectory_prediction.data.geometry import agent_to_global, global_to_agent
from trajectory_prediction.online.observations import ObservationView


@dataclass(frozen=True)
class MapConfig:
    radius_m: float = 50.0
    max_lanes: int = 8
    points_per_lane: int = 20
    resolution_m: float = 2.0

    def __post_init__(self):
        for name in ("radius_m", "resolution_m"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("max_lanes", "points_per_lane"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True)
class MapView:
    features: np.ndarray
    mask: np.ndarray
    lane_tokens: tuple[str, ...]

    def __post_init__(self):
        features = np.array(self.features, dtype=np.float32, copy=True)
        mask = np.array(self.mask, copy=True)
        tokens = tuple(self.lane_tokens)
        if features.ndim != 3 or features.shape[-1] != 4 or min(features.shape[:2]) < 1:
            raise ValueError("map features must have shape [K, P, 4]")
        if mask.shape != features.shape[:2] or mask.dtype != np.bool_:
            raise ValueError("map mask must be bool [K, P]")
        if len(tokens) != len(features) or any(not isinstance(t, str) for t in tokens):
            raise ValueError("lane tokens must match K")
        if not np.isfinite(features).all() or np.any(features[~mask] != 0):
            raise ValueError("map features must be finite with zero padding")
        if any(bool(mask[k].any()) != bool(token) for k, token in enumerate(tokens)):
            raise ValueError("lane tokens must agree with mask")
        features.setflags(write=False)
        mask.setflags(write=False)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "lane_tokens", tokens)


class LaneMap:
    def __init__(self, lanes: dict[str, np.ndarray], config: MapConfig):
        self.config = config
        self.lanes = {}
        self.cells = {}
        for token, poses in lanes.items():
            points = np.asarray(poses, dtype=np.float64)
            if not token or points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
                raise ValueError("lane poses must be finite [P, 3] with nonempty token")
            if not len(points):
                continue
            self.lanes[token] = points.copy()
            for cell in set(map(tuple, np.floor(points[:, :2] / config.radius_m).astype(int))):
                self.cells.setdefault(cell, set()).add(token)

    def query(self, observation: ObservationView, anchor_xy: np.ndarray, anchor_yaw: float) -> MapView:
        if not np.isfinite(anchor_xy).all() or not np.isfinite(anchor_yaw):
            raise ValueError("anchor must be finite")
        current = agent_to_global(observation.observed_position, anchor_xy, anchor_yaw).astype(float)
        cell = np.floor(current / self.config.radius_m).astype(int)
        candidates = set()
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                candidates.update(self.cells.get((cell[0] + dx, cell[1] + dy), ()))
        ranked = []
        for token in candidates:
            distance = np.linalg.norm(self.lanes[token][:, :2] - current, axis=1)
            nearest = int(distance.argmin())
            if distance[nearest] <= self.config.radius_m:
                ranked.append((float(distance[nearest]), token, nearest))
        features = np.zeros((self.config.max_lanes, self.config.points_per_lane, 4), np.float32)
        mask = np.zeros(features.shape[:2], bool)
        tokens = [""] * self.config.max_lanes
        angle = np.pi / 2 - anchor_yaw
        rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        for k, (_, token, nearest) in enumerate(sorted(ranked)[:self.config.max_lanes]):
            poses = self.lanes[token]
            start = max(0, min(nearest - self.config.points_per_lane // 2,
                               len(poses) - self.config.points_per_lane))
            poses = poses[start:start + self.config.points_per_lane]
            size = len(poses)
            features[k, :size, :2] = global_to_agent(poses[:, :2], anchor_xy, anchor_yaw)
            directions = np.column_stack((np.cos(poses[:, 2]), np.sin(poses[:, 2])))
            features[k, :size, 2:] = directions @ rotation.T
            mask[k, :size] = True
            tokens[k] = token
        return MapView(features, mask, tuple(tokens))


class MapProvider:
    """Load and discretize each required map once; never read sensor files."""
    def __init__(self, dataroot, config):
        self.dataroot = dataroot
        self.config = config
        self.maps = {}

    def get(self, location):
        if location not in self.maps:
            from nuscenes.map_expansion.map_api import NuScenesMap
            api = NuScenesMap(dataroot=str(self.dataroot), map_name=location)
            tokens = [item["token"] for item in api.lane + api.lane_connector]
            discrete = api.discretize_lanes(tokens, self.config.resolution_m)
            self.maps[location] = LaneMap(
                {token: np.asarray(poses).reshape(-1, 3) for token, poses in discrete.items()}, self.config)
        return self.maps[location]

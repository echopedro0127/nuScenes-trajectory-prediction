"""Causal oracle interaction histories without interpolation or future annotation access."""
import json
import math
from bisect import bisect_right
from dataclasses import dataclass

import numpy as np

from trajectory_prediction.context.schema import validate_timestamp_us
from trajectory_prediction.data.geometry import global_to_agent
from trajectory_prediction.utils.provenance import sha256_file

CATEGORY_PREFIXES = ("vehicle.", "human.pedestrian.")


@dataclass(frozen=True)
class NeighborConfig:
    radius_m: float = 50.0
    max_neighbors: int = 8
    max_age_us: int = 500_000
    latency_us: int = 0

    def __post_init__(self):
        if isinstance(self.radius_m, bool) or not isinstance(self.radius_m, (int, float)) or not math.isfinite(self.radius_m) or self.radius_m <= 0:
            raise ValueError("radius must be finite and positive")
        if type(self.max_neighbors) is not int or not 1 <= self.max_neighbors <= 8:
            raise ValueError("max_neighbors must be in [1, 8]")
        validate_timestamp_us(self.max_age_us)
        validate_timestamp_us(self.latency_us)
        if self.max_age_us > 500_000:
            raise ValueError("max age cannot exceed one history interval")


@dataclass(frozen=True)
class NeighborView:
    features: np.ndarray
    mask: np.ndarray
    velocity_mask: np.ndarray
    instance_tokens: tuple[str, ...]
    annotation_tokens: np.ndarray
    source_timestamp_us: np.ndarray
    available_timestamp_us: np.ndarray
    timestamp_us: int

    def __post_init__(self):
        validate_timestamp_us(self.timestamp_us)
        features = np.array(self.features, dtype=np.float32, copy=True)
        if features.ndim != 3 or features.shape[1:] != (5, 6) or not 1 <= len(features) <= 8:
            raise ValueError("neighbor features must be [N<=8, 5, 6]")
        shape = features.shape[:2]
        mask = np.array(self.mask, copy=True)
        velocity = np.array(self.velocity_mask, copy=True)
        source = np.array(self.source_timestamp_us, copy=True)
        available = np.array(self.available_timestamp_us, copy=True)
        annotations = np.array(self.annotation_tokens, copy=True)
        for value in (mask, velocity):
            if value.shape != shape or value.dtype != np.bool_:
                raise ValueError("neighbor masks must be bool [N, 5]")
        for value in (source, available):
            if value.shape != shape or value.dtype != np.int64:
                raise ValueError("neighbor timestamps must be int64 [N, 5]")
        tokens = tuple(self.instance_tokens)
        if len(tokens) != len(features) or any(not isinstance(t, str) for t in tokens):
            raise ValueError("neighbor tokens must match N")
        if annotations.shape != shape or annotations.dtype.kind != "U" or np.any((annotations != "") != mask):
            raise ValueError("annotation tokens must agree with history mask")
        present = [i for i, token in enumerate(tokens) if token]
        if present != list(range(len(present))) or len({tokens[i] for i in present}) != len(present):
            raise ValueError("neighbor tokens must be unique with trailing padding")
        if any(bool(mask[i, -1]) != bool(token) for i, token in enumerate(tokens)):
            raise ValueError("neighbor current mask must agree with token")
        if np.any(mask[~mask[:, -1]]):
            raise ValueError("absent current neighbors cannot retain history")
        if not np.isfinite(features).all() or np.any(features[~mask] != 0):
            raise ValueError("neighbor features must be finite with zero padding")
        if np.any(source[~mask] != 0) or np.any(available[~mask] != 0):
            raise ValueError("missing neighbor timestamps must be zero")
        if np.any(source[mask] < max(0, self.timestamp_us - 2_000_000)) or np.any(available[mask] < source[mask]) or np.any(available[mask] > self.timestamp_us):
            raise ValueError("neighbor history must be available within past two seconds")
        slots = self.timestamp_us - np.arange(4, -1, -1) * 500_000
        if np.any((source > slots[None]) & mask):
            raise ValueError("neighbor source cannot follow its history slot")
        eligible = np.zeros(shape, bool)
        differences = source[:, 1:] - source[:, :-1]
        eligible[:, 1:] = mask[:, 1:] & mask[:, :-1] & (differences > 0) & (differences <= 1_000_000)
        if np.any(velocity & ~eligible) or np.any(features[..., 2:4][~velocity] != 0):
            raise ValueError("neighbor velocity requires adjacent valid past observations")
        expected_velocity = np.diff(features[..., :2], axis=1) / np.maximum(differences, 1)[..., None] * 1e6
        if not np.allclose(features[:, 1:, 2:4][velocity[:, 1:]],
                           expected_velocity[velocity[:, 1:]], atol=1e-4, rtol=1e-4):
            raise ValueError("neighbor velocity must match real-dt backward differences")
        category = features[..., 4:]
        if np.any((category[mask] != 0) & (category[mask] != 1)) or np.any(category.sum(-1)[mask] != 1):
            raise ValueError("valid neighbors require fixed category one-hot")
        for i in present:
            times = source[i, mask[i]]
            if np.any(np.diff(times) <= 0):
                raise ValueError("neighbor histories cannot duplicate or reverse source times")
            if not np.all(category[i, mask[i]] == category[i, -1]):
                raise ValueError("neighbor category cannot change across history")
        order = [(float(np.linalg.norm(features[i, -1, :2])), tokens[i]) for i in present]
        if order != sorted(order):
            raise ValueError("neighbors must be sorted by current distance then token")
        for name, value in (("features", features), ("mask", mask), ("velocity_mask", velocity),
                            ("annotation_tokens", annotations), ("source_timestamp_us", source),
                            ("available_timestamp_us", available)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "instance_tokens", tokens)


class NeighborIndex:
    def __init__(self, samples, annotations, categories_by_instance, scene_names, config=None):
        self.config = config or NeighborConfig()
        self.samples = {record["token"]: record for record in samples}
        self.categories = dict(categories_by_instance)
        self.annotations = {sample: {} for sample in self.samples}
        for record in annotations:
            self.annotations[record["sample_token"]][record["instance_token"]] = record
        self.scene_samples = {}
        for sample in samples:
            validate_timestamp_us(sample["timestamp"])
            self.scene_samples.setdefault(scene_names[sample["scene_token"]], []).append(sample["token"])
        self.times = {}
        for scene, tokens in self.scene_samples.items():
            tokens.sort(key=lambda token: self.samples[token]["timestamp"])
            self.times[scene] = [self.samples[token]["timestamp"] for token in tokens]
            if len(set(self.times[scene])) != len(tokens):
                raise ValueError("scene sample times must be unique")
        self.extra_hashes = {}

    @classmethod
    def from_metadata(cls, index, config):
        tables, hashes = {}, {}
        for name in ("instance", "category"):
            path = index.root / index.version / f"{name}.json"
            records = json.loads(path.read_text(encoding="utf-8"))
            tables[name] = {record["token"]: record for record in records}
            hashes[f"{index.version}/{name}.json"] = sha256_file(path)
        categories = {token: tables["category"][record["category_token"]]["name"]
                      for token, record in tables["instance"].items()}
        result = cls(list(index.tables["sample"].values()), list(index.tables["sample_annotation"].values()),
                     categories, {t: r["name"] for t, r in index.tables["scene"].items()}, config)
        result.extra_hashes = hashes
        return result

    def _sample(self, scene, slot, query_time):
        limit = min(slot, query_time - self.config.latency_us)
        times = self.times.get(scene, [])
        k = bisect_right(times, limit) - 1
        if k < 0 or slot - times[k] > self.config.max_age_us or times[k] < query_time - 2_000_000:
            return None
        return self.scene_samples[scene][k]

    def query(self, observation, target_instance, scene, anchor_xy, anchor_yaw):
        timestamp = observation.timestamp_us
        if not np.isfinite(anchor_xy).all() or not np.isfinite(anchor_yaw):
            raise ValueError("anchor must be finite")
        current_sample = self._sample(scene, timestamp, timestamp)
        candidates = []
        for token, annotation in self.annotations.get(current_sample, {}).items():
            category = self.categories.get(token, "")
            category_id = next((i for i, prefix in enumerate(CATEGORY_PREFIXES) if category.startswith(prefix)), None)
            if token == target_instance or category_id is None:
                continue
            relative = global_to_agent(annotation["translation"][:2], anchor_xy, anchor_yaw) - observation.observed_position
            distance = float(np.linalg.norm(relative))
            if distance <= self.config.radius_m:
                candidates.append((distance, token, category_id))
        n = self.config.max_neighbors
        features, mask, velocity = np.zeros((n, 5, 6), np.float32), np.zeros((n, 5), bool), np.zeros((n, 5), bool)
        source, available = np.zeros((n, 5), np.int64), np.zeros((n, 5), np.int64)
        annotations = np.full((n, 5), "", dtype="U64")
        tokens = [""] * n
        slots = [timestamp - j * 500_000 for j in range(4, -1, -1)]
        history_samples = [self._sample(scene, slot, timestamp) for slot in slots]
        # Keep a repeated sample only at its newest slot.
        for j, token in enumerate(history_samples):
            if token in history_samples[j + 1:]:
                history_samples[j] = None
        for i, (_, token, category_id) in enumerate(sorted(candidates)[:n]):
            tokens[i] = token
            for j, sample_token in enumerate(history_samples):
                annotation = self.annotations.get(sample_token, {}).get(token)
                if annotation is None:
                    continue
                features[i, j, :2] = global_to_agent(annotation["translation"][:2], anchor_xy, anchor_yaw) - observation.observed_position
                features[i, j, 4 + category_id] = 1
                mask[i, j] = True
                source[i, j] = self.samples[sample_token]["timestamp"]
                available[i, j] = source[i, j] + self.config.latency_us
                annotations[i, j] = annotation["token"]
                if j > 0 and mask[i, j - 1] and 0 < source[i, j] - source[i, j - 1] <= 1_000_000:
                    features[i, j, 2:4] = (features[i, j, :2] - features[i, j - 1, :2]) / ((source[i, j] - source[i, j - 1]) / 1e6)
                    velocity[i, j] = True
        return NeighborView(features, mask, velocity, tuple(tokens), annotations, source, available, timestamp)

"""Schema-checked NPY candidate caches and separate input/label batch access."""
import json
from pathlib import Path

import numpy as np
import torch

from trajectory_prediction.context.cache import IDENTIFIERS, digest_json, key_hash
from trajectory_prediction.online.quality_gate import (
    CONDITIONS,
    GateConfig,
    packet_schedule,
    quality_features,
)
from trajectory_prediction.utils.provenance import sha256_file


def validate_gate_arrays(data, contract):
    config = GateConfig(**contract["gate_config"])
    n, c = len(data["instance_token"]), len(CONDITIONS)
    shapes = {**{name: (n,) for name in IDENTIFIERS}, "base": (n, 12, 2), "target": (n, 12, 2), "history": (n, 5, 2),
        "anchor_timestamp_us": (n,), "timestamp_us": (n, 11), "position": (n, 11, 2), "velocity": (n, 11, 2), "previous_velocity": (n, 11, 2),
        "cv": (n, 11, 12, 2), "filtered": (n, 11, 12, 2), "original": (n, 11, 12, 2), "map": (c, n, 11, 12, 2),
        "quality": (c, n, 11, 8), **{name: (c, n, 11) for name in ("valid", "packet_step", "source_us", "available_us", "age_s", "coverage")}}
    if set(data) != set(shapes) or n < 1:
        raise ValueError("gate array schema mismatch")
    for name, shape in shapes.items():
        value = data[name]
        expected = None if name in IDENTIFIERS else np.bool_ if name == "valid" else np.int64 if name in ("anchor_timestamp_us", "timestamp_us", "packet_step", "source_us", "available_us") else np.float32
        if value.shape != shape or (expected is None and value.dtype.kind != "U") or (expected is not None and value.dtype != expected) or (expected is not None and not np.isfinite(value).all()):
            raise ValueError("gate shape/dtype mismatch: " + name)
    if any(np.any(data[name] == "") for name in IDENTIFIERS) or len(set(zip(data["instance_token"].tolist(), data["sample_token"].tolist(), strict=True))) != n or ("keys_sha256" in contract and key_hash(data) != contract["keys_sha256"]):
        raise ValueError("gate sample key mismatch")
    anchor = data["anchor_timestamp_us"]
    if np.any(anchor < 2000000) or not np.array_equal(data["timestamp_us"], anchor[:, None] + np.arange(1, 12) * 500000):
        raise ValueError("gate anchor timeline mismatch")
    if not np.array_equal(data["position"], data["target"][:, :11]):
        raise ValueError("gate causal observed position mismatch")
    previous = np.concatenate((data["history"][:, -1:], data["position"][:, :-1]), 1)
    velocity = (data["position"] - previous) / .5
    previous_velocity = np.concatenate(((data["history"][:, -1:] - data["history"][:, -2:-1]) / .5, velocity[:, :-1]), 1)
    if not np.array_equal(data["velocity"], velocity) or not np.array_equal(data["previous_velocity"], previous_velocity):
        raise ValueError("gate causal velocity mismatch")
    filter_position = data["history"][:, -1].copy()
    filter_velocity = (data["history"][:, -1] - data["history"][:, -2]) / .5
    for k in range(11):
        lead = (np.arange(12, dtype=np.float32) - k) * .5
        expected_cv = data["position"][:, k, None] + velocity[:, k, None] * lead[None, :, None]
        prediction = filter_position + filter_velocity * .5
        residual = data["position"][:, k] - prediction
        filter_position, filter_velocity = prediction + .6 * residual, filter_velocity + .2 / .5 * residual
        expected_filtered = filter_position[:, None] + filter_velocity[:, None] * lead[None, :, None]
        if not np.array_equal(expected_cv, data["cv"][:, k]) or not np.array_equal(expected_filtered, data["filtered"][:, k]):
            raise ValueError("CV/filter candidate reconstruction mismatch")
    for ci, condition in enumerate(CONDITIONS):
        schedule = packet_schedule(condition, config=config)
        if not np.all(data["packet_step"][ci] == schedule["packet_step"]) or not np.all(data["age_s"][ci] == schedule["age_s"]):
            raise ValueError("gate packet schedule mismatch")
        source = np.where(schedule["packet_step"] > 0, anchor[:, None] + schedule["packet_step"] * 500000, 0)
        available = np.where(schedule["packet_step"] > 0, anchor[:, None] + schedule["available_step"] * 500000, 0)
        if not np.array_equal(source, data["source_us"][ci]) or not np.array_equal(available, data["available_us"][ci]) or np.any(available > data["timestamp_us"]):
            raise ValueError("gate packet availability mismatch")
        coverage = data["coverage"][ci]
        if np.any((coverage < 0) | (coverage > 1)) or np.any(coverage[:, schedule["packet_step"] == 0] != 0) or not np.array_equal(data["valid"][ci], schedule["age_valid"] & (coverage > 0)):
            raise ValueError("gate validity/coverage mismatch")
        for start in range(0, n, 256):
            end = min(start + 256, n)
            rows = np.repeat(np.arange(start, end), 11)
            updates = np.tile(np.arange(11), end - start)
            expected = quality_features(data["position"][rows, updates], velocity[rows, updates], previous_velocity[rows, updates],
                data["base"][rows], updates + 1, coverage[rows, updates], data["age_s"][ci, rows, updates],
                data["cv"][rows, updates], data["map"][ci, rows, updates])
            if not np.allclose(data["quality"][ci, rows, updates], expected, atol=1e-6, rtol=1e-6):
                raise ValueError("gate quality reconstruction mismatch")
            missing = ~data["valid"][ci, rows, updates]
            if not np.array_equal(data["map"][ci, rows[missing], updates[missing]], data["original"][rows[missing], updates[missing]]):
                raise ValueError("ungated map fallback mismatch")


def save_gate_cache(root, arrays, contract):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    validate_gate_arrays(arrays, contract)
    entries = {}
    for name, array in arrays.items():
        path = root / (name + ".npy")
        if isinstance(array, np.memmap) and Path(array.filename).resolve() == path.resolve():
            array.flush()
        else:
            np.save(path, array, allow_pickle=False)
        entries[name] = {"file": path.name, "sha256": sha256_file(path), "shape": list(array.shape), "dtype": array.dtype.str}
    manifest = {"schema_version": 1, "contract": contract, "contract_sha256": digest_json(contract), "arrays": entries}
    temporary = root / ".manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(root / "manifest.json")
    return manifest


def load_gate_cache(root, expected_contract):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or manifest.get("contract") != expected_contract or manifest.get("contract_sha256") != digest_json(expected_contract):
        raise ValueError("gate cache contract mismatch")
    arrays = {}
    for name, entry in manifest["arrays"].items():
        if name not in {"instance_token", "sample_token", "scene_name", "base", "target", "history", "anchor_timestamp_us", "timestamp_us", "position", "velocity", "previous_velocity", "cv", "filtered", "original", "map", "quality", "valid", "packet_step", "source_us", "available_us", "age_s", "coverage"} or entry["file"] != name + ".npy":
            raise ValueError("unexpected gate array file")
        path = root / entry["file"]
        if sha256_file(path) != entry["sha256"]:
            raise ValueError("gate array checksum mismatch")
        value = np.load(path, mmap_mode="r", allow_pickle=False)
        if list(value.shape) != entry["shape"] or value.dtype.str != entry["dtype"]:
            raise ValueError("gate declared array schema mismatch")
        arrays[name] = value
    validate_gate_arrays(arrays, expected_contract)
    return arrays, manifest


class GateDataset:
    def __init__(self, arrays):
        self.arrays = arrays
        self.n = len(arrays["instance_token"])
        self.size = 6 * self.n * 11

    def fingerprint(self):
        from trajectory_prediction.online.batched_context import BatchedContext
        return BatchedContext(self.arrays, {}, {}, self.arrays["base"]).fingerprint()

    @property
    def scenes(self):
        return sorted(set(self.arrays["scene_name"].tolist()))

    def indices(self, indices):
        indices = np.asarray(indices)
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer) or np.any((indices < 0) | (indices >= self.size)):
            raise ValueError("gate indices out of bounds")
        return indices // (self.n * 11), (indices // 11) % self.n, indices % 11

    def inputs(self, indices, device):
        c, i, k = self.indices(indices)
        mask = np.arange(12)[None] >= (k + 1)[:, None]
        values = {"cv": self.arrays["cv"][i, k], "map": self.arrays["map"][c, i, k], "quality": self.arrays["quality"][c, i, k],
            "valid": self.arrays["valid"][c, i, k], "mask": mask, "age_s": self.arrays["age_s"][c, i, k]}
        return {name: torch.from_numpy(np.ascontiguousarray(value)).to(device) for name, value in values.items()}

    def labels(self, indices, device):
        _, i, _ = self.indices(indices)
        return torch.from_numpy(np.ascontiguousarray(self.arrays["target"][i])).to(device)

"""Independent neighbor archive, bound to map cache, keys and source provenance."""
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from trajectory_prediction.context.cache import IDENTIFIERS, digest_json, key_hash
from trajectory_prediction.context.neighbors import NeighborConfig, NeighborView
from trajectory_prediction.data.cache import _replace_manifest
from trajectory_prediction.utils.provenance import sha256_file

FIELDS = ("features", "mask", "velocity_mask", "instance_tokens", "annotation_tokens",
          "source_timestamp_us", "available_timestamp_us")


def neighbor_view(arrays, i, k):
    return NeighborView(*(arrays[name][i, k] for name in FIELDS), int(arrays["timestamp_us"][i, k]))


def validate_neighbors(arrays, contract):
    config = NeighborConfig(**contract["config"])
    if set(arrays) != {*IDENTIFIERS, *FIELDS, "anchor_timestamp_us", "timestamp_us"}:
        raise ValueError("neighbor array schema mismatch")
    n = len(arrays["sample_token"])
    if n < 1:
        raise ValueError("neighbor cache must not be empty")
    for name in IDENTIFIERS:
        array = arrays[name]
        if array.shape != (n,) or array.dtype.kind != "U" or any(not token for token in array):
            raise ValueError("invalid neighbor keys")
    keys = list(zip(arrays["instance_token"].tolist(), arrays["sample_token"].tolist(), strict=True))
    if len(set(keys)) != n:
        raise ValueError("duplicate neighbor keys")
    if key_hash(arrays) != contract["keys_sha256"]:
        raise ValueError("neighbor key order mismatch")
    expected = (n, 11, config.max_neighbors, 5)
    shapes = {"features": (*expected, 6), "mask": expected, "velocity_mask": expected,
              "source_timestamp_us": expected, "available_timestamp_us": expected,
              "annotation_tokens": expected, "instance_tokens": expected[:3]}
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"invalid neighbor {name} shape")
    if arrays["features"].dtype != np.float32:
        raise ValueError("neighbor features must be float32")
    anchor, times = arrays["anchor_timestamp_us"], arrays["timestamp_us"]
    if anchor.shape != (n,) or anchor.dtype != np.int64 or np.any(anchor < 0):
        raise ValueError("invalid neighbor anchor times")
    if times.shape != (n, 11) or times.dtype != np.int64 or not np.array_equal(
        times, anchor[:, None] + np.arange(1, 12, dtype=np.int64) * 500_000
    ):
        raise ValueError("neighbor timeline mismatch")
    for i in range(n):
        for k in range(11):
            view = neighbor_view(arrays, i, k)
            if arrays["instance_token"][i] in view.instance_tokens:
                raise ValueError("target cannot be its own neighbor")
            if np.any(view.available_timestamp_us[view.mask] - view.source_timestamp_us[view.mask] != config.latency_us):
                raise ValueError("neighbor availability/config mismatch")
            if np.any(view.timestamp_us - view.source_timestamp_us[:, -1][view.mask[:, -1]] > config.max_age_us):
                raise ValueError("stale current neighbor")
            if np.any(np.linalg.norm(view.features[:, -1, :2], axis=-1) > config.radius_m + 1e-4):
                raise ValueError("neighbor outside configured radius")


def save_neighbors(root, arrays, contract, preparation=None):
    validate_neighbors(arrays, contract)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=root, prefix=".neighbors-", suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        digest = sha256_file(Path(temporary))
        name = f"neighbors-{digest}.npz"
        os.replace(temporary, root / name)
        _replace_manifest(root, {"schema_version": 1, "contract": contract,
            "contract_sha256": digest_json(contract), "npz_file": name, "npz_sha256": digest,
            "preparation": preparation or {}})
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_neighbors(root, expected_contract):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("contract") != expected_contract or manifest.get("contract_sha256") != digest_json(expected_contract):
        raise ValueError("neighbor contract mismatch")
    match = re.fullmatch(r"neighbors-([0-9a-f]{64})\.npz", manifest.get("npz_file", ""))
    if not match or match[1] != manifest.get("npz_sha256"):
        raise ValueError("invalid neighbor archive name")
    path = root / manifest["npz_file"]
    if sha256_file(path) != match[1]:
        raise ValueError("neighbor checksum mismatch")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    validate_neighbors(arrays, expected_contract)
    return arrays, manifest

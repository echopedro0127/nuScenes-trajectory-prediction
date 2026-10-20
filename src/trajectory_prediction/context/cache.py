"""Independent, content-addressed map cache; never extend trajectory NPZ schema."""
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np

from trajectory_prediction.context.maps import MapConfig, MapView
from trajectory_prediction.data.cache import _replace_manifest
from trajectory_prediction.utils.provenance import sha256_file

IDENTIFIERS = ("instance_token", "sample_token", "scene_name")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def key_hash(arrays):
    return digest_json([list(items) for items in zip(*(arrays[k].tolist() for k in IDENTIFIERS), strict=True)])


def validate_context(arrays, contract):
    required = {*IDENTIFIERS, "features", "mask", "lane_tokens", "anchor_timestamp_us", "timestamp_us"}
    if set(arrays) != required:
        raise ValueError("context array schema mismatch")
    config = MapConfig(**contract["config"])
    n = len(arrays["instance_token"])
    if n == 0:
        raise ValueError("context cache must not be empty")
    keys = []
    for name in IDENTIFIERS:
        values = arrays[name]
        if values.shape != (n,) or values.dtype.kind != "U" or any(not v for v in values):
            raise ValueError("invalid context keys")
    keys = list(zip(arrays["instance_token"].tolist(), arrays["sample_token"].tolist(), strict=True))
    if len(set(keys)) != n:
        raise ValueError("duplicate context keys")
    if key_hash(arrays) != contract["keys_sha256"]:
        raise ValueError("context key order mismatch")
    shape = (n, 11, config.max_lanes, config.points_per_lane)
    if arrays["features"].shape != (*shape, 4) or arrays["features"].dtype != np.float32:
        raise ValueError("invalid context features shape or dtype")
    if arrays["mask"].shape != shape or arrays["mask"].dtype != np.bool_:
        raise ValueError("invalid context mask")
    if arrays["lane_tokens"].shape != shape[:3] or arrays["lane_tokens"].dtype.kind != "U":
        raise ValueError("invalid lane tokens")
    anchor, times = arrays["anchor_timestamp_us"], arrays["timestamp_us"]
    if anchor.shape != (n,) or anchor.dtype != np.int64 or np.any(anchor < 0):
        raise ValueError("invalid anchor timestamps")
    if times.dtype != np.int64 or times.shape != (n, 11) or not np.array_equal(
        times, anchor[:, None] + np.arange(1, 12, dtype=np.int64) * 500_000
    ):
        raise ValueError("invalid update timestamps")
    for i in range(n):
        for k in range(11):
            MapView(arrays["features"][i, k], arrays["mask"][i, k], tuple(arrays["lane_tokens"][i, k]))


def save_context(root, arrays, contract, preparation=None):
    validate_context(arrays, contract)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=root, prefix=".context-", suffix=".npz")
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        digest = sha256_file(Path(temporary))
        filename = f"context-{digest}.npz"
        os.replace(temporary, root / filename)
        _replace_manifest(root, {"schema_version": 1, "contract": contract,
                                "contract_sha256": digest_json(contract), "npz_file": filename,
                                "npz_sha256": digest, "preparation": preparation or {}})
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_context(root, expected_contract):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("contract") != expected_contract or manifest.get("contract_sha256") != digest_json(expected_contract):
        raise ValueError("context contract mismatch")
    name = manifest.get("npz_file", "")
    match = re.fullmatch(r"context-([0-9a-f]{64})\.npz", name)
    if not match or match[1] != manifest.get("npz_sha256"):
        raise ValueError("invalid context archive name")
    if sha256_file(root / name) != match[1]:
        raise ValueError("context checksum mismatch")
    with np.load(root / name, allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    validate_context(arrays, expected_contract)
    return arrays, manifest

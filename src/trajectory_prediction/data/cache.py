import hashlib
import json
from pathlib import Path

import numpy as np

REQUIRED_ARRAYS = {
    "history_xy",
    "history_features",
    "history_mask",
    "future_xy",
    "anchor_xy",
    "anchor_yaw",
    "current_speed",
    "instance_token",
    "sample_token",
    "scene_name",
}


def _missing_arrays(names: object) -> set[str]:
    return REQUIRED_ARRAYS.difference(names)


def save_cache(root: Path, arrays: dict[str, np.ndarray], manifest: dict) -> None:
    missing = _missing_arrays(arrays.keys())
    if missing:
        raise ValueError(f"cache arrays missing: {sorted(missing)}")

    root.mkdir(parents=True, exist_ok=True)
    npz_path = root / "trajectories.npz"
    np.savez_compressed(npz_path, **arrays)

    payload = dict(manifest)
    payload["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    (root / "manifest.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_cache(
    root: Path, expected_split: str
) -> tuple[dict[str, np.ndarray], dict]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("unsupported cache schema")
    if manifest.get("split") != expected_split:
        raise ValueError("cache split does not match requested split")

    npz_path = root / "trajectories.npz"
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    if digest != manifest.get("npz_sha256"):
        raise ValueError("cache checksum mismatch")

    with np.load(npz_path, allow_pickle=False) as data:
        missing = _missing_arrays(data.files)
        if missing:
            raise ValueError(f"cache arrays missing: {sorted(missing)}")
        arrays = {name: data[name] for name in data.files}
    return arrays, manifest

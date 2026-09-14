import hashlib
import json
import os
import re
import tempfile
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

_FLOAT_ARRAY_SHAPES = {
    "history_xy": (5, 2),
    "history_features": (5, 6),
    "future_xy": (12, 2),
    "anchor_xy": (2,),
    "anchor_yaw": (),
    "current_speed": (),
}
_STRING_ARRAYS = ("instance_token", "sample_token", "scene_name")
_CONTENT_FILE_PATTERN = re.compile(r"trajectories-([0-9a-f]{64})\.npz")


def validate_cache_arrays(arrays: dict[str, np.ndarray]) -> None:
    names = set(arrays)
    missing = REQUIRED_ARRAYS.difference(names)
    if missing:
        raise ValueError(f"cache arrays missing: {sorted(missing)}")
    unexpected = names.difference(REQUIRED_ARRAYS)
    if unexpected:
        raise ValueError(f"unexpected cache arrays: {sorted(unexpected)}")

    for name, trailing_shape in _FLOAT_ARRAY_SHAPES.items():
        array = arrays[name]
        if not isinstance(array, np.ndarray) or array.dtype != np.float32:
            raise ValueError(f"{name} dtype must be float32")
        if array.ndim != len(trailing_shape) + 1 or array.shape[1:] != trailing_shape:
            raise ValueError(f"{name} has invalid shape: {array.shape}")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must contain only finite values")

    history_mask = arrays["history_mask"]
    if not isinstance(history_mask, np.ndarray) or history_mask.dtype != np.bool_:
        raise ValueError("history_mask dtype must be bool")
    if history_mask.ndim != 2 or history_mask.shape[1:] != (5,):
        raise ValueError(f"history_mask has invalid shape: {history_mask.shape}")

    for name in _STRING_ARRAYS:
        array = arrays[name]
        if not isinstance(array, np.ndarray) or array.dtype.kind != "U":
            raise ValueError(f"{name} must use a Unicode string dtype")
        if array.ndim != 1:
            raise ValueError(f"{name} has invalid shape: {array.shape}")

    sample_count = arrays["history_xy"].shape[0]
    for name, array in arrays.items():
        if array.shape[0] != sample_count:
            raise ValueError(
                f"{name} sample count {array.shape[0]} does not match {sample_count}"
            )


def _write_content_archive(
    root: Path, arrays: dict[str, np.ndarray]
) -> tuple[str, str]:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=root, prefix=".trajectories-", suffix=".npz"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        digest = hashlib.sha256(temporary_path.read_bytes()).hexdigest()
        content_name = f"trajectories-{digest}.npz"
        os.replace(temporary_path, root / content_name)
        return content_name, digest
    finally:
        temporary_path.unlink(missing_ok=True)


def _replace_manifest(root: Path, payload: dict) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=root, prefix=".manifest-", suffix=".json.tmp"
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, root / "manifest.json")
    finally:
        temporary_path.unlink(missing_ok=True)


def save_cache(root: Path, arrays: dict[str, np.ndarray], manifest: dict) -> None:
    validate_cache_arrays(arrays)
    root.mkdir(parents=True, exist_ok=True)
    content_name, digest = _write_content_archive(root, arrays)

    payload = dict(manifest)
    payload["npz_file"] = content_name
    payload["npz_sha256"] = digest
    _replace_manifest(root, payload)


def load_cache(
    root: Path,
    expected_split: str,
    expected_config_sha256: str | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError("cache manifest must be a mapping")  # noqa: TRY004
    schema_version = manifest.get("schema_version")
    if type(schema_version) is not int or schema_version != 1:
        raise ValueError("unsupported cache schema")
    if manifest.get("split") != expected_split:
        raise ValueError("cache split does not match requested split")
    if (
        expected_config_sha256 is not None
        and manifest.get("preprocessing_config_sha256") != expected_config_sha256
    ):
        raise ValueError("preprocessing config hash mismatch")

    content_name = manifest.get("npz_file")
    expected_digest = manifest.get("npz_sha256")
    if not isinstance(content_name, str) or not isinstance(expected_digest, str):
        raise ValueError("invalid cache archive filename")  # noqa: TRY004
    match = _CONTENT_FILE_PATTERN.fullmatch(content_name)
    if match is None or match.group(1) != expected_digest:
        raise ValueError("invalid cache archive filename")

    npz_path = root / content_name
    digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    if digest != expected_digest:
        raise ValueError("cache checksum mismatch")

    with np.load(npz_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    validate_cache_arrays(arrays)
    return arrays, manifest

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from trajectory_prediction.data.cache import REQUIRED_ARRAYS, load_cache, save_cache


def _arrays() -> dict[str, np.ndarray]:
    return {
        "history_xy": np.zeros((2, 5, 2), dtype=np.float32),
        "history_features": np.zeros((2, 5, 6), dtype=np.float32),
        "history_mask": np.ones((2, 5), dtype=np.bool_),
        "future_xy": np.zeros((2, 12, 2), dtype=np.float32),
        "anchor_xy": np.zeros((2, 2), dtype=np.float32),
        "anchor_yaw": np.zeros(2, dtype=np.float32),
        "current_speed": np.ones(2, dtype=np.float32),
        "instance_token": np.array(["instance-a", "instance-b"]),
        "sample_token": np.array(["sample-a", "sample-b"]),
        "scene_name": np.array(["scene-1", "scene-2"]),
    }


def test_cache_round_trip_preserves_all_arrays_and_manifest(tmp_path: Path) -> None:
    arrays = _arrays()

    save_cache(tmp_path, arrays, {"schema_version": 1, "split": "mini_train"})
    loaded, manifest = load_cache(tmp_path, expected_split="mini_train")

    assert set(loaded) == REQUIRED_ARRAYS
    for name, expected in arrays.items():
        np.testing.assert_array_equal(loaded[name], expected)
        assert loaded[name].dtype == expected.dtype
    assert manifest["schema_version"] == 1
    assert manifest["split"] == "mini_train"
    assert manifest["npz_sha256"] == hashlib.sha256(
        (tmp_path / "trajectories.npz").read_bytes()
    ).hexdigest()


def test_save_cache_rejects_missing_required_array(tmp_path: Path) -> None:
    arrays = _arrays()
    arrays.pop("future_xy")

    with pytest.raises(ValueError, match="cache arrays missing:.*future_xy"):
        save_cache(tmp_path, arrays, {"schema_version": 1, "split": "mini_train"})


@pytest.mark.parametrize(
    ("manifest_update", "message"),
    [
        ({"schema_version": 2}, "unsupported cache schema"),
        ({"split": "mini_val"}, "cache split does not match requested split"),
    ],
)
def test_load_cache_rejects_incompatible_manifest(
    tmp_path: Path, manifest_update: dict, message: str
) -> None:
    save_cache(tmp_path, _arrays(), {"schema_version": 1, "split": "mini_train"})
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(manifest_update)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_cache(tmp_path, expected_split="mini_train")


def test_load_cache_rejects_tampered_archive_before_loading(tmp_path: Path) -> None:
    save_cache(tmp_path, _arrays(), {"schema_version": 1, "split": "mini_train"})
    npz_path = tmp_path / "trajectories.npz"
    npz_path.write_bytes(npz_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="cache checksum mismatch"):
        load_cache(tmp_path, expected_split="mini_train")


def test_load_cache_rejects_archive_missing_required_array(tmp_path: Path) -> None:
    save_cache(tmp_path, _arrays(), {"schema_version": 1, "split": "mini_train"})
    npz_path = tmp_path / "trajectories.npz"
    incomplete = _arrays()
    incomplete.pop("history_mask")
    np.savez_compressed(npz_path, **incomplete)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["npz_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="cache arrays missing:.*history_mask"):
        load_cache(tmp_path, expected_split="mini_train")

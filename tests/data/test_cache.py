import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from trajectory_prediction.data import cache as cache_module
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
        (tmp_path / manifest["npz_file"]).read_bytes()
    ).hexdigest()
    assert manifest["npz_file"] == f"trajectories-{manifest['npz_sha256']}.npz"
    assert not (tmp_path / "trajectories.npz").exists()


def test_save_cache_rejects_missing_required_array(tmp_path: Path) -> None:
    arrays = _arrays()
    arrays.pop("future_xy")

    with pytest.raises(ValueError, match="cache arrays missing:.*future_xy"):
        save_cache(tmp_path, arrays, {"schema_version": 1, "split": "mini_train"})


def test_save_cache_rejects_extra_array(tmp_path: Path) -> None:
    arrays = _arrays()
    arrays["future_mask"] = np.ones((2, 12), dtype=np.bool_)

    with pytest.raises(ValueError, match="unexpected cache arrays:.*future_mask"):
        save_cache(tmp_path, arrays, {"schema_version": 1, "split": "mini_train"})


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("history_xy", np.zeros((2, 4, 2), dtype=np.float32), "history_xy.*shape"),
        ("current_speed", np.ones(2, dtype=np.float64), "current_speed.*dtype"),
        ("history_mask", np.ones((2, 5), dtype=np.uint8), "history_mask.*dtype"),
        ("scene_name", np.array(["scene-1"], dtype=np.str_), "scene_name.*sample count"),
        ("sample_token", np.array(["a", "b"], dtype=object), "sample_token.*Unicode"),
    ],
)
def test_save_cache_rejects_invalid_shape_dtype_or_count(
    tmp_path: Path, name: str, value: np.ndarray, message: str
) -> None:
    arrays = _arrays()
    arrays[name] = value

    with pytest.raises(ValueError, match=message):
        save_cache(tmp_path, arrays, {"schema_version": 1, "split": "mini_train"})


def test_save_cache_rejects_nonfinite_float_array(tmp_path: Path) -> None:
    arrays = _arrays()
    arrays["future_xy"][0, 0, 0] = np.nan

    with pytest.raises(ValueError, match="future_xy.*finite"):
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
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))
    npz_path = tmp_path / manifest["npz_file"]
    npz_path.write_bytes(npz_path.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="cache checksum mismatch"):
        load_cache(tmp_path, expected_split="mini_train")


def _replace_referenced_archive(root: Path, arrays: dict[str, np.ndarray]) -> None:
    replacement_path = root / "replacement.npz"
    np.savez_compressed(replacement_path, **arrays)
    digest = hashlib.sha256(replacement_path.read_bytes()).hexdigest()
    content_path = root / f"trajectories-{digest}.npz"
    replacement_path.replace(content_path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["npz_file"] = content_path.name
    manifest["npz_sha256"] = digest
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def test_load_cache_rejects_archive_missing_required_array(tmp_path: Path) -> None:
    save_cache(tmp_path, _arrays(), {"schema_version": 1, "split": "mini_train"})
    incomplete = _arrays()
    incomplete.pop("history_mask")
    _replace_referenced_archive(tmp_path, incomplete)

    with pytest.raises(ValueError, match="cache arrays missing:.*history_mask"):
        load_cache(tmp_path, expected_split="mini_train")


def test_load_cache_validates_arrays_after_checksum(tmp_path: Path) -> None:
    save_cache(tmp_path, _arrays(), {"schema_version": 1, "split": "mini_train"})
    invalid = _arrays()
    invalid["anchor_xy"] = np.zeros((2, 3), dtype=np.float32)
    _replace_referenced_archive(tmp_path, invalid)

    with pytest.raises(ValueError, match="anchor_xy.*shape"):
        load_cache(tmp_path, expected_split="mini_train")


def test_load_cache_rejects_unsafe_or_nonmatching_content_filename(tmp_path: Path) -> None:
    save_cache(tmp_path, _arrays(), {"schema_version": 1, "split": "mini_train"})
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["npz_file"] = "../trajectories.npz"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="invalid cache archive filename"):
        load_cache(tmp_path, expected_split="mini_train")


def test_load_cache_rejects_expected_config_hash_mismatch(tmp_path: Path) -> None:
    save_cache(
        tmp_path,
        _arrays(),
        {
            "schema_version": 1,
            "split": "mini_train",
            "preprocessing_config_sha256": "a" * 64,
        },
    )

    with pytest.raises(ValueError, match="preprocessing config hash mismatch"):
        load_cache(
            tmp_path,
            expected_split="mini_train",
            expected_config_sha256="b" * 64,
        )


def test_failed_manifest_swap_preserves_previous_readable_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = _arrays()
    save_cache(tmp_path, original, {"schema_version": 1, "split": "mini_train"})
    replacement = _arrays()
    replacement["current_speed"] = np.full(2, 7.0, dtype=np.float32)
    real_replace = cache_module.os.replace

    def fail_manifest_swap(source: Path, destination: Path) -> None:
        if Path(destination).name == "manifest.json":
            raise OSError("injected manifest swap failure")
        real_replace(source, destination)

    monkeypatch.setattr(cache_module.os, "replace", fail_manifest_swap)

    with pytest.raises(OSError, match="injected manifest swap failure"):
        save_cache(
            tmp_path,
            replacement,
            {"schema_version": 1, "split": "mini_train"},
        )

    loaded, _ = load_cache(tmp_path, expected_split="mini_train")
    np.testing.assert_array_equal(loaded["current_speed"], original["current_speed"])

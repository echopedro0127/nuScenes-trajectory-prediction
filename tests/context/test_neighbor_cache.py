from dataclasses import asdict

import numpy as np
import pytest

from trajectory_prediction.context.cache import key_hash
from trajectory_prediction.context.neighbor_cache import load_neighbors, save_neighbors
from trajectory_prediction.context.neighbors import NeighborConfig


def records():
    return {"features": np.zeros((1, 11, 8, 5, 6), np.float32),
        "mask": np.zeros((1, 11, 8, 5), bool), "velocity_mask": np.zeros((1, 11, 8, 5), bool),
        "instance_tokens": np.full((1, 11, 8), "", dtype="U64"),
        "annotation_tokens": np.full((1, 11, 8, 5), "", dtype="U64"),
        "source_timestamp_us": np.zeros((1, 11, 8, 5), np.int64),
        "available_timestamp_us": np.zeros((1, 11, 8, 5), np.int64),
        "anchor_timestamp_us": np.array([0], np.int64),
        "timestamp_us": np.arange(1, 12, dtype=np.int64)[None] * 500_000,
        "instance_token": np.array(["target"]), "sample_token": np.array(["sample"]),
        "scene_name": np.array(["scene"])}


def contract():
    return {"schema_version": 1, "modality": "oracle_neighbors", "split": "mini_train",
            "config": asdict(NeighborConfig()), "keys_sha256": key_hash(records()),
            "source": {"category": "a" * 64}, "map_npz_sha256": "b" * 64}


def test_neighbor_cache_roundtrip_and_bound_contract(tmp_path):
    values, expected = records(), contract()
    save_neighbors(tmp_path, values, expected)
    result, manifest = load_neighbors(tmp_path, expected)
    np.testing.assert_array_equal(result["features"], values["features"])
    assert manifest["contract"] == expected
    for field in ("split", "config", "keys_sha256", "source", "map_npz_sha256"):
        with pytest.raises(ValueError, match="contract"):
            load_neighbors(tmp_path, {**expected, field: "changed"})
    archive = next(tmp_path.glob("neighbors-*.npz"))
    archive.write_bytes(archive.read_bytes() + b"broken")
    with pytest.raises(ValueError, match="checksum"):
        load_neighbors(tmp_path, expected)


@pytest.mark.parametrize("field", ["timestamp_us", "features", "mask", "source_timestamp_us"])
def test_invalid_neighbor_cache_rejected(tmp_path, field):
    values = records()
    if field == "mask":
        values[field] = values[field].astype(int)
    else:
        values[field].flat[0] = 1
    with pytest.raises(ValueError):
        save_neighbors(tmp_path, values, contract())


def test_neighbor_cache_rejects_duplicate_or_reordered_keys(tmp_path):
    values = {name: np.repeat(value, 2, axis=0) for name, value in records().items()}
    with pytest.raises(ValueError, match="duplicate"):
        save_neighbors(tmp_path, values, contract())
    values["sample_token"] = np.array(["a", "b"])
    expected = {**contract(), "keys_sha256": key_hash(values)}
    with pytest.raises(ValueError, match="key"):
        save_neighbors(tmp_path, {name: value[::-1] for name, value in values.items()}, expected)



def test_stale_history_rejected_even_when_current_neighbor_is_fresh(tmp_path):
    values = records()
    k = 3  # tk=2s; previous slot=1.5s, but history source=0 is too old for that slot.
    values["instance_tokens"][0, k, 0] = "neighbor"
    values["mask"][0, k, 0, 3:] = True
    values["annotation_tokens"][0, k, 0, 3:] = ["old", "now"]
    values["source_timestamp_us"][0, k, 0, -1] = 2_000_000
    values["available_timestamp_us"][0, k, 0, -1] = 2_000_000
    values["features"][0, k, 0, 3:, 4] = 1
    with pytest.raises(ValueError, match="stale"):
        save_neighbors(tmp_path, values, contract())


def test_cached_velocity_must_match_backward_difference(tmp_path):
    values = records()
    k = 3
    values["instance_tokens"][0, k, 0] = "neighbor"
    values["mask"][0, k, 0, 3:] = True
    values["velocity_mask"][0, k, 0, -1] = True
    values["annotation_tokens"][0, k, 0, 3:] = ["past", "now"]
    values["source_timestamp_us"][0, k, 0, 3:] = [1_500_000, 2_000_000]
    values["available_timestamp_us"][0, k, 0, 3:] = [1_500_000, 2_000_000]
    values["features"][0, k, 0, 3:, 4] = 1
    values["features"][0, k, 0, 3:, 0] = [1, 3]
    with pytest.raises(ValueError, match="velocity"):
        save_neighbors(tmp_path, values, contract())

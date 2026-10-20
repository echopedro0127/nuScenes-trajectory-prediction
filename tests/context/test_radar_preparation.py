import json

import numpy as np
import pytest
from test_radar import radar_fixture

from trajectory_prediction.context.cache import key_hash
from trajectory_prediction.context.radar_index import RadarIndex
from trajectory_prediction.context.radar_preparation import load_radar, prepare_arrays, save_radar


def test_preparation_uses_shared_degraded_center_and_binds_source_files(tmp_path):
    source = tmp_path / "source"
    radar_fixture(source)
    index = RadarIndex(source)
    arrays = {"instance_token": np.array(["i"]), "sample_token": np.array(["sample"]), "scene_name": np.array(["scene"]),
        "anchor_xy": np.zeros((1, 2), np.float32), "anchor_yaw": np.array([np.pi / 2], np.float32),
        "history_xy": np.zeros((1, 5, 2), np.float32), "future_xy": np.zeros((1, 12, 2), np.float32)}
    # Future Radar file is intentionally absent: restrict this synthetic experiment before it.
    index.frames[("scene", "RADAR_FRONT")] = index.frames[("scene", "RADAR_FRONT")][:2]
    index.times[("scene", "RADAR_FRONT")] = index.times[("scene", "RADAR_FRONT")][:2]
    seen = []
    original = index.query
    def query(state, scene, anchor, yaw):
        seen.append(state.position.copy())
        return original(state, scene, anchor, yaw)
    index.query = query
    cache = prepare_arrays(arrays, index)
    assert cache["position"].shape == (5, 1, 11, 2)
    assert cache["radar_features"].shape == (5, 1, 11, 13)
    np.testing.assert_allclose(np.asarray(seen).reshape(1, 11, 5, 2).transpose(2, 0, 1, 3), cache["position"])
    contract = {"config": {"radius_m": 4., "window_us": 2000000, "sweeps_per_channel": 3},
                "keys_sha256": key_hash(arrays), "source": {**index.source_hashes, **index.used_files}, "split": "mini_train"}
    destination = tmp_path / "cache"
    save_radar(destination, cache, contract)
    loaded, manifest = load_radar(destination, contract, source)
    np.testing.assert_array_equal(loaded["position"], cache["position"])
    with pytest.raises(ValueError, match="contract"):
        load_radar(destination, {**contract, "split": "mini_val"}, source)
    (source / "current.pcd").write_bytes(b"changed")
    with pytest.raises(ValueError, match="source"):
        load_radar(destination, contract, source)
    # Contract hashes cannot authorize malformed mask/features even on publication.
    bad = {name: value.copy() for name, value in cache.items()}
    bad["present"][:] = False
    bad["radar_features"][:] = 1
    with pytest.raises(ValueError):
        save_radar(tmp_path / "bad", bad, contract)
    assert manifest["contract"]["keys_sha256"] == key_hash(arrays)


def test_publication_cannot_omit_all_raw_metadata_sources(tmp_path):
    source = tmp_path / "source"
    radar_fixture(source)
    index = RadarIndex(source)
    index.frames[("scene", "RADAR_FRONT")] = index.frames[("scene", "RADAR_FRONT")][:2]
    index.times[("scene", "RADAR_FRONT")] = index.times[("scene", "RADAR_FRONT")][:2]
    arrays = {"instance_token": np.array(["i"]), "sample_token": np.array(["sample"]), "scene_name": np.array(["scene"]),
        "anchor_xy": np.zeros((1, 2), np.float32), "anchor_yaw": np.array([np.pi / 2], np.float32),
        "history_xy": np.zeros((1, 5, 2), np.float32), "future_xy": np.zeros((1, 12, 2), np.float32)}
    cache = prepare_arrays(arrays, index)
    contract = {"config": {"radius_m": 4., "window_us": 2000000, "sweeps_per_channel": 3},
                "keys_sha256": key_hash(arrays), "source": {}, "split": "mini_train"}
    with pytest.raises(ValueError, match="source"):
        save_radar(tmp_path / "bad", cache, contract)


def test_state_audit_rejects_changed_centers_and_omitted_selected_pcd(tmp_path):
    from trajectory_prediction.context.radar_preparation import audit_bundle
    source = tmp_path / "source"
    tables = radar_fixture(source)
    tables["sample_data"] = tables["sample_data"][:2]
    (source / "v1.0-mini/sample_data.json").write_text(json.dumps(tables["sample_data"]))
    index = RadarIndex(source)
    arrays = {"instance_token": np.array(["i"]), "sample_token": np.array(["sample"]), "scene_name": np.array(["scene"]),
        "anchor_xy": np.zeros((1, 2), np.float32), "anchor_yaw": np.array([np.pi / 2], np.float32),
        "history_xy": np.zeros((1, 5, 2), np.float32), "future_xy": np.zeros((1, 12, 2), np.float32)}
    cache = prepare_arrays(arrays, index)
    contract = {"config": {"radius_m": 4., "window_us": 2000000, "sweeps_per_channel": 3},
                "keys_sha256": key_hash(arrays), "source": {**index.source_hashes, **index.used_files},
                "perturbation_seed": 2027, "dataset_version": "v1.0-mini", "split": "mini_train"}
    audit_bundle(arrays, cache, contract, source)
    cache["position"][:] = 99999
    with pytest.raises(ValueError, match="state"):
        audit_bundle(arrays, cache, contract, source)
    cache = prepare_arrays(arrays, index)
    contract["source"].pop("current.pcd")
    with pytest.raises(ValueError, match="source"):
        audit_bundle(arrays, cache, contract, source)

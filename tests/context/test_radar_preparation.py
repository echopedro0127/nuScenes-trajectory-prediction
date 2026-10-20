
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

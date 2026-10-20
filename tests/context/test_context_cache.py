import numpy as np
import pytest

from trajectory_prediction.context.cache import load_context, save_context


def records():
    return {"features": np.zeros((1, 11, 1, 2, 4), np.float32),
            "mask": np.zeros((1, 11, 1, 2), bool),
            "lane_tokens": np.full((1, 11, 1), "", dtype="U1"),
            "instance_token": np.array(["i"]), "sample_token": np.array(["s"]),
            "scene_name": np.array(["scene"]), "anchor_timestamp_us": np.array([0], np.int64),
            "timestamp_us": np.arange(1, 12, dtype=np.int64)[None] * 500_000}


def contract():
    from trajectory_prediction.context.cache import key_hash
    return {"schema_version": 1, "split": "mini_train", "config": {
        "max_lanes": 1, "points_per_lane": 2, "radius_m": 50., "resolution_m": 2.},
        "keys_sha256": key_hash(records()), "source": {"map": "abc"},
        "oracle": "target_position_annotations", "time_policy": "fixed_2hz_11_updates"}


def test_cache_roundtrip_and_contract_rejection(tmp_path):
    arrays, expected = records(), contract()
    save_context(tmp_path, arrays, expected)
    loaded, manifest = load_context(tmp_path, expected)
    np.testing.assert_array_equal(loaded["features"], arrays["features"])
    assert manifest["contract"] == expected
    for field in ("split", "config", "source", "keys_sha256"):
        with pytest.raises(ValueError, match="contract"):
            load_context(tmp_path, {**expected, field: "changed"})


def test_cache_checksum_and_key_order_rejection(tmp_path):
    from trajectory_prediction.context.cache import key_hash
    arrays = {name: np.repeat(value, 2, axis=0) for name, value in records().items()}
    arrays["sample_token"] = np.array(["a", "b"])
    expected = {**contract(), "keys_sha256": key_hash(arrays)}
    save_context(tmp_path, arrays, expected)
    reordered = {name: value[::-1] for name, value in arrays.items()}
    with pytest.raises(ValueError, match="key"):
        save_context(tmp_path / "bad", reordered, expected)
    archive = next(tmp_path.glob("context-*.npz"))
    archive.write_bytes(archive.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        load_context(tmp_path, expected)


@pytest.mark.parametrize("field", ["timestamp_us", "mask", "features"])
def test_cache_invalid_time_mask_or_padding_rejected(tmp_path, field):
    arrays = records()
    if field == "mask":
        arrays[field] = arrays[field].astype(np.int32)
    else:
        arrays[field].flat[0] = 1
    with pytest.raises(ValueError):
        save_context(tmp_path, arrays, contract())


def test_cache_rejects_duplicate_keys(tmp_path):
    arrays = {name: np.repeat(value, 2, axis=0) for name, value in records().items()}
    with pytest.raises(ValueError, match="duplicate"):
        save_context(tmp_path, arrays, contract())



def test_predictions_require_matching_identifier_sidecar_and_target(tmp_path):
    import pandas as pd

    from trajectory_prediction.context.preparation import load_predictions
    arrays = records()
    arrays["future_xy"] = np.zeros((1, 12, 2), np.float32)
    path = tmp_path / "predictions.npz"
    import json
    provenance = {"model_name": "gru", "checkpoint_sha256": "a" * 64,
                  "model_config_sha256": "b" * 64, "data_config_sha256": "c" * 64,
                  "history_seconds": 2.0, "history_steps": 5, "split": "mini_train"}
    (tmp_path / "metrics.json").write_text(json.dumps({"provenance": provenance}))
    np.savez(path, prediction=arrays["future_xy"], target=arrays["future_xy"])
    pd.DataFrame({name: arrays[name] for name in ("instance_token", "sample_token", "scene_name")}).to_csv(
        tmp_path / "per_sample.csv", index=False)
    predicted, hashes = load_predictions(path, arrays)
    assert predicted.shape == (1, 12, 2)
    assert len(hashes["npz_sha256"]) == 64
    np.savez(path, prediction=arrays["future_xy"], target=arrays["future_xy"] + 1)
    with pytest.raises(ValueError, match="target"):
        load_predictions(path, arrays)
    arrays["sample_token"] = np.array(["other"])
    with pytest.raises(ValueError, match="key order"):
        load_predictions(path, arrays)


def test_metadata_index_checks_anchor_and_source_hashes(tmp_path):
    import json

    from trajectory_prediction.context.nuscenes_index import MetadataIndex
    root = tmp_path / "v1.0-mini"
    root.mkdir()
    tables = {"sample": [{"token": "s", "timestamp": 10, "scene_token": "sc"}],
              "scene": [{"token": "sc", "name": "scene", "log_token": "l"}],
              "log": [{"token": "l", "location": "location"}],
              "sample_annotation": [{"token": "a", "instance_token": "i", "sample_token": "s",
                                     "translation": [1, 2, 0], "rotation": [1, 0, 0, 0]}]}
    for name, records in tables.items():
        (root / f"{name}.json").write_text(json.dumps(records))
    maps = tmp_path / "maps" / "expansion"
    maps.mkdir(parents=True)
    (maps / "location.json").write_text("{}")
    index = MetadataIndex(tmp_path, "v1.0-mini")
    assert index.anchor("i", "s", "scene", [1, 2], 0) == (10, "location")
    assert len(index.provenance()) == 5
    with pytest.raises(ValueError, match="scene"):
        index.anchor("i", "s", "wrong", [1, 2], 0)
    with pytest.raises(ValueError, match="position"):
        index.anchor("i", "s", "scene", [2, 2], 0)
    with pytest.raises(ValueError, match="heading"):
        index.anchor("i", "s", "scene", [1, 2], 1)



def test_prediction_requires_fixed_gru_model_identity(tmp_path):
    import json

    import pandas as pd

    from trajectory_prediction.context.preparation import load_predictions
    arrays = records()
    arrays["future_xy"] = np.zeros((1, 12, 2), np.float32)
    np.savez(tmp_path / "predictions.npz", prediction=arrays["future_xy"], target=arrays["future_xy"])
    pd.DataFrame({key: arrays[key] for key in ("instance_token", "sample_token", "scene_name")}).to_csv(
        tmp_path / "per_sample.csv", index=False)
    (tmp_path / "metrics.json").write_text(json.dumps({"provenance": {"model_name": "transformer"}}))
    with pytest.raises(ValueError, match="GRU"):
        load_predictions(tmp_path / "predictions.npz", arrays)



def test_prepare_reveals_only_current_position_for_map_queries(tmp_path, monkeypatch):
    import json

    from trajectory_prediction.context import preparation
    from trajectory_prediction.context.maps import LaneMap, MapConfig
    from trajectory_prediction.data.cache import save_cache
    root = tmp_path / "data"
    metadata = root / "v1.0-mini"
    metadata.mkdir(parents=True)
    tables = {"sample": [{"token": "s", "timestamp": 100, "scene_token": "sc"}],
              "scene": [{"token": "sc", "name": "scene", "log_token": "l"}],
              "log": [{"token": "l", "location": "location"}],
              "sample_annotation": [{"token": "a", "instance_token": "i", "sample_token": "s",
                                     "translation": [0, 0, 0], "rotation": [1, 0, 0, 0]}]}
    for name, value in tables.items():
        (metadata / f"{name}.json").write_text(json.dumps(value))
    maps = root / "maps" / "expansion"
    maps.mkdir(parents=True)
    (maps / "location.json").write_text("{}")
    lane_map = LaneMap({"lane": np.array([[0., 0., 0.], [10., 0., 0.]])}, MapConfig())

    class Provider:
        def __init__(self, dataroot, config):
            pass

        def get(self, location):
            return lane_map

    monkeypatch.setattr(preparation, "MapProvider", Provider)
    arrays = {"history_xy": np.zeros((1, 5, 2), np.float32),
              "history_features": np.zeros((1, 5, 6), np.float32),
              "history_mask": np.ones((1, 5), bool),
              "future_xy": np.arange(24, dtype=np.float32).reshape(1, 12, 2),
              "anchor_xy": np.zeros((1, 2), np.float32), "anchor_yaw": np.zeros(1, np.float32),
              "current_speed": np.zeros(1, np.float32), "instance_token": np.array(["i"]),
              "sample_token": np.array(["s"]), "scene_name": np.array(["scene"])}
    manifest = {"schema_version": 1, "dataset_version": "v1.0-mini", "split": "mini_train",
                "preprocessing_config_sha256": "a" * 64}
    observed = []
    for run in ("first", "changed"):
        trajectory, output = tmp_path / (run + "_trajectory"), tmp_path / (run + "_context")
        save_cache(trajectory, arrays, manifest)
        preparation.prepare(trajectory, "mini_train", root, MapConfig(), output)
        # Read verified context separately to avoid ever passing the target into map query.
        _, cached, _ = preparation.load_bundle(trajectory, output, "mini_train", root, MapConfig())
        observed.append(cached)
        arrays["future_xy"][:, 1:] += 10000
    np.testing.assert_array_equal(observed[0]["features"][:, 0], observed[1]["features"][:, 0])
    np.testing.assert_array_equal(observed[0]["mask"][:, 0], observed[1]["mask"][:, 0])
    assert observed[0]["mask"][:, 1:].any() and not observed[1]["mask"][:, 1:].any()

import json

import numpy as np
import pytest

from trajectory_prediction.context.mmap_cache import open_sidecar, publish_sidecar
from trajectory_prediction.context.streaming_index import CompactMetadata, stream_json_array


@pytest.mark.parametrize("chunk", [1, 7, 4096])
def test_streaming_json_handles_chunk_boundaries_and_escaped_strings(tmp_path, chunk):
    path = tmp_path / "data.json"
    records = [{"token": "a", "text": 'comma, brace} quote" newline\n'}, {"token": "b", "v": [1, 2]}]
    path.write_text(json.dumps(records), encoding="utf-8")
    assert list(stream_json_array(path, chunk)) == records


@pytest.mark.parametrize("text", ['[{}', '[{},]', '{}', '[{}] trailing', '[{broken}]'])
def test_streaming_json_rejects_malformed_arrays(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text)
    with pytest.raises(ValueError):
        list(stream_json_array(path, 3))


def test_readonly_mmap_sidecar_binds_manifest_shape_and_checksum(tmp_path):
    arrays = {"features": np.arange(12, dtype=np.float32).reshape(3, 4)}
    for name, array in arrays.items():
        np.save(tmp_path / (name + ".npy"), array)
    manifest = {"contract": {"split": "train"}, "npz_sha256": "a" * 64}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    publish_sidecar(tmp_path, arrays)
    loaded = open_sidecar(tmp_path, manifest["contract"])
    assert isinstance(loaded["features"], np.memmap)
    assert not loaded["features"].flags.writeable
    np.testing.assert_array_equal(loaded["features"], arrays["features"])
    with pytest.raises(ValueError):
        open_sidecar(tmp_path, {"split": "val"})
    loaded["features"]._mmap.close()
    del loaded
    (tmp_path / "features.npy").write_bytes((tmp_path / "features.npy").read_bytes() + b"broken")
    with pytest.raises(ValueError, match="checksum"):
        open_sidecar(tmp_path, manifest["contract"])


def test_compact_metadata_scene_query_and_source_rejection(tmp_path):
    root = tmp_path / "data"
    version = root / "v1.0-mini"
    version.mkdir(parents=True)
    tables = {"sample": [{"token": "s", "timestamp": 0, "scene_token": "sc"}],
        "scene": [{"token": "sc", "name": "scene", "log_token": "l"}],
        "log": [{"token": "l", "location": "location"}],
        "category": [{"token": "c", "name": "vehicle.car"}],
        "instance": [{"token": "i", "category_token": "c"}],
        "sample_annotation": [{"token": "ann", "instance_token": "i", "sample_token": "s",
                               "translation": [1, 2, 0], "rotation": [1, 0, 0, 0]}]}
    for name, records in tables.items():
        (version / (name + ".json")).write_text(json.dumps(records))
    maps = root / "maps" / "expansion"
    maps.mkdir(parents=True)
    (maps / "location.json").write_text("{}")
    db = tmp_path / "index.sqlite"
    metadata = CompactMetadata(root, "v1.0-mini", db)
    index = metadata.scene_index("scene")
    assert index.annotations["s"]["i"]["translation"][:2] == [1., 2.]
    assert metadata.scene_index("missing").annotations == {}
    (version / "sample_annotation.json").write_text("[]")
    with pytest.raises(ValueError, match="source"):
        CompactMetadata(root, "v1.0-mini", db)



def test_vectorized_neighbor_query_matches_reference_with_rotation_delay_and_missing():
    from trajectory_prediction.context.neighbors import NeighborConfig, NeighborIndex
    from trajectory_prediction.context.scaled_neighbors import query_neighbors
    from trajectory_prediction.online.observations import ObservationView
    samples = [{"token": str(k), "timestamp": k * 500_000, "scene_token": "sc"} for k in range(7)]
    records = [{"token": f"{token}-{k}", "instance_token": token, "sample_token": str(k),
                "translation": [float(k + i), float(i), 0]} for k in range(7)
               for i, token in enumerate(["self", "a", "b", "c", "cone"]) if not (token == "b" and k == 2)]
    categories = {"self": "vehicle.car", "a": "vehicle.car", "b": "vehicle.bicycle",
                  "c": "human.pedestrian.adult", "cone": "movable_object.trafficcone"}
    for delay in (0, 500_000):
        index = NeighborIndex(samples, records, categories, {"sc": "scene"}, NeighborConfig(latency_us=delay))
        for yaw in (0., .7, np.pi / 2):
            for timestamp in (500_000, 2_000_000, 4_000_001):
                obs = ObservationView(1, timestamp, np.zeros((5, 2)), np.array([.3, -.5]), ())
                args = (index, obs, "self", "scene", np.array([.2, .1]), yaw)
                actual = query_neighbors(*args)
                expected = index.query(*args[1:])
                assert actual.instance_tokens == expected.instance_tokens
                for field in ("mask", "velocity_mask", "source_timestamp_us", "available_timestamp_us", "annotation_tokens"):
                    np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))
                np.testing.assert_allclose(actual.features, expected.features, atol=1e-5, rtol=1e-5)



def archive_fixture(root):
    from test_context_cache import contract, records

    from trajectory_prediction.utils.provenance import sha256_file
    values, expected = records(), contract()
    root.mkdir(parents=True, exist_ok=True)
    np.savez(root / "archive.npz", **values)
    for name, value in values.items():
        np.save(root / (name + ".npy"), value)
    manifest = {"contract": expected, "npz_file": "archive.npz", "npz_sha256": sha256_file(root / "archive.npz")}
    (root / "manifest.json").write_text(json.dumps(manifest))
    return values, expected


def test_mmap_sidecars_cannot_change_content_or_omit_arrays_under_same_archive(tmp_path):
    values, _expected = archive_fixture(tmp_path / "changed")
    np.save(tmp_path / "changed" / "features.npy", values["features"] + 999)
    with pytest.raises(ValueError, match="archive"):
        publish_sidecar(tmp_path / "changed", values)
    values, _expected = archive_fixture(tmp_path / "missing")
    with pytest.raises(ValueError, match="schema|archive"):
        publish_sidecar(tmp_path / "missing", {name: value for name, value in values.items() if name != "mask"})


def test_mmap_contract_derived_shapes_are_checked_before_publication(tmp_path):
    values, expected = archive_fixture(tmp_path)
    values["features"] = np.zeros((1, 11, 1, 1, 4), np.float32)
    np.savez(tmp_path / "archive.npz", **values)
    for name, value in values.items():
        np.save(tmp_path / (name + ".npy"), value)
    from trajectory_prediction.utils.provenance import sha256_file
    manifest = {"contract": expected, "npz_file": "archive.npz", "npz_sha256": sha256_file(tmp_path / "archive.npz")}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="shape"):
        publish_sidecar(tmp_path, values)


@pytest.mark.parametrize("chunk", [1, 2, 7])
def test_streamed_numbers_cross_chunks(tmp_path, chunk):
    path = tmp_path / "numbers.json"
    path.write_text("[1e2,-3.25,1.0e-4]")
    assert list(stream_json_array(path, chunk)) == [100., -3.25, .0001]

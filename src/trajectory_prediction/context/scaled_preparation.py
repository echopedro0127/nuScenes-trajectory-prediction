"""Whole-split context preparation with bounded scene metadata and mmap output."""
import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import IDENTIFIERS, digest_json
from trajectory_prediction.context.maps import MapConfig, MapProvider
from trajectory_prediction.context.mmap_cache import open_sidecar, publish_sidecar
from trajectory_prediction.context.neighbor_cache import FIELDS
from trajectory_prediction.context.neighbor_preparation import neighbor_contract
from trajectory_prediction.context.neighbors import NeighborConfig
from trajectory_prediction.context.preparation import context_contract, load_predictions
from trajectory_prediction.context.scaled_neighbors import query_neighbors
from trajectory_prediction.context.streaming_index import CompactMetadata
from trajectory_prediction.data.cache import _replace_manifest, load_cache
from trajectory_prediction.data.nuscenes_adapter import quaternion_yaw
from trajectory_prediction.online.observations import replay_observations
from trajectory_prediction.utils.provenance import sha256_file


def _allocate(root, arrays, shapes):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    result = {}
    for name in IDENTIFIERS:
        np.save(root / (name + ".npy"), arrays[name])
        result[name] = np.load(root / (name + ".npy"), mmap_mode="r")
    for name, (shape, dtype) in shapes.items():
        result[name] = np.lib.format.open_memmap(root / (name + ".npy"), mode="w+", shape=shape, dtype=dtype)
    return result


def _publish(root, arrays, contract, prefix, report):
    root = Path(root)
    for array in arrays.values():
        if hasattr(array, "flush"):
            array.flush()
    fd, name = tempfile.mkstemp(dir=root, prefix=".building-", suffix=".npz")
    os.close(fd)
    temporary = Path(name)
    try:
        # Uncompressed NPZ streams mmap bytes; avoids a second dense resident copy.
        np.savez(temporary, **arrays)
        digest = sha256_file(temporary)
        filename = f"{prefix}-{digest}.npz"
        os.replace(temporary, root / filename)
        manifest = {"schema_version": 1, "contract": contract, "contract_sha256": digest_json(contract),
                    "npz_file": filename, "npz_sha256": digest, "preparation": report}
        _replace_manifest(root, manifest)
        publish_sidecar(root, arrays)
        return manifest
    finally:
        temporary.unlink(missing_ok=True)


def prepare_split(metadata, trajectory_root, split, output_root, map_config, config):
    start = time.perf_counter()
    arrays, original = load_cache(Path(trajectory_root), split)
    n, lanes, points, m = len(arrays["sample_token"]), map_config.max_lanes, map_config.points_per_lane, config.max_neighbors
    root = Path(output_root)
    maps = _allocate(root / "map", arrays, {
        "features": ((n, 11, lanes, points, 4), np.float32), "mask": ((n, 11, lanes, points), bool),
        "lane_tokens": ((n, 11, lanes), "U64"), "anchor_timestamp_us": ((n,), np.int64),
        "timestamp_us": ((n, 11), np.int64)})
    neighbors = _allocate(root / "neighbors", arrays, {
        "features": ((n, 11, m, 5, 6), np.float32), "mask": ((n, 11, m, 5), bool),
        "velocity_mask": ((n, 11, m, 5), bool), "instance_tokens": ((n, 11, m), "U32"),
        "annotation_tokens": ((n, 11, m, 5), "U32"), "source_timestamp_us": ((n, 11, m, 5), np.int64),
        "available_timestamp_us": ((n, 11, m, 5), np.int64), "anchor_timestamp_us": ((n,), np.int64),
        "timestamp_us": ((n, 11), np.int64)})
    provider = MapProvider(metadata.root, map_config)
    scenes = sorted(set(arrays["scene_name"].tolist()))
    done, map_covered, neighbor_covered = 0, 0, 0
    for scene_number, scene in enumerate(scenes, 1):
        index = metadata.scene_index(scene, config)
        for i in np.flatnonzero(arrays["scene_name"] == scene):
            sample = metadata.tables["sample"][str(arrays["sample_token"][i])]
            annotation = index.annotations[sample["token"]][str(arrays["instance_token"][i])]
            anchor, yaw = arrays["anchor_xy"][i], float(arrays["anchor_yaw"][i])
            delta = (quaternion_yaw(annotation["rotation"]) - yaw + np.pi) % (2 * np.pi) - np.pi
            if not np.allclose(annotation["translation"][:2], anchor, atol=1e-3, rtol=0) or abs(delta) > 1e-4:
                raise ValueError("compact metadata anchor mismatch")
            scene_record = metadata.tables["scene"][sample["scene_token"]]
            if scene_record["name"] != scene:
                raise ValueError("compact metadata scene mismatch")
            location = metadata.tables["log"][scene_record["log_token"]]["location"]
            lane_map = provider.get(location)
            timestamp = int(sample["timestamp"])
            maps["anchor_timestamp_us"][i] = neighbors["anchor_timestamp_us"][i] = timestamp
            for k, (observation, _) in enumerate(replay_observations(arrays["history_xy"][i], arrays["future_xy"][i], anchor_timestamp_us=timestamp)):
                view = lane_map.query(observation, anchor, yaw)
                adjacent = query_neighbors(index, observation, str(arrays["instance_token"][i]), scene, anchor, yaw)
                if any(len(token) > 64 for token in view.lane_tokens) or any(
                    len(token) > 32 for token in (*adjacent.instance_tokens, *adjacent.annotation_tokens.ravel())
                ):
                    raise ValueError("scaled token storage width exceeded")
                maps["features"][i, k], maps["mask"][i, k], maps["lane_tokens"][i, k] = view.features, view.mask, view.lane_tokens
                for name in FIELDS:
                    neighbors[name][i, k] = getattr(adjacent, name)
                maps["timestamp_us"][i, k] = neighbors["timestamp_us"][i, k] = observation.timestamp_us
                map_covered += int(view.mask.any())
                neighbor_covered += int(adjacent.mask.any())
            done += 1
        if scene_number % 5 == 0 or scene_number == len(scenes):
            print(json.dumps({"split": split, "scenes_done": scene_number, "scenes_total": len(scenes),
                "samples_done": done, "samples_total": n, "elapsed_seconds": round(time.perf_counter() - start, 2)}), flush=True)
    if done != n:
        raise ValueError("not all trajectory rows were prepared")
    report = {"sample_count": n, "update_count": n * 11, "map_covered": map_covered,
              "neighbor_covered": neighbor_covered, "scenes": scenes, "sensor_io": False,
              "query_seconds": time.perf_counter() - start, "storage": "uncompressed_npz_with_readonly_npy_sidecars"}
    map_manifest = _publish(root / "map", maps, context_contract(arrays, original, metadata, map_config), "context", report)
    bound = neighbor_contract(map_manifest, metadata, SimpleNamespace(extra_hashes=metadata.extra_hashes), config)
    neighbor_manifest = _publish(root / "neighbors", neighbors, bound, "neighbors", report)
    print(json.dumps({"split": split, "prepared": n, "total_seconds": time.perf_counter() - start,
                      "map_sha256": map_manifest["npz_sha256"], "neighbor_sha256": neighbor_manifest["npz_sha256"]}), flush=True)
    return map_manifest, neighbor_manifest


def file_provenance(dataroot, version):
    root = Path(dataroot)
    source = {f"{version}/{name}.json": sha256_file(root / version / (name + ".json"))
              for name in ("sample", "scene", "log", "sample_annotation")}
    source.update({f"maps/expansion/{p.name}": sha256_file(p) for p in sorted((root / "maps" / "expansion").glob("*.json"))})
    extra = {f"{version}/{name}.json": sha256_file(root / version / (name + ".json")) for name in ("instance", "category")}
    return SimpleNamespace(provenance=lambda: source, extra_hashes=extra)


def load_scaled_bundle(trajectory_root, context_root, split, dataroot, map_config, config, prediction_path):
    arrays, original = load_cache(Path(trajectory_root), split)
    source = file_provenance(dataroot, original["dataset_version"])
    expected = context_contract(arrays, original, source, map_config)
    root = Path(context_root)
    maps = open_sidecar(root / "map", expected)
    map_manifest = json.loads((root / "map" / "manifest.json").read_text(encoding="utf-8"))
    expected_neighbors = neighbor_contract(map_manifest, source, source, config)
    neighbors = open_sidecar(root / "neighbors", expected_neighbors)
    neighbor_manifest = json.loads((root / "neighbors" / "manifest.json").read_text(encoding="utf-8"))
    if not np.array_equal(maps["timestamp_us"], neighbors["timestamp_us"]):
        raise ValueError("scaled map/neighbor timeline mismatch")
    base, hashes = load_predictions(prediction_path, arrays, expected)
    return (arrays, maps, neighbors, base), map_manifest, neighbor_manifest, hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare full contexts with bounded scene metadata")
    for name in ("trajectory-root", "dataroot", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--map-config", type=Path, default=Path("configs/context/map.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/context/neighbors.yaml"))
    args = parser.parse_args(argv)
    metadata = CompactMetadata(args.dataroot, args.version, args.output_root / "metadata.sqlite")
    for split in args.splits:
        root = args.output_root / split
        if (root / "neighbors" / "mmap_manifest.json").exists():
            arrays, original = load_cache(args.trajectory_root / split, split)
            map_contract = context_contract(arrays, original, metadata, MapConfig(**load_yaml(args.map_config)))
            open_sidecar(root / "map", map_contract)
            manifest = json.loads((root / "map" / "manifest.json").read_text())
            open_sidecar(root / "neighbors", neighbor_contract(manifest, metadata,
                SimpleNamespace(extra_hashes=metadata.extra_hashes), NeighborConfig(**load_yaml(args.config))))
            print(json.dumps({"split": split, "reused_verified_cache": True}), flush=True)
            continue
        prepare_split(metadata, args.trajectory_root / split, split, root,
                      MapConfig(**load_yaml(args.map_config)), NeighborConfig(**load_yaml(args.config)))

"""Neighbor preparation reuses and strictly binds the unchanged P2 map cache."""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import IDENTIFIERS
from trajectory_prediction.context.maps import MapConfig
from trajectory_prediction.context.neighbor_cache import FIELDS, load_neighbors, save_neighbors
from trajectory_prediction.context.neighbors import CATEGORY_PREFIXES, NeighborConfig, NeighborIndex
from trajectory_prediction.context.nuscenes_index import MetadataIndex
from trajectory_prediction.context.preparation import load_bundle
from trajectory_prediction.online.observations import replay_observations


def neighbor_contract(map_manifest, metadata, index, config):
    source = {**metadata.provenance(), **index.extra_hashes}
    parent = map_manifest["contract"]
    return {"schema_version": 1, "modality": "oracle_neighbors", "split": parent["split"],
        "config": asdict(config), "category_prefixes": list(CATEGORY_PREFIXES),
        "source": source, "keys_sha256": parent["keys_sha256"],
        "trajectory_sha256": parent["trajectory_sha256"], "map_contract": parent,
        "map_npz_sha256": map_manifest["npz_sha256"],
        "coordinate_policy": "fixed_anchor_relative_to_current_target",
        "time_policy": "past_two_seconds_five_slots_no_interpolation",
        "velocity_policy": "adjacent_valid_slots_real_dt_anchor_rotated_not_target_relative",
        "oracle": "annotation_interaction_context"}


def load_interaction_bundle(trajectory_root, map_root, neighbor_root, split, dataroot, map_config, neighbor_config):
    arrays, maps, map_manifest = load_bundle(trajectory_root, map_root, split, dataroot, map_config)
    metadata = MetadataIndex(dataroot, map_manifest["contract"]["dataset_version"])
    index = NeighborIndex.from_metadata(metadata, neighbor_config)
    contract = neighbor_contract(map_manifest, metadata, index, neighbor_config)
    neighbors, manifest = load_neighbors(neighbor_root, contract)
    if not np.array_equal(neighbors["timestamp_us"], maps["timestamp_us"]):
        raise ValueError("map/neighbor timeline mismatch")
    return arrays, maps, map_manifest, neighbors, manifest


def prepare_neighbors(trajectory_root, map_root, split, dataroot, map_config, config, output_dir):
    start = time.perf_counter()
    arrays, maps, map_manifest = load_bundle(trajectory_root, map_root, split, dataroot, map_config)
    metadata = MetadataIndex(dataroot, map_manifest["contract"]["dataset_version"])
    index = NeighborIndex.from_metadata(metadata, config)
    contract = neighbor_contract(map_manifest, metadata, index, config)
    n, m = len(arrays["sample_token"]), config.max_neighbors
    result = {name: arrays[name].copy() for name in IDENTIFIERS}
    for name in ("anchor_timestamp_us", "timestamp_us"):
        result[name] = maps[name].copy()
    result.update(features=np.zeros((n, 11, m, 5, 6), np.float32),
        mask=np.zeros((n, 11, m, 5), bool), velocity_mask=np.zeros((n, 11, m, 5), bool),
        instance_tokens=np.full((n, 11, m), "", dtype="U64"),
        annotation_tokens=np.full((n, 11, m, 5), "", dtype="U64"),
        source_timestamp_us=np.zeros((n, 11, m, 5), np.int64),
        available_timestamp_us=np.zeros((n, 11, m, 5), np.int64))
    loaded = time.perf_counter()
    for i in range(n):
        for k, (observation, _) in enumerate(replay_observations(
            arrays["history_xy"][i], arrays["future_xy"][i],
            anchor_timestamp_us=int(maps["anchor_timestamp_us"][i])
        )):
            view = index.query(observation, str(arrays["instance_token"][i]), str(arrays["scene_name"][i]),
                               arrays["anchor_xy"][i], float(arrays["anchor_yaw"][i]))
            for name in FIELDS:
                result[name][i, k] = getattr(view, name)
    report = {"sample_count": n, "update_count": n * 11,
        "covered_updates": int(result["mask"].any(axis=(2, 3)).sum()),
        "selected_neighbors": int(result["mask"][..., -1].sum()),
        "valid_history_points": int(result["mask"].sum()), "valid_velocity_points": int(result["velocity_mask"].sum()),
        "metadata_load_seconds": loaded - start, "total_seconds": time.perf_counter() - start,
        "scenes": sorted(set(arrays["scene_name"].tolist())), "oracle": "annotation_interaction_context", "sensor_io": False}
    save_neighbors(output_dir, result, contract, report)
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare causal oracle neighbor histories")
    for name in ("trajectory-cache", "map-cache", "dataroot", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/context/neighbors.yaml"))
    parser.add_argument("--map-config", type=Path, default=Path("configs/context/map.yaml"))
    args = parser.parse_args(argv)
    prepare_neighbors(args.trajectory_cache, args.map_cache, args.split, args.dataroot,
        MapConfig(**load_yaml(args.map_config)), NeighborConfig(**load_yaml(args.config)), args.output_dir)

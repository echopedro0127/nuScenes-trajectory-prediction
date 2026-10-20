"""Prepare and validate map context with keys and source provenance."""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import IDENTIFIERS, key_hash, load_context, save_context
from trajectory_prediction.context.maps import MapConfig, MapProvider
from trajectory_prediction.context.nuscenes_index import MetadataIndex
from trajectory_prediction.data.cache import load_cache
from trajectory_prediction.online.observations import replay_observations
from trajectory_prediction.utils.provenance import sha256_file


def context_contract(arrays, manifest, index, config):
    return {"schema_version": 1, "split": manifest["split"], "config": asdict(config),
            "keys_sha256": key_hash(arrays), "source": index.provenance(),
            "dataset_version": manifest["dataset_version"],
            "trajectory_sha256": manifest["npz_sha256"],
            "preprocessing_sha256": manifest["preprocessing_config_sha256"],
            "oracle": "target_position_annotations", "map_source_timestamp_us": 0,
            "map_available_timestamp_us": 0, "map_age": "static_not_applicable",
            "coordinate_policy": "fixed_anchor_forward_positive_y",
            "time_policy": "fixed_2hz_11_updates"}


def load_predictions(path, arrays):
    path = Path(path)
    sidecar = path.parent / "per_sample.csv"
    keys = pd.read_csv(sidecar, dtype=str, keep_default_na=False)
    for name in IDENTIFIERS:
        if name not in keys or keys[name].tolist() != arrays[name].tolist():
            raise ValueError("base prediction key order mismatch")
    with np.load(path, allow_pickle=False) as data:
        prediction = data["prediction"].copy()
        if "target" not in data or not np.array_equal(data["target"], arrays["future_xy"]):
            raise ValueError("base prediction target/cache mismatch")
    if prediction.shape != arrays["future_xy"].shape or not np.issubdtype(prediction.dtype, np.floating) or not np.isfinite(prediction).all():
        raise ValueError("invalid base prediction")
    return prediction.astype(np.float32), {"npz_sha256": sha256_file(path), "keys_sha256": sha256_file(sidecar)}


def load_bundle(trajectory_root, context_root, split, dataroot, config):
    arrays, trajectory_manifest = load_cache(Path(trajectory_root), split)
    index = MetadataIndex(dataroot, trajectory_manifest["dataset_version"])
    expected = context_contract(arrays, trajectory_manifest, index, config)
    context, manifest = load_context(context_root, expected)
    return arrays, context, manifest


def prepare(trajectory_root, split, dataroot, config, output_dir):
    start = time.perf_counter()
    arrays, manifest = load_cache(Path(trajectory_root), split)
    index = MetadataIndex(dataroot, manifest["dataset_version"])
    contract = context_contract(arrays, manifest, index, config)
    provider = MapProvider(dataroot, config)
    n = len(arrays["sample_token"])
    context = {name: arrays[name].copy() for name in IDENTIFIERS}
    context.update(features=np.zeros((n, 11, config.max_lanes, config.points_per_lane, 4), np.float32),
                   mask=np.zeros((n, 11, config.max_lanes, config.points_per_lane), bool),
                   lane_tokens=np.full((n, 11, config.max_lanes), "", dtype="U64"),
                   anchor_timestamp_us=np.zeros(n, np.int64), timestamp_us=np.zeros((n, 11), np.int64))
    metadata_seconds = time.perf_counter() - start
    for i in range(n):
        anchor_time, location = index.anchor(*(arrays[name][i] for name in IDENTIFIERS),
                                             arrays["anchor_xy"][i], float(arrays["anchor_yaw"][i]))
        context["anchor_timestamp_us"][i] = anchor_time
        lane_map = provider.get(location)
        for k, (observation, _) in enumerate(replay_observations(
            arrays["history_xy"][i], arrays["future_xy"][i], anchor_timestamp_us=anchor_time
        )):
            view = lane_map.query(observation, arrays["anchor_xy"][i], float(arrays["anchor_yaw"][i]))
            context["features"][i, k] = view.features
            context["mask"][i, k] = view.mask
            context["lane_tokens"][i, k] = view.lane_tokens
            context["timestamp_us"][i, k] = observation.timestamp_us
    report = {"metadata_seconds": metadata_seconds, "total_seconds": time.perf_counter() - start,
              "sample_count": n, "update_count": n * 11,
              "covered_updates": int(context["mask"].any(axis=(2, 3)).sum()),
              "scenes": sorted(set(arrays["scene_name"].tolist())),
              "sensor_io": False, "timing_scope": "metadata_map_load_discretization_query_cache_prepare"}
    save_context(output_dir, context, contract, report)
    print(json.dumps(report, indent=2))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="Prepare causal lane map context without sensor I/O")
    parser.add_argument("--trajectory-cache", type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--dataroot", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/context/map.yaml"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    prepare(args.trajectory_cache, args.split, args.dataroot, MapConfig(**load_yaml(args.config)), args.output_dir)

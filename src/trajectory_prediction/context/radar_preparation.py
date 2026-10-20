"""Independent mini Radar caches bound to measurements, configuration and raw files."""
import argparse
import json
import os
import re
import tempfile
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import IDENTIFIERS, digest_json, key_hash
from trajectory_prediction.context.preparation import load_predictions
from trajectory_prediction.context.radar import RadarConfig
from trajectory_prediction.context.radar_index import CHANNELS, TABLES, RadarIndex
from trajectory_prediction.data.cache import _replace_manifest, load_cache
from trajectory_prediction.online.observation_noise import (
    CONDITIONS,
    filtered_positions,
    replay_states,
)
from trajectory_prediction.utils.provenance import sha256_file

STATE_FIELDS = ("position", "velocity", "filtered_position", "filtered_velocity", "source_us", "available_us", "fresh")


def prepare_arrays(arrays, index):
    n, slots = len(arrays["sample_token"]), 5 * index.config.sweeps_per_channel
    shape = (len(CONDITIONS), n, 11)
    output = {name: arrays[name].copy() for name in IDENTIFIERS}
    output["anchor_timestamp_us"] = np.zeros(n, np.int64)
    output["timestamp_us"] = np.zeros((n, 11), np.int64)
    for name in ("position", "velocity", "filtered_position", "filtered_velocity"):
        output[name] = np.zeros((*shape, 2), np.float32)
    for name in ("source_us", "available_us"):
        output[name] = np.zeros(shape, np.int64)
    for name in ("fresh", "present"):
        output[name] = np.zeros(shape, bool)
    output["radar_features"] = np.zeros((*shape, 13), np.float32)
    output["point_count"] = np.zeros(shape, np.int32)
    output["pdh0_median"] = np.zeros(shape, np.float32)
    output["frame_tokens"] = np.full((*shape, slots), "", dtype="U32")
    for name in ("frame_source_us", "frame_available_us"):
        output[name] = np.zeros((*shape, slots), np.int64)
    started = time.perf_counter()
    for i in range(n):
        sample = index.samples[str(arrays["sample_token"][i])]
        if index.scenes[sample["scene_token"]]["name"] != arrays["scene_name"][i]:
            raise ValueError("radar anchor scene mismatch")
        anchor = sample["timestamp"]
        output["anchor_timestamp_us"][i] = anchor
        output["timestamp_us"][i] = anchor + np.arange(1, 12) * 500000
        key = str(arrays["instance_token"][i]) + "/" + str(arrays["sample_token"][i])
        states = [list(replay_states(arrays["history_xy"][i], arrays["future_xy"][i, :11], anchor, key, condition)) for condition in CONDITIONS]
        for c, stream in enumerate(states):
            output["filtered_position"][c, i], output["filtered_velocity"][c, i] = filtered_positions(arrays["history_xy"][i], anchor, stream)
        for k in range(11):
            for c in range(len(CONDITIONS)):
                state = states[c][k]
                view = index.query(state, str(arrays["scene_name"][i]), arrays["anchor_xy"][i], float(arrays["anchor_yaw"][i]))
                output["position"][c, i, k], output["velocity"][c, i, k] = state.position, state.velocity
                for name in ("source_us", "available_us", "fresh"):
                    output[name][c, i, k] = getattr(state, name)
                output["radar_features"][c, i, k] = view.features
                output["point_count"][c, i, k], output["present"][c, i, k] = view.point_count, view.present
                output["pdh0_median"][c, i, k] = view.pdh0_median
                length = len(view.frame_tokens)
                if any(len(token) > 32 for token in view.frame_tokens):
                    raise ValueError("radar token too long")
                for name in ("frame_tokens", "frame_source_us", "frame_available_us"):
                    output[name][c, i, k, :length] = getattr(view, name)
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(json.dumps({"prepared_samples": i + 1, "total_samples": n, "condition_updates": (i + 1) * 55,
                "seconds": round(time.perf_counter() - started, 2), "used_pcd_files": len(index.used_files)}), flush=True)
    return output


def validate_radar(arrays, contract):
    config = RadarConfig(**contract["config"])
    source = contract.get("source", {})
    required = {contract.get("dataset_version", "v1.0-mini") + "/" + name + ".json" for name in TABLES}
    if not isinstance(source, dict) or not required <= set(source):
        raise ValueError("radar source metadata list incomplete")
    n, c = len(arrays["instance_token"]), len(CONDITIONS)
    expected = {**{name: (n,) for name in IDENTIFIERS}, "anchor_timestamp_us": (n,), "timestamp_us": (n, 11)}
    expected.update({name: (c, n, 11, 2) for name in ("position", "velocity", "filtered_position", "filtered_velocity")})
    expected.update({name: (c, n, 11) for name in ("source_us", "available_us", "fresh", "present", "point_count", "pdh0_median")})
    expected["radar_features"] = (c, n, 11, 13)
    expected.update({name: (c, n, 11, 5 * config.sweeps_per_channel) for name in ("frame_tokens", "frame_source_us", "frame_available_us")})
    if set(arrays) != set(expected) or not n:
        raise ValueError("radar array schema mismatch")
    for name, shape in expected.items():
        value = arrays[name]
        dtype = "U" if name in (*IDENTIFIERS, "frame_tokens") else "b" if name in ("fresh", "present") else "i" if name in ("anchor_timestamp_us", "timestamp_us", "source_us", "available_us", "point_count", "frame_source_us", "frame_available_us") else "f"
        if value.shape != shape or value.dtype.kind != dtype or (dtype not in ("U", "b") and not np.isfinite(value).all()):
            raise ValueError("radar array shape/dtype mismatch: " + name)
    if key_hash(arrays) != contract["keys_sha256"] or len(set(zip(arrays["instance_token"].tolist(), arrays["sample_token"].tolist(), strict=True))) != n:
        raise ValueError("radar key order mismatch")
    times = arrays["timestamp_us"]
    if np.any(arrays["anchor_timestamp_us"] < 2000000) or not np.array_equal(times, arrays["anchor_timestamp_us"][:, None] + np.arange(1, 12) * 500000):
        raise ValueError("radar update timeline mismatch")
    source, available = arrays["source_us"], arrays["available_us"]
    if np.any(source < 0) or np.any(source > available) or np.any(available > times) or np.any(np.diff(source, axis=-1) < 0):
        raise ValueError("position availability mismatch")
    features, count, present = arrays["radar_features"], arrays["point_count"], arrays["present"]
    if np.any(count < 0) or not np.array_equal(present, count > 0) or np.any(features[~present] != 0):
        raise ValueError("radar mask/count mismatch")
    if not np.allclose(features[..., 0], np.log1p(count) / np.log(129), atol=1e-6) or np.any(np.abs(features[..., 1:3]) > 1.00001) or np.any(features[..., 5:7] < 0) or np.any((features[..., 8:] < 0) | (features[..., 8:] > 1)) or np.any(np.diff(features[..., 9:12], axis=-1) < 0):
        raise ValueError("radar statistics semantics mismatch")
    tokens, frame_source, frame_available = (arrays[name] for name in ("frame_tokens", "frame_source_us", "frame_available_us"))
    valid = tokens != ""
    if np.any(present & ~valid.any(-1)):
        raise ValueError("radar points without source frames")
    if np.any(frame_source[~valid] != 0) or np.any(frame_available[~valid] != 0) or np.any(frame_source[valid] < 0) or np.any(frame_available[valid] != frame_source[valid]) or np.any((frame_source > times[None, ..., None]) & valid) or np.any(((times[None, ..., None] - frame_source) > config.window_us) & valid):
        raise ValueError("radar frame availability mismatch")


def save_radar(root, arrays, contract):
    validate_radar(arrays, contract)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=root, prefix=".radar-", suffix=".npz")
    os.close(fd)
    try:
        np.savez_compressed(temporary, **arrays)
        digest = sha256_file(Path(temporary))
        filename = "radar-" + digest + ".npz"
        os.replace(temporary, root / filename)
        _replace_manifest(root, {"schema_version": 1, "contract": contract, "contract_sha256": digest_json(contract), "npz_file": filename, "npz_sha256": digest})
    finally:
        Path(temporary).unlink(missing_ok=True)


def load_radar(root, expected_contract, dataroot):
    root, dataroot = Path(root), Path(dataroot).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    contract = manifest.get("contract")
    if manifest.get("schema_version") != 1 or contract != expected_contract or manifest.get("contract_sha256") != digest_json(contract):
        raise ValueError("radar contract mismatch")
    match = re.fullmatch(r"radar-([0-9a-f]{64})\.npz", manifest.get("npz_file", ""))
    if not match or match[1] != manifest.get("npz_sha256") or sha256_file(root / match[0]) != match[1]:
        raise ValueError("radar archive checksum mismatch")
    for name, digest in contract["source"].items():
        path = (dataroot / name).resolve()
        if not path.is_relative_to(dataroot) or sha256_file(path) != digest:
            raise ValueError("radar source checksum mismatch")
    with np.load(root / match[0], allow_pickle=False) as payload:
        arrays = {name: payload[name] for name in payload.files}
    validate_radar(arrays, contract)
    return arrays, manifest


def audit_bundle(arrays, cache, contract, dataroot):
    """Rebuild the required source set and causal state independently of cached claims."""
    if contract.get("perturbation_seed") != 2027:
        raise ValueError("radar perturbation seed mismatch")
    index = RadarIndex(dataroot, contract["dataset_version"], RadarConfig(**contract["config"]))
    required = set(index.source_hashes)
    if any(contract["source"].get(name) != digest for name, digest in index.source_hashes.items()):
        raise ValueError("radar metadata source mismatch")
    slots = 5 * index.config.sweeps_per_channel
    for i, sample_token in enumerate(arrays["sample_token"]):
        sample = index.samples[str(sample_token)]
        anchor = sample["timestamp"]
        scene = str(arrays["scene_name"][i])
        if anchor != cache["anchor_timestamp_us"][i] or index.scenes[sample["scene_token"]]["name"] != scene:
            raise ValueError("radar anchor/source mismatch")
        key = str(arrays["instance_token"][i]) + "/" + str(sample_token)
        for c, condition in enumerate(CONDITIONS):
            states = list(replay_states(arrays["history_xy"][i], arrays["future_xy"][i, :11], anchor, key, condition))
            for name in ("position", "velocity", "source_us", "available_us", "fresh"):
                expected = np.array([getattr(state, name) for state in states], dtype=cache[name].dtype)
                if not np.array_equal(expected, cache[name][c, i]):
                    raise ValueError("cached position state differs from fixed causal replay")
            position, velocity = filtered_positions(arrays["history_xy"][i], anchor, states)
            if not np.array_equal(position, cache["filtered_position"][c, i]) or not np.array_equal(velocity, cache["filtered_velocity"][c, i]):
                raise ValueError("cached filter state differs from causal replay")
        for k, timestamp in enumerate(cache["timestamp_us"][i]):
            tokens, times = [], []
            for channel in CHANNELS:
                key = scene, channel
                records = index.frames.get(key, [])
                end = int(np.searchsorted(index.times.get(key, []), timestamp, side="right"))
                for row in records[max(0, end - index.config.sweeps_per_channel):end][::-1]:
                    if timestamp - row["timestamp"] <= index.config.window_us:
                        tokens.append(row["token"])
                        times.append(row["timestamp"])
                        required.add(row["filename"])
            tokens += [""] * (slots - len(tokens))
            times += [0] * (slots - len(times))
            if not np.all(cache["frame_tokens"][:, i, k] == np.asarray(tokens)) or not np.all(cache["frame_source_us"][:, i, k] == np.asarray(times)) or not np.all(cache["frame_available_us"][:, i, k] == np.asarray(times)):
                raise ValueError("radar frame source selection differs from metadata")
    if set(contract["source"]) != required:
        raise ValueError("radar source PCD list incomplete or unexpected")


def load_radar_bundle(trajectory_root, context_root, split, prediction_path, dataroot, config):
    arrays, original = load_cache(Path(trajectory_root), split)
    manifest = json.loads((Path(context_root) / "manifest.json").read_text(encoding="utf-8"))
    contract = manifest["contract"]
    base, hashes = load_predictions(prediction_path, arrays, {"split": split, "preprocessing_sha256": original["preprocessing_config_sha256"]})
    if contract["trajectory_sha256"] != original["npz_sha256"] or contract["split"] != split or contract["config"] != asdict(config) or contract["keys_sha256"] != key_hash(arrays) or contract["base"] != hashes or contract["conditions"] != list(CONDITIONS):
        raise ValueError("radar source/base/config contract mismatch")
    cache, manifest = load_radar(context_root, contract, dataroot)
    for name in IDENTIFIERS:
        if not np.array_equal(arrays[name], cache[name]):
            raise ValueError("radar trajectory keys differ")
    if contract.get("perturbation_seed") != 2027 or contract.get("filters") != {"invalid_states": [0], "dynprop_states": list(range(7)), "ambig_states": [3]}:
        raise ValueError("radar fixed policy mismatch")
    audit_bundle(arrays, cache, contract, dataroot)
    return arrays, cache, base, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trajectory-root", "prediction-root", "dataroot", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/context/radar.yaml"))
    args = parser.parse_args(argv)
    config = RadarConfig(**load_yaml(args.config))
    for split, prediction in (("mini_train", "gru_train_eval"), ("mini_val", "gru_eval")):
        arrays, original = load_cache(args.trajectory_root / split, split)
        base_contract = {"split": split, "preprocessing_sha256": original["preprocessing_config_sha256"]}
        _, hashes = load_predictions(args.prediction_root / prediction / "predictions.npz", arrays, base_contract)
        index = RadarIndex(args.dataroot, config=config)
        started = time.perf_counter()
        cache = prepare_arrays(arrays, index)
        contract = {"schema_version": 1, "config": asdict(config), "conditions": list(CONDITIONS), "perturbation_seed": 2027,
            "trajectory_sha256": original["npz_sha256"], "split": split, "keys_sha256": key_hash(arrays),
            "source": {**index.source_hashes, **index.used_files}, "base": hashes, "dataset_version": "v1.0-mini",
            "oracle": "initial_anchor_history_and_released_position_annotations_no_bbox_radar_association",
            "filters": {"invalid_states": [0], "dynprop_states": list(range(7)), "ambig_states": [3]},
            "nuscenes_version": version("nuscenes-devkit"), "time_policy": "native_radar_source_equals_availability_fixed_2hz_positions",
            "coordinate_policy": "fixed_anchor_forward_positive_y_velocity_rotation_only", "motion_compensation": "ego_only_no_target_motion"}
        save_radar(args.output_root / split, cache, contract)
        print(json.dumps({"split": split, "samples": len(arrays["sample_token"]), "total_updates": len(arrays["sample_token"]) * 55,
            "covered": int(cache["present"].sum()), "used_pcd_files": len(index.used_files), "seconds": time.perf_counter() - started}), flush=True)

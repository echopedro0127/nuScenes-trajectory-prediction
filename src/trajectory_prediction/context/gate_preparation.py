"""Frozen P4 forecasts under causal map-query packet delivery."""
import argparse
import io
import json
import time
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import IDENTIFIERS, digest_json, key_hash
from trajectory_prediction.context.maps import MapConfig
from trajectory_prediction.context.mmap_cache import open_sidecar
from trajectory_prediction.context.preparation import context_contract, load_predictions
from trajectory_prediction.context.scaled_preparation import file_provenance
from trajectory_prediction.data.cache import load_cache
from trajectory_prediction.online.batched_context import residual_batch
from trajectory_prediction.online.context_training import choose_device, shared_contract
from trajectory_prediction.online.corrector import ResidualCorrector
from trajectory_prediction.online.gate_dataset import load_gate_cache, save_gate_cache
from trajectory_prediction.online.interaction_corrector import InteractionCorrector
from trajectory_prediction.online.interaction_provenance import validate_interaction_metadata
from trajectory_prediction.online.quality_gate import (
    CONDITIONS,
    GateConfig,
    packet_schedule,
    quality_features,
)
from trajectory_prediction.utils.provenance import sha256_file


def generate_candidates(arrays, maps, base, original, map_model, *, root=None, config=None, batch_size=4096, device=None):
    config, device = config or GateConfig(), device or torch.device("cpu")
    original, map_model = original.to(device).eval(), map_model.to(device).eval()
    original.requires_grad_(False)
    map_model.requires_grad_(False)
    n = len(base)
    def allocate(name, shape, dtype):
        if root is None:
            return np.zeros(shape, dtype)
        Path(root).mkdir(parents=True, exist_ok=True)
        return np.lib.format.open_memmap(Path(root) / (name + ".npy"), mode="w+", dtype=dtype, shape=shape)
    values = {}
    fixed = {**{name: arrays[name] for name in IDENTIFIERS}, "base": base, "target": arrays["future_xy"], "history": arrays["history_xy"],
        "anchor_timestamp_us": maps["anchor_timestamp_us"], "timestamp_us": maps["timestamp_us"]}
    for name, value in fixed.items():
        values[name] = allocate(name, value.shape, value.dtype)
        values[name][:] = value
    for name in ("position", "velocity", "previous_velocity"):
        values[name] = allocate(name, (n, 11, 2), np.float32)
    for name in ("cv", "filtered", "original"):
        values[name] = allocate(name, (n, 11, 12, 2), np.float32)
    values["map"] = allocate("map", (6, n, 11, 12, 2), np.float32)
    values["quality"] = allocate("quality", (6, n, 11, 8), np.float32)
    for name in ("age_s", "coverage", "valid", "packet_step", "source_us", "available_us"):
        dtype = np.bool_ if name == "valid" else np.int64 if name in ("packet_step", "source_us", "available_us") else np.float32
        values[name] = allocate(name, (6, n, 11), dtype)
    values["position"][:] = arrays["future_xy"][:, :11]
    previous = np.concatenate((arrays["history_xy"][:, -1:], arrays["future_xy"][:, :10]), 1)
    values["velocity"][:] = (values["position"] - previous) / .5
    initial_velocity = (arrays["history_xy"][:, -1] - arrays["history_xy"][:, -2]) / .5
    values["previous_velocity"][:] = np.concatenate((initial_velocity[:, None], values["velocity"][:, :-1]), 1)
    filter_position, filter_velocity = arrays["history_xy"][:, -1].copy(), initial_velocity.copy()
    for k in range(11):
        lead = (np.arange(12, dtype=np.float32) - k) * .5
        values["cv"][:, k] = values["position"][:, k, None] + values["velocity"][:, k, None] * lead[None, :, None]
        prediction = filter_position + filter_velocity * .5
        residual = values["position"][:, k] - prediction
        filter_position, filter_velocity = prediction + .6 * residual, filter_velocity + .2 / .5 * residual
        values["filtered"][:, k] = filter_position[:, None] + filter_velocity[:, None] * lead[None, :, None]
    for start in range(0, n * 11, batch_size):
        flat = np.arange(start, min(n * 11, start + batch_size))
        i, k = flat // 11, flat % 11
        features, mask = residual_batch(values["position"][i, k], values["position"][i, k] - .5 * values["velocity"][i, k], base[i], k + 1)
        args = torch.from_numpy(features).to(device), torch.from_numpy(mask).to(device)
        with torch.inference_mode():
            offsets = original(*args).cpu().numpy()
        values["original"][i, k] = base[i] + offsets
    for ci, condition in enumerate(CONDITIONS):
        schedule = packet_schedule(condition, config=config)
        values["packet_step"][ci] = schedule["packet_step"]
        values["age_s"][ci] = schedule["age_s"]
        anchor = maps["anchor_timestamp_us"][:, None]
        values["source_us"][ci] = np.where(schedule["packet_step"] > 0, anchor + schedule["packet_step"] * 500000, 0)
        values["available_us"][ci] = np.where(schedule["packet_step"] > 0, anchor + schedule["available_step"] * 500000, 0)
        for start in range(0, n * 11, batch_size):
            flat = np.arange(start, min(n * 11, start + batch_size))
            i, k = flat // 11, flat % 11
            packet = np.maximum(schedule["packet_step"][k] - 1, 0)
            map_points, map_mask = maps["features"][i, packet].copy(), maps["mask"][i, packet].copy()
            coverage = map_mask.mean((1, 2), dtype=np.float64).astype(np.float32)
            coverage[schedule["packet_step"][k] == 0] = 0
            valid = schedule["age_valid"][k] & (coverage > 0)
            map_points[~valid], map_mask[~valid] = 0, False
            features, mask = residual_batch(values["position"][i, k], values["position"][i, k] - .5 * values["velocity"][i, k], base[i], k + 1)
            tensors = [torch.from_numpy(value).to(device) for value in (features, mask, map_points, map_mask,
                np.zeros((len(flat), 1, 1, 8), np.float32), np.zeros((len(flat), 1, 1), bool))]
            with torch.inference_mode():
                candidate = base[i] + map_model(*tensors).cpu().numpy()
            candidate[~valid] = values["original"][i[~valid], k[~valid]]
            values["map"][ci, i, k] = candidate
            values["coverage"][ci, i, k], values["valid"][ci, i, k] = coverage, valid
            values["quality"][ci, i, k] = quality_features(values["position"][i, k], values["velocity"][i, k], values["previous_velocity"][i, k],
                base[i], k + 1, coverage, schedule["age_s"][k], values["cv"][i, k], candidate)
        print(json.dumps({"candidate_condition": condition, "samples": n, "valid": int(values["valid"][ci].sum())}), flush=True)
    return values


def audit_candidates(arrays, maps, base, original, model, cached, *, config=None, batch_size=4096, device=None):
    """Re-forward frozen models in bounded sample batches; labels never enter forward."""
    for name in ("anchor_timestamp_us", "timestamp_us"):
        if not np.array_equal(cached[name], maps[name]):
            raise ValueError("gate parent timeline mismatch")
    for start in range(0, len(base), 128):
        end = min(start + 128, len(base))
        with redirect_stdout(io.StringIO()):
            rebuilt = generate_candidates({k: v[start:end] for k,v in arrays.items()},
                {k: v[start:end] for k,v in maps.items()}, base[start:end], original, model,
                config=config, batch_size=batch_size, device=device)
        for name in ("original", "map"):
            actual = cached[name][:, start:end] if name == "map" else cached[name][start:end]
            if not np.allclose(actual, rebuilt[name], atol=1e-5, rtol=1e-5):
                raise ValueError("P4 candidate reconstruction mismatch: " + name)


def load_frozen_models(checkpoint, map_manifest, base_hashes, scenes, split):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    if payload.get("checkpoint_schema") != 1 or metadata.get("architecture") != "interaction_suite_v1":
        raise ValueError("P6 requires a P4 interaction checkpoint")
    validate_interaction_metadata(metadata)
    if metadata["base_model"] != base_hashes["model_identity"] or metadata["shared_contract"] != shared_contract(map_manifest["contract"]):
        raise ValueError("P4 candidate base/map source mismatch")
    if split in ("train", "train_val"):
        field = "train_provenance" if split == "train" else "selection_provenance"
        expected_scenes = metadata["train_scenes" if split == "train" else "selection_scenes"]
        p = metadata[field]
        if sorted(scenes) != expected_scenes or p["context_sha256"] != map_manifest["npz_sha256"] or p["contract"] != map_manifest["contract"] or p["base"] != base_hashes:
            raise ValueError("P4 candidate train/selection provenance mismatch")
    elif split == "val":
        if set(scenes) & set(metadata["train_scenes"] + metadata["selection_scenes"]):
            raise ValueError("P4 candidate evaluation scene overlap")
    else:
        raise ValueError("unsupported P6 split")
    package = Path(__file__).parents[1]
    for name in ("corrector.py", "interaction_corrector.py", "batched_context.py"):
        if metadata["source_code_sha256"].get(name) != sha256_file(package / "online" / name):
            raise ValueError("P4 candidate executable source mismatch")
    original = ResidualCorrector(hidden_dim=metadata["hidden_dim"])
    original.load_state_dict({name.removeprefix("original."): value for name, value in payload["model"].items() if name.startswith("original.")}, strict=True)
    model = InteractionCorrector(deepcopy(original), "map", metadata["hidden_dim"])
    model.load_state_dict({name.removeprefix("map."): value for name, value in payload["model"].items() if name.startswith("map.")}, strict=True)
    if any(not torch.equal(value, model.fallback.state_dict()[name]) for name, value in original.state_dict().items()):
        raise ValueError("P4 frozen fallback mismatch")
    return original.eval(), model.eval(), metadata


SOURCE_FILES = ("online/quality_gate.py", "online/gate_dataset.py", "context/gate_preparation.py", "online/batched_context.py", "online/corrector.py", "online/interaction_corrector.py")


def preparation_sources():
    package = Path(__file__).parents[1]
    return {name: sha256_file(package / name) for name in SOURCE_FILES}


def verify_references(contract):
    required = {"trajectory_manifest", "trajectory_archive", "map_manifest", "map_archive", "map_sidecar", "map_mask", "base_npz", "base_keys", "base_metrics", "checkpoint"}
    required.update("metadata:" + name for name in contract.get("map_contract", {}).get("source", {}))
    refs = contract.get("references", {})
    if set(refs) != required or not any(name.startswith("metadata:") for name in refs):
        raise ValueError("gate source references incomplete or unexpected")
    for reference in refs.values():
        if sha256_file(Path(reference["path"])) != reference["sha256"]:
            raise ValueError("gate source reference checksum mismatch")


def load_verified_gate_cache(root, split, seed, config=None, device=None):
    root, config = Path(root), config or GateConfig()
    manifest = json.loads((root / "manifest.json").read_text())
    contract = manifest["contract"]
    if contract.get("seed") != seed or contract.get("split") != split or contract.get("gate_config") != asdict(config) or contract.get("conditions") != list(CONDITIONS) or contract.get("preparation_sources") != preparation_sources():
        raise ValueError("gate experiment/source contract mismatch")
    verify_references(contract)
    refs = contract["references"]
    trajectory_root = Path(refs["trajectory_manifest"]["path"]).parent
    map_root = Path(refs["map_manifest"]["path"]).parent
    arrays, trajectory = load_cache(trajectory_root, split)
    map_manifest = json.loads((map_root / "manifest.json").read_text())
    expected_map = map_manifest["contract"]
    if contract["map_contract"] != expected_map or expected_map["trajectory_sha256"] != trajectory["npz_sha256"] or expected_map["keys_sha256"] != key_hash(arrays):
        raise ValueError("gate parent trajectory/map contract mismatch")
    expected_paths = {"trajectory_archive": trajectory_root / trajectory["npz_file"], "map_archive": map_root / map_manifest["npz_file"],
        "map_sidecar": map_root / "mmap_manifest.json", "map_mask": map_root / "mask.npy"}
    base_path = Path(refs["base_npz"]["path"])
    expected_paths.update(base_keys=base_path.parent / "per_sample.csv", base_metrics=base_path.parent / "metrics.json")
    if any(Path(refs[name]["path"]).resolve() != path.resolve() for name, path in expected_paths.items()):
        raise ValueError("gate parent reference paths mismatch")
    for name, digest in expected_map["source"].items():
        reference = refs["metadata:" + name]
        if reference["sha256"] != digest or not Path(reference["path"]).as_posix().endswith("/" + name):
            raise ValueError("gate metadata source mismatch")
    base, hashes = load_predictions(base_path, arrays, expected_map)
    if hashes != contract["base_hashes"] or contract["dataset_identity"] != digest_json({"map_contract": expected_map, "map_sha256": map_manifest["npz_sha256"], "base": hashes}):
        raise ValueError("gate base/dataset identity mismatch")
    original, model, p4 = load_frozen_models(Path(refs["checkpoint"]["path"]), map_manifest, hashes, sorted(set(arrays["scene_name"].tolist())), split)
    if p4["seed"] != seed or p4["run_identity"] != contract["p4_training_identity"]:
        raise ValueError("gate P4 identity mismatch")
    data, manifest = load_gate_cache(root, contract)
    for name, value in {**{key: arrays[key] for key in IDENTIFIERS}, "target": arrays["future_xy"], "history": arrays["history_xy"], "base": base}.items():
        if not np.array_equal(data[name], value):
            raise ValueError("gate cached source arrays mismatch")
    sidecar = json.loads((map_root / "mmap_manifest.json").read_text())
    if sidecar["archive_manifest_sha256"] != refs["map_manifest"]["sha256"] or sidecar["arrays"]["mask"]["sha256"] != refs["map_mask"]["sha256"]:
        raise ValueError("gate parent sidecar identity mismatch")
    maps = open_sidecar(map_root, expected_map)
    mask = maps["mask"]
    for c, condition in enumerate(CONDITIONS):
        schedule = packet_schedule(condition, config=config)
        packet = np.maximum(schedule["packet_step"] - 1, 0)
        for start in range(0, len(base), 256):
            coverage = mask[start:start + 256, packet].mean((2, 3), dtype=np.float64).astype(np.float32)
            coverage[:, schedule["packet_step"] == 0] = 0
            if not np.array_equal(coverage, data["coverage"][c, start:start + 256]):
                raise ValueError("gate parent map coverage mismatch")
    audit_candidates(arrays, maps, base, original, model, data, config=config, device=device)
    return data, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trajectory-root", "context-root", "prediction-root", "dataroot", "p4-training-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--splits", nargs="+", default=["train", "train_val", "val"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029])
    parser.add_argument("--config", type=Path, default=Path("configs/context/gate.yaml"))
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    config, device = GateConfig(**load_yaml(args.config)), choose_device(args.device)
    executable_sources = preparation_sources()
    for split in args.splits:
        begun = time.perf_counter()
        arrays, trajectory = load_cache(args.trajectory_root / split, split)
        provenance = file_provenance(args.dataroot, trajectory["dataset_version"])
        expected = context_contract(arrays, trajectory, provenance, MapConfig())
        map_root = args.context_root / split / "map"
        maps = open_sidecar(map_root, expected)
        map_manifest = json.loads((map_root / "manifest.json").read_text())
        prediction_name = {"train": "gru_train_eval", "train_val": "gru_train_val_eval", "val": "gru_eval"}[split]
        prediction = args.prediction_root / prediction_name / "predictions.npz"
        base, hashes = load_predictions(prediction, arrays, expected)
        sources = {"trajectory_manifest": args.trajectory_root / split / "manifest.json", "trajectory_archive": args.trajectory_root / split / trajectory["npz_file"],
            "map_manifest": map_root / "manifest.json", "map_archive": map_root / map_manifest["npz_file"], "map_sidecar": map_root / "mmap_manifest.json", "map_mask": map_root / "mask.npy", "base_npz": prediction,
            "base_keys": prediction.parent / "per_sample.csv", "base_metrics": prediction.parent / "metrics.json"}
        sources.update({"metadata:" + name: args.dataroot / name for name in expected["source"]})
        refs = {name: {"path": str(path.resolve()), "sha256": sha256_file(path)} for name, path in sources.items()}
        for seed in args.seeds:
            checkpoint = args.p4_training_root / ("seed-" + str(seed)) / "best.pt"
            original, model, metadata = load_frozen_models(checkpoint, map_manifest, hashes, sorted(set(arrays["scene_name"].tolist())), split)
            if metadata["seed"] != seed:
                raise ValueError("P4 seed mismatch")
            root = args.output_root / split / ("seed-" + str(seed))
            candidate_started = time.perf_counter()
            values = generate_candidates(arrays, maps, base, original, model, root=root, config=config, device=device)
            candidate_seconds = time.perf_counter() - candidate_started
            if preparation_sources() != executable_sources:
                raise ValueError("preparation executable changed during run")
            contract = {"candidate_preparation_seconds": candidate_seconds, "candidate_updates_per_second": len(base) * 11 * 6 / candidate_seconds, "gate_config": asdict(config), "split": split, "seed": seed, "conditions": list(CONDITIONS), "keys_sha256": key_hash(arrays),
                "dataset_identity": digest_json({"map_contract": expected, "map_sha256": map_manifest["npz_sha256"], "base": hashes}),
                "map_contract": expected, "base_hashes": hashes, "p4_training_identity": metadata["run_identity"],
                "references": {**refs, "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)}},
                "preparation_sources": executable_sources, "feature_policy": "quality8_observation_and_predicted_candidates_only", "fallback_policy": "expired_map_to_original_gated_to_CV"}
            save_gate_cache(root, values, contract)
            print(json.dumps({"prepared_split": split, "seed": seed, "samples": len(base), "seconds_since_split_start": time.perf_counter() - begun}), flush=True)
        del maps

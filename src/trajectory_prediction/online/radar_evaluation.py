"""Matched-condition Radar forecasts, censored recovery, and raw-input latency."""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.analysis.maneuvers import describe_motion, tag_maneuvers
from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.context.radar import RadarConfig, RadarView
from trajectory_prediction.context.radar_index import RadarIndex
from trajectory_prediction.context.radar_preparation import load_radar_bundle
from trajectory_prediction.online.batched_evaluation import forecast_metrics
from trajectory_prediction.online.context_corrector import _sync
from trajectory_prediction.online.context_evaluator import DEFAULT_THRESHOLDS
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.observation_noise import (
    CONDITIONS,
    replay_states,
    state_residual,
)
from trajectory_prediction.online.radar_corrector import RadarDataset, forward_model
from trajectory_prediction.online.radar_training import load_radar_suite
from trajectory_prediction.online.scaled_evaluation import benchmark_keys
from trajectory_prediction.utils.provenance import sha256_file

STRATEGIES = ("frozen", "sliding", "filtered", "original", "radar", "matched_radar")


def evaluate_radar(data, models, batch_size=1024, device=None):
    if set(models) != {"original", "radar", "matched_radar"} or type(batch_size) is not int or batch_size < 1:
        raise ValueError("complete Radar models and positive batch size required")
    device = device or torch.device("cpu")
    models = models.to(device).eval()
    n = len(data.sample_indices)
    results, timings = {}, {}
    for strategy in STRATEGIES:
        forecast = np.empty((data.size, 12, 2), np.float32)
        _sync(device)
        started = time.perf_counter()
        for start in range(0, data.size, batch_size):
            selected = np.arange(start, min(start + batch_size, data.size))
            c, i, k = data.indices(selected)
            if strategy == "frozen":
                value = data.base[i]
            elif strategy in ("sliding", "filtered"):
                prefix = "filtered_" if strategy == "filtered" else ""
                position = data.cache[prefix + "position"][c, i, k]
                velocity = data.cache[prefix + "velocity"][c, i, k]
                lead = (np.arange(12)[None] - k[:, None]).astype(np.float32) * .5
                value = position[:, None] + velocity[:, None] * lead[..., None]
            else:
                inputs = data.inputs(selected, device)
                with torch.inference_mode():
                    offsets = forward_model(models[strategy], inputs).detach().cpu().numpy()
                value = data.base[i] + offsets
            forecast[start:start + len(selected)] = value
        _sync(device)
        timings[strategy] = time.perf_counter() - started
        shaped = forecast.reshape(5, n, 11, 12, 2)
        target = data.arrays["future_xy"][data.sample_indices]
        results[strategy] = np.stack([forecast_metrics(values, target) for values in shaped])
        print(json.dumps({"evaluated": strategy, "updates": data.size, "ade": float(results[strategy][..., 0].mean()), "batch_seconds": timings[strategy]}), flush=True)
    return results, timings


def recovery_table(results, arrays, seed, threshold=.25):
    rows = []
    for strategy, values in results.items():
        gap = values[4, :, :, 0] - values[0, :, :, 0]
        for i in range(len(gap)):
            matches = [k for k in range(6, 10) if np.all(gap[i, k:k + 2] <= threshold)]
            rows.append({"seed": seed, "strategy": strategy, "instance_token": arrays["instance_token"][i],
                "sample_token": arrays["sample_token"][i], "scene_name": arrays["scene_name"][i], "recovered": bool(matches),
                "recovery_seconds": (matches[0] - 6) * .5 if matches else np.nan, "censor_seconds": 2., "threshold_m": threshold})
    return pd.DataFrame(rows)


def row_key_hash(frame):
    digest = hashlib.sha256()
    fields = ["condition", "instance_token", "sample_token", "scene_name", "update_step", "timestamp_us"]
    for row in frame[fields].itertuples(index=False, name=None):
        digest.update(json.dumps(row, separators=(",", ":")).encode() + b"\n")
    return digest.hexdigest()


def write_radar_outputs(data, results, timings, root, provenance):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    selected = np.arange(data.size)
    c, i, k = data.indices(selected)
    arrays, cache = data.arrays, data.cache
    tags = ["|".join(tag_maneuvers(describe_motion(future, .5), DEFAULT_THRESHOLDS)) for future in arrays["future_xy"]]
    keys = pd.DataFrame({"condition": np.array(CONDITIONS)[c], "instance_token": arrays["instance_token"][i],
        "sample_token": arrays["sample_token"][i], "scene_name": arrays["scene_name"][i], "sample_index": i,
        "update_step": k + 1, "remaining_steps": 11 - k, "timestamp_us": cache["timestamp_us"][i, k],
        "maneuver": np.asarray(tags)[i], "radar_covered": cache["present"][c, i, k], "radar_points": cache["point_count"][c, i, k],
        "position_age_s": (cache["timestamp_us"][i, k] - cache["source_us"][c, i, k]) / 1e6,
        "position_fresh": cache["fresh"][c, i, k], "radar_age_max_s": cache["radar_features"][c, i, k, 11] * 2})
    if keys.duplicated(["condition", "instance_token", "sample_token", "update_step"]).any():
        raise ValueError("duplicate radar evaluation keys")
    scene_frames, condition_frames = [], []
    grouped = {name: [] for name in ("update_step", "remaining_steps", "maneuver")}
    path = root / "per_update.csv.gz"
    for j, strategy in enumerate(STRATEGIES):
        metrics = results[strategy].reshape(-1, 3)
        if metrics.shape != (data.size, 3) or not np.isfinite(metrics).all():
            raise ValueError("invalid Radar evaluation metrics")
        frame = keys.copy()
        frame["seed"], frame["strategy"] = provenance["seed"], strategy
        for col, name in enumerate(("ade", "fde", "jitter")):
            frame[name] = metrics[:, col]
        frame["fallback"] = ~frame.radar_covered if strategy == "radar" else False
        frame.to_csv(path, index=False, mode="w" if j == 0 else "a", header=j == 0, compression="gzip")
        scene = frame.groupby(["condition", "scene_name"], as_index=False).agg(update_count=("ade", "size"),
            ade_sum=("ade", "sum"), fde_sum=("fde", "sum"), jitter_sum=("jitter", "sum"))
        scene["strategy"], scene["seed"] = strategy, provenance["seed"]
        scene_frames.append(scene)
        aggregate = frame.groupby("condition", as_index=False).agg(update_count=("ade", "size"), ade=("ade", "mean"),
            fde=("fde", "mean"), jitter=("jitter", "mean"), radar_coverage=("radar_covered", "mean"),
            mean_points=("radar_points", "mean"), fallback_updates=("fallback", "sum"))
        aggregate["strategy"], aggregate["seed"] = strategy, provenance["seed"]
        condition_frames.append(aggregate)
        for field, frames in grouped.items():
            part = frame.copy()
            if field == "maneuver":
                part[field] = part[field].str.split("|")
                part = part.explode(field)
            values = part.groupby(["condition", field], as_index=False).agg(update_count=("ade", "size"), ade=("ade", "mean"),
                fde=("fde", "mean"), jitter=("jitter", "mean"), radar_coverage=("radar_covered", "mean"), fallback_updates=("fallback", "sum"))
            values["strategy"], values["seed"] = strategy, provenance["seed"]
            frames.append(values)
    pd.concat(scene_frames).to_csv(root / "scene_stats.csv", index=False)
    pd.concat(condition_frames).to_csv(root / "by_condition.csv", index=False)
    for field, filename in (("update_step", "by_update.csv"), ("remaining_steps", "by_horizon.csv"), ("maneuver", "by_maneuver.csv")):
        pd.concat(grouped[field]).to_csv(root / filename, index=False)
    recovery_table(results, arrays, provenance["seed"]).to_csv(root / "recovery.csv", index=False)
    files = ("per_update.csv.gz", "scene_stats.csv", "by_condition.csv", "by_update.csv", "by_horizon.csv", "by_maneuver.csv", "recovery.csv")
    summary = {"provenance": provenance, "sample_count": len(data.sample_indices), "updates_per_condition_strategy": len(data.sample_indices) * 11,
        "all_condition_updates_per_strategy": data.size, "row_count": data.size * 6, "failure_count": 0,
        "ordered_key_sha256": row_key_hash(keys), "sha256": {name: sha256_file(root / name) for name in files},
        "batch_forward_seconds": timings, "timing_scope": "batch_forward_throughput_only_not_update_latency"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def predict_radar(model, state, base, view, device):
    _sync(device)
    started = time.perf_counter_ns()
    features, mask = state_residual(state, base)
    inputs = {name: torch.from_numpy(np.asarray(value)[None].copy()).to(device) for name, value in
        (("features", features), ("mask", mask), ("radar", view.features), ("present", np.asarray(view.present)))}
    _sync(device)
    prepared = time.perf_counter_ns()
    with torch.inference_mode():
        offsets = forward_model(model, inputs)[0].detach().cpu().numpy()
    _sync(device)
    return offsets, {"feature_transfer_ms": (prepared - started) / 1e6, "head_return_ms": (time.perf_counter_ns() - prepared) / 1e6}


def benchmark_radar(data, models, contract, dataroot, root, device, count=256, warmup=16):
    root = Path(root)
    index = RadarIndex(dataroot, config=RadarConfig(**contract["config"]), decoded_cache=False)
    index.used_files = dict(contract["source"])
    keys = benchmark_keys(data.size, count)
    c, i, k = data.indices(keys)
    key_frame = pd.DataFrame({"flat_index": keys, "condition": np.array(CONDITIONS)[c], "instance_token": data.arrays["instance_token"][i],
        "sample_token": data.arrays["sample_token"][i], "scene_name": data.arrays["scene_name"][i], "update_step": k + 1,
        "timestamp_us": data.cache["timestamp_us"][i, k]})
    key_frame.to_csv(root / "latency_keys.csv", index=False)
    measurements = []
    models = models.to(device).eval()
    for mode in ("cached_radar", "raw_radar"):
        sequence = [int(keys[j % len(keys)]) for j in range(warmup)] + keys.tolist()
        for j, selected in enumerate(sequence):
            c, i, k = (int(v[0]) for v in data.indices(np.array([selected])))
            _sync(device)
            started = time.perf_counter_ns()
            key = str(data.arrays["instance_token"][i]) + "/" + str(data.arrays["sample_token"][i])
            state = list(replay_states(data.arrays["history_xy"][i], data.arrays["future_xy"][i, :k + 1],
                int(data.cache["anchor_timestamp_us"][i]), key, CONDITIONS[c]))[-1]
            if mode == "raw_radar":
                view = index.query(state, str(data.arrays["scene_name"][i]), data.arrays["anchor_xy"][i], float(data.arrays["anchor_yaw"][i]))
                profile = dict(index.last_profile)
                if not np.allclose(view.features, data.cache["radar_features"][c, i, k], atol=1e-6, rtol=1e-6):
                    raise ValueError("raw/cached Radar feature parity failure")
            else:
                view = RadarView(data.cache["radar_features"][c, i, k], int(data.cache["point_count"][c, i, k]))
                profile = {"read_decode_ms": 0., "transform_ms": 0., "aggregate_ms": 0., "radar_total_ms": 0.}
            offsets, model_profile = predict_radar(models["radar"], state, data.base[i], view, device)
            forecast = data.base[i, k + 1:] + offsets[k + 1:]
            _sync(device)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if not np.isfinite(forecast).all():
                raise ValueError("nonfinite latency forecast")
            if j >= warmup:
                measurements.append({"mode": mode, "flat_index": selected, "total_ms": elapsed, **profile, **model_profile})
    frame = pd.DataFrame(measurements)
    frame.to_csv(root / "latency_measurements.csv", index=False)
    fields = [name for name in frame if name.endswith("_ms")]
    summary = {"key_count": len(keys), "warmup": warmup, "keys_sha256": sha256_file(root / "latency_keys.csv"),
        "measurements_sha256": sha256_file(root / "latency_measurements.csv"), "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "torch_version": str(torch.__version__),
        "decoded_frame_cache": False, "raw_scope": "position_replay_frame_selection_PCD_read_decode_transform_gate_aggregate_features_transfer_head_output",
        "excluded": "metadata_index_startup_and_physical_cold_disk_guarantee", "inference_batch_size": 1,
        "modes": {mode: {field: {"p50": float(part[field].quantile(.5)), "p95": float(part[field].quantile(.95))} for field in fields}
                  for mode, part in frame.groupby("mode")}}
    (root / "latency.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trajectory-root", "prediction-root", "dataroot", "context-root", "training-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/context/radar.yaml"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    device = choose_device(args.device)
    arrays, cache, base, manifest = load_radar_bundle(args.trajectory_root / "mini_val", args.context_root / "mini_val",
        "mini_val", args.prediction_root / "gru_eval/predictions.npz", args.dataroot, RadarConfig(**load_yaml(args.config)))
    data = RadarDataset(arrays, cache, base)
    for seed in args.seeds:
        checkpoint = args.training_root / ("seed-" + str(seed)) / "best.pt"
        models, training = load_radar_suite(checkpoint, manifest["contract"], data.scenes)
        if training["seed"] != seed:
            raise ValueError("radar checkpoint seed mismatch")
        results, timings = evaluate_radar(data, models, device=device)
        root = args.output_root / ("seed-" + str(seed))
        settings = {field: training[field] for field in ("epochs", "batch_size", "hidden_dim", "patience", "learning_rate", "weight_decay", "smoothness_weight")}
        provenance = {"seed": seed, "evaluation_identity": digest_json(manifest), "evaluation_manifest": manifest,
            "checkpoint_sha256": sha256_file(checkpoint), "training_identity": training["identity"],
            "training_config_sha256": digest_json(settings), "evaluation_source_sha256": sha256_file(Path(__file__)),
            "validation_role": "mini_two_scene_feasibility_not_generalization", "oracle": manifest["contract"]["oracle"]}
        write_radar_outputs(data, results, timings, root, provenance)
        benchmark_radar(data, models, manifest["contract"], args.dataroot, root, device)
        print(json.dumps({"seed": seed, "evaluation_complete": True}), flush=True)

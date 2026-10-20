"""Matched-row causal evaluator; labels terminate at diagnostics and metrics."""
import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.analysis.maneuvers import describe_motion, tag_maneuvers
from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import IDENTIFIERS
from trajectory_prediction.context.maps import MapConfig, MapView
from trajectory_prediction.context.preparation import load_bundle, load_predictions
from trajectory_prediction.models.physics import finite_difference_cv
from trajectory_prediction.online.context_corrector import _sync, profiled_predict_offsets
from trajectory_prediction.online.context_features import residual_features
from trajectory_prediction.online.context_training import choose_device, load_map_checkpoint
from trajectory_prediction.online.evaluator import aggregate_metrics, update_metrics
from trajectory_prediction.online.observations import replay_observations
from trajectory_prediction.utils.provenance import sha256_file

STRATEGIES = ("frozen", "sliding", "corrected", "map_corrected")
DEFAULT_THRESHOLDS = {"low_speed_mps": 2., "turn_yaw_deg": 15.,
                      "high_curvature_yaw_deg": 45., "acceleration_mps2": 1.}


def validate_update_rows(frame):
    required = set(STRATEGIES)
    key_names = ["instance_token", "sample_token", "update_step"]
    if frame.empty or set(frame.strategy) != required or frame.duplicated(key_names + ["strategy"]).any():
        raise ValueError("all strategies must have unique matched update rows")
    if not (frame.groupby(key_names).strategy.nunique() == len(required)).all():
        raise ValueError("strategy update key mismatch")


def evaluate_context_arrays(arrays, context, base, model, device=None,
                            predictor=profiled_predict_offsets, thresholds=None):
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    future = arrays["future_xy"]
    if base.shape != future.shape or base.ndim != 3 or base.shape[1:] != (12, 2) or not len(base):
        raise ValueError("base and target must be nonempty matching [N, 12, 2]")
    if not np.isfinite(base).all():
        raise ValueError("base must be finite")
    rows = []
    for i in range(len(base)):
        # These tags are evaluation-only diagnostics; never model arguments.
        tags = tag_maneuvers(describe_motion(future[i], .5), thresholds or DEFAULT_THRESHOLDS)
        previous = {name: None for name in STRATEGIES}
        iterator = iter(replay_observations(arrays["history_xy"][i], future[i],
                                           anchor_timestamp_us=int(context["anchor_timestamp_us"][i])))
        for k in range(11):
            shared_start = time.perf_counter_ns()
            observation, label = next(iterator)
            step = observation.update_step
            view = MapView(context["features"][i, k], context["mask"][i, k], tuple(context["lane_tokens"][i, k]))
            shared_ms = (time.perf_counter_ns() - shared_start) / 1e6
            covered = bool(view.mask.any())
            for strategy in STRATEGIES:
                _sync(device)
                start = time.perf_counter_ns()
                profile = {"feature_ms": 0., "encoder_head_ms": 0.}
                if strategy == "frozen":
                    forecast = base[i, step:].copy()
                elif strategy == "sliding":
                    with torch.inference_mode():
                        forecast = finite_difference_cv(
                            torch.from_numpy(observation.observed_history[None].copy()).to(device),
                            dt=.5, future_steps=12 - step).detach().cpu().numpy()[0]
                elif strategy == "corrected":
                    features, mask = residual_features(observation, base[i])
                    feature_tensor = torch.from_numpy(features[None]).to(device)
                    mask_tensor = torch.from_numpy(mask[None]).to(device)
                    _sync(device)
                    prepared = time.perf_counter_ns()
                    with torch.inference_mode():
                        offsets = model.fallback(feature_tensor, mask_tensor)[0].detach().cpu().numpy()
                    _sync(device)
                    profile = {"feature_ms": (prepared - start) / 1e6,
                               "encoder_head_ms": (time.perf_counter_ns() - prepared) / 1e6}
                    forecast = base[i, step:] + offsets[step:]
                else:
                    offsets, profile = predictor(model, observation, base[i], view, device)
                    forecast = base[i, step:] + offsets[step:]
                _sync(device)
                latency = (time.perf_counter_ns() - start) / 1e6 + shared_ms
                overlap = forecast if previous[strategy] is None else previous[strategy][1:]
                values = update_metrics(forecast, label.remaining_target, overlap, latency)
                rows.append({**{name: str(arrays[name][i]) for name in IDENTIFIERS},
                    "sample_index": i, "update_step": step, "timestamp_us": observation.timestamp_us,
                    "remaining_steps": 12 - step, "strategy": strategy, "maneuver": "|".join(tags),
                    "map_covered": covered, "valid_map_points": int(view.mask.sum()),
                    "fallback": strategy == "map_corrected" and not covered,
                    "cached_replay_prepare_ms": shared_ms, **profile, **values})
                previous[strategy] = forecast
    frame = pd.DataFrame(rows)
    validate_update_rows(frame)
    return frame


def write_context_outputs(frame, output_dir, provenance):
    validate_update_rows(frame)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "per_update.csv", index=False, lineterminator="\n")
    summaries = {}
    for strategy, group in frame.groupby("strategy"):
        summaries[strategy] = aggregate_metrics(group.to_dict(orient="records"))
        for name in ("feature_ms", "encoder_head_ms", "cached_replay_prepare_ms"):
            summaries[strategy][name + "_p50"] = float(np.percentile(group[name], 50))
            summaries[strategy][name + "_p95"] = float(np.percentile(group[name], 95))
    mapped = frame[frame.strategy == "map_corrected"]
    summary = {"provenance": provenance, "strategies": summaries,
        "sample_count": int(mapped.sample_index.nunique()), "update_count": len(mapped),
        "coverage": float(mapped.map_covered.mean()), "fallback_updates": int(mapped.fallback.sum()),
        "failure_count": 0, "oracle": "target_position_annotations",
        "map_timestamps": {"source_us": 0, "available_us": 0, "age": "static_not_applicable"},
        "timing_scope": "cached_replay_including_label_snapshot_plus_input_transfer_and_prediction; excludes_disk_map_load_and_discretization",
        "encoder_head_scope": "encoder_head_and_return_transfer; synchronized; no_warmup; feasibility_only"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for column, filename in (("update_step", "by_update.csv"), ("remaining_steps", "by_horizon.csv"), ("maneuver", "by_maneuver.csv")):
        grouped = frame.copy()
        if column == "maneuver":
            grouped[column] = grouped[column].str.split("|")
            grouped = grouped.explode(column)
        grouped.groupby(["strategy", column], as_index=False).agg(
            update_count=("sample_index", "size"), ade=("ade", "mean"), fde=("fde", "mean"),
            jitter=("jitter", "mean"), coverage=("map_covered", "mean"), latency_ms=("latency_ms", "mean")
        ).to_csv(root / filename, index=False, lineterminator="\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate matched-row causal Map-only corrections")
    for name in ("trajectory-cache", "context-cache", "base-predictions", "checkpoint", "dataroot", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/context/map.yaml"))
    parser.add_argument("--maneuver-config", type=Path, default=Path("configs/analysis/maneuvers.yaml"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    arrays, context, manifest = load_bundle(args.trajectory_cache, args.context_cache, args.split,
                                           args.dataroot, MapConfig(**load_yaml(args.config)))
    base, prediction_hashes = load_predictions(args.base_predictions, arrays)
    model, checkpoint = load_map_checkpoint(args.checkpoint, manifest["contract"], arrays["scene_name"].tolist())
    device = choose_device(args.device)
    frame = evaluate_context_arrays(arrays, context, base, model, device,
                                     thresholds=load_yaml(args.maneuver_config))
    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    summary = write_context_outputs(frame, args.output_dir, {
        "git_commit": git_commit, "device": str(device), "split": args.split,
        "context_sha256": manifest["npz_sha256"], "context_contract": manifest["contract"],
        "base_predictions": prediction_hashes, "checkpoint_sha256": sha256_file(args.checkpoint),
        "checkpoint_metadata": checkpoint["metadata"], "scenes": sorted(set(arrays["scene_name"].tolist())),
        "maneuver_config_sha256": sha256_file(args.maneuver_config)})
    print(json.dumps({name: summary[name] for name in ("sample_count", "update_count", "coverage", "fallback_updates", "failure_count")}, indent=2))

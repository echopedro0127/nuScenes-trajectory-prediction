"""Batched forecasts and exactly aligned remaining-horizon metrics."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.analysis.maneuvers import describe_motion, tag_maneuvers
from trajectory_prediction.online.batched_context import forward_inputs
from trajectory_prediction.online.context_corrector import _sync
from trajectory_prediction.online.context_evaluator import DEFAULT_THRESHOLDS
from trajectory_prediction.online.interaction_corrector import InteractionCorrector
from trajectory_prediction.online.interaction_report import STRATEGIES
from trajectory_prediction.utils.provenance import sha256_file


def forecast_metrics(forecast, target):
    if forecast.shape != (len(target), 11, 12, 2) or not np.isfinite(forecast).all():
        raise ValueError("forecasts must be finite [N, 11, 12, 2]")
    mask = np.arange(12)[None] >= np.arange(1, 12)[:, None]
    count = mask.sum(-1)
    error = np.linalg.norm(forecast - target[:, None], axis=-1)
    ade = (error * mask[None]).sum(-1, dtype=np.float64) / count[None]
    fde = error[..., -1]
    jitter = np.zeros_like(ade)
    delta = np.linalg.norm(forecast[:, 1:] - forecast[:, :-1], axis=-1)
    jitter[:, 1:] = (delta * mask[None, 1:]).sum(-1, dtype=np.float64) / count[None, 1:]
    return np.stack((ade, fde, jitter), axis=-1)


def evaluate_batches(data, models, batch_size=4096, device=None):
    if set(models) != set(STRATEGIES) - {"frozen", "sliding"}:
        raise ValueError("complete learned suite required")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch size must be positive integer")
    device = device or torch.device("cpu")
    models = models.to(device).eval()
    results, timings = {}, {}
    n = len(data.base)
    for strategy in STRATEGIES:
        _sync(device)
        started = time.perf_counter()
        forecast = np.empty((n * 11, 12, 2), np.float32)
        for start in range(0, data.size, batch_size):
            indices = np.arange(start, min(start + batch_size, data.size))
            rows, _, steps = data._indices(indices)
            if strategy == "frozen":
                value = data.base[rows]
            elif strategy == "sliding":
                current = data.arrays["future_xy"][rows, steps - 1]
                previous = data.arrays["future_xy"][rows, np.maximum(steps - 2, 0)].copy()
                previous[steps == 1] = data.arrays["history_xy"][rows[steps == 1], -1]
                velocity = (current - previous) / np.float32(.5)
                lead = (np.arange(12, dtype=np.float32)[None] - steps[:, None] + 1).astype(np.float32) * np.float32(.5)
                value = current[:, None] + velocity[:, None] * lead[..., None]
            else:
                model = models[strategy]
                inputs = data.inputs(indices, device, context=model.mode if isinstance(model, InteractionCorrector) else False)
                with torch.inference_mode():
                    offsets = forward_inputs(model, inputs).detach().cpu().numpy()
                value = data.base[rows] + offsets
            forecast[start:start + len(indices)] = value
        _sync(device)
        timings[strategy] = time.perf_counter() - started
        results[strategy] = forecast_metrics(forecast.reshape(n, 11, 12, 2), data.arrays["future_xy"])
        print(json.dumps({"evaluated_strategy": strategy, "update_count": data.size,
            "batch_seconds": timings[strategy], "ade": float(results[strategy][..., 0].mean())}), flush=True)
    return results, timings


def key_frame(data):
    n = len(data.base)
    rows = np.repeat(np.arange(n), 11)
    steps = np.tile(np.arange(1, 12), n)
    arrays, maps, neighbors = data.arrays, data.maps, data.neighbors
    valid = neighbors["mask"]
    counts = valid[..., -1].sum(-1).reshape(-1)
    age = np.where(valid[..., -1], neighbors["timestamp_us"][..., None] - neighbors["source_timestamp_us"][..., -1], 0).max(-1).reshape(-1)
    tags = ["|".join(tag_maneuvers(describe_motion(future, .5), DEFAULT_THRESHOLDS)) for future in arrays["future_xy"]]
    return pd.DataFrame({"instance_token": arrays["instance_token"][rows], "sample_token": arrays["sample_token"][rows],
        "scene_name": arrays["scene_name"][rows], "sample_index": rows, "update_step": steps,
        "remaining_steps": 12 - steps, "timestamp_us": maps["timestamp_us"].reshape(-1),
        "maneuver": np.repeat(tags, 11), "map_covered": maps["mask"].any(axis=(2, 3)).reshape(-1),
        "neighbor_covered": counts > 0, "neighbor_count": counts,
        "neighbor_history_points": valid.sum(axis=(2, 3)).reshape(-1),
        "neighbor_velocity_points": neighbors["velocity_mask"].sum(axis=(2, 3)).reshape(-1),
        "neighbor_frame_age_us": age})


def ordered_key_hash(keys):
    digest = hashlib.sha256()
    names = ("instance_token", "sample_token", "scene_name", "update_step", "timestamp_us")
    for values in keys[list(names)].itertuples(index=False, name=None):
        digest.update(json.dumps(values, separators=(",", ":")).encode() + b"\n")
    return digest.hexdigest()


def write_batch_outputs(data, results, timings, output_dir, provenance):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    keys = key_frame(data)
    if keys.duplicated(["instance_token", "sample_token", "update_step"]).any():
        raise ValueError("duplicate evaluation update keys")
    path = root / "per_update.csv.gz"
    summaries, scene_frames, group_frames = {}, [], {"update_step": [], "remaining_steps": [], "maneuver": []}
    for index, strategy in enumerate(STRATEGIES):
        metric = results[strategy]
        if metric.shape != (len(data.base), 11, 3) or not np.isfinite(metric).all():
            raise ValueError("evaluation metric shape/finite mismatch")
        frame = keys.copy()
        frame["seed"], frame["strategy"] = provenance["seed"], strategy
        for j, name in enumerate(("ade", "fde", "jitter")):
            frame[name] = metric[..., j].reshape(-1)
        if strategy == "map":
            frame["fallback"] = ~frame.map_covered
        elif strategy == "neighbors":
            frame["fallback"] = ~frame.neighbor_covered
        elif strategy == "fusion":
            frame["fallback"] = ~(frame.map_covered | frame.neighbor_covered)
        else:
            frame["fallback"] = False
        frame.to_csv(path, index=False, mode="w" if index == 0 else "a", header=index == 0,
                     compression={"method": "gzip", "compresslevel": 1}, lineterminator="\n")
        summaries[strategy] = {name: float(frame[name].mean()) for name in ("ade", "fde", "jitter")}
        summaries[strategy].update(sample_count=len(data.base), update_count=data.size,
            fallback_updates=int(frame.fallback.sum()), batch_forward_seconds=timings[strategy],
            batch_updates_per_second=data.size / timings[strategy] if timings[strategy] else None)
        stats = frame.groupby("scene_name", as_index=False).agg(update_count=("ade", "size"),
            ade_sum=("ade", "sum"), fde_sum=("fde", "sum"), jitter_sum=("jitter", "sum"))
        stats["seed"], stats["strategy"] = provenance["seed"], strategy
        scene_frames.append(stats)
        for field, frames in group_frames.items():
            grouped = frame.copy()
            if field == "maneuver":
                grouped[field] = grouped[field].str.split("|")
                grouped = grouped.explode(field)
            aggregate = grouped.groupby(field, as_index=False).agg(update_count=("ade", "size"),
                ade=("ade", "mean"), fde=("fde", "mean"), jitter=("jitter", "mean"),
                map_coverage=("map_covered", "mean"), neighbor_coverage=("neighbor_covered", "mean"),
                fallback_updates=("fallback", "sum"))
            aggregate["strategy"], aggregate["seed"] = strategy, provenance["seed"]
            frames.append(aggregate)
    pd.concat(scene_frames, ignore_index=True).to_csv(root / "scene_stats.csv", index=False)
    for field, filename in (("update_step", "by_update.csv"), ("remaining_steps", "by_horizon.csv"), ("maneuver", "by_maneuver.csv")):
        pd.concat(group_frames[field], ignore_index=True).to_csv(root / filename, index=False)
    summary = {"provenance": provenance, "strategies": summaries, "sample_count": len(data.base),
        "update_count": data.size, "ordered_key_sha256": ordered_key_hash(keys),
        "per_update_sha256": sha256_file(path), "scene_stats_sha256": sha256_file(root / "scene_stats.csv"),
        "coverage": {"map": float(keys.map_covered.mean()), "neighbors": float(keys.neighbor_covered.mean()),
                     "either": float((keys.map_covered | keys.neighbor_covered).mean())},
        "failure_count": 0, "timing_scope": "batch_forward_throughput_not_single_update_latency", "oracle": "annotation_target_and_neighbor_positions"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return summary

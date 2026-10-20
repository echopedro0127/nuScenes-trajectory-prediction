"""Nine causal strategies with evaluation labels excluded from all predictor calls."""
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
from trajectory_prediction.context.neighbor_cache import neighbor_view
from trajectory_prediction.context.neighbor_preparation import load_interaction_bundle
from trajectory_prediction.context.neighbors import NeighborConfig
from trajectory_prediction.context.preparation import load_predictions
from trajectory_prediction.models.physics import finite_difference_cv
from trajectory_prediction.online.context_corrector import _sync
from trajectory_prediction.online.context_evaluator import DEFAULT_THRESHOLDS
from trajectory_prediction.online.context_features import residual_features
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.evaluator import update_metrics
from trajectory_prediction.online.interaction_corrector import MODES, predict_interaction
from trajectory_prediction.online.interaction_provenance import (
    SUITE_NAMES,
    load_interaction_checkpoint,
)
from trajectory_prediction.online.interaction_report import (
    STRATEGIES,
    validate_rows,
    write_interaction_outputs,
)
from trajectory_prediction.online.observations import replay_observations
from trajectory_prediction.utils.provenance import sha256_file


def predict_residual(model, observation, base, device):
    _sync(device)
    start = time.perf_counter_ns()
    features, mask = residual_features(observation, base)
    args = torch.from_numpy(features[None]).to(device), torch.from_numpy(mask[None]).to(device)
    _sync(device)
    prepared = time.perf_counter_ns()
    with torch.inference_mode():
        result = model(*args)[0].detach().cpu().numpy()
    _sync(device)
    end = time.perf_counter_ns()
    return result, {"feature_ms": (prepared - start) / 1e6, "encoder_head_ms": (end - prepared) / 1e6}


def evaluate_interactions(arrays, maps, neighbors, base, models, device=None,
                          predictor=predict_interaction, thresholds=None):
    if set(models) != set(SUITE_NAMES):
        raise ValueError("all seven learned ablation models are required")
    device = device or torch.device("cpu")
    models = models.to(device).eval()
    target = arrays["future_xy"]
    if base.shape != target.shape or base.ndim != 3 or base.shape[1:] != (12, 2) or not len(base) or not np.isfinite(base).all():
        raise ValueError("base and target must be finite nonempty matching [N, 12, 2]")
    rows = []
    for i in range(len(base)):
        tags = tag_maneuvers(describe_motion(target[i], .5), thresholds or DEFAULT_THRESHOLDS)
        previous = {name: None for name in STRATEGIES}
        iterator = iter(replay_observations(arrays["history_xy"][i], target[i],
                                           anchor_timestamp_us=int(maps["anchor_timestamp_us"][i])))
        for k in range(11):
            start = time.perf_counter_ns()
            observation, label = next(iterator)
            step = observation.update_step
            map_view = MapView(maps["features"][i, k], maps["mask"][i, k], tuple(maps["lane_tokens"][i, k]))
            adjacent = neighbor_view(neighbors, i, k)
            if adjacent.timestamp_us != observation.timestamp_us:
                raise ValueError("observation/neighbor timeline mismatch")
            shared_ms = (time.perf_counter_ns() - start) / 1e6
            map_present, neighbor_present = bool(map_view.mask.any()), bool(adjacent.mask.any())
            current_mask = adjacent.mask[:, -1]
            age = int((observation.timestamp_us - adjacent.source_timestamp_us[:, -1][current_mask]).max()) if current_mask.any() else 0
            for strategy in STRATEGIES:
                _sync(device)
                start = time.perf_counter_ns()
                profile = {"feature_ms": 0., "encoder_head_ms": 0.}
                fallback = False
                if strategy == "frozen":
                    forecast = base[i, step:].copy()
                elif strategy == "sliding":
                    with torch.inference_mode():
                        forecast = finite_difference_cv(torch.from_numpy(observation.observed_history[None].copy()).to(device),
                            dt=.5, future_steps=12 - step).detach().cpu().numpy()[0]
                else:
                    if strategy in MODES:
                        offsets, profile = predictor(models[strategy], observation, base[i], map_view, adjacent, device)
                        covered = map_present if strategy == "map" else neighbor_present if strategy == "neighbors" else map_present or neighbor_present
                        fallback = not covered
                    else:
                        offsets, profile = predict_residual(models[strategy], observation, base[i], device)
                    forecast = base[i, step:] + offsets[step:]
                _sync(device)
                latency = (time.perf_counter_ns() - start) / 1e6 + shared_ms
                overlap = forecast if previous[strategy] is None else previous[strategy][1:]
                values = update_metrics(forecast, label.remaining_target, overlap, latency)
                rows.append({**{name: str(arrays[name][i]) for name in IDENTIFIERS},
                    "sample_index": i, "update_step": step, "timestamp_us": observation.timestamp_us,
                    "remaining_steps": 12 - step, "strategy": strategy, "maneuver": "|".join(tags),
                    "map_covered": map_present, "neighbor_covered": neighbor_present,
                    "fallback": fallback, "valid_map_points": int(map_view.mask.sum()),
                    "neighbor_count": int(current_mask.sum()), "neighbor_history_points": int(adjacent.mask.sum()),
                    "neighbor_velocity_points": int(adjacent.velocity_mask.sum()), "neighbor_frame_age_us": age,
                    "cached_replay_prepare_ms": shared_ms, **profile, **values})
                previous[strategy] = forecast
    frame = pd.DataFrame(rows)
    validate_rows(frame)
    return frame


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate matched causal map/neighbor ablation strategies")
    for name in ("trajectory-cache", "map-cache", "neighbor-cache", "base-predictions", "dataroot", "checkpoint", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/context/neighbors.yaml"))
    parser.add_argument("--map-config", type=Path, default=Path("configs/context/map.yaml"))
    parser.add_argument("--maneuver-config", type=Path, default=Path("configs/analysis/maneuvers.yaml"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    arrays, maps, map_manifest, neighbors, neighbor_manifest = load_interaction_bundle(
        args.trajectory_cache, args.map_cache, args.neighbor_cache, args.split, args.dataroot,
        MapConfig(**load_yaml(args.map_config)), NeighborConfig(**load_yaml(args.config)))
    base, hashes = load_predictions(args.base_predictions, arrays, map_manifest["contract"])
    models, payload = load_interaction_checkpoint(args.checkpoint, map_manifest["contract"], neighbor_manifest["contract"],
                                                 arrays["scene_name"].tolist(), hashes["model_identity"])
    device = choose_device(args.device)
    frame = evaluate_interactions(arrays, maps, neighbors, base, models, device,
                                  thresholds=load_yaml(args.maneuver_config))
    summary = write_interaction_outputs(frame, args.output_dir, {
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
        "device": str(device), "split": args.split, "map_sha256": map_manifest["npz_sha256"],
        "map_contract": map_manifest["contract"], "neighbor_sha256": neighbor_manifest["npz_sha256"],
        "neighbor_contract": neighbor_manifest["contract"], "base_predictions": hashes,
        "checkpoint_sha256": sha256_file(args.checkpoint), "checkpoint_metadata": payload["metadata"],
        "scenes": sorted(set(arrays["scene_name"].tolist())), "maneuver_config_sha256": sha256_file(args.maneuver_config)})
    print(json.dumps({name: summary[name] for name in ("sample_count", "update_count", "coverage", "fallback_updates", "failure_count")}, indent=2))

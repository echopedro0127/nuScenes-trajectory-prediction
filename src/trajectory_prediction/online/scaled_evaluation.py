"""Whole-split evaluation and a separate synchronized single-update benchmark."""
import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.context.maps import MapConfig, MapView
from trajectory_prediction.context.neighbor_cache import neighbor_view
from trajectory_prediction.context.neighbors import NeighborConfig
from trajectory_prediction.context.scaled_preparation import load_scaled_bundle
from trajectory_prediction.models.physics import finite_difference_cv
from trajectory_prediction.online.batched_context import BatchedContext
from trajectory_prediction.online.batched_evaluation import evaluate_batches, write_batch_outputs
from trajectory_prediction.online.context_corrector import _sync
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.interaction_corrector import MODES, predict_interaction
from trajectory_prediction.online.interaction_evaluator import predict_residual
from trajectory_prediction.online.interaction_provenance import load_interaction_checkpoint
from trajectory_prediction.online.interaction_report import STRATEGIES
from trajectory_prediction.online.observations import ObservationView
from trajectory_prediction.utils.provenance import sha256_file


def benchmark_keys(size, count=256):
    if type(size) is not int or type(count) is not int or min(size, count) < 1:
        raise ValueError("positive benchmark size required")
    return np.sort(np.random.default_rng(2027).choice(size, min(size, count), replace=False))


def benchmark_updates(data, models, output_dir, device=None, count=256, warmup=16):
    device = device or torch.device("cpu")
    if type(warmup) is not int or warmup < 0:
        raise ValueError("nonnegative warmup required")
    models = models.to(device).eval()
    keys = benchmark_keys(data.size, count)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    records = []
    for index in keys:
        i, k = int(index // 11), int(index % 11)
        records.append({"flat_index": int(index), "instance_token": str(data.arrays["instance_token"][i]),
            "sample_token": str(data.arrays["sample_token"][i]), "scene_name": str(data.arrays["scene_name"][i]),
            "update_step": k + 1, "timestamp_us": int(data.maps["timestamp_us"][i, k])})
    pd.DataFrame(records).to_csv(root / "latency_keys.csv", index=False)
    profiles = {}
    for strategy in STRATEGIES:
        measurements = []
        sequence = [int(keys[j % len(keys)]) for j in range(warmup)] + keys.tolist()
        for iteration, index in enumerate(sequence):
            i, k, step = index // 11, index % 11, index % 11 + 1
            _sync(device)
            started = time.perf_counter_ns()
            # Construct only visible observations; no evaluation labels enter this path.
            history = np.concatenate((data.arrays["history_xy"][i], data.arrays["future_xy"][i, :step]))[-5:]
            observation = ObservationView(step, int(data.maps["timestamp_us"][i, k]), history, history[-1], ())
            map_view = MapView(data.maps["features"][i, k], data.maps["mask"][i, k], tuple(data.maps["lane_tokens"][i, k]))
            adjacent = neighbor_view(data.neighbors, i, k)
            base = data.base[i]
            feature_ms, head_ms = 0., 0.
            if strategy == "frozen":
                forecast = base[step:].copy()
            elif strategy == "sliding":
                with torch.inference_mode():
                    forecast = finite_difference_cv(torch.from_numpy(observation.observed_history[None].copy()).to(device),
                        dt=.5, future_steps=12 - step).detach().cpu().numpy()[0]
            else:
                if strategy in MODES:
                    offsets, profile = predict_interaction(models[strategy], observation, base, map_view, adjacent, device)
                else:
                    offsets, profile = predict_residual(models[strategy], observation, base, device)
                feature_ms, head_ms = profile["feature_ms"], profile["encoder_head_ms"]
                forecast = base[step:] + offsets[step:]
            _sync(device)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if not np.isfinite(forecast).all():
                raise ValueError("nonfinite latency benchmark forecast")
            if iteration >= warmup:
                measurements.append({"flat_index": index, "total_ms": elapsed, "feature_ms": feature_ms, "encoder_head_ms": head_ms})
        frame = pd.DataFrame(measurements)
        frame["strategy"] = strategy
        profiles[strategy] = {field: {"p50": float(frame[field].quantile(.5)), "p95": float(frame[field].quantile(.95))}
                               for field in ("total_ms", "feature_ms", "encoder_head_ms")}
        frame.to_csv(root / ("latency_" + strategy + ".csv"), index=False)
    result = {"key_count": len(keys), "warmup_per_strategy": warmup, "key_selection_seed": 2027,
        "keys_sha256": sha256_file(root / "latency_keys.csv"), "strategies": profiles, "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": str(torch.__version__), "scope": "single_update_cached_context_read_view_construction_features_transfer_head_output_cuda_synchronized",
        "excluded": ["raw_sensor_IO", "cold_cache_load", "batch_forward_throughput"], "inference_batch_size": 1}
    (root / "latency.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trajectory-root", "context-root", "prediction-root", "dataroot", "training-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029])
    parser.add_argument("--split", default="val")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--map-config", type=Path, default=Path("configs/context/map.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/context/neighbors.yaml"))
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    device = choose_device(args.device)
    bundle, map_manifest, neighbor_manifest, hashes = load_scaled_bundle(args.trajectory_root / args.split,
        args.context_root / args.split, args.split, args.dataroot, MapConfig(**load_yaml(args.map_config)),
        NeighborConfig(**load_yaml(args.config)), args.prediction_root / "gru_eval" / "predictions.npz")
    data = BatchedContext(*bundle)
    evaluation = {"map": map_manifest["npz_sha256"], "neighbors": neighbor_manifest["npz_sha256"],
                  "map_contract": map_manifest["contract"], "neighbor_contract": neighbor_manifest["contract"], "base": hashes}
    evaluation_identity = digest_json(evaluation)
    scenes = sorted(set(data.arrays["scene_name"].tolist()))
    for seed in args.seeds:
        checkpoint = args.training_root / ("seed-" + str(seed)) / "best.pt"
        models, payload = load_interaction_checkpoint(checkpoint, map_manifest["contract"], neighbor_manifest["contract"], scenes, hashes["model_identity"])
        if payload["metadata"]["seed"] != seed:
            raise ValueError("checkpoint seed mismatch")
        root = args.output_root / ("seed-" + str(seed))
        results, timings = evaluate_batches(data, models, args.batch_size, device)
        training_settings = json.loads((checkpoint.parent / "run.json").read_text())["settings"]
        if digest_json(training_settings) != payload["metadata"]["training_config_sha256"]:
            raise ValueError("training configuration identity mismatch")
        training_settings.pop("seed")
        source = {name: sha256_file(Path(__file__).with_name(name + ".py")) for name in ("scaled_evaluation", "batched_evaluation", "batched_context")}
        provenance = {"seed": seed, "evaluation_identity": evaluation_identity, "evaluation": evaluation,
            "training_config_sha256": digest_json(training_settings), "checkpoint_sha256": sha256_file(checkpoint),
            "training_run_identity": payload["metadata"]["run_identity"], "source_code_sha256": source,
            "heldout_scene_count": len(scenes), "validation_role": "historical_fixed_regression_not_new_blind_test", "device": str(device)}
        write_batch_outputs(data, results, timings, root, provenance)
        benchmark_updates(data, models, root, device)
        print(json.dumps({"seed": seed, "complete": True, "output": str(root)}), flush=True)

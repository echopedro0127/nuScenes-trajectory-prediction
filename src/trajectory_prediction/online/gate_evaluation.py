"""Map-packet evaluation and a separately scoped cached gate benchmark."""

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.analysis.maneuvers import describe_motion, tag_maneuvers
from trajectory_prediction.context.gate_preparation import load_verified_gate_cache
from trajectory_prediction.online.batched_evaluation import forecast_metrics
from trajectory_prediction.online.context_corrector import _sync
from trajectory_prediction.online.context_evaluator import DEFAULT_THRESHOLDS
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.gate_dataset import GateDataset
from trajectory_prediction.online.gate_training import load_gate_checkpoint
from trajectory_prediction.online.quality_gate import CONDITIONS, blend_forecasts, fixed_weights
from trajectory_prediction.online.scaled_evaluation import benchmark_keys
from trajectory_prediction.utils.provenance import sha256_file

STRATEGIES = (
    "frozen",
    "sliding",
    "filtered",
    "original",
    "ungated_map",
    "fixed_gate",
    "learned_gate",
)
METRICS = ("ade", "fde", "jitter")
KEY_FIELDS = (
    "condition",
    "instance_token",
    "sample_token",
    "scene_name",
    "update_step",
    "timestamp_us",
)
DIAGNOSTICS = (
    "sample_index",
    "remaining_steps",
    "maneuver",
    "packet_step",
    "source_us",
    "available_us",
    "age_s",
    "coverage",
    "valid",
)
ARTIFACTS = (
    "per_update.csv.gz",
    "scene_stats.csv",
    "by_condition.csv",
    "by_update.csv",
    "by_horizon.csv",
    "by_maneuver.csv",
    "recovery.csv",
    "gate_distribution.csv",
)
LATENCY_SCOPE = "cached_candidate_read_quality_gate_blend_return_cuda_synchronized"


def update_hash(digest, frame, fields):
    for row in frame[list(fields)].itertuples(index=False, name=None):
        digest.update(json.dumps(row, separators=(",", ":")).encode() + b"\n")


def predict_gate(data, model, selected, device, fixed=False):
    inputs = data.inputs(selected, device)
    with torch.inference_mode():
        weight = (
            fixed_weights(inputs["age_s"], inputs["valid"])
            if fixed
            else model(inputs["quality"], inputs["valid"])
        )
        forecast = blend_forecasts(inputs["cv"], inputs["map"], weight)
    return forecast.cpu().numpy(), weight.cpu().numpy()


def evaluate_gate(data, model, batch_size=4096, device=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive evaluation batch required")
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    a, n = data.arrays, data.n
    results, timings, weights = {}, {}, {}
    for strategy in STRATEGIES:
        started = time.perf_counter()
        result = np.empty((6, n, 11, 3), np.float64)
        if strategy in ("fixed_gate", "learned_gate"):
            weights[strategy] = np.empty((6, n, 11), np.float32)
        for c in range(6):
            if strategy == "frozen":
                forecast = np.broadcast_to(a["base"][:, None], (n, 11, 12, 2))
            elif strategy in ("sliding", "filtered", "original"):
                forecast = a["cv" if strategy == "sliding" else strategy]
            elif strategy == "ungated_map":
                forecast = a["map"][c]
            else:
                forecast = np.empty((n, 11, 12, 2), np.float32)
                for start in range(0, n * 11, batch_size):
                    local = np.arange(start, min(start + batch_size, n * 11))
                    value, weight = predict_gate(
                        data, model, c * n * 11 + local, device, strategy == "fixed_gate"
                    )
                    i, k = local // 11, local % 11
                    forecast[i, k] = value
                    weights[strategy][c, i, k] = weight
                invalid = ~a["valid"][c]
                if not np.array_equal(forecast[invalid], a["cv"][invalid]) or np.any(
                    weights[strategy][c][invalid] != 0
                ):
                    raise ValueError("gate hard fallback differs from CV")
            result[c] = forecast_metrics(forecast, a["target"])
        results[strategy] = result
        timings[strategy] = time.perf_counter() - started
        print(
            json.dumps(
                {
                    "evaluated": strategy,
                    "updates": data.size,
                    "ade": float(result[..., 0].mean()),
                    "batch_seconds": timings[strategy],
                }
            ),
            flush=True,
        )
    return results, timings, weights


def recovery_table(results, arrays, seed, threshold=0.25):
    rows = []
    for strategy, values in results.items():
        gap = values[3, :, :, 0] - values[0, :, :, 0]
        for i in range(len(gap)):
            matches = [k for k in range(6, 10) if np.all(gap[i, k : k + 2] <= threshold)]
            rows.append(
                {
                    "seed": seed,
                    "strategy": strategy,
                    **{
                        name: arrays[name][i]
                        for name in ("instance_token", "sample_token", "scene_name")
                    },
                    "recovered": bool(matches),
                    "recovery_seconds": (matches[0] - 6) * 0.5 if matches else np.nan,
                    "censor_seconds": 2.0,
                    "threshold_m": threshold,
                }
            )
    return pd.DataFrame(rows)


def key_frame(data, c, tags):
    a = data.arrays
    i, k = np.repeat(np.arange(data.n), 11), np.tile(np.arange(11), data.n)
    return pd.DataFrame(
        {
            "condition": CONDITIONS[c],
            **{name: a[name][i] for name in ("instance_token", "sample_token", "scene_name")},
            "sample_index": i,
            "update_step": k + 1,
            "remaining_steps": 11 - k,
            "timestamp_us": a["timestamp_us"][i, k],
            "maneuver": np.asarray(tags)[i],
            **{
                name: a[name][c, i, k]
                for name in (
                    "packet_step",
                    "source_us",
                    "available_us",
                    "age_s",
                    "coverage",
                    "valid",
                )
            },
        }
    )


def aggregate_rows(frame, fields):
    return frame.groupby(fields, as_index=False).agg(
        update_count=("ade", "size"),
        ade_sum=("ade", "sum"),
        fde_sum=("fde", "sum"),
        jitter_sum=("jitter", "sum"),
        valid_sum=("valid", "sum"),
        fallback_sum=("fallback", "sum"),
        gate_sum=("gate_weight", "sum"),
        missing_sum=("missing", "sum"),
        expired_sum=("expired", "sum"),
    )


def mean_rows(frame):
    result = frame.copy()
    for name in METRICS:
        result[name] = result.pop(name + "_sum") / result.update_count
    result["coverage_fraction"] = result.pop("valid_sum") / result.update_count
    result["mean_gate"] = result.pop("gate_sum") / result.update_count
    return result


def write_gate_outputs(data, results, timings, weights, root, provenance):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    tags = [
        "|".join(tag_maneuvers(describe_motion(future, 0.5), DEFAULT_THRESHOLDS))
        for future in data.arrays["target"]
    ]
    groups = {
        field: []
        for field in ("scene_name", "condition", "update_step", "remaining_steps", "maneuver")
    }
    distributions = []
    hashes = {}
    path = root / "per_update.csv.gz"
    first = True
    for strategy in STRATEGIES:
        key_digest, diagnostic_digest = hashlib.sha256(), hashlib.sha256()
        for c, condition in enumerate(CONDITIONS):
            frame = key_frame(data, c, tags)
            frame["seed"], frame["strategy"] = provenance["seed"], strategy
            metric = results[strategy][c].reshape(-1, 3)
            if metric.shape != (data.n * 11, 3) or not np.isfinite(metric).all():
                raise ValueError("invalid gate metrics")
            for j, name in enumerate(METRICS):
                frame[name] = metric[:, j]
            frame["gate_weight"] = weights[strategy][c].reshape(-1) if strategy in weights else 0.0
            frame["fallback"] = (
                ~frame.valid if strategy in ("ungated_map", "fixed_gate", "learned_gate") else False
            )
            frame["missing"] = frame.packet_step == 0
            frame["expired"] = (frame.packet_step > 0) & (frame.age_s > 2.0)
            for column in frame.select_dtypes(include=["floating"]).columns:
                frame[column] = frame[column].astype(np.float64)
            update_hash(key_digest, frame, KEY_FIELDS)
            update_hash(diagnostic_digest, frame, DIAGNOSTICS)
            frame.to_csv(
                path, index=False, mode="w" if first else "a", header=first, compression="gzip"
            )
            first = False
            for field, parts in groups.items():
                grouped = frame
                if field == "maneuver":
                    grouped = frame.assign(maneuver=frame.maneuver.str.split("|")).explode(
                        "maneuver"
                    )
                fields = ["seed", "strategy", "condition"] + (
                    [] if field == "condition" else [field]
                )
                parts.append(aggregate_rows(grouped, fields))
            if strategy in weights:
                g = frame.gate_weight
                distributions.append(
                    {
                        "seed": provenance["seed"],
                        "condition": condition,
                        "strategy": strategy,
                        "count": len(g),
                        "zero_count": int((g == 0).sum()),
                        "mean": float(g.mean()),
                        "std": float(g.std(ddof=0)),
                        "p05": float(g.quantile(0.05)),
                        "p50": float(g.quantile(0.5)),
                        "p95": float(g.quantile(0.95)),
                        "max": float(g.max()),
                    }
                )
        hashes[strategy] = {
            "keys": key_digest.hexdigest(),
            "diagnostics": diagnostic_digest.hexdigest(),
        }
    if len({tuple(v.values()) for v in hashes.values()}) != 1:
        raise ValueError("cross-strategy key/diagnostic mismatch")
    for field, filename in (
        ("scene_name", "scene_stats.csv"),
        ("condition", "by_condition.csv"),
        ("update_step", "by_update.csv"),
        ("remaining_steps", "by_horizon.csv"),
        ("maneuver", "by_maneuver.csv"),
    ):
        combined = pd.concat(groups[field], ignore_index=True)
        (combined if field == "scene_name" else mean_rows(combined)).to_csv(
            root / filename, index=False
        )
    recovery_table(results, data.arrays, provenance["seed"]).to_csv(
        root / "recovery.csv", index=False
    )
    pd.DataFrame(distributions).to_csv(root / "gate_distribution.csv", index=False)
    summary = {
        "provenance": provenance,
        "sample_count": data.n,
        "updates_per_condition_strategy": data.n * 11,
        "all_condition_updates_per_strategy": data.size,
        "row_count": data.size * 7,
        "failure_count": 0,
        "strategies": list(STRATEGIES),
        "conditions": list(CONDITIONS),
        "ordered_hashes": hashes,
        "sha256": {name: sha256_file(root / name) for name in ARTIFACTS},
        "batch_evaluation_seconds": timings,
        "timing_scope": "cached_batch_forecast_metrics_throughput_not_online_latency",
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def benchmark_gate(data, model, root, device, count=256, warmup=16):
    root = Path(root)
    keys = benchmark_keys(data.size, count)
    tags = ["unused"] * data.n
    all_keys = pd.concat(
        [key_frame(data, c, tags)[list(KEY_FIELDS)] for c in range(6)], ignore_index=True
    )
    key_rows = all_keys.iloc[keys].copy()
    key_rows.insert(0, "flat_index", keys)
    key_rows.to_csv(root / "latency_keys.csv", index=False)
    rows = []
    model = model.to(device).eval()
    for strategy in ("fixed_gate", "learned_gate"):
        sequence = [int(keys[j % len(keys)]) for j in range(warmup)] + keys.tolist()
        for iteration, index in enumerate(sequence):
            _sync(device)
            begun = time.perf_counter_ns()
            value, weight = predict_gate(
                data, model, np.array([index]), device, strategy == "fixed_gate"
            )
            _sync(device)
            elapsed = (time.perf_counter_ns() - begun) / 1e6
            if not np.isfinite(value).all() or not np.isfinite(weight).all():
                raise ValueError("invalid timed gate prediction")
            if iteration >= warmup:
                rows.append({"strategy": strategy, "flat_index": index, "total_ms": elapsed})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "latency_measurements.csv", index=False)
    result = {
        "key_count": len(keys),
        "warmup": warmup,
        "inference_batch_size": 1,
        "key_selection_seed": 2027,
        "scope": LATENCY_SCOPE,
        "excluded": ["map_query", "P4_candidate_forward", "cold_cache_load"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "torch_version": str(torch.__version__),
        "keys_sha256": sha256_file(root / "latency_keys.csv"),
        "measurements_sha256": sha256_file(root / "latency_measurements.csv"),
        "strategies": {
            s: {"p50": float(f.total_ms.quantile(0.5)), "p95": float(f.total_ms.quantile(0.95))}
            for s, f in frame.groupby("strategy")
        },
    }
    (root / "latency.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache-root", "training-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027, 2028, 2029])
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    device = choose_device(args.device)
    for seed in args.seeds:
        cache = args.cache_root / "val" / ("seed-" + str(seed))
        arrays, manifest = load_verified_gate_cache(cache, "val", seed, device=device)
        data = GateDataset(arrays)
        checkpoint = args.training_root / ("seed-" + str(seed)) / "best.pt"
        model, metadata = load_gate_checkpoint(
            checkpoint, manifest["contract"], data.scenes, device
        )
        results, timings, weights = evaluate_gate(data, model, device=device)
        provenance = {
            "seed": seed,
            "candidate_contract": manifest["contract"],
            "candidate_manifest": {
                "path": str((cache / "manifest.json").resolve()),
                "sha256": sha256_file(cache / "manifest.json"),
            },
            "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
            "training_run_identity": metadata["run_identity"],
            "evaluation_source_sha256": sha256_file(Path(__file__)),
        }
        root = args.output_root / ("seed-" + str(seed))
        write_gate_outputs(data, results, timings, weights, root, provenance)
        benchmark_gate(data, model, root, device)


if __name__ == "__main__":
    main()

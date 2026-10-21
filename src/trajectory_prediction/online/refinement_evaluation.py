"""Frozen train_val selection and complete P6.1 evaluation on bound numeric keys."""

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.analysis.maneuvers import describe_motion, tag_maneuvers
from trajectory_prediction.context.cache import key_hash
from trajectory_prediction.context.gate_preparation import load_verified_gate_cache
from trajectory_prediction.online.batched_evaluation import forecast_metrics
from trajectory_prediction.online.context_corrector import _sync
from trajectory_prediction.online.context_evaluator import DEFAULT_THRESHOLDS
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.gate_evaluation import (
    CONDITIONS,
    KEY_FIELDS,
    aggregate_rows,
    key_frame,
    mean_rows,
    recovery_table,
)
from trajectory_prediction.online.gate_refinement import (
    RefinementDataset,
    paired_revision,
    shrink_weights,
)
from trajectory_prediction.online.quality_gate import QualityGate, blend_forecasts, fixed_weights
from trajectory_prediction.online.refinement_training import (
    MODEL_NAMES,
    load_initial,
    load_refinement,
    sources,
    state_fingerprint,
)
from trajectory_prediction.online.scaled_evaluation import benchmark_keys
from trajectory_prediction.utils.provenance import sha256_file

STRATEGIES = (
    "sliding",
    "ungated_map",
    "p6_fixed",
    "p6_learned",
    "constant_gate",
    "time_age_gate",
    "original_gate",
    "short_gate",
    "finetune_0",
    "consistent_002",
    "consistent_010",
    "consistent_short",
)
METRICS = ("ade", "fde", "jitter")
SEEDS = (2027, 2028, 2029)
TAUS = (0.0, 0.5, 1.0, 1.5)
ARTIFACTS = (
    "metrics.npz",
    "weights.npz",
    "scene_stats.csv",
    "by_condition.csv",
    "by_update.csv",
    "by_horizon.csv",
    "by_maneuver.csv",
    "quantiles.csv",
    "recovery.csv",
    "gate_distribution.csv",
)
SCOPE = "cached_candidate_read_adapt_quality_gate_cap_blend_return_cuda_synchronized"


def evaluation_sources():
    package = Path(__file__).parents[1]
    names = (
        "online/refinement_evaluation.py",
        "online/gate_evaluation.py",
        "online/scaled_evaluation.py",
        "online/context_corrector.py",
        "online/context_evaluator.py",
        "analysis/maneuvers.py",
    )
    return {
        "execution": {name: sha256_file(package / name) for name in names},
        "training": sources(),
    }


def validate_model_role(name, meta, p6_identity, random_identity):
    expected = {
        "candidate": "original" if name == "original_gate" else "map",
        "revision_weight": {"consistent_002": 0.02, "consistent_010": 0.1}.get(name, 0.0),
        "epochs": 20 if name == "original_gate" else 10,
        "patience": 5 if name == "original_gate" else 3,
        "batch_size": 4096,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "smoothness_weight": 0.05,
        "hidden_dim": 16,
        "initial_state_sha256": random_identity if name == "original_gate" else p6_identity,
    }
    if (
        name not in MODEL_NAMES
        or any(meta.get(key) != value for key, value in expected.items())
        or not 1
        <= meta.get("best_epoch", 0)
        <= meta.get("completed_epochs", 0)
        <= expected["epochs"]
    ):
        raise ValueError("refinement model role/settings/initialization mismatch")


def load_suite(training_root, p6_root, contract, scenes, split, seed, device):
    training_root, p6_root = Path(training_root), Path(p6_root)
    root = training_root / f"seed-{seed}"
    models = {}
    references = {}
    parent = None
    model_metadata = {}
    for name in MODEL_NAMES:
        path = root / name / "best.pt"
        model, meta = load_refinement(
            path, contract, scenes, device, selection=split == "train_val"
        )
        parents = {
            key: meta[key]
            for key in ("candidate_contracts", "candidate_manifests", "p6_checkpoint")
        }
        if parent is not None and parents != parent:
            raise ValueError("refinement suite parent source mismatch")
        parent = parents
        model_metadata[name] = meta
        models[name] = model
        references[name] = {
            "path": str(path.resolve()),
            "sha256": sha256_file(path),
            "run_identity": meta["run_identity"],
        }
    for reference in parent["candidate_manifests"].values():
        if sha256_file(Path(reference["path"])) != reference["sha256"]:
            raise ValueError("refinement training cache manifest changed")
    p6 = p6_root / f"seed-{seed}" / "best.pt"
    if parent["p6_checkpoint"] != {"path": str(p6.resolve()), "sha256": sha256_file(p6)}:
        raise ValueError("refinement P6 checkpoint mismatch")
    state = load_initial(p6, parent["candidate_contracts"])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        random_identity = state_fingerprint(QualityGate().state_dict())
    for name, meta in model_metadata.items():
        validate_model_role(name, meta, state_fingerprint(state), random_identity)
    model = QualityGate().to(device)
    model.load_state_dict(state, strict=True)
    models["p6_learned"] = model.eval()
    references["p6_learned"] = {"path": str(p6.resolve()), "sha256": sha256_file(p6)}
    path = root / "calibration.json"
    calibration = json.loads(path.read_text())
    if calibration["metadata"] != parent or calibration["source_code_sha256"] != sources():
        raise ValueError("refinement calibration source mismatch")
    references["calibration"] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
    constant = calibration["train_val"]["constant"]
    lookup = np.asarray(calibration["train"]["lookup"])
    if (
        not np.isfinite(constant)
        or not 0 <= constant <= 1
        or lookup.shape != (11, 5)
        or not np.isfinite(lookup).all()
        or np.any((lookup < 0) | (lookup > 1))
    ):
        raise ValueError("invalid calibrated weights")
    return {
        "models": models,
        "calibration": {"constant": constant, "lookup": lookup.tolist()},
        "references": references,
        "parents": parent,
    }


def predict_strategy(data, suite, policy, strategy, indices, device):
    if strategy not in STRATEGIES:
        raise ValueError("unknown refinement strategy")
    candidate = "original" if strategy == "original_gate" else "map"
    adapted = (
        data if candidate == data.candidate else RefinementDataset(data.arrays, candidate=candidate)
    )
    inputs = adapted.inputs(indices, device)
    _, _, k = data.indices(indices)
    with torch.inference_mode():
        if strategy == "sliding":
            return inputs["cv"].cpu().numpy(), np.zeros(len(indices), np.float32)
        if strategy == "ungated_map":
            return inputs["map"].cpu().numpy(), np.ones(len(indices), np.float32)
        if strategy == "p6_fixed":
            weight = fixed_weights(inputs["age_s"], inputs["valid"])
        elif strategy == "constant_gate":
            weight = (
                torch.full((len(indices),), float(suite["calibration"]["constant"]), device=device)
                * inputs["valid"]
            )
        elif strategy == "time_age_gate":
            age = torch.clamp(torch.round(inputs["age_s"] * 2).long(), 0, 4)
            table = torch.tensor(suite["calibration"]["lookup"], dtype=torch.float32, device=device)
            weight = table[torch.from_numpy(k).to(device), age] * inputs["valid"]
        else:
            name = (
                "p6_learned"
                if strategy == "short_gate"
                else policy["consistent_model"]
                if strategy == "consistent_short"
                else strategy
            )
            weight = suite["models"][name](inputs["quality"], inputs["valid"])
            if strategy in ("short_gate", "consistent_short"):
                tau = policy["short_tau"] if strategy == "short_gate" else policy["consistent_tau"]
                weight = shrink_weights(weight, inputs["mask"].sum(1).to(weight.dtype) * 0.5, tau)
        forecast = blend_forecasts(inputs["cv"], inputs["map"], weight)
    return forecast.cpu().numpy(), weight.cpu().numpy()


def measure_strategy(data, suite, policy, strategy, *, batch_size=4096, device=None):
    device = device or torch.device("cpu")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive evaluation batch required")
    begun = time.perf_counter()
    metrics = np.empty((6, data.n, 11, 3), np.float64)
    weights = np.empty((6, data.n, 11), np.float32)
    for c in range(6):
        forecast = np.empty((data.n, 11, 12, 2), np.float32)
        for start in range(0, data.n * 11, batch_size):
            local = np.arange(start, min(start + batch_size, data.n * 11))
            value, w = predict_strategy(
                data, suite, policy, strategy, c * data.n * 11 + local, device
            )
            i, k = local // 11, local % 11
            forecast[i, k] = value
            weights[c, i, k] = w
        invalid = ~data.arrays["valid"][c]
        if strategy != "ungated_map" and (
            np.any(weights[c][invalid] != 0)
            or not np.array_equal(forecast[invalid], data.arrays["cv"][invalid])
        ):
            raise ValueError("refinement hard fallback mismatch")
        metrics[c] = forecast_metrics(forecast, data.arrays["target"])
    return metrics, weights, time.perf_counter() - begun


def evaluate_suite(data, suite, policy, *, batch_size=4096, device=None):
    results, weights, timings = {}, {}, {}
    for name in STRATEGIES:
        values, w, elapsed = measure_strategy(
            data, suite, policy, name, batch_size=batch_size, device=device
        )
        results[name], weights[name], timings[name] = values, w, elapsed
        print(
            json.dumps(
                {
                    "strategy": name,
                    "ade": float(values[..., 0].mean()),
                    "jitter": float(values[..., 2].mean()),
                    "seconds": elapsed,
                }
            ),
            flush=True,
        )
    return results, weights, timings


def condition_table(results, seed):
    return pd.DataFrame(
        [
            {
                "seed": seed,
                "strategy": name,
                "condition": condition,
                **{metric: float(values[c, ..., j].mean()) for j, metric in enumerate(METRICS)},
            }
            for name, values in results.items()
            for c, condition in enumerate(CONDITIONS)
        ]
    )


def choose_champions(frame):
    if set(frame.seed) != set(SEEDS) or frame.duplicated(["seed", "strategy", "condition"]).any():
        raise ValueError("complete fixed seed rows required")
    expected = {
        (seed, strategy, condition)
        for seed in SEEDS
        for strategy in STRATEGIES
        for condition in CONDITIONS
    }
    if (
        set(frame[["seed", "strategy", "condition"]].itertuples(index=False, name=None)) != expected
        or len(frame) != len(expected)
        or not np.isfinite(frame[["ade", "jitter"]]).all().all()
        or (frame[["ade", "jitter"]] < 0).any().any()
    ):
        raise ValueError("complete finite experiment matrix required")
    sizes = frame.groupby(["strategy", "condition"]).seed.nunique()
    if not (sizes == 3).all():
        raise ValueError("incomplete strategy seeds")
    means = frame.groupby("strategy")[["ade", "jitter"]].mean()
    clean = frame[frame.condition == "clean"].groupby("strategy").ade.mean()
    reference = means.loc["p6_learned", "ade"]
    reference_clean = clean["p6_learned"]
    accuracy = means.reset_index().sort_values(["ade", "strategy"]).iloc[0].strategy
    eligible = means[
        (means.ade <= 1.01 * reference) & (clean <= 1.01 * reference_clean)
    ].reset_index()
    stable = eligible.sort_values(["jitter", "ade", "strategy"]).iloc[0].strategy
    return {
        "accuracy": str(accuracy),
        "stable": str(stable),
        "ade_tolerance": 0.01,
        "reference": "p6_learned",
        "reference_ade": float(reference),
        "reference_clean_ade": float(reference_clean),
        "eligible": eligible.strategy.tolist(),
    }


def fit_selection(cache_root, training_root, p6_root, output_root, device):
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    executable = evaluation_sources()
    sweeps = []
    references = {}
    for seed in SEEDS:
        cache = Path(cache_root) / "train_val" / f"seed-{seed}"
        arrays, manifest = load_verified_gate_cache(cache, "train_val", seed, device=device)
        data = RefinementDataset(arrays)
        suite = load_suite(
            training_root, p6_root, manifest["contract"], data.scenes, "train_val", seed, device
        )
        references[str(seed)] = {
            "candidate_manifest": {
                "path": str((cache / "manifest.json").resolve()),
                "sha256": sha256_file(cache / "manifest.json"),
            },
            "suite": suite["references"],
        }
        for name in ("p6_learned", "consistent_002", "consistent_010"):
            for tau in TAUS:
                strategy = "short_gate" if name == "p6_learned" else "consistent_short"
                policy = {"short_tau": tau, "consistent_model": name, "consistent_tau": tau}
                values, _, _ = measure_strategy(data, suite, policy, strategy, device=device)
                sweeps.append(
                    {
                        "seed": seed,
                        "model": name,
                        "tau": tau,
                        "ade": float(values[..., 0].mean()),
                        "jitter": float(values[..., 2].mean()),
                    }
                )
    sweep = pd.DataFrame(sweeps)
    mean = sweep.groupby(["model", "tau"], as_index=False)[["ade", "jitter"]].mean()
    short = mean[mean.model == "p6_learned"].sort_values(["ade", "tau"]).iloc[0]
    consistent = mean[mean.model != "p6_learned"].sort_values(["ade", "tau", "model"]).iloc[0]
    policy = {
        "short_tau": float(short.tau),
        "consistent_model": str(consistent.model),
        "consistent_tau": float(consistent.tau),
    }
    rows = []
    for seed in SEEDS:
        cache = Path(cache_root) / "train_val" / f"seed-{seed}"
        arrays, manifest = load_verified_gate_cache(cache, "train_val", seed, device=device)
        data = RefinementDataset(arrays)
        suite = load_suite(
            training_root, p6_root, manifest["contract"], data.scenes, "train_val", seed, device
        )
        results, _, _ = evaluate_suite(data, suite, policy, device=device)
        rows.append(condition_table(results, seed))
    table = pd.concat(rows, ignore_index=True)
    champions = choose_champions(table)
    sweep.to_csv(root / "shrink_sweep.csv", index=False)
    table.to_csv(root / "selection_conditions.csv", index=False)
    if evaluation_sources() != executable:
        raise ValueError("selection executable changed")
    result = {
        "schema_version": 1,
        "split": "train_val",
        "seeds": list(SEEDS),
        "policy": policy,
        "champions": champions,
        "references": references,
        "source_code_sha256": executable,
        "sha256": {
            name: sha256_file(root / name)
            for name in ("shrink_sweep.csv", "selection_conditions.csv")
        },
    }
    path = root / ".selection.tmp"
    path.write_text(json.dumps(result, indent=2) + "\n")
    path.replace(root / "selection.json")
    print(json.dumps({"frozen_selection": policy, **champions}), flush=True)
    return result


def load_selection(root):
    root = Path(root)
    value = json.loads((root / "selection.json").read_text())
    if (
        value.get("schema_version") != 1
        or value.get("split") != "train_val"
        or value.get("seeds") != list(SEEDS)
        or value.get("source_code_sha256") != evaluation_sources()
    ):
        raise ValueError("selection source/schema mismatch")
    for name in ("shrink_sweep.csv", "selection_conditions.csv"):
        if sha256_file(root / name) != value["sha256"][name]:
            raise ValueError("selection table checksum mismatch")
    for record in value["references"].values():
        for ref in (record["candidate_manifest"], *record["suite"].values()):
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                raise ValueError("selection input reference mismatch")
    frame = pd.read_csv(root / "selection_conditions.csv", float_precision="round_trip")
    if choose_champions(frame) != value["champions"]:
        raise ValueError("selection champion mismatch")
    sweep = pd.read_csv(root / "shrink_sweep.csv", float_precision="round_trip")
    expected = {
        (seed, name, tau)
        for seed in SEEDS
        for name in ("p6_learned", "consistent_002", "consistent_010")
        for tau in TAUS
    }
    if set(sweep[["seed", "model", "tau"]].itertuples(index=False, name=None)) != expected or len(
        sweep
    ) != len(expected):
        raise ValueError("selection sweep incomplete")
    mean = sweep.groupby(["model", "tau"], as_index=False)[["ade", "jitter"]].mean()
    short = mean[mean.model == "p6_learned"].sort_values(["ade", "tau"]).iloc[0]
    consistent = mean[mean.model != "p6_learned"].sort_values(["ade", "tau", "model"]).iloc[0]
    if value["policy"] != {
        "short_tau": float(short.tau),
        "consistent_model": str(consistent.model),
        "consistent_tau": float(consistent.tau),
    }:
        raise ValueError("selection policy mismatch")
    return value


def derive_tables(data, results, weights, seed):
    tags = [
        "|".join(tag_maneuvers(describe_motion(t, 0.5), DEFAULT_THRESHOLDS))
        for t in data.arrays["target"]
    ]
    grouped = {
        field: []
        for field in ("scene_name", "condition", "update_step", "remaining_steps", "maneuver")
    }
    quantiles = []
    distributions = []
    for strategy in STRATEGIES:
        for c, condition in enumerate(CONDITIONS):
            frame = key_frame(data, c, tags)
            frame["seed"], frame["strategy"] = seed, strategy
            for j, metric in enumerate(METRICS):
                frame[metric] = results[strategy][c, ..., j].reshape(-1)
            frame["gate_weight"] = weights[strategy][c].reshape(-1).astype(np.float64)
            frame["fallback"] = ~frame.valid if strategy != "sliding" else False
            frame["missing"] = frame.packet_step == 0
            frame["expired"] = (frame.packet_step > 0) & (frame.age_s > 2.0)
            for field, parts in grouped.items():
                part = (
                    frame.assign(maneuver=frame.maneuver.str.split("|")).explode("maneuver")
                    if field == "maneuver"
                    else frame
                )
                keys = ["seed", "strategy", "condition"] + ([] if field == "condition" else [field])
                parts.append(aggregate_rows(part, keys))
            for metric in ("ade", "fde"):
                quantiles.append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "strategy": strategy,
                        "metric": metric,
                        "update_count": len(frame),
                        "p90": float(frame[metric].quantile(0.9)),
                        "p95": float(frame[metric].quantile(0.95)),
                    }
                )
            g = frame.gate_weight
            distributions.append(
                {
                    "seed": seed,
                    "condition": condition,
                    "strategy": strategy,
                    "count": len(g),
                    "zero_count": int((g == 0).sum()),
                    "mean": float(g.mean()),
                    "p05": float(g.quantile(0.05)),
                    "p50": float(g.quantile(0.5)),
                    "p95": float(g.quantile(0.95)),
                    "max": float(g.max()),
                }
            )
    tables = {}
    for field, filename in (
        ("scene_name", "scene_stats.csv"),
        ("condition", "by_condition.csv"),
        ("update_step", "by_update.csv"),
        ("remaining_steps", "by_horizon.csv"),
        ("maneuver", "by_maneuver.csv"),
    ):
        frame = pd.concat(grouped[field], ignore_index=True)
        tables[filename] = frame if field == "scene_name" else mean_rows(frame)
    tables["quantiles.csv"] = pd.DataFrame(quantiles)
    tables["gate_distribution.csv"] = pd.DataFrame(distributions)
    tables["recovery.csv"] = recovery_table(results, data.arrays, seed)
    return tables


def write_suite_outputs(data, results, weights, timings, root, provenance):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if tuple(results) != STRATEGIES or tuple(weights) != STRATEGIES:
        raise ValueError("complete ordered refinement strategies required")
    for name in STRATEGIES:
        if (
            results[name].shape != (6, data.n, 11, 3)
            or weights[name].shape != (6, data.n, 11)
            or not np.isfinite(results[name]).all()
            or not np.isfinite(weights[name]).all()
        ):
            raise ValueError("refinement numeric array mismatch")
    np.savez_compressed(root / "metrics.npz", **results)
    np.savez_compressed(root / "weights.npz", **weights)
    for filename, frame in derive_tables(data, results, weights, provenance["seed"]).items():
        frame.to_csv(root / filename, index=False)
    summary = {
        "schema_version": 1,
        "provenance": provenance,
        "sample_count": data.n,
        "condition_count": 6,
        "updates_per_condition_strategy": data.n * 11,
        "row_count": data.size * len(STRATEGIES),
        "strategies": list(STRATEGIES),
        "conditions": list(CONDITIONS),
        "key_shape": [6, data.n, 11],
        "key_order": "condition_then_bound_sample_then_update1to11",
        "sample_keys_sha256": key_hash(data.arrays),
        "metric_dtype": "float64",
        "weight_dtype": "float32",
        "failure_count": 0,
        "sha256": {name: sha256_file(root / name) for name in ARTIFACTS},
        "batch_evaluation_seconds": timings,
        "evaluation_source_sha256": evaluation_sources(),
        "timing_scope": "cached_batch_forecast_metrics_only_not_online_latency",
    }
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def benchmark_pair_forward(data, suite, root, device, iterations=64, warmup=16):
    root = Path(root)
    keys = benchmark_keys(data.size, 4096)
    np.save(root / "pair_forward_keys.npy", keys, allow_pickle=False)
    model = suite["models"]["p6_learned"].to(device).eval()
    _, _, k = data.indices(keys)
    has_previous = torch.from_numpy(k > 0).to(device)
    rows = []
    for mode in ("single", "paired"):
        for iteration in range(warmup + iterations):
            _sync(device)
            started = time.perf_counter_ns()
            with torch.inference_mode():
                inputs = data.inputs(keys, device)
                weight = model(inputs["quality"], inputs["valid"])
                now = blend_forecasts(inputs["cv"], inputs["map"], weight)
                if mode == "paired":
                    previous = data.inputs(data.previous_indices(keys), device)
                    past_weight = model(previous["quality"], previous["valid"])
                    past = blend_forecasts(previous["cv"], previous["map"], past_weight)
                    revision = paired_revision(now, past, inputs["mask"], has_previous)
                else:
                    revision = now.sum() * 0.0
            _sync(device)
            elapsed = (time.perf_counter_ns() - started) / 1e6
            if not torch.isfinite(revision):
                raise ValueError("nonfinite timed pair forward")
            if iteration >= warmup:
                rows.append({"mode": mode, "iteration": iteration - warmup, "total_ms": elapsed})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "pair_forward_measurements.csv", index=False)
    result = {
        "batch_size": len(keys),
        "iterations": iterations,
        "warmup": warmup,
        "key_selection_seed": 2027,
        "scope": "cached_inputs_gate_blend_optional_previous_forward_and_revision_no_backward_optimizer_or_labels_cuda_synchronized",
        "device": str(device),
        "keys_sha256": sha256_file(root / "pair_forward_keys.npy"),
        "measurements_sha256": sha256_file(root / "pair_forward_measurements.csv"),
        "model_reference": suite["references"].get("p6_learned", {}),
        "modes": {
            name: {
                "p50": float(part.total_ms.quantile(0.5)),
                "p95": float(part.total_ms.quantile(0.95)),
            }
            for name, part in frame.groupby("mode")
        },
    }
    (root / "pair_forward.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def benchmark_suite(data, suite, policy, root, device, count=256, warmup=16):
    root = Path(root)
    keys = benchmark_keys(data.size, count)
    tags = ["unused"] * data.n
    allkeys = pd.concat(
        [key_frame(data, c, tags)[list(KEY_FIELDS)] for c in range(6)], ignore_index=True
    )
    selected = allkeys.iloc[keys].copy()
    selected.insert(0, "flat_index", keys)
    selected.to_csv(root / "latency_keys.csv", index=False)
    rows = []
    for strategy in STRATEGIES:
        for iteration, index in enumerate(
            [int(keys[j % len(keys)]) for j in range(warmup)] + keys.tolist()
        ):
            _sync(device)
            start = time.perf_counter_ns()
            value, weight = predict_strategy(
                data, suite, policy, strategy, np.array([index]), device
            )
            _sync(device)
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if not np.isfinite(value).all() or not np.isfinite(weight).all():
                raise ValueError("nonfinite timed refinement")
            if iteration >= warmup:
                rows.append({"strategy": strategy, "flat_index": index, "total_ms": elapsed})
    frame = pd.DataFrame(rows)
    frame.to_csv(root / "latency_measurements.csv", index=False)
    result = {
        "key_count": len(keys),
        "warmup": warmup,
        "inference_batch_size": 1,
        "key_selection_seed": 2027,
        "scope": SCOPE,
        "excluded": ["map_query", "P4_candidate_forward", "cold_cache_load"],
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device)
        if device.type == "cuda"
        else platform.processor(),
        "torch_version": str(torch.__version__),
        "keys_sha256": sha256_file(root / "latency_keys.csv"),
        "measurements_sha256": sha256_file(root / "latency_measurements.csv"),
        "strategies": {
            name: {"p50": float(f.total_ms.quantile(0.5)), "p95": float(f.total_ms.quantile(0.95))}
            for name, f in frame.groupby("strategy")
        },
    }
    (root / "latency.json").write_text(json.dumps(result, indent=2) + "\n")
    benchmark_pair_forward(data, suite, root, device)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "cache-root",
        "training-root",
        "p6-training-root",
        "selection-root",
        "output-root",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--select-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    device = choose_device(args.device)
    if not (args.selection_root / "selection.json").exists():
        fit_selection(
            args.cache_root, args.training_root, args.p6_training_root, args.selection_root, device
        )
    selection = load_selection(args.selection_root)
    if args.select_only:
        return
    for seed in SEEDS:
        cache = args.cache_root / "val" / f"seed-{seed}"
        arrays, manifest = load_verified_gate_cache(cache, "val", seed, device=device)
        data = RefinementDataset(arrays)
        suite = load_suite(
            args.training_root,
            args.p6_training_root,
            manifest["contract"],
            data.scenes,
            "val",
            seed,
            device,
        )
        if suite["references"] != selection["references"][str(seed)]["suite"]:
            raise ValueError("evaluation differs from frozen selected suite")
        results, weights, timings = evaluate_suite(data, suite, selection["policy"], device=device)
        provenance = {
            "seed": seed,
            "split": "val",
            "candidate_contract": manifest["contract"],
            "candidate_manifest": {
                "path": str((cache / "manifest.json").resolve()),
                "sha256": sha256_file(cache / "manifest.json"),
            },
            "training_root": str(args.training_root.resolve()),
            "p6_training_root": str(args.p6_training_root.resolve()),
            "suite": suite["references"],
            "selection": {
                "path": str((args.selection_root / "selection.json").resolve()),
                "sha256": sha256_file(args.selection_root / "selection.json"),
            },
        }
        root = args.output_root / f"seed-{seed}"
        write_suite_outputs(data, results, weights, timings, root, provenance)
        benchmark_suite(data, suite, selection["policy"], root, device)
        print(json.dumps({"evaluated_seed": seed, "rows": data.size * len(STRATEGIES)}), flush=True)


if __name__ == "__main__":
    main()

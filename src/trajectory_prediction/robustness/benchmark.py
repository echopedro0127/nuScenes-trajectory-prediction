"""Fixed, development-only packet/filter pilot with auditable sampling and metrics."""
import argparse
import csv
import hashlib
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.data.cache import load_cache
from trajectory_prediction.online.batched_evaluation import forecast_metrics
from trajectory_prediction.robustness.filters import SETTINGS, STRATEGIES, run_packets
from trajectory_prediction.robustness.packets import PROFILES, make_packets
from trajectory_prediction.utils.provenance import sha256_file

ROLES = ("independent_fit", "independent_select")
SEEDS = (2027, 2028, 2029)
CAPS = {"independent_fit": 16, "independent_select": 64}
METRICS = ("ade", "fde", "jitter", "position_rmse", "recovery_7_8_ade", "inflated_update_fraction")
PROJECT = Path(__file__).resolve().parents[3]
DESIGN = PROJECT / "docs/superpowers/specs/2026-11-10-noise-aware-phase1-design.md"


def sources():
    names = ("src/trajectory_prediction/robustness/packets.py", "src/trajectory_prediction/robustness/filters.py",
             "src/trajectory_prediction/robustness/benchmark.py", "src/trajectory_prediction/online/batched_evaluation.py",
             "src/trajectory_prediction/data/cache.py", "scripts/run_noise_aware_baselines.py",
             DESIGN.relative_to(PROJECT).as_posix())
    return {name: sha256_file(PROJECT / name) for name in names}


def sample_indices(arrays, logs, role, cap):
    if role not in ROLES or type(cap) is not int or cap < 1:
        raise ValueError("only fixed development roles and positive log caps are allowed")
    selected = []
    for log in sorted(set(logs.tolist())):
        indices = np.flatnonzero(logs == log).tolist()
        def key(i):
            text = f"v2phase1:{role}:{arrays['instance_token'][i]}:{arrays['sample_token'][i]}"
            return hashlib.sha256(text.encode()).hexdigest(), text
        selected.extend(sorted(indices, key=key)[:cap])
    return np.array(sorted(selected), np.int64)


def score_result(result, future):
    future = np.asarray(future, np.float64)
    if future.shape != (12, 2) or not np.isfinite(future).all():
        raise ValueError("finite 12-point evaluation labels required")
    metrics = forecast_metrics(result["forecasts"][None], future[None])[0]
    fresh = result["fresh"]
    return {"ade": float(metrics[:, 0].mean()), "fde": float(metrics[:, 1].mean()),
            "jitter": float(metrics[:, 2].mean()),
            "position_rmse": float(np.sqrt(np.mean(np.sum((result["positions"] - future[:11]) ** 2, axis=1)))),
            "recovery_7_8_ade": float(metrics[6:8, 0].mean()),
            "inflated_update_fraction": float(np.sum((result["r_scales"] > 1) & fresh) / max(1, fresh.sum()))}


def load_development(cache_root, scene_metadata):
    records = json.loads(Path(scene_metadata).read_text(encoding="utf-8"))
    if (len({r["name"] for r in records}) != len(records)
            or any(not r.get("name") or not r.get("log_token") for r in records)):
        raise ValueError("unique scene names and nonempty log tokens required")
    lookup = {r["name"]: r["log_token"] for r in records}
    loaded, manifests, used_scenes, used_logs = {}, {}, set(), set()
    for role in ROLES:
        arrays, manifest = load_cache(Path(cache_root) / role, role)
        scene_set = set(arrays["scene_name"].tolist())
        logs = np.array([lookup[name] for name in arrays["scene_name"]])
        if used_scenes & scene_set or used_logs & set(logs):
            raise ValueError("development scene/log overlap")
        used_scenes.update(scene_set)
        used_logs.update(logs)
        loaded[role] = arrays, logs
        manifests[role] = manifest
    for field in ("partition_identity", "preprocessing_config_sha256"):
        if not manifests[ROLES[0]].get(field) or manifests[ROLES[0]][field] != manifests[ROLES[1]][field]:
            raise ValueError("development parent split/config identity mismatch")
    return loaded, manifests


def summaries(output):
    output = Path(output)
    frame = pd.read_csv(output / "per_trajectory.csv", float_precision="round_trip", dtype={"log": str})
    result = {"development_only": True, "holdout_evaluated": False, "neural_training": False,
              "rows": len(frame), "noise_seeds": list(SEEDS), "conditions": list(PROFILES),
              "timing_scope": "select_seed2027_arrived_packet_updates_and_state_extrapolation_and_12_point_forecast_ms"}
    for label, group in (("by_condition", ["role", "condition", "strategy"]),
                         ("by_log", ["role", "log", "condition", "strategy"]),
                         ("by_seed", ["role", "seed", "condition", "strategy"]),
                         ("aggregate", ["role", "strategy"])):
        records = []
        for key, rows in frame.groupby(group, sort=True):
            item = dict(zip(group, (int(v) if isinstance(v, (int, np.integer)) else str(v) for v in key), strict=True))
            item.update({metric: float(rows[metric].mean()) for metric in METRICS})
            item["trajectory_seed_records"] = len(rows)
            records.append(item)
        result[label] = records
    with np.load(output / "latency_ms.npz", allow_pickle=False) as arrays:
        result["timing"] = {name: {"updates": len(arrays[name]), "p95_ms": float(np.quantile(arrays[name], .95)),
                                   "mean_ms": float(arrays[name].mean())} for name in arrays.files}
    return result


def run(cache_root, scene_metadata, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError("use a new noise-aware experiment directory")
    cache_root, scene_metadata = Path(cache_root).resolve(), Path(scene_metadata).resolve()
    loaded, manifests = load_development(cache_root, scene_metadata)
    samples = []
    for role, (arrays, logs) in loaded.items():
        for i in sample_indices(arrays, logs, role, CAPS[role]):
            samples.append({"role": role, "row": int(i), "log": str(logs[i]), "scene": str(arrays["scene_name"][i]),
                            "instance": str(arrays["instance_token"][i]), "sample": str(arrays["sample_token"][i])})
    binding = {"date": "2026-11-10", "sources": sources(), "cache_root": str(cache_root),
               "scene_metadata": {"path": str(scene_metadata), "sha256": sha256_file(scene_metadata)},
               "parents": {role: {"manifest_sha256": sha256_file(cache_root / role / "manifest.json"),
                                   "archive_sha256": manifests[role]["npz_sha256"]} for role in ROLES},
               "samples": samples, "caps": CAPS, "noise_seeds": list(SEEDS),
               "profiles": {name: asdict(value) for name, value in PROFILES.items()}, "filter_settings": SETTINGS,
               "strategies": list(STRATEGIES), "development_only": True,
               "runtime": {"python": platform.python_version(), "numpy": str(np.__version__),
                           "platform": platform.platform(), "processor": platform.processor()},
               "git_commit": subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
                                            check=True, capture_output=True, text=True).stdout.strip()}
    identity = digest_json(binding)
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.json").write_text(json.dumps({"identity": identity, "binding": binding}, indent=2) + "\n")
    (output / "samples.json").write_text(json.dumps(samples, indent=2) + "\n")
    latency = {condition + "__" + strategy: [] for condition in PROFILES for strategy in STRATEGIES}
    fields = ("role", "log", "scene", "row", "condition", "seed", "strategy", *METRICS)
    started = time.perf_counter()
    with (output / "per_trajectory.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for role in ROLES:
            arrays, _ = loaded[role]
            selected = [sample for sample in samples if sample["role"] == role]
            for condition, profile in PROFILES.items():
                for seed in SEEDS:
                    for sample in selected:
                        i = sample["row"]
                        key = sample["instance"] + ":" + sample["sample"]
                        packets = make_packets(arrays["history_xy"][i], arrays["future_xy"][i, :11], key, profile, seed)
                        for strategy in STRATEGIES:
                            result = run_packets(packets, strategy)
                            row = {name: sample[name] for name in ("role", "log", "scene", "row")}
                            writer.writerow({**row, "condition": condition, "seed": seed, "strategy": strategy,
                                             **score_result(result, arrays["future_xy"][i])})
                            if role == "independent_select" and seed == SEEDS[0]:
                                latency[condition + "__" + strategy].extend(result["latency_ms"].tolist())
                stream.flush()
                print(json.dumps({"role": role, "condition": condition, "sample_count": len(selected),
                                  "noise_seeds": len(SEEDS), "elapsed_seconds": time.perf_counter() - started}), flush=True)
    if sources() != binding["sources"]:
        raise ValueError("experiment source changed during run")
    np.savez_compressed(output / "latency_ms.npz", **{name: np.array(values) for name, values in latency.items()})
    summary = summaries(output)
    summary.update(identity=identity, seconds=time.perf_counter() - started,
                   samples_by_role={role: sum(s["role"] == role for s in samples) for role in ROLES},
                   logs_by_role={role: len({s["log"] for s in samples if s["role"] == role}) for role in ROLES})
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    names = ("run.json", "samples.json", "per_trajectory.csv", "latency_ms.npz", "summary.json")
    (output / "completed.json").write_text(json.dumps({"identity": identity,
        "sha256": {name: sha256_file(output / name) for name in names}}, indent=2) + "\n")
    return summary


def verify(output):
    output = Path(output)
    marker = json.loads((output / "completed.json").read_text())
    for name, digest in marker["sha256"].items():
        if sha256_file(output / name) != digest:
            raise ValueError("noise-aware output checksum mismatch")
    run_record = json.loads((output / "run.json").read_text())
    binding = run_record["binding"]
    if digest_json(binding) != marker["identity"] or run_record["identity"] != marker["identity"] or sources() != binding["sources"]:
        raise ValueError("noise-aware source/run identity mismatch")
    loaded, manifests = load_development(binding["cache_root"], binding["scene_metadata"]["path"])
    if sha256_file(Path(binding["scene_metadata"]["path"])) != binding["scene_metadata"]["sha256"]:
        raise ValueError("scene metadata changed")
    for role in ROLES:
        if (manifests[role]["npz_sha256"] != binding["parents"][role]["archive_sha256"]
                or sha256_file(Path(binding["cache_root"]) / role / "manifest.json") != binding["parents"][role]["manifest_sha256"]):
            raise ValueError("noise-aware parent cache changed")
        arrays, logs = loaded[role]
        expected = sample_indices(arrays, logs, role, CAPS[role]).tolist()
        if expected != [s["row"] for s in binding["samples"] if s["role"] == role]:
            raise ValueError("noise-aware sample replay mismatch")
    summary = json.loads((output / "summary.json").read_text())
    for name, value in summaries(output).items():
        if summary[name] != value:
            raise ValueError("noise-aware summary replay mismatch")
    frame = pd.read_csv(output / "per_trajectory.csv", float_precision="round_trip")
    checked = 0
    for i in np.linspace(0, len(frame) - 1, min(64, len(frame)), dtype=int):
        row = frame.iloc[i]
        arrays, _ = loaded[row.role]
        index = int(row.row)
        key = str(arrays["instance_token"][index]) + ":" + str(arrays["sample_token"][index])
        packets = make_packets(arrays["history_xy"][index], arrays["future_xy"][index, :11], key, PROFILES[row.condition], int(row.seed))
        result = score_result(run_packets(packets, row.strategy), arrays["future_xy"][index])
        for name in METRICS:
            if not np.isclose(result[name], row[name], atol=1e-10, rtol=0):
                raise ValueError("noise-aware deterministic prediction replay mismatch")
        checked += 1
    return {"identity": marker["identity"], "all_summaries_verified": True,
            "prediction_spot_checks": checked, "holdout_evaluated": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=Path("../causal-context-online/outputs/context/p66-independent/cache"))
    parser.add_argument("--scene-metadata", type=Path, default=Path("../../data/nuscenes/v1.0-trainval/scene.json"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/noise-aware/phase1"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        print(json.dumps(verify(args.output_root)), flush=True)
    else:
        summary = run(args.cache_root, args.scene_metadata, args.output_root)
        print(json.dumps({k: summary[k] for k in ("identity", "samples_by_role", "logs_by_role", "seconds")}), flush=True)


if __name__ == "__main__":
    main()

"""Paired scene-cluster uncertainty conditional on a fixed three-seed suite."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trajectory_prediction.online.interaction_report import STRATEGIES
from trajectory_prediction.utils.provenance import sha256_file

METRICS = ("ade", "fde", "jitter")


def paired_scene_interval(delta_sums, counts, replicates=2000, seed=0):
    delta_sums, counts = np.asarray(delta_sums, float), np.asarray(counts, float)
    if delta_sums.shape != counts.shape or counts.ndim != 1 or not len(counts) or not np.isfinite(delta_sums).all() or not np.isfinite(counts).all() or np.any(counts <= 0) or type(replicates) is not int or replicates < 1:
        raise ValueError("invalid scene bootstrap inputs")
    draw = np.random.default_rng(seed).integers(0, len(counts), (replicates, len(counts)))
    values = delta_sums[draw].sum(1) / counts[draw].sum(1)
    return {"delta": float(delta_sums.sum() / counts.sum()), "ci95": np.quantile(values, [.025, .975]).tolist()}


def _read_run(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    for name, field in (("per_update.csv.gz", "per_update_sha256"), ("scene_stats.csv", "scene_stats_sha256")):
        if sha256_file(root / name) != summary[field]:
            raise ValueError("evaluation artifact checksum mismatch")
    if set(summary["strategies"]) != set(STRATEGIES) or summary.get("failure_count") != 0:
        raise ValueError("incomplete strategy evaluation")
    stats = pd.read_csv(root / "scene_stats.csv")
    if stats.duplicated(["strategy", "scene_name"]).any() or set(stats.strategy) != set(STRATEGIES) or set(stats.seed) != {summary["provenance"]["seed"]}:
        raise ValueError("duplicate or incomplete scene/seed rows")
    numeric = stats[["update_count", *(name + "_sum" for name in METRICS)]].to_numpy()
    if not np.isfinite(numeric).all() or (numeric < 0).any() or (stats.update_count <= 0).any():
        raise ValueError("invalid scene metrics")
    reference = None
    for strategy in STRATEGIES:
        selected = stats[stats.strategy == strategy].sort_values("scene_name")
        entry = summary["strategies"][strategy]
        if entry["sample_count"] != summary["sample_count"] or entry["update_count"] != summary["update_count"]:
            raise ValueError("strategy sample/update count mismatch")
        counts = selected.set_index("scene_name").update_count
        if counts.sum() != summary["update_count"] or (reference is not None and not counts.equals(reference)):
            raise ValueError("scene update counts differ across strategies")
        reference = counts
        for metric in METRICS:
            if not np.isclose(selected[metric + "_sum"].sum() / counts.sum(), summary["strategies"][strategy][metric], rtol=1e-9, atol=1e-9):
                raise ValueError("scene and summary metrics differ")
    for name in ("by_update.csv", "by_horizon.csv", "by_maneuver.csv"):
        if sha256_file(root / name) != summary["group_sha256"][name]:
            raise ValueError("group artifact checksum mismatch")
    # Stream keys and metrics; at most one strategy's duplicate-key set is retained.
    digests = {name: hashlib.sha256() for name in STRATEGIES}
    totals = dict.fromkeys(STRATEGIES, 0)
    seen, current = set(), None
    completed = set()
    row_stats = []
    names = ["instance_token", "sample_token", "scene_name", "update_step", "timestamp_us"]
    for chunk in pd.read_csv(root / "per_update.csv.gz", chunksize=32768):
        if set(chunk.seed) != {summary["provenance"]["seed"]} or not set(chunk.strategy) <= set(STRATEGIES):
            raise ValueError("evaluation row seed/strategy mismatch")
        if not np.isfinite(chunk[list(METRICS)].to_numpy()).all() or (chunk[list(METRICS)] < 0).any().any():
            raise ValueError("invalid row metrics")
        for strategy, frame in chunk.groupby("strategy", sort=False):
            if current != strategy:
                if current is not None:
                    completed.add(current)
                if strategy in completed:
                    raise ValueError("unordered strategy rows")
                seen, current = set(), strategy
            for values in frame[names].itertuples(index=False, name=None):
                key = values[0], values[1], values[3]
                if key in seen:
                    raise ValueError("duplicate evaluation key")
                seen.add(key)
                digests[strategy].update(json.dumps(values, separators=(",", ":")).encode() + b"\n")
            totals[strategy] += len(frame)
            aggregate = frame.groupby("scene_name", as_index=False).agg(update_count=("ade", "size"),
                ade_sum=("ade", "sum"), fde_sum=("fde", "sum"), jitter_sum=("jitter", "sum"))
            aggregate["strategy"] = strategy
            row_stats.append(aggregate)
    if any(count != summary["update_count"] for count in totals.values()) or any(digest.hexdigest() != summary["ordered_key_sha256"] for digest in digests.values()):
        raise ValueError("missing rows or ordered key mismatch")
    derived = pd.concat(row_stats).groupby(["strategy", "scene_name"]).sum(numeric_only=True).sort_index()
    recorded = stats.set_index(["strategy", "scene_name"])[derived.columns].sort_index()
    if not derived.index.equals(recorded.index) or not np.allclose(derived.to_numpy(), recorded.to_numpy(), rtol=1e-9, atol=1e-8):
        raise ValueError("per-update and scene statistics differ")
    return summary, stats


def report_ablation(run_dirs, output_dir, seeds=(2027, 2028, 2029), replicates=2000):
    if len(run_dirs) != len(seeds) or len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("three unique seeds required")
    runs = [_read_run(root) for root in run_dirs]
    actual = [summary["provenance"]["seed"] for summary, _ in runs]
    if sorted(actual) != sorted(seeds):
        raise ValueError("missing or duplicate seeds")
    first = runs[0][0]
    for summary, _ in runs[1:]:
        if any(summary[field] != first[field] for field in ("ordered_key_sha256", "sample_count", "update_count", "coverage")) or any(summary["provenance"][field] != first["provenance"][field] for field in ("evaluation_identity", "training_config_sha256")):
            raise ValueError("cross-seed key/identity mismatch")
        if summary["provenance"].get("source_code_sha256") != first["provenance"].get("source_code_sha256"):
            raise ValueError("evaluation source identity mismatch")
        if any(summary["strategies"][strategy]["fallback_updates"] != first["strategies"][strategy]["fallback_updates"] for strategy in STRATEGIES):
            raise ValueError("cross-seed fallback counts differ")
        for strategy in ("frozen", "sliding"):
            if any(summary["strategies"][strategy][metric] != first["strategies"][strategy][metric] for metric in METRICS):
                raise ValueError("deterministic baseline differs across seeds")
    all_stats = pd.concat([stats for _, stats in runs], ignore_index=True)
    scenes = sorted(all_stats.scene_name.unique())
    blocks = {}
    counts = None
    for strategy in STRATEGIES:
        block = all_stats[all_stats.strategy == strategy].pivot(index="scene_name", columns="seed", values="update_count").reindex(index=scenes, columns=seeds)
        if block.isna().any().any() or not np.equal(block.to_numpy(), block.iloc[:, :1].to_numpy()).all():
            raise ValueError("cross-seed scene count mismatch")
        current = block.iloc[:, 0].to_numpy()
        if counts is not None and not np.array_equal(current, counts):
            raise ValueError("cross-strategy scene mismatch")
        counts = current
        blocks[strategy] = {}
        for metric in METRICS:
            values = all_stats[all_stats.strategy == strategy].pivot(index="scene_name", columns="seed", values=metric + "_sum").reindex(index=scenes, columns=seeds)
            if values.isna().any().any():
                raise ValueError("missing scene metrics")
            blocks[strategy][metric] = values.to_numpy().mean(1)
    metrics = {strategy: {metric: {"mean": float(np.mean([summary["strategies"][strategy][metric] for summary, _ in runs])),
                    "seed_std": float(np.std([summary["strategies"][strategy][metric] for summary, _ in runs], ddof=1))}
                for metric in METRICS} for strategy in STRATEGIES}
    for strategy in STRATEGIES:
        for metric in METRICS:
            metrics[strategy][metric]["ci95"] = paired_scene_interval(blocks[strategy][metric], counts, replicates)["ci95"]
    contrasts = {}
    for mode in ("map", "neighbors", "fusion"):
        for reference in ("original", "matched_" + mode):
            contrasts[mode + "-" + reference] = {metric: paired_scene_interval(blocks[mode][metric] - blocks[reference][metric], counts, replicates)
                                                   for metric in METRICS}
    result = {"seeds": list(seeds), "sample_count": first["sample_count"], "update_count_per_strategy_seed": first["update_count"],
        "scene_count": len(scenes), "coverage": first["coverage"],
        "fallback_updates": {strategy: first["strategies"][strategy]["fallback_updates"] for strategy in STRATEGIES}, "metrics": metrics, "contrasts": contrasts,
        "uncertainty": {"unit": "paired_scene_cluster", "replicates": replicates, "bootstrap_seed": 0,
            "seed_reduction": "average_fixed_three_seeds_before_scene_resampling", "seed_std_ddof": 1,
            "scope": "conditional_on_fixed_seeds_and_historical_validation_split", "negative_delta": "improvement"},
        "sources": [{"path": str(Path(root)), "summary_sha256": sha256_file(Path(root) / "summary.json")} for root in run_dirs]}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    rows = [{"strategy": strategy, **{metric + "_" + field: value for metric, values in entry.items() for field, value in values.items() if field != "ci95"},
             **{metric + "_ci_" + bound: values["ci95"][i] for metric, values in entry.items() for i, bound in enumerate(("low", "high"))}}
            for strategy, entry in metrics.items()]
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame([{"contrast": name, "metric": metric, "delta": entry["delta"], "ci_low": entry["ci95"][0], "ci_high": entry["ci95"][1]}
                  for name, values in contrasts.items() for metric, entry in values.items()]).to_csv(output / "contrasts.csv", index=False)
    for filename, group in (("by_update.csv", "update_step"), ("by_horizon.csv", "remaining_steps"), ("by_maneuver.csv", "maneuver")):
        frames = pd.concat([pd.read_csv(Path(root) / filename) for root in run_dirs], ignore_index=True)
        if frames.duplicated(["strategy", group, "seed"]).any():
            raise ValueError("duplicate grouped rows")
        if set(frames.seed) != set(seeds) or not (frames.groupby(["strategy", group]).seed.nunique() == 3).all() or not (frames.groupby(["strategy", group]).update_count.nunique() == 1).all():
            raise ValueError("missing grouped seed rows/counts")
        frames.groupby(["strategy", group], as_index=False).agg(**{metric: (metric, "mean") for metric in METRICS},
            update_count=("update_count", "first"), map_coverage=("map_coverage", "mean"), neighbor_coverage=("neighbor_coverage", "mean"),
            fallback_updates=("fallback_updates", "first")).to_csv(output / filename, index=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=3, type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = report_ablation(args.runs, args.output)
    print(json.dumps({"scenes": result["scene_count"], "seeds": result["seeds"], "contrasts": result["contrasts"]}, indent=2))

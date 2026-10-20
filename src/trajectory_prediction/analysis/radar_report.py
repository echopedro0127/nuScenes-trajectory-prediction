"""Audited mini Radar condition tables; two-scene uncertainty is descriptive only."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trajectory_prediction.analysis.context_ablation import paired_scene_interval
from trajectory_prediction.online.observation_noise import CONDITIONS
from trajectory_prediction.online.radar_evaluation import STRATEGIES, recovery_table, row_key_hash
from trajectory_prediction.online.scaled_evaluation import benchmark_keys
from trajectory_prediction.utils.provenance import sha256_file

METRICS = ("ade", "fde", "jitter")
LATENCY_FIELDS = ("total_ms", "read_decode_ms", "transform_ms", "aggregate_ms", "radar_total_ms", "feature_transfer_ms", "head_return_ms")
DIAGNOSTICS = ("radar_covered", "radar_points", "position_age_s", "position_fresh", "radar_age_max_s", "sample_index", "maneuver", "remaining_steps")
ARTIFACTS = ("per_update.csv.gz", "scene_stats.csv", "by_condition.csv", "by_update.csv", "by_horizon.csv", "by_maneuver.csv", "recovery.csv")


def _equal_frame(actual, expected, keys):
    try:
        pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True), expected[actual.columns].sort_values(keys).reset_index(drop=True),
            check_dtype=False, rtol=1e-8, atol=1e-9)
    except AssertionError as exc:
        raise ValueError("inconsistent evaluation row/group/recovery statistics") from exc


def _load_run(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    if summary["failure_count"] != 0 or set(summary["sha256"]) != set(ARTIFACTS):
        raise ValueError("incomplete Radar evaluation")
    for name in ARTIFACTS:
        if sha256_file(root / name) != summary["sha256"][name]:
            raise ValueError("Radar artifact checksum mismatch")
    frame = pd.read_csv(root / "per_update.csv.gz")
    seed, n = summary["provenance"]["seed"], summary["sample_count"]
    if len(frame) != n * 55 * 6 or summary["row_count"] != len(frame) or summary["all_condition_updates_per_strategy"] != n * 55 or summary["updates_per_condition_strategy"] != n * 11 or set(frame.seed) != {seed} or set(frame.strategy) != set(STRATEGIES) or set(frame.condition) != set(CONDITIONS):
        raise ValueError("missing Radar evaluation rows/seed")
    if frame.duplicated(["strategy", "condition", "instance_token", "sample_token", "update_step"]).any() or not np.isfinite(frame[list(METRICS)]).all().all() or (frame[list(METRICS)] < 0).any().any():
        raise ValueError("invalid/duplicate Radar metric rows")
    if not np.all(frame.remaining_steps == 12 - frame.update_step) or not np.all(frame[frame.update_step == 1].jitter == 0):
        raise ValueError("Radar remaining horizon/jitter mismatch")
    if any(frame[name].dtype.kind != "b" for name in ("radar_covered", "position_fresh", "fallback")) or not np.isfinite(frame[["radar_points", "position_age_s", "radar_age_max_s"]]).all().all() or (frame[["radar_points", "position_age_s", "radar_age_max_s"]] < 0).any().any():
        raise ValueError("invalid sensor diagnostics")
    reference_diagnostics = frame[frame.strategy == "frozen"][list(DIAGNOSTICS)].reset_index(drop=True)
    values = {}
    for strategy in STRATEGIES:
        part = frame[frame.strategy == strategy]
        if not part[list(DIAGNOSTICS)].reset_index(drop=True).equals(reference_diagnostics):
            raise ValueError("cross-strategy sensor diagnostics differ")
        expected_fallback = ~part.radar_covered if strategy == "radar" else np.zeros(len(part), bool)
        if not np.array_equal(part.fallback, expected_fallback):
            raise ValueError("fallback/coverage diagnostics differ")
        if len(part) != n * 55 or row_key_hash(part) != summary["ordered_key_sha256"]:
            raise ValueError("Radar cross-strategy ordered key mismatch")
        if not np.array_equal(part.condition.to_numpy(), np.repeat(CONDITIONS, n * 11)) or not np.array_equal(part.update_step.to_numpy(), np.tile(np.arange(1, 12), n * 5)):
            raise ValueError("Radar condition/update order mismatch")
        values[strategy] = part[list(METRICS)].to_numpy().reshape(5, n, 11, 3)
        if strategy == "frozen" and (not np.all(values[strategy][..., 2] == 0) or not np.allclose(values[strategy], values[strategy][:1], rtol=0, atol=0)):
            raise ValueError("frozen baseline must ignore condition")
    scenes = frame.groupby(["condition", "scene_name", "strategy", "seed"], as_index=False).agg(update_count=("ade", "size"),
        ade_sum=("ade", "sum"), fde_sum=("fde", "sum"), jitter_sum=("jitter", "sum"))
    _equal_frame(pd.read_csv(root / "scene_stats.csv"), scenes, ["condition", "scene_name", "strategy", "seed"])
    condition = frame.groupby(["condition", "strategy", "seed"], as_index=False).agg(update_count=("ade", "size"), ade=("ade", "mean"),
        fde=("fde", "mean"), jitter=("jitter", "mean"), radar_coverage=("radar_covered", "mean"), mean_points=("radar_points", "mean"), fallback_updates=("fallback", "sum"))
    if not (condition.update_count == n * 11).all():
        raise ValueError("condition row counts differ")
    _equal_frame(pd.read_csv(root / "by_condition.csv"), condition, ["condition", "strategy", "seed"])
    base_keys = frame[(frame.strategy == "frozen") & (frame.condition == "clean") & (frame.update_step == 1)]
    expected_recovery = recovery_table(values, {name: base_keys[name].to_numpy() for name in ("instance_token", "sample_token", "scene_name")}, seed)
    recovery = pd.read_csv(root / "recovery.csv")
    _equal_frame(recovery, expected_recovery, ["strategy", "instance_token", "sample_token", "seed"])
    for filename, field in (("by_update.csv", "update_step"), ("by_horizon.csv", "remaining_steps"), ("by_maneuver.csv", "maneuver")):
        grouped = frame.copy()
        if field == "maneuver":
            grouped[field] = grouped[field].str.split("|")
            grouped = grouped.explode(field)
        expected = grouped.groupby(["condition", "strategy", "seed", field], as_index=False).agg(update_count=("ade", "size"), ade=("ade", "mean"),
            fde=("fde", "mean"), jitter=("jitter", "mean"), radar_coverage=("radar_covered", "mean"), fallback_updates=("fallback", "sum"))
        _equal_frame(pd.read_csv(root / filename), expected, ["condition", "strategy", "seed", field])
    latency = _load_latency(root, summary, frame[frame.strategy == "frozen"])
    return summary, frame, scenes, condition, recovery, latency


def _load_latency(root, summary, baseline):
    latency = json.loads((root / "latency.json").read_text())
    for name, field in (("latency_keys.csv", "keys_sha256"), ("latency_measurements.csv", "measurements_sha256")):
        if sha256_file(root / name) != latency[field]:
            raise ValueError("latency artifact checksum mismatch")
    keys, measurements = pd.read_csv(root / "latency_keys.csv"), pd.read_csv(root / "latency_measurements.csv")
    count = min(256, summary["all_condition_updates_per_strategy"])
    if latency["key_count"] != count or len(keys) != count or latency["warmup"] != 16 or latency["inference_batch_size"] != 1 or latency["decoded_frame_cache"] is not False or set(latency["modes"]) != {"raw_radar", "cached_radar"}:
        raise ValueError("incomplete latency benchmark")
    if set(measurements) != {"mode", "flat_index", *LATENCY_FIELDS} or any(set(profile) != set(LATENCY_FIELDS) for profile in latency["modes"].values()):
        raise ValueError("latency total/stage fields missing or unexpected")
    expected_indices = benchmark_keys(summary["all_condition_updates_per_strategy"])
    if not np.array_equal(keys.flat_index, expected_indices):
        raise ValueError("latency key selection mismatch")
    names = ["condition", "instance_token", "sample_token", "scene_name", "update_step", "timestamp_us"]
    if not keys[names].reset_index(drop=True).equals(baseline.iloc[expected_indices][names].reset_index(drop=True)):
        raise ValueError("latency/evaluation keys differ")
    if len(measurements) != count * 2 or measurements.duplicated(["mode", "flat_index"]).any() or set(measurements["mode"]) != {"raw_radar", "cached_radar"}:
        raise ValueError("missing latency measurement rows")
    for mode, part in measurements.groupby("mode"):
        if not np.array_equal(part.flat_index, expected_indices):
            raise ValueError("latency measurements use different keys")
        for field, values in latency["modes"][mode].items():
            if not field.endswith("_ms") or field not in part or not np.isfinite(part[field]).all() or (part[field] < 0).any() or not np.allclose([values["p50"], values["p95"]], part[field].quantile([.5, .95]), rtol=1e-9, atol=1e-9):
                raise ValueError("latency percentile mismatch")
    return latency


def report_radar(run_dirs, output_dir):
    if len(run_dirs) != 3:
        raise ValueError("three unique seeds required")
    loaded = [_load_run(root) for root in run_dirs]
    seeds = [run[0]["provenance"]["seed"] for run in loaded]
    if sorted(seeds) != [2027, 2028, 2029]:
        raise ValueError("missing or duplicate seeds")
    first = loaded[0][0]
    for summary, frame, _, _, _, latency in loaded[1:]:
        if summary["ordered_key_sha256"] != first["ordered_key_sha256"] or any(summary["provenance"][name] != first["provenance"][name] for name in ("evaluation_identity", "training_config_sha256", "evaluation_source_sha256")):
            raise ValueError("cross-seed evaluation key/source mismatch")
        if any(latency[name] != loaded[0][5][name] for name in ("keys_sha256", "device", "device_name", "torch_version", "raw_scope")):
            raise ValueError("cross-seed latency identity mismatch")
        baseline_diagnostics = loaded[0][1][loaded[0][1].strategy == "frozen"][list(DIAGNOSTICS)].reset_index(drop=True)
        actual_diagnostics = frame[frame.strategy == "frozen"][list(DIAGNOSTICS)].reset_index(drop=True)
        if not baseline_diagnostics.equals(actual_diagnostics):
            raise ValueError("cross-seed sensor diagnostics differ")
        for strategy in ("frozen", "sliding", "filtered"):
            reference = loaded[0][1][loaded[0][1].strategy == strategy]
            actual = frame[frame.strategy == strategy]
            if not np.array_equal(reference[list(METRICS)].to_numpy(), actual[list(METRICS)].to_numpy()):
                raise ValueError("deterministic baseline differs across seeds")
    stats = pd.concat([run[2] for run in loaded])
    conditions = pd.concat([run[3] for run in loaded])
    recovery_rows = pd.concat([run[4] for run in loaded])
    scenes = sorted(stats.scene_name.unique())
    metrics, contrasts, coverage = {}, {}, {}
    for condition in CONDITIONS:
        block = stats[stats.condition == condition]
        counts = block.pivot(index="scene_name", columns=["strategy", "seed"], values="update_count").reindex(scenes)
        if counts.isna().any().any() or not np.all(counts.to_numpy() == counts.iloc[:, :1].to_numpy()):
            raise ValueError("cross-seed scene count mismatch")
        count = counts.iloc[:, 0].to_numpy()
        sums = {}
        metrics[condition] = {}
        for strategy in STRATEGIES:
            sums[strategy], metrics[condition][strategy] = {}, {}
            for metric in METRICS:
                sums[strategy][metric] = block[block.strategy == strategy].groupby("scene_name")[metric + "_sum"].mean().reindex(scenes).to_numpy()
                mean = conditions[(conditions.condition == condition) & (conditions.strategy == strategy)][metric].to_numpy()
                interval = paired_scene_interval(sums[strategy][metric], count)
                metrics[condition][strategy][metric] = {"mean": float(mean.mean()), "seed_std": float(np.std(mean, ddof=1)), "ci95": interval["ci95"]}
        contrasts[condition] = {"radar-" + reference: {metric: paired_scene_interval(sums["radar"][metric] - sums[reference][metric], count)
            for metric in METRICS} for reference in ("original", "matched_radar")}
        part = conditions[(conditions.condition == condition) & (conditions.strategy == "radar")]
        if part.radar_coverage.nunique() != 1 or part.fallback_updates.nunique() != 1:
            raise ValueError("cross-seed Radar coverage differs")
        coverage[condition] = {"radar_coverage": float(part.radar_coverage.iloc[0]), "mean_points": float(part.mean_points.iloc[0]), "fallback_updates_per_seed": int(part.fallback_updates.iloc[0])}
    recovery = {}
    for strategy, part in recovery_rows.groupby("strategy"):
        recovered = part[part.recovered]
        recovery[strategy] = {"sample_seed_rows": len(part), "recovered": len(recovered), "unrecovered": len(part) - len(recovered),
            "recovered_fraction": len(recovered) / len(part), "mean_seconds_among_recovered": float(recovered.recovery_seconds.mean()) if len(recovered) else None,
            "right_censor_seconds": 2., "scope": "descriptive_repeated_fixed_seeds_not_independent_trials"}
    result = {"seeds": seeds, "sample_count": first["sample_count"], "scene_count": len(scenes), "rows_per_seed": first["row_count"],
        "failure_count": 0, "metrics": metrics, "contrasts": contrasts, "coverage": coverage, "recovery": recovery,
        "latency": {str(seed): run[5] for seed, run in zip(seeds, loaded, strict=True)},
        "uncertainty": {"scene_clusters": len(scenes), "replicates": 2000, "bootstrap_seed": 0,
            "scope": "mini_two_scene_description_not_generalization", "seed_std_ddof": 1, "seed_reduction": "mean_fixed_seeds_before_paired_scene_resampling"},
        "source_code_sha256": sha256_file(Path(__file__)), "sources": [{"path": str(root), "summary_sha256": sha256_file(Path(root) / "summary.json")} for root in run_dirs]}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    rows = [{"condition": condition, "strategy": strategy, **{metric + "_" + field: value for metric, item in entry.items() for field, value in item.items() if field != "ci95"},
             **{metric + "_ci_" + bound: item["ci95"][j] for metric, item in entry.items() for j, bound in enumerate(("low", "high"))}}
            for condition, strategies in metrics.items() for strategy, entry in strategies.items()]
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame([{"condition": condition, "contrast": name, "metric": metric, "delta": item["delta"], "ci_low": item["ci95"][0], "ci_high": item["ci95"][1]}
        for condition, values in contrasts.items() for name, entries in values.items() for metric, item in entries.items()]).to_csv(output / "contrasts.csv", index=False)
    recovery_rows.to_csv(output / "recovery.csv", index=False)
    pd.DataFrame([{"strategy": strategy, **entry} for strategy, entry in recovery.items()]).to_csv(output / "recovery_summary.csv", index=False)
    for filename, field in (("by_update.csv", "update_step"), ("by_horizon.csv", "remaining_steps"), ("by_maneuver.csv", "maneuver")):
        values = pd.concat([pd.read_csv(Path(root) / filename) for root in run_dirs])
        values.groupby(["condition", "strategy", field], as_index=False).agg(**{metric: (metric, "mean") for metric in METRICS},
            update_count=("update_count", "first"), radar_coverage=("radar_coverage", "mean"), fallback_updates=("fallback_updates", "first")).to_csv(output / filename, index=False)
    pd.DataFrame([{"seed": seed, "mode": mode, **{field + "_" + percentile: value for field, profile in profiles.items() for percentile, value in profile.items()}}
        for seed, run in zip(seeds, loaded, strict=True) for mode, profiles in run[5]["modes"].items()]).to_csv(output / "latency.csv", index=False)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs=3, required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = report_radar(args.runs, args.output)
    print(json.dumps({"scenes": result["scene_count"], "seeds": result["seeds"], "metrics": result["metrics"], "recovery": result["recovery"]}, indent=2))

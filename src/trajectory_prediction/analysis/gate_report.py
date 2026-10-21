"""Stream and reconstruct gate results before scene-paired three-seed reporting."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.analysis.context_ablation import paired_scene_interval
from trajectory_prediction.analysis.maneuvers import describe_motion, tag_maneuvers
from trajectory_prediction.analysis.radar_report import _equal_frame as _compare_frame
from trajectory_prediction.context.gate_preparation import (
    load_verified_gate_cache,
    preparation_sources,
)
from trajectory_prediction.online.context_evaluator import DEFAULT_THRESHOLDS
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.gate_dataset import GateDataset
from trajectory_prediction.online.gate_evaluation import (
    ARTIFACTS,
    CONDITIONS,
    DIAGNOSTICS,
    KEY_FIELDS,
    LATENCY_SCOPE,
    METRICS,
    STRATEGIES,
    aggregate_rows,
    evaluate_gate,
    key_frame,
    mean_rows,
    recovery_table,
    update_hash,
)
from trajectory_prediction.online.gate_training import load_gate_checkpoint
from trajectory_prediction.online.scaled_evaluation import benchmark_keys
from trajectory_prediction.utils.provenance import sha256_file

SEEDS = (2027, 2028, 2029)
CONTROLS = ("sliding", "filtered", "fixed_gate", "ungated_map")


def _equal_frame(actual, expected, keys):
    if set(actual.columns) != set(expected.columns):
        raise ValueError("artifact columns mismatch")
    _compare_frame(actual, expected, keys)


def combine_scene_intervals(stats, replicates=2000):
    if set(stats.seed) != set(SEEDS):
        raise ValueError("complete fixed seeds required")
    if stats.duplicated(["seed", "condition", "strategy", "scene_name"]).any():
        raise ValueError("duplicate scene rows")
    rows = []
    for condition, part in stats.groupby("condition"):
        for control in CONTROLS:
            a = part[part.strategy == "learned_gate"].set_index(["seed", "scene_name"]).sort_index()
            b = part[part.strategy == control].set_index(["seed", "scene_name"]).sort_index()
            if not a.index.equals(b.index) or not np.array_equal(a.update_count, b.update_count):
                raise ValueError("paired scene counts mismatch")
            if not (a.reset_index().groupby("scene_name").seed.nunique() == 3).all():
                raise ValueError("incomplete scene seeds")
            count = a.update_count.groupby("scene_name").mean()
            if not (a.update_count.groupby("scene_name").nunique() == 1).all():
                raise ValueError("scene seed counts mismatch")
            for metric in METRICS:
                delta = (a[metric + "_sum"] - b[metric + "_sum"]).groupby("scene_name").mean()
                rows.append(
                    {
                        "condition": condition,
                        "contrast": "learned_gate-" + control,
                        "metric": metric,
                        **paired_scene_interval(
                            delta.to_numpy(), count.to_numpy(), replicates=replicates, seed=0
                        ),
                    }
                )
    return rows


def audit_latency(root, size, expected_keys):
    root = Path(root)
    value = json.loads((root / "latency.json").read_text())
    fields = {
        "key_count",
        "warmup",
        "inference_batch_size",
        "key_selection_seed",
        "scope",
        "excluded",
        "device",
        "device_name",
        "torch_version",
        "keys_sha256",
        "measurements_sha256",
        "strategies",
    }
    if (
        set(value) != fields
        or value["scope"] != LATENCY_SCOPE
        or value["warmup"] != 16
        or value["inference_batch_size"] != 1
        or value["key_selection_seed"] != 2027
        or value["key_count"] != min(256, size)
        or value["excluded"] != ["map_query", "P4_candidate_forward", "cold_cache_load"]
    ):
        raise ValueError("incomplete latency scope/fields")
    for name, field in (
        ("latency_keys.csv", "keys_sha256"),
        ("latency_measurements.csv", "measurements_sha256"),
    ):
        if sha256_file(root / name) != value[field]:
            raise ValueError("latency checksum mismatch")
    keys = pd.read_csv(root / "latency_keys.csv")
    measurements = pd.read_csv(root / "latency_measurements.csv")
    if set(keys) != {"flat_index", *KEY_FIELDS}:
        raise ValueError("latency key columns missing")
    if not np.array_equal(keys.flat_index, benchmark_keys(size)):
        raise ValueError("latency fixed key mismatch")
    _equal_frame(keys, expected_keys, ["flat_index"])
    if (
        set(measurements) != {"strategy", "flat_index", "total_ms"}
        or measurements.duplicated(["strategy", "flat_index"]).any()
        or set(measurements.strategy) != {"fixed_gate", "learned_gate"}
        or set(value["strategies"]) != {"fixed_gate", "learned_gate"}
    ):
        raise ValueError("incomplete latency measurements")
    for strategy, part in measurements.groupby("strategy"):
        if (
            not np.array_equal(part.flat_index, keys.flat_index)
            or not np.isfinite(part.total_ms).all()
            or (part.total_ms < 0).any()
        ):
            raise ValueError("invalid latency rows")
        profile = value["strategies"][strategy]
        if set(profile) != {"p50", "p95"} or not np.allclose(
            [profile["p50"], profile["p95"]],
            part.total_ms.quantile([0.5, 0.95]),
            rtol=1e-9,
            atol=1e-9,
        ):
            raise ValueError("latency percentile mismatch")
    return value


def audit_run(root, device=None):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    p = summary["provenance"]
    if (
        summary["failure_count"] != 0
        or summary["strategies"] != list(STRATEGIES)
        or summary["conditions"] != list(CONDITIONS)
        or set(summary["sha256"]) != set(ARTIFACTS)
    ):
        raise ValueError("incomplete gate evaluation")
    for name in ARTIFACTS:
        if sha256_file(root / name) != summary["sha256"][name]:
            raise ValueError("gate artifact checksum mismatch")
    for name in ("candidate_manifest", "checkpoint"):
        if sha256_file(Path(p[name]["path"])) != p[name]["sha256"]:
            raise ValueError("gate source checksum mismatch")
    source = Path(__file__).parents[1] / "online" / "gate_evaluation.py"
    if p["evaluation_source_sha256"] != sha256_file(source):
        raise ValueError("gate evaluation executable mismatch")
    contract = p["candidate_contract"]
    if (
        contract["seed"] != p["seed"]
        or contract["split"] != "val"
        or contract["preparation_sources"] != preparation_sources()
    ):
        raise ValueError("gate evaluation candidate source mismatch")
    arrays, _manifest = load_verified_gate_cache(
        Path(p["candidate_manifest"]["path"]).parent, "val", p["seed"], device=device
    )
    if _manifest["contract"] != contract:
        raise ValueError("gate provenance candidate contract mismatch")
    data = GateDataset(arrays)
    model, meta = load_gate_checkpoint(
        p["checkpoint"]["path"], contract, data.scenes, torch.device("cpu")
    )
    if meta["run_identity"] != p["training_run_identity"]:
        raise ValueError("gate training identity mismatch")
    if (
        summary["sample_count"] != data.n
        or summary["updates_per_condition_strategy"] != data.n * 11
        or summary["all_condition_updates_per_strategy"] != data.size
        or summary["row_count"] != data.size * 7
    ):
        raise ValueError("gate evaluation size mismatch")
    results, _, weights = evaluate_gate(data, model, device=device or torch.device("cpu"))
    tags = [
        "|".join(tag_maneuvers(describe_motion(t, 0.5), DEFAULT_THRESHOLDS))
        for t in arrays["target"]
    ]
    accumulated = {
        field: []
        for field in ("scene_name", "condition", "update_step", "remaining_steps", "maneuver")
    }
    counts = {s: 0 for s in STRATEGIES}
    digests = {s: (hashlib.sha256(), hashlib.sha256()) for s in STRATEGIES}
    current_pair, expected = None, None
    for chunk in pd.read_csv(root / "per_update.csv.gz", chunksize=32768, float_precision="round_trip"):
        if (
            set(chunk.seed) != {p["seed"]}
            or not set(chunk.strategy) <= set(STRATEGIES)
            or not set(chunk.condition) <= set(CONDITIONS)
        ):
            raise ValueError("row seed/strategy/condition mismatch")
        for (strategy, condition), part in chunk.groupby(["strategy", "condition"], sort=False):
            c = CONDITIONS.index(condition)
            offset = counts[strategy] - c * data.n * 11
            if offset < 0 or offset + len(part) > data.n * 11:
                raise ValueError("row order/count mismatch")
            pair = (strategy, condition)
            if pair != current_pair:
                expected = key_frame(data, c, tags)
                current_pair = pair
            expected_part = expected.iloc[offset : offset + len(part)]
            _equal_frame(
                part[list(KEY_FIELDS) + list(DIAGNOSTICS)], expected_part, list(KEY_FIELDS)
            )
            # Exact order is also hashed; comparison above deliberately ignores pandas chunk indices.
            update_hash(digests[strategy][0], part, KEY_FIELDS)
            update_hash(digests[strategy][1], part, DIAGNOSTICS)
            values = results[strategy][c].reshape(-1, 3)[offset : offset + len(part)]
            if not np.allclose(part[list(METRICS)], values, rtol=1e-6, atol=1e-6):
                raise ValueError("row candidate metric reconstruction mismatch")
            g = (
                weights[strategy][c].reshape(-1)[offset : offset + len(part)]
                if strategy in weights
                else np.zeros(len(part))
            )
            if not np.allclose(part.gate_weight, g, rtol=1e-6, atol=1e-7):
                raise ValueError("row gate weight mismatch")
            for name in ("valid", "fallback", "missing", "expired"):
                if part[name].dtype.kind != "b":
                    raise ValueError("invalid diagnostic bool")
            fallback = (
                ~part.valid
                if strategy in ("ungated_map", "fixed_gate", "learned_gate")
                else np.zeros(len(part), bool)
            )
            if (
                not np.array_equal(part.fallback, fallback)
                or not np.array_equal(part.missing, part.packet_step == 0)
                or not np.array_equal(part.expired, (part.packet_step > 0) & (part.age_s > 2))
            ):
                raise ValueError("row gate fallback diagnostics mismatch")
            if (
                not np.isfinite(part[list(METRICS)]).all().all()
                or (part[list(METRICS)] < 0).any().any()
                or not (part[part.update_step == 1].jitter == 0).all()
            ):
                raise ValueError("invalid row metrics/jitter")
            for field, parts in accumulated.items():
                grouped = part
                if field == "maneuver":
                    grouped = part.assign(maneuver=part.maneuver.str.split("|")).explode("maneuver")
                fields = ["seed", "strategy", "condition"] + (
                    [] if field == "condition" else [field]
                )
                parts.append(aggregate_rows(grouped, fields))
            counts[strategy] += len(part)
    for strategy in STRATEGIES:
        actual = {
            "keys": digests[strategy][0].hexdigest(),
            "diagnostics": digests[strategy][1].hexdigest(),
        }
        if (
            counts[strategy] != data.size
            or actual != summary["ordered_hashes"][strategy]
            or actual != summary["ordered_hashes"]["frozen"]
        ):
            raise ValueError("missing rows or cross-strategy diagnostics mismatch")
    audited = {}
    for field, filename in (
        ("scene_name", "scene_stats.csv"),
        ("condition", "by_condition.csv"),
        ("update_step", "by_update.csv"),
        ("remaining_steps", "by_horizon.csv"),
        ("maneuver", "by_maneuver.csv"),
    ):
        keys = ["seed", "strategy", "condition"] + ([] if field == "condition" else [field])
        combined = (
            pd.concat(accumulated[field]).groupby(keys, as_index=False).sum(numeric_only=True)
        )
        derived = combined if field == "scene_name" else mean_rows(combined)
        _equal_frame(pd.read_csv(root / filename), derived, keys)
        audited[field] = derived
    recovery = pd.read_csv(root / "recovery.csv")
    _equal_frame(
        recovery,
        recovery_table(results, arrays, p["seed"]),
        ["strategy", "instance_token", "sample_token", "seed"],
    )
    distribution = []
    for strategy, values in weights.items():
        for c, condition in enumerate(CONDITIONS):
            g = pd.Series(values[c].reshape(-1).astype(np.float64))
            distribution.append(
                {
                    "seed": p["seed"],
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
    _equal_frame(
        pd.read_csv(root / "gate_distribution.csv"),
        pd.DataFrame(distribution),
        ["seed", "condition", "strategy"],
    )
    indices = benchmark_keys(data.size)
    expected_keys = (
        pd.concat([key_frame(data, c, tags)[list(KEY_FIELDS)] for c in range(6)], ignore_index=True)
        .iloc[indices]
        .copy()
    )
    expected_keys.insert(0, "flat_index", indices)
    latency = audit_latency(root, data.size, expected_keys)
    return summary, audited, recovery, latency


def report_gate(run_dirs, output_dir, replicates=2000, device=None):
    if len(run_dirs) != 3:
        raise ValueError("three seed runs required")
    summaries, groups, recoveries, latencies = [], [], [], []
    for root in run_dirs:
        summary, audited, recovery, latency = audit_run(root, device=device)
        print(json.dumps({"audited_seed": summary["provenance"]["seed"], "rows": summary["row_count"]}), flush=True)
        summaries.append(summary)
        groups.append(audited)
        recoveries.append(recovery)
        latencies.append(latency)
    if sorted(s["provenance"]["seed"] for s in summaries) != list(SEEDS):
        raise ValueError("fixed seed set mismatch")
    reference = summaries[0]
    for s in summaries[1:]:
        if (
            s["ordered_hashes"]["frozen"] != reference["ordered_hashes"]["frozen"]
            or s["sample_count"] != reference["sample_count"]
            or s["provenance"]["candidate_contract"]["dataset_identity"]
            != reference["provenance"]["candidate_contract"]["dataset_identity"]
        ):
            raise ValueError("cross-seed evaluation source/keys mismatch")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    scene = pd.concat([g["scene_name"] for g in groups], ignore_index=True)
    condition = pd.concat([g["condition"] for g in groups], ignore_index=True)
    mean = condition.groupby(["condition", "strategy"], as_index=False).agg(
        **{m + "_mean": (m, "mean") for m in METRICS}, **{m + "_std": (m, "std") for m in METRICS}
    )
    mean.to_csv(output / "condition_seed_mean_std.csv", index=False)
    intervals = combine_scene_intervals(scene, replicates)
    (output / "paired_scene_intervals.json").write_text(json.dumps(intervals, indent=2) + "\n")
    for field, filename in (
        ("update_step", "by_update.csv"),
        ("remaining_steps", "by_horizon.csv"),
        ("maneuver", "by_maneuver.csv"),
    ):
        pd.concat([g[field] for g in groups], ignore_index=True).to_csv(
            output / filename, index=False
        )
    recovery = pd.concat(recoveries, ignore_index=True)
    recovery.groupby(["seed", "strategy"], as_index=False).agg(
        sample_count=("recovered", "size"),
        recovered_fraction=("recovered", "mean"),
        conditional_recovery_seconds=("recovery_seconds", "mean"),
    ).to_csv(output / "recovery_summary.csv", index=False)
    result = {
        "seeds": list(SEEDS),
        "sample_count": reference["sample_count"],
        "total_rows": sum(s["row_count"] for s in summaries),
        "bootstrap": {
            "replicates": replicates,
            "seed": 0,
            "unit": "scene_after_fixed_three_seed_average",
        },
        "recovery_interpretation": "stateless_candidates_match_clean_when_packet_7_arrives_so_zero_recovery_is_structural",
        "latency_scope": LATENCY_SCOPE,
        "latencies": latencies,
        "preparation": {
            str(s["provenance"]["seed"]): {
                k: s["provenance"]["candidate_contract"][k]
                for k in ("candidate_preparation_seconds", "candidate_updates_per_second")
            }
            for s in summaries
        },
        "runs": [
            {
                "path": str(Path(root).resolve()),
                "summary_sha256": sha256_file(Path(root) / "summary.json"),
            }
            for root in run_dirs
        ],
        "report_source_sha256": sha256_file(Path(__file__)),
    }
    (output / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    report_gate(
        [args.evaluation_root / ("seed-" + str(seed)) for seed in SEEDS],
        args.output_root,
        device=choose_device(args.device),
    )


if __name__ == "__main__":
    main()

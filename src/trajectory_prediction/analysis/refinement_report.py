"""Re-fit calibration, replay frozen selection and every P6.1 numeric result."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.analysis.context_ablation import paired_scene_interval
from trajectory_prediction.analysis.gate_report import _equal_frame
from trajectory_prediction.context.cache import key_hash
from trajectory_prediction.context.gate_preparation import load_verified_gate_cache
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.gate_evaluation import KEY_FIELDS, key_frame
from trajectory_prediction.online.gate_refinement import RefinementDataset
from trajectory_prediction.online.refinement_evaluation import (
    ARTIFACTS,
    CONDITIONS,
    METRICS,
    SCOPE,
    SEEDS,
    STRATEGIES,
    TAUS,
    condition_table,
    derive_tables,
    evaluate_suite,
    evaluation_sources,
    load_selection,
    load_suite,
    measure_strategy,
)
from trajectory_prediction.online.refinement_training import fit_calibration
from trajectory_prediction.online.scaled_evaluation import benchmark_keys
from trajectory_prediction.utils.provenance import sha256_file

NEW_STRATEGIES = (
    "constant_gate",
    "time_age_gate",
    "original_gate",
    "short_gate",
    "finetune_0",
    "consistent_002",
    "consistent_010",
    "consistent_short",
)


def report_sources():
    package = Path(__file__).parents[1]
    return {
        "report": {
            name: sha256_file(package / name)
            for name in (
                "analysis/refinement_report.py",
                "analysis/gate_report.py",
                "analysis/radar_report.py",
                "analysis/context_ablation.py",
            )
        },
        "evaluation": evaluation_sources(),
    }


def load_numeric_artifacts(root, summary):
    root = Path(root)
    n = summary["sample_count"]
    result = []
    for name, shape, dtype in (
        ("metrics.npz", (6, n, 11, 3), np.float64),
        ("weights.npz", (6, n, 11), np.float32),
    ):
        if sha256_file(root / name) != summary["sha256"][name]:
            raise ValueError("numeric artifact checksum mismatch")
        with np.load(root / name, allow_pickle=False) as archive:
            if set(archive.files) != set(STRATEGIES) or len(archive.files) != len(STRATEGIES):
                raise ValueError("incomplete numeric strategies")
            values = {key: archive[key] for key in STRATEGIES}
        for array in values.values():
            if (
                array.shape != shape
                or array.dtype != dtype
                or not np.isfinite(array).all()
                or np.any(array < 0)
                or (name == "weights.npz" and np.any(array > 1))
            ):
                raise ValueError("numeric shape/dtype/range mismatch")
        result.append(values)
    return tuple(result)


def paired_comparisons(stats, champions, replicates=2000):
    if (
        set(stats.seed) != set(SEEDS)
        or stats.duplicated(["seed", "condition", "strategy", "scene_name"]).any()
    ):
        raise ValueError("complete unique scene seeds required")
    contrasts = [(name, "p6_learned", name + "-p6_learned") for name in NEW_STRATEGIES]
    contrasts += [
        (champions[role], "sliding", role + "(" + champions[role] + ")-sliding")
        for role in ("accuracy", "stable")
    ]
    rows = []
    for condition, part in stats.groupby("condition"):
        for name, control, label in contrasts:
            a = part[part.strategy == name].set_index(["seed", "scene_name"]).sort_index()
            b = part[part.strategy == control].set_index(["seed", "scene_name"]).sort_index()
            if (
                not a.index.equals(b.index)
                or not len(a)
                or not np.array_equal(a.update_count, b.update_count)
                or not (a.reset_index().groupby("scene_name").seed.nunique() == 3).all()
                or not (a.update_count.groupby("scene_name").nunique() == 1).all()
            ):
                raise ValueError("paired scene seed/count mismatch")
            count = a.update_count.groupby("scene_name").mean()
            for metric in METRICS:
                delta = (a[metric + "_sum"] - b[metric + "_sum"]).groupby("scene_name").mean()
                rows.append(
                    {
                        "condition": condition,
                        "contrast": label,
                        "metric": metric,
                        **paired_scene_interval(
                            delta.to_numpy(), count.to_numpy(), replicates=replicates, seed=0
                        ),
                    }
                )
    return rows


def audit_latency(root, data):
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
        or value["scope"] != SCOPE
        or value["warmup"] != 16
        or value["key_count"] != min(256, data.size)
        or value["inference_batch_size"] != 1
        or value["key_selection_seed"] != 2027
        or value["excluded"] != ["map_query", "P4_candidate_forward", "cold_cache_load"]
        or set(value["strategies"]) != set(STRATEGIES)
    ):
        raise ValueError("incomplete refinement latency scope")
    for name, field in (
        ("latency_keys.csv", "keys_sha256"),
        ("latency_measurements.csv", "measurements_sha256"),
    ):
        if sha256_file(root / name) != value[field]:
            raise ValueError("refinement latency checksum mismatch")
    keys = pd.read_csv(root / "latency_keys.csv")
    measured = pd.read_csv(root / "latency_measurements.csv", float_precision="round_trip")
    indices = benchmark_keys(data.size)
    expected = (
        pd.concat(
            [key_frame(data, c, ["unused"] * data.n)[list(KEY_FIELDS)] for c in range(6)],
            ignore_index=True,
        )
        .iloc[indices]
        .copy()
    )
    expected.insert(0, "flat_index", indices)
    _equal_frame(keys, expected, ["flat_index"])
    if (
        set(measured) != {"strategy", "flat_index", "total_ms"}
        or measured.duplicated(["strategy", "flat_index"]).any()
        or set(measured.strategy) != set(STRATEGIES)
    ):
        raise ValueError("incomplete refinement latency rows")
    for name, part in measured.groupby("strategy"):
        if (
            not np.array_equal(part.flat_index, indices)
            or not np.isfinite(part.total_ms).all()
            or (part.total_ms < 0).any()
            or set(value["strategies"][name]) != {"p50", "p95"}
        ):
            raise ValueError("invalid refinement latency samples")
        if not np.allclose(
            [value["strategies"][name][k] for k in ("p50", "p95")],
            part.total_ms.quantile([0.5, 0.95]),
            rtol=1e-9,
            atol=1e-9,
        ):
            raise ValueError("refinement latency percentile mismatch")
    return value


def stability_outcome(frame, strategy):
    means = frame.groupby("strategy")[["ade", "jitter"]].mean()
    clean = frame[frame.condition == "clean"].groupby("strategy").ade.mean()
    reference = means.loc["p6_learned"]
    chosen = means.loc[strategy]
    within = bool(
        chosen.ade <= reference.ade * 1.01 and clean[strategy] <= clean["p6_learned"] * 1.01
    )
    improved = bool(chosen.jitter < reference.jitter - 1e-12)
    baseline = strategy in ("sliding", "ungated_map", "p6_fixed", "p6_learned")
    return {
        "strategy": strategy,
        "selected_baseline": baseline,
        "within_ade_tolerance": within,
        "jitter_improved": improved,
        "target_met": within and improved and not baseline,
        "ade_delta": float(chosen.ade - reference.ade),
        "jitter_delta": float(chosen.jitter - reference.jitter),
        "clean_ade_delta": float(clean[strategy] - clean["p6_learned"]),
    }


def audit_pair_forward(root, data, suite):
    root = Path(root)
    value = json.loads((root / "pair_forward.json").read_text())
    scope = "cached_inputs_gate_blend_optional_previous_forward_and_revision_no_backward_optimizer_or_labels_cuda_synchronized"
    expected_fields = {
        "batch_size",
        "iterations",
        "warmup",
        "key_selection_seed",
        "scope",
        "device",
        "keys_sha256",
        "measurements_sha256",
        "model_reference",
        "modes",
    }
    if (
        set(value) != expected_fields
        or value["scope"] != scope
        or value["batch_size"] != min(4096, data.size)
        or value["iterations"] != 64
        or value["warmup"] != 16
        or value["key_selection_seed"] != 2027
        or value["model_reference"] != suite["references"]["p6_learned"]
        or set(value["modes"]) != {"single", "paired"}
    ):
        raise ValueError("incomplete pair forward benchmark")
    for name, field in (
        ("pair_forward_keys.npy", "keys_sha256"),
        ("pair_forward_measurements.csv", "measurements_sha256"),
    ):
        if sha256_file(root / name) != value[field]:
            raise ValueError("pair forward checksum mismatch")
    keys = np.load(root / "pair_forward_keys.npy", allow_pickle=False)
    if keys.dtype != np.int64 or not np.array_equal(keys, benchmark_keys(data.size, 4096)):
        raise ValueError("pair forward fixed keys mismatch")
    frame = pd.read_csv(root / "pair_forward_measurements.csv", float_precision="round_trip")
    if (
        set(frame) != {"mode", "iteration", "total_ms"}
        or frame.duplicated(["mode", "iteration"]).any()
        or set(frame["mode"]) != {"single", "paired"}
        or not np.isfinite(frame.total_ms).all()
        or (frame.total_ms < 0).any()
    ):
        raise ValueError("pair forward rows mismatch")
    for name, part in frame.groupby("mode"):
        if (
            not np.array_equal(part.iteration, np.arange(64))
            or set(value["modes"][name]) != {"p50", "p95"}
            or not np.allclose(
                [value["modes"][name][q] for q in ("p50", "p95")],
                part.total_ms.quantile([0.5, 0.95]),
                atol=1e-9,
                rtol=1e-9,
            )
        ):
            raise ValueError("pair forward sample/percentile mismatch")
    return value


def audit_run(root, selection, device):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    p = summary["provenance"]
    seed = p["seed"]
    if (
        summary.get("schema_version") != 1
        or summary.get("failure_count") != 0
        or summary.get("strategies") != list(STRATEGIES)
        or summary.get("conditions") != list(CONDITIONS)
        or summary.get("evaluation_source_sha256") != evaluation_sources()
        or set(summary["sha256"]) != set(ARTIFACTS)
    ):
        raise ValueError("incomplete refinement evaluation source/schema")
    for name in ARTIFACTS:
        if sha256_file(root / name) != summary["sha256"][name]:
            raise ValueError("refinement artifact checksum mismatch")
    for name in ("candidate_manifest", "selection"):
        if sha256_file(Path(p[name]["path"])) != p[name]["sha256"]:
            raise ValueError("refinement provenance checksum mismatch")
    if (
        p.get("split") != "val"
        or seed not in SEEDS
        or load_selection(Path(p["selection"]["path"]).parent) != selection
    ):
        raise ValueError("refinement frozen selection mismatch")
    arrays, manifest = load_verified_gate_cache(
        Path(p["candidate_manifest"]["path"]).parent, "val", seed, device=device
    )
    if manifest["contract"] != p["candidate_contract"]:
        raise ValueError("refinement candidate provenance mismatch")
    data = RefinementDataset(arrays)
    if (
        summary["sample_count"] != data.n
        or summary["condition_count"] != 6
        or summary["updates_per_condition_strategy"] != data.n * 11
        or summary["row_count"] != data.size * 12
        or summary["key_shape"] != [6, data.n, 11]
        or summary["sample_keys_sha256"] != key_hash(arrays)
        or summary["metric_dtype"] != "float64"
        or summary["weight_dtype"] != "float32"
        or summary["key_order"] != "condition_then_bound_sample_then_update1to11"
    ):
        raise ValueError("refinement numeric key contract mismatch")
    suite = load_suite(
        p["training_root"],
        p["p6_training_root"],
        manifest["contract"],
        data.scenes,
        "val",
        seed,
        device,
    )
    if (
        suite["references"] != p["suite"]
        or p["suite"] != selection["references"][str(seed)]["suite"]
    ):
        raise ValueError("refinement suite provenance mismatch")
    stored, stored_weights = load_numeric_artifacts(root, summary)
    rebuilt, weights, _ = evaluate_suite(data, suite, selection["policy"], device=device)
    for name in STRATEGIES:
        if not np.allclose(stored[name], rebuilt[name], rtol=1e-6, atol=1e-6) or not np.allclose(
            stored_weights[name], weights[name], rtol=1e-6, atol=1e-7
        ):
            raise ValueError("refinement numeric candidate reconstruction mismatch")
    tables = derive_tables(data, stored, stored_weights, seed)
    for filename, expected in tables.items():
        keys = (
            ["seed", "strategy", "condition"]
            if filename != "recovery.csv"
            else ["seed", "strategy", "instance_token", "sample_token"]
        )
        if filename == "scene_stats.csv":
            keys += ["scene_name"]
        elif filename == "by_update.csv":
            keys += ["update_step"]
        elif filename == "by_horizon.csv":
            keys += ["remaining_steps"]
        elif filename == "by_maneuver.csv":
            keys += ["maneuver"]
        elif filename == "quantiles.csv":
            keys += ["metric"]
        _equal_frame(pd.read_csv(root / filename, float_precision="round_trip"), expected, keys)
    p6root = Path(p["p6_training_root"]).parent / "evaluation" / f"seed-{seed}"
    old = json.loads((p6root / "summary.json").read_text())
    if old["provenance"]["candidate_manifest"] != p["candidate_manifest"] or old["provenance"][
        "checkpoint"
    ] != {k: suite["references"]["p6_learned"][k] for k in ("path", "sha256")}:
        raise ValueError("P6 regression source mismatch")
    if sha256_file(p6root / "by_condition.csv") != old["sha256"]["by_condition.csv"]:
        raise ValueError("P6 regression table checksum mismatch")
    oldframe = pd.read_csv(p6root / "by_condition.csv", float_precision="round_trip")
    for name, reference in (
        ("sliding", "sliding"),
        ("ungated_map", "ungated_map"),
        ("p6_fixed", "fixed_gate"),
        ("p6_learned", "learned_gate"),
    ):
        a = tables["by_condition.csv"].query("strategy == @name").sort_values("condition")
        b = oldframe.query("strategy == @reference").sort_values("condition")
        if not np.allclose(a[list(METRICS)], b[list(METRICS)], rtol=1e-9, atol=1e-9):
            raise ValueError("P6 numeric regression mismatch")
    latency = audit_latency(root, data)
    latency["pair_forward"] = audit_pair_forward(root, data, suite)
    print(json.dumps({"audited_seed": seed, "rows": summary["row_count"]}), flush=True)
    return summary, tables, latency


def audit_selection(selection, training_root, p6_root, device):
    rows = []
    sweeps = []
    for seed in SEEDS:
        reference = selection["references"][str(seed)]
        path = Path(reference["candidate_manifest"]["path"])
        arrays, manifest = load_verified_gate_cache(path.parent, "train_val", seed, device=device)
        data = RefinementDataset(arrays)
        suite = load_suite(
            training_root, p6_root, manifest["contract"], data.scenes, "train_val", seed, device
        )
        if suite["references"] != reference["suite"]:
            raise ValueError("selection replay suite mismatch")
        train_path = Path(suite["parents"]["candidate_manifests"]["train"]["path"])
        train_arrays, _ = load_verified_gate_cache(train_path.parent, "train", seed, device=device)
        calibration = json.loads(Path(suite["references"]["calibration"]["path"]).read_text())
        for split, dataset in (("train", RefinementDataset(train_arrays)), ("train_val", data)):
            actual = fit_calibration(dataset, device=device)
            recorded = calibration[split]
            for key in actual:
                if key == "candidate":
                    if actual[key] != recorded[key]:
                        raise ValueError("calibration candidate source mismatch")
                elif not np.allclose(
                    np.asarray(actual[key]), np.asarray(recorded[key]), rtol=1e-9, atol=1e-9
                ):
                    raise ValueError("calibration fitting reconstruction mismatch")
        for name in ("p6_learned", "consistent_002", "consistent_010"):
            for tau in TAUS:
                policy = {"short_tau": tau, "consistent_model": name, "consistent_tau": tau}
                values, _, _ = measure_strategy(
                    data,
                    suite,
                    policy,
                    "short_gate" if name == "p6_learned" else "consistent_short",
                    device=device,
                )
                sweeps.append(
                    {
                        "seed": seed,
                        "model": name,
                        "tau": tau,
                        "ade": float(values[..., 0].mean()),
                        "jitter": float(values[..., 2].mean()),
                    }
                )
        results, _, _ = evaluate_suite(data, suite, selection["policy"], device=device)
        rows.append(condition_table(results, seed))
        print(json.dumps({"audited_selection_seed": seed}), flush=True)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(sweeps)


def report_refinement(run_dirs, selection_root, output_root, device):
    if len(run_dirs) != 3:
        raise ValueError("three evaluation seed runs required")
    selection = load_selection(selection_root)
    first = json.loads((Path(run_dirs[0]) / "summary.json").read_text())["provenance"]
    selected, sweep = audit_selection(
        selection, first["training_root"], first["p6_training_root"], device
    )
    _equal_frame(
        pd.read_csv(
            Path(selection_root) / "selection_conditions.csv", float_precision="round_trip"
        ),
        selected,
        ["seed", "strategy", "condition"],
    )
    _equal_frame(
        pd.read_csv(Path(selection_root) / "shrink_sweep.csv", float_precision="round_trip"),
        sweep,
        ["seed", "model", "tau"],
    )
    audited = [audit_run(root, selection, device) for root in run_dirs]
    if sorted(s[0]["provenance"]["seed"] for s in audited) != list(SEEDS):
        raise ValueError("fixed evaluation seed mismatch")
    reference = audited[0][0]
    if any(
        s[0]["sample_keys_sha256"] != reference["sample_keys_sha256"]
        or s[0]["provenance"]["candidate_contract"]["dataset_identity"]
        != reference["provenance"]["candidate_contract"]["dataset_identity"]
        for s in audited
    ):
        raise ValueError("cross-seed dataset/key mismatch")
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    frame = pd.concat([v[1]["by_condition.csv"] for v in audited], ignore_index=True)
    average = frame.groupby(["condition", "strategy"], as_index=False).agg(
        **{m + "_mean": (m, "mean") for m in METRICS}, **{m + "_std": (m, "std") for m in METRICS}
    )
    average.to_csv(root / "condition_seed_mean_std.csv", index=False)
    stats = pd.concat([v[1]["scene_stats.csv"] for v in audited], ignore_index=True)
    intervals = paired_comparisons(stats, selection["champions"])
    (root / "paired_scene_intervals.json").write_text(json.dumps(intervals, indent=2) + "\n")
    for filename in (
        "by_update.csv",
        "by_horizon.csv",
        "by_maneuver.csv",
        "quantiles.csv",
        "gate_distribution.csv",
    ):
        pd.concat([v[1][filename] for v in audited], ignore_index=True).to_csv(
            root / filename, index=False
        )
    recovery = pd.concat([v[1]["recovery.csv"] for v in audited], ignore_index=True)
    recovery.groupby(["seed", "strategy"], as_index=False).agg(
        sample_count=("recovered", "size"),
        recovered_fraction=("recovered", "mean"),
        conditional_recovery_seconds=("recovery_seconds", "mean"),
    ).to_csv(root / "recovery_summary.csv", index=False)
    means = frame.groupby("strategy")[["ade", "fde", "jitter"]].mean()
    training_timing = []
    for seed in SEEDS:
        for name in ("original_gate", "finetune_0", "consistent_002", "consistent_010"):
            checkpoint = Path(selection["references"][str(seed)]["suite"][name]["path"])
            log = json.loads((checkpoint.parent / "training.json").read_text())
            seconds = np.array([row["seconds"] for row in log["epochs"]], np.float64)
            if not np.isfinite(seconds).all() or np.any(seconds < 0):
                raise ValueError("invalid training epoch timings")
            training_timing.append(
                {
                    "seed": seed,
                    "strategy": name,
                    "completed_epochs": len(seconds),
                    "sum_epoch_seconds": float(seconds.sum()),
                    "mean_epoch_seconds": float(seconds.mean()),
                    "scope": "training_forward_backward_optimizer_plus_full_selection_metrics_not_pure_pair_forward",
                    "training_log_sha256": sha256_file(checkpoint.parent / "training.json"),
                }
            )
    result = {
        "stability_assessment": {
            "train_val": stability_outcome(selected, selection["champions"]["stable"]),
            "val_fixed_choice": stability_outcome(frame, selection["champions"]["stable"]),
        },
        "training_epoch_timings": training_timing,
        "seeds": list(SEEDS),
        "sample_count": reference["sample_count"],
        "total_rows": sum(v[0]["row_count"] for v in audited),
        "selection": selection,
        "overall_condition_seed_means": means.to_dict(orient="index"),
        "bootstrap": {
            "replicates": 2000,
            "seed": 0,
            "unit": "scene_after_fixed_three_seed_average",
        },
        "latency_scope": SCOPE,
        "latencies": [v[2] for v in audited],
        "quantile_scope": "per_update_error_quantiles_per_seed_not_trajectory_tail_or_pooled_independent_seeds",
        "report_sources": report_sources(),
        "runs": [
            {
                "path": str(Path(path).resolve()),
                "summary_sha256": sha256_file(Path(path) / "summary.json"),
            }
            for path in run_dirs
        ],
    }
    (root / "report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("evaluation-root", "selection-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    report_refinement(
        [args.evaluation_root / f"seed-{s}" for s in SEEDS],
        args.selection_root,
        args.output_root,
        choose_device(args.device),
    )


if __name__ == "__main__":
    main()

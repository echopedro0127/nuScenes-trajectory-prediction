"""Production replay, fixed-selection outcomes and scene-paired P6.3 reporting."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.analysis.context_ablation import paired_scene_interval
from trajectory_prediction.context.gate_preparation import load_verified_gate_cache
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.horizon_artifacts import BASELINES, STRATEGIES, verify_run
from trajectory_prediction.online.horizon_evaluation import (
    CONDITIONS,
    METRICS,
    SCOPE,
    SEEDS,
    THRESHOLDS,
    choose_horizon,
    derive_tables,
    evaluate_models,
    evaluation_sources,
    load_context,
    load_horizon_selection,
    parse_args,
    selection_rows,
)
from trajectory_prediction.online.horizon_features import HorizonDataset
from trajectory_prediction.online.horizon_gate import VARIANTS
from trajectory_prediction.online.horizon_streaming import new_state, stream_inputs
from trajectory_prediction.online.horizon_training import check_initialization, configure_runtime
from trajectory_prediction.online.refinement_training import completed_payload
from trajectory_prediction.online.scaled_evaluation import benchmark_keys
from trajectory_prediction.utils.provenance import sha256_file


def report_sources():
    return {"report": sha256_file(Path(__file__)), "bootstrap": sha256_file(Path(__file__).with_name("context_ablation.py")),
        "evaluation": evaluation_sources()}


def paired_intervals(stats, decision):
    if set(stats.seed) != set(SEEDS) or stats.duplicated(["seed", "condition", "strategy", "scene_name"]).any():
        raise ValueError("complete paired scene seeds required")
    contrasts = [(v, "B", v+"-B") for v in VARIANTS if v != "B"]
    contrasts += [(v, "consistent_short", v+"-consistent_short") for v in VARIANTS]
    contrasts += [(decision[role], "sliding", role+"("+decision[role]+")-sliding") for role in ("accuracy", "stable")]
    contrasts += [(decision["stable"], "consistent_010", "stable("+decision["stable"]+")-consistent_010")]
    if set(stats.condition) != set(CONDITIONS):
        raise ValueError("complete paired conditions required")
    fields = ["seed", "strategy", "scene_name"]
    if not (stats.groupby(fields).condition.nunique() == len(CONDITIONS)).all():
        raise ValueError("complete paired scene conditions required")
    overall = stats.groupby(fields, as_index=False)[["update_count", *[m+"_sum" for m in METRICS]]].sum()
    overall["condition"] = "overall"
    combined = pd.concat((stats, overall), ignore_index=True)
    result = []
    for condition, part in combined.groupby("condition"):
        for name, control, label in contrasts:
            a = part[part.strategy == name].set_index(["seed", "scene_name"]).sort_index()
            b = part[part.strategy == control].set_index(["seed", "scene_name"]).sort_index()
            if (not a.index.equals(b.index) or not len(a) or not np.array_equal(a.update_count, b.update_count)
                or not (a.reset_index().groupby("scene_name").seed.nunique() == 3).all()
                or not (a.update_count.groupby("scene_name").nunique() == 1).all()):
                raise ValueError("paired horizon scene/seed/count mismatch")
            count = a.update_count.groupby("scene_name").mean()
            for metric in METRICS:
                delta = (a[metric+"_sum"]-b[metric+"_sum"]).groupby("scene_name").mean()
                result.append({"condition": condition, "contrast": label, "metric": metric,
                    **paired_scene_interval(delta.to_numpy(), count.to_numpy(), replicates=2000, seed=0)})
    return result


def factorial_effects(frame):
    rows = []
    for scope in ("overall", "clean"):
        part = frame if scope == "overall" else frame[frame.condition == "clean"]
        per_seed = part[part.strategy.isin(VARIANTS)].groupby(["seed", "strategy"])[list(METRICS)].mean()
        for factor in "DRH":
            on = [v for v in VARIANTS if factor in v]
            off = [v for v in VARIANTS if factor not in v]
            for metric in METRICS:
                delta = per_seed.loc[(slice(None), on), metric].groupby("seed").mean()-per_seed.loc[(slice(None), off), metric].groupby("seed").mean()
                rows.append({"scope": scope, "factor": factor, "metric": metric, "delta": float(delta.mean()), "seed_std": float(delta.std(ddof=1))})
    return pd.DataFrame(rows)


def fixed_outcomes(frame, decision):
    means = frame.groupby("strategy")[[*METRICS, "fde_p95"]].mean()
    clean = frame[frame.condition == "clean"].groupby("strategy")[[*METRICS, "fde_p95"]].mean()
    ref = means.loc["consistent_short"]
    result = {}
    for role in ("accuracy", "stable"):
        name = decision[role]
        value = means.loc[name]
        common = value.fde <= THRESHOLDS["fde_ratio"]*ref.fde and clean.loc[name, "fde_p95"] <= THRESHOLDS["clean_fde_p95_ratio"]*clean.loc["consistent_short", "fde_p95"]
        constraints = common and (value.jitter <= THRESHOLDS["accuracy_jitter_ratio"]*ref.jitter if role == "accuracy" else value.ade <= THRESHOLDS["stable_ade_ratio"]*ref.ade and clean.loc[name, "ade"] <= THRESHOLDS["stable_ade_ratio"]*clean.loc["consistent_short", "ade"])
        result[role] = {"strategy": name, "is_new": name in VARIANTS, "constraints_met": bool(constraints),
            "ade_delta_vs_reference": float(value.ade-ref.ade), "fde_delta_vs_reference": float(value.fde-ref.fde),
            "jitter_delta_vs_reference": float(value.jitter-ref.jitter), "clean_fde_p95_delta_vs_reference": float(clean.loc[name, "fde_p95"]-clean.loc["consistent_short", "fde_p95"]),
            "research_target_met": bool(name in VARIANTS and constraints and (value.ade <= .99*ref.ade and value.fde <= ref.fde if role == "accuracy" else value.jitter <= .97*means.loc["consistent_010", "jitter"]))}
    return result


def audit_latency(root, data, device):
    root = Path(root)
    meta = json.loads((root/"latency.json").read_text())
    fields = {"key_count", "warmup", "batch_size", "key_selection_seed", "new_strategy_scope", "baseline_scope", "excluded",
        "device", "device_name", "torch_version", "keys_sha256", "measurements_sha256", "strategies"}
    indices = benchmark_keys(data.size, 256)
    if (set(meta) != fields or meta["key_count"] != len(indices) or meta["warmup"] != 16 or meta["batch_size"] != 1
        or meta["key_selection_seed"] != 2027 or meta["new_strategy_scope"] != SCOPE
        or meta["baseline_scope"] != "cached_read_quality_gate_cap_blend_return_cuda_synchronized"
        or meta["excluded"] != ["prefix_state_reconstruction", "map_query", "P4_forward", "cold_cache_load"]
        or set(meta["strategies"]) != set(STRATEGIES) or meta["device"] != str(device)):
        raise ValueError("horizon latency profile mismatch")
    for filename, field in (("latency_keys.csv", "keys_sha256"), ("latency_measurements.csv", "measurements_sha256")):
        if sha256_file(root/filename) != meta[field]:
            raise ValueError("horizon latency checksum mismatch")
    keys = pd.read_csv(root/"latency_keys.csv")
    frame = pd.read_csv(root/"latency_measurements.csv", float_precision="round_trip")
    if (set(keys) != {"key_index"} or not np.array_equal(keys.key_index, indices)
        or set(frame) != {"strategy", "key_index", "milliseconds"} or frame.duplicated(["strategy", "key_index"]).any()
        or set(frame.strategy) != set(STRATEGIES) or len(frame) != len(indices)*len(STRATEGIES)):
        raise ValueError("horizon latency keys/rows mismatch")
    for name, part in frame.groupby("strategy"):
        expected = part.milliseconds.quantile([.5, .95]).to_numpy()
        if (not np.array_equal(part.key_index, indices) or not np.isfinite(part.milliseconds).all() or (part.milliseconds < 0).any()
            or set(meta["strategies"][name]) != {"p50", "p95"}
            or not np.allclose([meta["strategies"][name][k] for k in ("p50", "p95")], expected, atol=1e-9, rtol=1e-9)):
            raise ValueError("horizon latency percentile/raw mismatch")
    # Independently execute real prefix updates and compare current model inputs.
    for index in indices:
        c, i, k = (int(x[0]) for x in data.indices(np.array([index])))
        state = new_state(data, c, i)
        for step in range(k+1):
            actual = stream_inputs(data, c, i, step, state, torch.device("cpu"))
        expected = data.inputs(np.array([index]), torch.device("cpu"))
        torch.testing.assert_close(actual["features"], expected["features"], atol=1e-6, rtol=1e-6)
    return meta


def audit_training(args, seed, device):
    cache = args.cache_root/"train"/f"seed-{seed}"
    arrays, manifest = load_verified_gate_cache(cache, "train", seed, device=device)
    data = HorizonDataset(arrays)
    fingerprint = data.fingerprint()
    logs = []
    for variant in VARIANTS:
        root = args.training_root/f"seed-{seed}"/variant
        payload = completed_payload(root)
        meta = payload["metadata"]
        if (meta["train_content"] != fingerprint or meta["candidate_contracts"]["train"] != manifest["contract"]
            or meta["cublas_workspace_config"] != ":4096:8"):
            raise ValueError("horizon actual training feature/contract mismatch")
        record = json.loads((root/"training.json").read_text())
        seconds = np.array([row["seconds"] for row in record["epochs"]])
        if not np.isfinite(seconds).all() or np.any(seconds < 0):
            raise ValueError("invalid horizon epoch times")
        logs.append({"seed": seed, "variant": variant, "best_epoch": meta["best_epoch"], "completed_epochs": meta["completed_epochs"],
            "selection_ade": meta["selection_ade"], "epoch_seconds_sum": float(seconds.sum()), "epoch_seconds_mean": float(seconds.mean()),
            "scope": "training_forward_backward_optimizer_plus_complete_selection_metrics_not_online_latency",
            "checkpoint_sha256": sha256_file(root/"best.pt"), "training_log_sha256": sha256_file(root/"training.json")})
    return logs


def check_reference_parity(args, split, seed, table, manifest):
    if split == "train_val":
        root = args.reference_selection_root
        path = root/"selection_conditions.csv"
        old = pd.read_csv(path, float_precision="round_trip")
        old = old[old.seed == seed]
    else:
        root = args.reference_training_root.parent/"evaluation"/f"seed-{seed}"
        summary = json.loads((root/"summary.json").read_text())
        if sha256_file(root/"by_condition.csv") != summary["sha256"]["by_condition.csv"]:
            raise ValueError("P6.1 historical baseline checksum mismatch")
        provenance = summary["provenance"]
        if provenance["candidate_contract"] != manifest:
            raise ValueError("P6.1 historical val candidate source mismatch")
        old = pd.read_csv(root/"by_condition.csv", float_precision="round_trip")
    fields = ["seed", "strategy", "condition"]
    a = table[table.strategy.isin(BASELINES)].sort_values(fields).reset_index(drop=True)
    b = old[old.strategy.isin(BASELINES)].sort_values(fields).reset_index(drop=True)
    if not a[fields].equals(b[fields]) or not np.allclose(a[list(METRICS)], b[list(METRICS)], atol=1e-9, rtol=0):
        raise ValueError("P6.1 frozen baseline parity mismatch")


def audit_one(args, split, seed, selection, device):
    root = args.output_root/split/f"seed-{seed}"
    data, suite, policy, models, metadata = load_context(args, split, seed, device)
    if split == "val":
        metadata["frozen_selection"] = {"path": str((args.selection_root/"selection.json").resolve()), "sha256": sha256_file(args.selection_root/"selection.json")}
    result, weights = evaluate_models(data, suite, policy, models, device)
    tables = derive_tables(data, result, weights, seed)
    saved = verify_run(root, result, weights, tables, metadata)
    latency = audit_latency(root, data, device)
    check_reference_parity(args, split, seed, tables["by_condition.csv"], metadata["candidate_contract"])
    if split == "train_val":
        initial = completed_payload(args.training_root/f"seed-{seed}"/"B")["metadata"]
        from trajectory_prediction.online.refinement_training import load_initial
        parity = check_initialization(data, load_initial(initial["p6_checkpoint"]["path"], initial["candidate_contracts"]), device)
        for variant in VARIANTS:
            meta = completed_payload(args.training_root/f"seed-{seed}"/variant)["metadata"]
            if meta["initial_full_metrics"] != parity[variant]:
                raise ValueError("horizon full initial metrics replay mismatch")
    print(json.dumps({"audited_horizon_seed": seed, "split": split, "rows": saved["row_count"]}), flush=True)
    return saved, tables, latency


def write_report(args, selection, audited, logs):
    root = args.output_root.parent/"report"
    root.mkdir(parents=True, exist_ok=True)
    summary = {"schema_version": 1, "selection": selection, "sources": report_sources(), "splits": {},
        "training_epoch_scope": "forward_backward_optimizer_plus_complete_selection_metrics", "training_runs": logs,
        "bootstrap": {"replicates": 2000, "seed": 0, "unit": "scene_after_fixed_three_seed_average"}}
    filenames = []
    for split, entries in audited.items():
        frame = pd.concat([x[1]["by_condition.csv"] for x in entries], ignore_index=True)
        matrix = pd.concat([selection_rows(x[1]) for x in entries], ignore_index=True)
        if split == "train_val":
            chosen = choose_horizon(matrix)
            if chosen != selection["decision"]:
                raise ValueError("horizon train_val selection replay mismatch")
            expected = pd.read_csv(args.selection_root/"selection_conditions.csv", float_precision="round_trip")
            pd.testing.assert_frame_equal(matrix, expected, check_dtype=False, atol=1e-10, rtol=1e-9)
        stats = pd.concat([x[1]["scene_stats.csv"] for x in entries], ignore_index=True)
        intervals = paired_intervals(stats, selection["decision"])
        prefix = split+"_"
        interval_name = prefix+"paired_scene_intervals.json"
        (root/interval_name).write_text(json.dumps(intervals, indent=2)+"\n")
        filenames.append(interval_name)
        mean_std = frame.groupby(["strategy", "condition"], as_index=False).agg(**{m+"_mean": (m, "mean") for m in METRICS}, **{m+"_std": (m, "std") for m in METRICS})
        effects = factorial_effects(frame)
        for name, table in ((prefix+"condition_seed_mean_std.csv", mean_std), (prefix+"factorial_effects.csv", effects)):
            table.to_csv(root/name, index=False)
            filenames.append(name)
        for name in ("by_update.csv", "by_horizon.csv", "by_maneuver.csv", "quantiles.csv", "gate_distribution.csv", "recovery.csv"):
            pd.concat([x[1][name] for x in entries], ignore_index=True).to_csv(root/(prefix+name), index=False)
            filenames.append(prefix+name)
        summary["splits"][split] = {"seeds": list(SEEDS), "sample_count": entries[0][0]["metadata"]["sample_count"],
            "scene_count": entries[0][0]["metadata"]["scene_count"], "row_count": sum(x[0]["row_count"] for x in entries),
            "condition_strategy_seed_means": frame.groupby("strategy")[list(METRICS)].mean().to_dict(orient="index"),
            "fixed_choice_outcomes": fixed_outcomes(matrix, selection["decision"]), "interval_count": len(intervals),
            "latencies": [x[2] for x in entries], "runs": [{"path": str((args.output_root/split/f"seed-{seed}"/"summary.json").resolve()),
                "sha256": sha256_file(args.output_root/split/f"seed-{seed}"/"summary.json")} for seed in SEEDS]}
    summary["val_run"] = "val" in audited
    summary["sha256"] = {name: sha256_file(root/name) for name in filenames}
    (root/"report.json").write_text(json.dumps(summary, indent=2)+"\n")
    return summary


def main(argv=None):
    args = parse_args(argv)
    configure_runtime()
    torch.set_num_threads(2)
    device = choose_device(args.device)
    selection = load_horizon_selection(args.selection_root)
    source = report_sources()
    audited, logs = {"train_val": []}, []
    if selection["decision"]["advance"]:
        audited["val"] = []
    for seed in SEEDS:
        logs.extend(audit_training(args, seed, device))
        audited["train_val"].append(audit_one(args, "train_val", seed, selection, device))
        if "val" in audited:
            audited["val"].append(audit_one(args, "val", seed, selection, device))
    for entries in audited.values():
        reference = entries[0][0]["metadata"]
        if any(x[0]["metadata"]["sample_keys_sha256"] != reference["sample_keys_sha256"]
            or x[0]["metadata"]["candidate_contract"]["dataset_identity"] != reference["candidate_contract"]["dataset_identity"] for x in entries):
            raise ValueError("horizon cross-seed dataset/key mismatch")
    if report_sources() != source:
        raise ValueError("horizon report sources changed")
    result = write_report(args, selection, audited, logs)
    print(json.dumps({"horizon_report_complete": True, "val_run": result["val_run"]}), flush=True)


if __name__ == "__main__":
    main()

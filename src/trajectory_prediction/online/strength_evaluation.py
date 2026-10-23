"""Frozen P6.5 matched consistency strength evaluation, selection constraints and conditional val."""

import argparse
import copy
import json
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.analysis.maneuvers import describe_motion, tag_maneuvers
from trajectory_prediction.context.gate_preparation import load_verified_gate_cache
from trajectory_prediction.online.consistency_evaluation import (
    evaluate_models as parent_evaluate_models,
)
from trajectory_prediction.online.consistency_evaluation import (
    evaluation_sources as parent_evaluation_sources,
)
from trajectory_prediction.online.consistency_evaluation import load_context as parent_load_context
from trajectory_prediction.online.context_corrector import _sync
from trajectory_prediction.online.context_evaluator import DEFAULT_THRESHOLDS
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.gate_evaluation import (
    aggregate_rows,
    key_frame,
    mean_rows,
    recovery_table,
)
from trajectory_prediction.online.horizon_evaluation import load_horizon_selection
from trajectory_prediction.online.horizon_gate import VARIANTS, blend_horizon, measure_horizon
from trajectory_prediction.online.horizon_streaming import new_state, stream_inputs
from trajectory_prediction.online.innovation_consistency import ARMS as OLD_ARMS
from trajectory_prediction.online.innovation_consistency import InnovationDataset
from trajectory_prediction.online.refinement_evaluation import (
    SEEDS,
)
from trajectory_prediction.online.refinement_training import RefinementDataset
from trajectory_prediction.online.scaled_evaluation import benchmark_keys
from trajectory_prediction.online.strength_artifacts import (
    HISTORICAL,
    STRATEGIES,
    publish_run,
)
from trajectory_prediction.online.strength_consistency import ARMS
from trajectory_prediction.online.strength_training import (
    configure_runtime,
    load_strength,
    training_sources,
)
from trajectory_prediction.utils.provenance import sha256_file

METRICS = ("ade", "fde", "jitter")
CONDITIONS = ("clean", "delay_500ms", "delay_1000ms", "dropout", "missing", "stale")
THRESHOLDS = {"ade_ratio":1.002, "fde_ratio":1.005, "clean_fde_p95_ratio":1.01,
    "advance_jitter_fraction":.005, "target_jitter_fraction":.01}
SCOPE = "candidate_cache_read_state_update_direction_feedback_lead_features_gate_blend_return_cuda_synchronized"


def evaluation_sources():
    package = Path(__file__).parents[1]
    return {"execution": {name:sha256_file(package/name) for name in (
        "online/strength_artifacts.py", "online/strength_evaluation.py")},
        "training":training_sources(), "parent_evaluation":parent_evaluation_sources()}


def choose_strength(frame):
    expected={(s,c,n) for s in SEEDS for c in CONDITIONS for n in STRATEGIES}
    fields=["ade","fde","jitter","fde_p95"]
    if (len(frame)!=len(expected) or set(frame[["seed","condition","strategy"]].itertuples(index=False,name=None))!=expected
        or frame.duplicated(["seed","condition","strategy"]).any()
        or not np.isfinite(frame[fields]).all().all() or (frame[fields]<0).any().any()):
        raise ValueError("complete finite unique consistency experiment matrix required")
    means=frame.groupby("strategy")[fields].mean()
    clean=frame[frame.condition.eq("clean")].groupby("strategy")[fields].mean()
    ref=means.loc["RH"]; candidates=means.loc[["RH","revision_fixed",*ARMS]]
    eligible=(candidates.ade<=THRESHOLDS["ade_ratio"]*ref.ade)&(candidates.fde<=THRESHOLDS["fde_ratio"]*ref.fde)
    eligible &= clean.loc[candidates.index,"ade"]<=THRESHOLDS["ade_ratio"]*clean.loc["RH","ade"]
    eligible &= clean.loc[candidates.index,"fde_p95"]<=THRESHOLDS["clean_fde_p95_ratio"]*clean.loc["RH","fde_p95"]
    chosen=str(candidates[eligible].reset_index().sort_values(["jitter","ade","strategy"]).iloc[0].strategy)
    advance=chosen in ARMS and means.loc[chosen,"jitter"]<means.loc["revision_fixed","jitter"] and means.loc[chosen,"jitter"]<means.loc["revision_0","jitter"] and means.loc[chosen,"jitter"]<=(1-THRESHOLDS["advance_jitter_fraction"])*ref.jitter
    return {"balanced":chosen,"advance":bool(advance),"reference":"RH","thresholds":THRESHOLDS}


def selection_rows(tables):
    frame = tables["by_condition.csv"][["seed", "strategy", "condition", *METRICS]].copy()
    tail = tables["quantiles.csv"].query("metric == 'fde'")[["seed", "strategy", "condition", "p95"]].rename(columns={"p95": "fde_p95"})
    return frame.merge(tail, on=["seed", "strategy", "condition"], validate="one_to_one")


def evaluate_models(data, suite, policy, models, device):
    results, weights=parent_evaluate_models(data,suite,policy,models,device)
    for arm in ARMS:
        values,w=measure_horizon(data,models[arm],device=device)
        results[arm],weights[arm]=values,w
        print(json.dumps({"consistency_evaluated":arm,"ade":float(values[...,0].mean()),
            "fde":float(values[...,1].mean()),"jitter":float(values[...,2].mean())}),flush=True)
    np.testing.assert_allclose(results["revision_0"],results["RH"],atol=1e-9,rtol=0)
    np.testing.assert_allclose(weights["revision_0"],weights["RH"],atol=1e-9,rtol=0)
    return results,weights


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
            mask = np.arange(12)[None] >= np.arange(1, 12)[:, None]
            frame["gate_weight"] = ((weights[strategy][c]*mask[None]).sum(-1, dtype=np.float64)/mask.sum(-1)[None]).reshape(-1)
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
            g = pd.Series(weights[strategy][c][:, mask].reshape(-1).astype(np.float64))
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


def benchmark_strength(data, suite, policy, models, root, device):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    baseline_data = RefinementDataset(data.arrays)
    keys = benchmark_keys(data.size, 256)
    profile = []
    state_cache = {}
    for index in keys:
        c, i, k = (int(value[0]) for value in data.indices(np.array([index])))
        state = new_state(data, c, i)
        for previous in range(k):
            stream_inputs(data, c, i, previous, state, torch.device("cpu"))
        state_cache[int(index)] = state
    from trajectory_prediction.online.refinement_evaluation import predict_strategy
    for name in STRATEGIES:
        for serial, index in enumerate(np.concatenate((keys[:16], keys))):
            c, i, k = (int(value[0]) for value in data.indices(np.array([index])))
            state = copy.deepcopy(state_cache[int(index)])
            _sync(device)
            begun = time.perf_counter_ns()
            with torch.inference_mode():
                if name in (*VARIANTS, *OLD_ARMS, *ARMS):
                    values = stream_inputs(data, c, i, k, state, device)
                    weight = models[name](values["features"], values["valid"], values["mask"])
                    value = blend_horizon(values["cv"], values["map"], weight).cpu().numpy()
                else:
                    value, _ = predict_strategy(baseline_data, suite, policy, name, np.array([index]), device)
            _sync(device)
            elapsed = (time.perf_counter_ns()-begun)/1e6
            if not np.isfinite(value).all():
                raise ValueError("nonfinite online horizon benchmark forecast")
            if serial >= 16:
                profile.append({"strategy": name, "key_index": int(index), "milliseconds": elapsed})
    pd.DataFrame({"key_index": keys}).to_csv(root/"latency_keys.csv", index=False)
    frame = pd.DataFrame(profile)
    frame.to_csv(root/"latency_measurements.csv", index=False)
    summary = {name: {"p50": float(part.milliseconds.quantile(.5)), "p95": float(part.milliseconds.quantile(.95))} for name, part in frame.groupby("strategy")}
    meta = {"key_count": len(keys), "warmup": 16, "batch_size": 1, "key_selection_seed": 2027,
        "new_strategy_scope": SCOPE, "baseline_scope": "cached_read_quality_gate_cap_blend_return_cuda_synchronized",
        "excluded": ["prefix_state_reconstruction", "map_query", "P4_forward", "cold_cache_load"],
        "device": str(device), "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor(),
        "torch_version": str(torch.__version__), "keys_sha256": sha256_file(root/"latency_keys.csv"),
        "measurements_sha256": sha256_file(root/"latency_measurements.csv"), "strategies": summary}
    (root/"latency.json").write_text(json.dumps(meta, indent=2)+"\n")


def load_context(args, split, seed, device):
    frozen=load_horizon_selection(args.horizon_selection_root)
    if frozen["decision"]["accuracy"]!="RH":
        raise ValueError("consistency requires frozen RH reference")
    parent_args=copy.copy(args)
    parent_args.training_root=args.consistency_training_root
    parent_data,suite,policy,models,metadata=parent_load_context(parent_args,split,seed,device)
    data=InnovationDataset(parent_data.arrays)
    train_arrays,train_manifest=load_verified_gate_cache(args.cache_root/"train"/f"seed-{seed}","train",seed,device=device)
    trusted=InnovationDataset(train_arrays)
    train_mean,train_content=trusted.train_mean_weight,trusted.fingerprint()
    del trusted
    references=dict(metadata["models"])
    for arm in ARMS:
        path=args.training_root/f"seed-{seed}"/arm/"best.pt"
        model,meta=load_strength(path,metadata["candidate_contract"],data.scenes,device,split=split,arm=arm,
            train_mean_weight=train_mean,train_content=train_content)
        if (meta["candidate_contracts"]["train"]!=train_manifest["contract"]
            or (split=="train_val" and meta["selection_content"]!=data.fingerprint())):
            raise ValueError("consistency actual feature/parent content mismatch")
        models[arm]=model
        references[arm]={"path":str(path.resolve()),"sha256":sha256_file(path),"run_identity":meta["run_identity"]}
    metadata={**metadata,"models":references,"feature_fingerprint":data.fingerprint(),"sources":evaluation_sources(),
        "trusted_train_content":train_content,"trusted_train_mean_weight":train_mean,
        "horizon_selection":{"path":str((args.horizon_selection_root/"selection.json").resolve()),
            "sha256":sha256_file(args.horizon_selection_root/"selection.json")}}
    return data,suite,policy,models,metadata


def run_split(args, split, seed, device, phase_selection=None):
    if split == "val" and (phase_selection is None or not phase_selection["decision"]["advance"]):
        raise ValueError("val requires frozen successful advancement")
    source = evaluation_sources()
    data, suite, policy, models, meta = load_context(args, split, seed, device)
    if split == "val":
        meta["frozen_selection"] = {"path": str((args.selection_root/"selection.json").resolve()), "sha256": sha256_file(args.selection_root/"selection.json")}
    root = args.output_root/split/f"seed-{seed}"
    result, weights = evaluate_models(data, suite, policy, models, device)
    tables = derive_tables(data, result, weights, seed)
    benchmark_strength(data, suite, policy, models, root, device)
    if evaluation_sources() != source:
        raise ValueError("horizon evaluation executable changed")
    return publish_run(root, result, weights, tables, meta)


def fit_selection(args):
    rows, references = [], {}
    for seed in SEEDS:
        root = args.output_root/"train_val"/f"seed-{seed}"
        summary = json.loads((root/"summary.json").read_text())
        if summary["metadata"]["sources"] != evaluation_sources() or summary["metadata"]["split"] != "train_val" or summary["metadata"]["seed"] != seed:
            raise ValueError("horizon selection source/split mismatch")
        for name, checksum in summary["sha256"].items():
            if sha256_file(root/name) != checksum:
                raise ValueError("horizon selection artifact checksum mismatch")
        tables = {name: pd.read_csv(root/name, float_precision="round_trip") for name in ("by_condition.csv", "quantiles.csv")}
        rows.append(selection_rows(tables))
        references[str(seed)] = {"path": str((root/"summary.json").resolve()), "sha256": sha256_file(root/"summary.json")}
    frame = pd.concat(rows, ignore_index=True)
    previous=[]
    for seed in SEEDS:
        root=args.consistency_evaluation_root/"train_val"/f"seed-{seed}"
        parent=json.loads((root/"summary.json").read_text())
        if (parent["metadata"]["sources"]!=parent_evaluation_sources()
            or sha256_file(root/"by_condition.csv")!=parent["sha256"]["by_condition.csv"]):
            raise ValueError("P6.4 baseline source/checksum changed")
        previous.append(pd.read_csv(root/"by_condition.csv",float_precision="round_trip"))
    old=pd.concat(previous,ignore_index=True)
    fields=["seed","strategy","condition"]
    a=frame[frame.strategy.isin(HISTORICAL)].sort_values(fields).reset_index(drop=True)
    b=old.sort_values(fields).reset_index(drop=True)
    if not a[fields].equals(b[fields]) or not np.allclose(a[list(METRICS)],b[list(METRICS)],atol=1e-9,rtol=0):
        raise ValueError("P6.4 frozen train_val baseline parity failed")
    args.selection_root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.selection_root/"selection_conditions.csv", index=False)
    value = {"schema_version": 1, "split": "train_val", "seeds": list(SEEDS), "decision": choose_strength(frame),
        "references": references, "sources": evaluation_sources(), "selection_conditions_sha256": sha256_file(args.selection_root/"selection_conditions.csv")}
    temp = args.selection_root/".selection.tmp"
    temp.write_text(json.dumps(value, indent=2)+"\n")
    temp.replace(args.selection_root/"selection.json")
    return value


def load_strength_selection(root):
    root = Path(root)
    value = json.loads((root/"selection.json").read_text())
    if (set(value) != {"schema_version", "split", "seeds", "decision", "references", "sources", "selection_conditions_sha256"}
        or value["schema_version"] != 1 or value["split"] != "train_val" or value["seeds"] != list(SEEDS)
        or value["sources"] != evaluation_sources() or sha256_file(root/"selection_conditions.csv") != value["selection_conditions_sha256"]
        or set(value["references"]) != {str(s) for s in SEEDS}):
        raise ValueError("horizon frozen selection source/schema mismatch")
    for reference in value["references"].values():
        if sha256_file(Path(reference["path"])) != reference["sha256"]:
            raise ValueError("horizon frozen selection run changed")
    frame = pd.read_csv(root/"selection_conditions.csv", float_precision="round_trip")
    if value["decision"] != choose_strength(frame):
        raise ValueError("horizon frozen selection decision mismatch")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache-root", "training-root", "p6-training-root", "reference-training-root", "reference-selection-root", "horizon-training-root", "horizon-selection-root", "horizon-evaluation-root", "consistency-training-root", "consistency-selection-root", "consistency-evaluation-root", "consistency-report-root", "selection-root", "output-root"):
        parser.add_argument("--"+name, required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configure_runtime()
    torch.set_num_threads(2)
    device = choose_device(args.device)
    if not (args.selection_root/"selection.json").is_file():
        for seed in SEEDS:
            run_split(args, "train_val", seed, device)
        fit_selection(args)
    selection = load_strength_selection(args.selection_root)
    print(json.dumps({"frozen_horizon_decision": selection["decision"]}), flush=True)
    if selection["decision"]["advance"]:
        for seed in SEEDS:
            run_split(args, "val", seed, device, selection)
    else:
        print(json.dumps({"val_not_run": "predefined_advancement_not_met"}), flush=True)


if __name__ == "__main__":
    main()

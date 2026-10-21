"""Train-val-only, frozen candidate ceilings and stratified feature dependence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.context.cache import key_hash
from trajectory_prediction.context.gate_preparation import load_verified_gate_cache
from trajectory_prediction.online.batched_evaluation import forecast_metrics
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.gate_refinement import RefinementDataset
from trajectory_prediction.online.quality_gate import CONDITIONS, blend_forecasts
from trajectory_prediction.online.refinement_evaluation import (
    SEEDS,
    evaluation_sources,
    load_selection,
    load_suite,
)
from trajectory_prediction.utils.provenance import sha256_file

FEATURE_NAMES = (
    "coverage", "age", "position_residual", "velocity_residual", "speed",
    "acceleration", "remaining_time", "disagreement",
)
FEATURE_GROUPS = {name: (i,) for i, name in enumerate(FEATURE_NAMES)}
FEATURE_GROUPS.update(timing_coverage=(0, 1, 6), motion=(2, 3, 4, 5), all=tuple(range(8)))
REPEATS = (0, 1, 2)
METRICS = ("ade", "fde", "jitter")
BASELINES = ("cv", "map_endpoint", "original_endpoint", "filtered_endpoint", "p6_learned", "original_gate")


def oracle_mix(cv, candidate, target, mask, valid, reference_weights):
    """Label-using diagnostic only: finite grid plus the actual reference weight."""
    n = len(cv)
    if (cv.shape != (n, 12, 2) or candidate.shape != cv.shape or target.shape != cv.shape
        or mask.shape != (n, 12) or mask.dtype != torch.bool
        or valid.shape != (n,) or valid.dtype != torch.bool
        or reference_weights.shape != (n,) or not mask.any(1).all()
        or len({x.device for x in (cv, candidate, target, mask, valid, reference_weights)}) != 1
        or not all(torch.isfinite(x).all() for x in (cv, candidate, target, reference_weights))
        or ((reference_weights < 0) | (reference_weights > 1)).any()):
        raise ValueError("invalid oracle inputs/mask")
    best = torch.full((n,), float("inf"), dtype=torch.float64, device=cv.device)
    chosen = torch.zeros(n, dtype=cv.dtype, device=cv.device)
    with torch.inference_mode():
        for index in range(22):
            weight = (torch.full_like(reference_weights, index / 20) if index < 21 else reference_weights) * valid
            forecast = blend_forecasts(cv, candidate, weight)
            distance = torch.linalg.vector_norm(forecast - target, dim=-1)
            ade = (distance * mask).double().sum(1) / mask.sum(1)
            better = ade < best
            best = torch.where(better, ade, best)
            chosen = torch.where(better, weight, chosen)
    return best, chosen


def permutation_donors(valid, ages, seed):
    if (valid.ndim != 3 or valid.shape[0] != 6 or valid.shape[2] != 11
        or ages.shape != valid.shape or valid.dtype != np.bool_
        or not np.isfinite(ages).all() or np.any(ages < 0)
        or type(seed) is not int or seed < 0):
        raise ValueError("invalid permutation strata")
    n = valid.shape[1]
    donors = np.broadcast_to(np.arange(n)[None, :, None], valid.shape).copy()
    rng = np.random.default_rng(seed)
    for c in range(6):
        for k in range(11):
            for age in np.unique(ages[c, valid[c, :, k], k]):
                rows = np.flatnonzero(valid[c, :, k] & (ages[c, :, k] == age))
                donors[c, rows, k] = rng.permutation(rows)
    return donors


def _prediction(data, suite, c, method, device, batch_size, donors=None, columns=()):
    forecast = np.empty((data.n, 11, 12, 2), np.float32)
    weights = np.zeros((data.n, 11), np.float32)
    changes = np.zeros((data.n, 11), bool)
    delta = np.zeros((data.n, 11), np.float64)
    adapted = RefinementDataset(data.arrays, "original") if method == "original_gate" else data
    for start in range(0, data.n * 11, batch_size):
        local = np.arange(start, min(start + batch_size, data.n * 11))
        i, k = local // 11, local % 11
        values = adapted.inputs(c * data.n * 11 + local, device)
        with torch.inference_mode():
            if method == "cv":
                value = values["cv"]
                weight = torch.zeros(len(local), device=device)
            elif method.endswith("_endpoint"):
                candidate = method.removesuffix("_endpoint")
                raw = data.arrays[candidate][c, i, k] if candidate == "map" else data.arrays[candidate][i, k]
                weight = values["valid"].to(torch.float32)
                value = blend_forecasts(values["cv"], torch.from_numpy(np.ascontiguousarray(raw)).to(device), weight)
            else:
                model = suite["models"]["original_gate" if method == "original_gate" else "p6_learned"]
                quality = values["quality"]
                original_weight = model(quality, values["valid"])
                if donors is not None:
                    donor = data.arrays["quality"][c, donors[c, i, k], k]
                    changed = quality.clone()
                    changed[:, list(columns)] = torch.from_numpy(np.ascontiguousarray(donor[:, columns])).to(device)
                    changes[i, k] = (changed != quality).any(1).cpu().numpy()
                    quality = changed
                weight = model(quality, values["valid"])
                delta[i, k] = (weight - original_weight).abs().cpu().numpy()
                value = blend_forecasts(values["cv"], values["map"], weight)
        forecast[i, k], weights[i, k] = value.cpu().numpy(), weight.cpu().numpy()
    invalid = ~data.arrays["valid"][c]
    if np.any(weights[invalid] != 0) or not np.array_equal(forecast[invalid], data.arrays["cv"][invalid]):
        raise ValueError("diagnostic hard fallback mismatch")
    return forecast, weights, changes, delta


def diagnose(data, suite, device, batch_size=4096):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive batch size required")
    arrays, changes, donor_records = {}, [], []
    reference = {}
    for method in BASELINES:
        result = np.empty((6, data.n, 11, 3), np.float64)
        w = np.empty((6, data.n, 11), np.float32)
        for c in range(6):
            forecast, w[c], _, _ = _prediction(data, suite, c, method, device, batch_size)
            result[c] = forecast_metrics(forecast, data.arrays["target"])
        arrays[method] = result
        if method in ("p6_learned", "original_gate"):
            reference[method] = w
    oracle_rows = []
    for candidate in ("map", "original", "filtered"):
        result = np.empty((6, data.n, 11), np.float64)
        chosen = np.empty((6, data.n, 11), np.float32)
        reference_ade = np.empty_like(result)
        ref = reference["original_gate" if candidate == "original" else "p6_learned"]
        for c in range(6):
            for start in range(0, data.n * 11, batch_size):
                local = np.arange(start, min(start + batch_size, data.n * 11))
                i, k = local // 11, local % 11
                values = data.inputs(c * data.n * 11 + local, device)
                raw = data.arrays[candidate][c, i, k] if candidate == "map" else data.arrays[candidate][i, k]
                ade, weight = oracle_mix(values["cv"], torch.from_numpy(np.ascontiguousarray(raw)).to(device),
                    data.labels(c * data.n * 11 + local, device), values["mask"], values["valid"],
                    torch.from_numpy(np.ascontiguousarray(ref[c, i, k])).to(device))
                result[c, i, k], chosen[c, i, k] = ade.cpu().numpy(), weight.cpu().numpy()
                # Compare the oracle to this candidate at the same reference g.
                reference_forecast = blend_forecasts(values["cv"], torch.from_numpy(np.ascontiguousarray(raw)).to(device),
                    torch.from_numpy(np.ascontiguousarray(ref[c, i, k])).to(device))
                distance = torch.linalg.vector_norm(reference_forecast-data.labels(c * data.n * 11 + local, device), dim=-1)
                reference_ade[c, i, k] = ((distance*values["mask"]).double().sum(1)/values["mask"].sum(1)).cpu().numpy()
            valid = data.arrays["valid"][c]
            count = int(valid.sum())
            endpoint = arrays[candidate + "_endpoint"][c, ..., 0]
            actual = reference_ade[c]
            oracle_rows.append({"condition": CONDITIONS[c], "candidate": candidate,
                "ade": result[c].mean(), "reference_ade": actual.mean(), "reference_gap": (actual-result[c]).mean(),
                "valid_count": count, "endpoint_better_cv_fraction": float(((endpoint < arrays["cv"][c, ..., 0]) & valid).sum()) / max(count, 1),
                "oracle_zero_fraction": float(((chosen[c] == 0) & valid).sum()) / max(count, 1),
                "weight_distance": float(np.abs(chosen[c]-ref[c])[valid].sum(dtype=np.float64)) / max(count, 1)})
        arrays["oracle_" + candidate + "_ade"] = result
        arrays["oracle_" + candidate + "_weight"] = chosen
    for repeat in REPEATS:
        donors = permutation_donors(data.arrays["valid"], data.arrays["age_s"], repeat)
        donor_records.append({"repeat": repeat, "donor_sha256": hashlib.sha256(donors.tobytes()).hexdigest()})
        for feature, columns in FEATURE_GROUPS.items():
            method = f"perm_{feature}_r{repeat}"
            result = np.empty((6, data.n, 11, 3), np.float64)
            for c in range(6):
                forecast, _, changed, delta = _prediction(data, suite, c, method, device, batch_size, donors, columns)
                result[c] = forecast_metrics(forecast, data.arrays["target"])
                valid = data.arrays["valid"][c]
                count = int(valid.sum())
                changes.append({"condition": CONDITIONS[c], "feature": feature, "repeat": repeat,
                    "valid_count": count, "changed_fraction": float(changed[valid].sum()) / max(count, 1),
                    "mean_abs_weight_change": float(delta[valid].sum()) / max(count, 1)})
            arrays[method] = result
    conditions, scenes = [], []
    names = data.arrays["scene_name"]
    for method, values in arrays.items():
        if values.ndim != 4:
            continue
        for c, condition in enumerate(CONDITIONS):
            conditions.append({"method": method, "condition": condition,
                **dict(zip(METRICS, values[c].mean(axis=(0, 1)).tolist(), strict=True))})
            for scene in sorted(set(names.tolist())):
                value = values[c, names == scene]
                scenes.append({"method": method, "condition": condition, "scene": scene, "count": len(value) * 11,
                    **dict(zip((m + "_sum" for m in METRICS), value.sum(axis=(0, 1)).tolist(), strict=True))})
    return arrays, {"conditions.csv": pd.DataFrame(conditions), "scene_stats.csv": pd.DataFrame(scenes),
        "oracle_summary.csv": pd.DataFrame(oracle_rows), "feature_change.csv": pd.DataFrame(changes),
        "donors.csv": pd.DataFrame(donor_records)}


def write_outputs(root, arrays, tables, metadata):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "manifest.json"
    marker.unlink(missing_ok=True)
    np.savez_compressed(root / "diagnostics.npz", **arrays)
    for filename, table in tables.items():
        table.to_csv(root / filename, index=False)
    manifest = {"schema_version": 1, "metadata": metadata,
        "arrays": {k: {"shape": list(v.shape), "dtype": v.dtype.str} for k, v in arrays.items()},
        "sha256": {name: sha256_file(root / name) for name in ("diagnostics.npz", *tables)}}
    temporary = root / ".manifest.tmp"
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(marker)


def verify_outputs(root, arrays, tables, metadata):
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    expected = {k: {"shape": list(v.shape), "dtype": v.dtype.str} for k, v in arrays.items()}
    if (set(manifest) != {"schema_version", "metadata", "arrays", "sha256"}
        or manifest["schema_version"] != 1 or manifest["metadata"] != metadata or manifest["arrays"] != expected
        or set(manifest["sha256"]) != {"diagnostics.npz", *tables}):
        raise ValueError("diagnostic replay metadata/schema mismatch")
    for name, checksum in manifest["sha256"].items():
        if sha256_file(root / name) != checksum:
            raise ValueError("diagnostic replay checksum mismatch")
    with np.load(root / "diagnostics.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(arrays):
            raise ValueError("diagnostic replay array set mismatch")
        for name, value in arrays.items():
            actual = saved[name]
            if (actual.shape != value.shape or actual.dtype != value.dtype or not np.isfinite(actual).all()
                or not np.allclose(actual, value, atol=1e-8, rtol=1e-7)):
                raise ValueError("diagnostic replay numeric mismatch: " + name)
    for name, frame in tables.items():
        try:
            pd.testing.assert_frame_equal(pd.read_csv(root / name, float_precision="round_trip"), frame,
                check_dtype=False, atol=1e-10, rtol=1e-9)
        except AssertionError as error:
            raise ValueError("diagnostic replay table mismatch: " + name) from error


def diagnostic_sources():
    package = Path(__file__).parents[1]
    return {"diagnostic": sha256_file(Path(__file__)), "parents": evaluation_sources(),
        "cache_loader": sha256_file(package / "context/gate_preparation.py")}


def combined_tables(all_tables):
    conditions = pd.concat([t["conditions.csv"].assign(seed=s) for s, t in all_tables.items()], ignore_index=True)
    oracle = pd.concat([t["oracle_summary.csv"].assign(seed=s) for s, t in all_tables.items()], ignore_index=True)
    changes = pd.concat([t["feature_change.csv"].assign(seed=s) for s, t in all_tables.items()], ignore_index=True)
    stats = pd.concat([t["scene_stats.csv"].assign(seed=s) for s, t in all_tables.items()], ignore_index=True)
    candidate = conditions[conditions.method.isin(BASELINES)].groupby(["method", "condition"], as_index=False).agg(
        **{m + "_mean": (m, "mean") for m in METRICS}, **{m + "_std": (m, "std") for m in METRICS})
    oracle_mean = oracle.groupby(["candidate", "condition"], as_index=False).agg(
        **{m + "_mean": (m, "mean") for m in ("ade", "reference_gap", "endpoint_better_cv_fraction", "oracle_zero_fraction", "weight_distance")},
        ade_std=("ade", "std"), valid_count=("valid_count", "first"))
    intervals = []
    for scope in ("overall", "clean"):
        selected = stats if scope == "overall" else stats[stats.condition == "clean"]
        baseline = selected[selected.method == "p6_learned"].groupby(["seed", "scene"], as_index=False)[["count", *[m + "_sum" for m in METRICS]]].sum()
        for feature in FEATURE_GROUPS:
            methods = [f"perm_{feature}_r{r}" for r in REPEATS]
            perm = selected[selected.method.isin(methods)].groupby(["seed", "scene", "condition"], as_index=False)[["count", *[m + "_sum" for m in METRICS]]].mean()
            perm = perm.groupby(["seed", "scene"], as_index=False)[["count", *[m + "_sum" for m in METRICS]]].sum()
            merged = baseline.merge(perm, on=["seed", "scene"], suffixes=("_base", "_perm"), validate="one_to_one")
            if not np.array_equal(merged.count_base, merged.count_perm):
                raise ValueError("diagnostic scene count mismatch")
            per_scene = merged.groupby("scene", sort=True).mean(numeric_only=True)
            idx = np.random.default_rng(0).integers(0, len(per_scene), (2000, len(per_scene)))
            count = per_scene.count_base.to_numpy()
            for metric in METRICS:
                delta = per_scene[metric + "_sum_perm"].to_numpy() - per_scene[metric + "_sum_base"].to_numpy()
                draws = delta[idx].sum(1) / count[idx].sum(1)
                seed_delta = merged.assign(difference=merged[metric + "_sum_perm"]-merged[metric + "_sum_base"]).groupby("seed")[["difference", "count_base"]].sum()
                values = seed_delta.difference / seed_delta.count_base
                low, high = np.quantile(draws, [.025, .975])
                intervals.append({"scope": scope, "feature": feature, "metric": metric,
                    "delta": float(delta.sum()/count.sum()), "seed_std": float(values.std(ddof=1)),
                    "ci95_low": float(low), "ci95_high": float(high)})
    return {"candidates.csv": candidate, "oracle_opportunity.csv": oracle_mean,
        "feature_importance.csv": pd.DataFrame(intervals), "feature_change.csv": changes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache-root", "training-root", "p6-training-root", "selection-root", "output-root"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    device = choose_device(args.device)
    source = diagnostic_sources()
    selection = load_selection(args.selection_root)
    all_tables, references = {}, {}
    keys = None
    for seed in SEEDS:
        cache = args.cache_root / "train_val" / f"seed-{seed}"
        print(json.dumps({"loading_train_val_seed": seed, "verify": args.verify}), flush=True)
        arrays, manifest = load_verified_gate_cache(cache, "train_val", seed, device=device)
        data = RefinementDataset(arrays)
        suite = load_suite(args.training_root, args.p6_training_root, manifest["contract"], data.scenes, "train_val", seed, device)
        reference = {"candidate_manifest": {"path": str((cache / "manifest.json").resolve()), "sha256": sha256_file(cache / "manifest.json")}, "suite": suite["references"]}
        if reference != selection["references"][str(seed)]:
            raise ValueError("diagnostic parent differs from frozen selection")
        sample_keys = key_hash(arrays)
        if keys is not None and sample_keys != keys:
            raise ValueError("diagnostic cross-seed sample keys mismatch")
        keys = sample_keys
        if seed == SEEDS[0]:
            dataset_identity = manifest["contract"]["dataset_identity"]
        elif manifest["contract"]["dataset_identity"] != dataset_identity:
            raise ValueError("diagnostic cross-seed dataset mismatch")
        metadata = {"split": "train_val", "seed": seed, "sample_count": data.n, "scene_count": len(data.scenes),
            "keys_sha256": keys, "references": reference, "sources": source, "device": str(device), "torch_version": str(torch.__version__),
            "selection_sha256": sha256_file(args.selection_root / "selection.json"),
            "feature_groups": {k: list(v) for k, v in FEATURE_GROUPS.items()}, "repeats": list(REPEATS),
            "oracle_scope": "future_label_using_grid_plus_reference_not_deployable_or_continuous_optimum"}
        values, tables = diagnose(data, suite, device)
        if diagnostic_sources() != source:
            raise ValueError("diagnostic source changed while running")
        root = args.output_root / f"seed-{seed}"
        if args.verify:
            verify_outputs(root, values, tables, metadata)
        else:
            write_outputs(root, values, tables, metadata)
        references[str(seed)] = {"path": str((root / "manifest.json").resolve()), "sha256": sha256_file(root / "manifest.json")}
        all_tables[seed] = tables
        print(json.dumps({"diagnosed_seed": seed, "samples": data.n, "scenes": len(data.scenes), "methods": len(values), "verified": args.verify}), flush=True)
        del values, arrays, data, suite
    tables = combined_tables(all_tables)
    meta = {"schema_version": 1, "split": "train_val", "seeds": list(SEEDS), "keys_sha256": keys,
        "references": references, "sources": source, "bootstrap": {"unit": "scene_after_repeat_then_fixed_seed_average", "replicates": 2000, "seed": 0},
        "interval_count": len(tables["feature_importance.csv"])}
    if args.verify:
        verify_outputs(args.output_root, {}, tables, meta)
    else:
        write_outputs(args.output_root, {}, tables, meta)
    print(json.dumps({"complete": True, "verified": args.verify, "interval_count": meta["interval_count"]}), flush=True)


if __name__ == "__main__":
    main()

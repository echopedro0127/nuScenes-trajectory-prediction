"""Retrospective train_val-only precision/Pareto diagnostic of frozen P6.5 heads."""

import argparse
import json
from pathlib import Path

import pandas as pd

from trajectory_prediction.online.strength_consistency import ARMS
from trajectory_prediction.online.strength_evaluation import (
    SEEDS,
    choose_strength,
    load_strength_selection,
    selection_rows,
)
from trajectory_prediction.utils.provenance import sha256_file

PARENT_SELECTION_SHA256 = "508bdef16942f9ce4996f9e15649d4f68768d1131557e039a2f9950d40d94a34"
FIELDS = ("ade", "fde", "jitter", "fde_p95")
POOL = ("RH", "revision_fixed", *ARMS)
MARGINS = (0., .0005, .001, .0015, .002)
PRIMARY_MARGIN = .0005
RATIOS = ("overall_ade_ratio", "clean_ade_ratio", "overall_fde_ratio", "clean_fde_p95_ratio")
OBJECTIVES = ("ade", "fde", "jitter", "clean_ade", "clean_fde_p95")


def _choice(candidates: pd.DataFrame, margin: float) -> tuple[str, pd.Series]:
    eligible = (
        candidates.overall_ade_ratio.le(1 + margin)
        & candidates.clean_ade_ratio.le(1 + margin)
        & candidates.overall_fde_ratio.le(1.0025)
        & candidates.clean_fde_p95_ratio.le(1.005)
    )
    selected = candidates[eligible].sort_values(["jitter", "ade", "strategy"]).iloc[0]
    return str(selected.strategy), eligible


def diagnose(frame: pd.DataFrame):
    """Validate the full frozen matrix, then diagnose a fixed six-head pool."""
    choose_strength(frame)  # Parent validation requires all 22 strategies, 3 seeds, 6 conditions.
    means = frame.groupby("strategy")[list(FIELDS)].mean()
    clean = frame[frame.condition.eq("clean")].groupby("strategy")[list(FIELDS)].mean()
    if (means.loc["RH", list(FIELDS)] <= 0).any() or (clean.loc["RH", list(FIELDS)] <= 0).any():
        raise ValueError("positive RH reference metrics required")
    candidates = means.loc[list(POOL), ["ade", "fde", "jitter"]].reset_index()
    candidates["clean_ade"] = candidates.strategy.map(clean.ade)
    candidates["clean_fde_p95"] = candidates.strategy.map(clean.fde_p95)
    candidates["overall_ade_ratio"] = candidates.ade / means.loc["RH", "ade"]
    candidates["clean_ade_ratio"] = candidates.clean_ade / clean.loc["RH", "ade"]
    candidates["overall_fde_ratio"] = candidates.fde / means.loc["RH", "fde"]
    candidates["clean_fde_p95_ratio"] = candidates.clean_fde_p95 / clean.loc["RH", "fde_p95"]
    candidates["jitter_ratio"] = candidates.jitter / means.loc["RH", "jitter"]
    dominated_by = []
    values = candidates[list(OBJECTIVES)].to_numpy()
    for value in values:
        dominates = (values <= value).all(axis=1) & (values < value).any(axis=1)
        dominated_by.append(";".join(sorted(candidates.loc[dominates, "strategy"])))
    candidates["dominated_by"] = dominated_by
    candidates["pareto"] = candidates.dominated_by.eq("")
    selected, eligible = _choice(candidates, PRIMARY_MARGIN)
    candidates["eligible_primary"] = eligible
    grid = pd.DataFrame([
        {"ade_margin": margin, "selected": _choice(candidates, margin)[0],
         "eligible_count": int(_choice(candidates, margin)[1].sum())}
        for margin in MARGINS
    ])
    per_seed = []
    for seed in SEEDS:
        part = frame[frame.seed.eq(seed)]
        overall = part.groupby("strategy")[list(FIELDS)].mean()
        clean_seed = part[part.condition.eq("clean")].set_index("strategy")
        for name in POOL:
            per_seed.append({
                "seed": seed, "strategy": name,
                "overall_ade_ratio": overall.loc[name, "ade"] / overall.loc["RH", "ade"],
                "clean_ade_ratio": clean_seed.loc[name, "ade"] / clean_seed.loc["RH", "ade"],
                "overall_fde_ratio": overall.loc[name, "fde"] / overall.loc["RH", "fde"],
                "clean_fde_p95_ratio": clean_seed.loc[name, "fde_p95"] / clean_seed.loc["RH", "fde_p95"],
                "jitter_ratio": overall.loc[name, "jitter"] / overall.loc["RH", "jitter"],
            })
    selected_row = candidates.set_index("strategy").loc[selected]
    gate = (selected in ARMS and selected_row.jitter_ratio <= .995
            and selected_row.jitter < means.loc["revision_fixed", "jitter"]
            and selected_row.jitter < means.loc["revision_0", "jitter"])
    decision = {
        "selected": selected, "reference": "RH", "ade_margin": PRIMARY_MARGIN,
        "fde_margin": .0025, "clean_fde_p95_margin": .005,
        "ratios": {name: float(selected_row[name]) for name in (*RATIOS, "jitter_ratio")},
        "meets_diagnostic_gate": bool(gate), "advance": False, "val_run": False,
        "scope": "retrospective_selection_diagnostic_not_independent_generalization",
    }
    return candidates, pd.DataFrame(per_seed), grid, decision


def load_parent(root: Path):
    """Bind the frozen parent and reconstruct its train_val selection table."""
    root = Path(root).resolve()
    if sha256_file(root / "selection.json") != PARENT_SELECTION_SHA256:
        raise ValueError("frozen P6.5 selection identity mismatch")
    parent = load_strength_selection(root)
    expected = pd.read_csv(root / "selection_conditions.csv", float_precision="round_trip")
    reconstructed = []
    bindings = {"selection.json": PARENT_SELECTION_SHA256,
                "selection_conditions.csv": sha256_file(root / "selection_conditions.csv")}
    for seed in SEEDS:
        folder = root.parent / "evaluation" / "train_val" / f"seed-{seed}"
        reference = parent["references"][str(seed)]
        summary_path = folder / "summary.json"
        if Path(reference["path"]).resolve() != summary_path or sha256_file(summary_path) != reference["sha256"]:
            raise ValueError("train_val reference identity mismatch")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        metadata = summary["metadata"]
        if (metadata["split"] != "train_val" or metadata["seed"] != seed
                or metadata.get("sources") != parent.get("sources")):
            raise ValueError("train_val source or split mismatch")
        tables = {}
        for name in ("by_condition.csv", "quantiles.csv"):
            if sha256_file(folder / name) != summary["sha256"][name]:
                raise ValueError("train_val table checksum mismatch")
            tables[name] = pd.read_csv(folder / name, float_precision="round_trip")
            bindings[f"seed-{seed}/{name}"] = summary["sha256"][name]
        reconstructed.append(selection_rows(tables))
        bindings[f"seed-{seed}/summary.json"] = reference["sha256"]
    rebuilt = pd.concat(reconstructed, ignore_index=True)
    keys = ["seed", "condition", "strategy"]
    try:
        pd.testing.assert_frame_equal(
            expected.sort_values(keys).reset_index(drop=True),
            rebuilt[expected.columns].sort_values(keys).reset_index(drop=True),
            check_exact=True,
        )
    except AssertionError as exc:
        raise ValueError("train_val selection reconstruction mismatch") from exc
    return expected, bindings


def run(selection_root: Path, output_root: Path):
    """Write into a new directory; no val reader or training path exists here."""
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError("diagnostic output already exists; choose a new run directory")
    frame, bindings = load_parent(selection_root)
    candidates, seeds, grid, decision = diagnose(frame)
    output_root.mkdir(parents=True, exist_ok=False)
    for name, table in (("candidates.csv", candidates), ("per_seed.csv", seeds), ("margins.csv", grid)):
        table.to_csv(output_root / name, index=False)
    project = Path(__file__).resolve().parents[3]
    sources = {name: sha256_file(project / name) for name in (
        "src/trajectory_prediction/analysis/precision_selection.py",
        "scripts/audit_precision_selection.py",
        "docs/superpowers/specs/2026-11-06-causal-context-phase66-selection-design.md",
    )}
    report = {"schema_version": 1, "split": "train_val", "decision": decision,
              "parent_inputs": bindings, "sources": sources, "training_runs": 0,
              "production_replay": False, "candidate_count": len(candidates),
              "input_row_count": len(frame), "seeds": list(SEEDS)}
    (output_root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    marker = {"schema_version": 1, "sha256": {name: sha256_file(output_root / name) for name in (
        "candidates.csv", "per_seed.csv", "margins.csv", "report.json")}}
    temporary = output_root / ".completion.tmp"
    temporary.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output_root / "completion.json")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-root", type=Path, default=Path("outputs/context/p65/selection"))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/context/p66-selection"))
    args = parser.parse_args(argv)
    print(json.dumps(run(args.selection_root, args.output_root)["decision"], indent=2))
    return 0

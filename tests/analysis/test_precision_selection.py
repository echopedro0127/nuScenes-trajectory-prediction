import json

import numpy as np
import pandas as pd
import pytest

from trajectory_prediction.analysis import precision_selection as audit
from trajectory_prediction.online.strength_artifacts import STRATEGIES
from trajectory_prediction.online.strength_evaluation import CONDITIONS, SEEDS
from trajectory_prediction.utils.provenance import sha256_file


def matrix():
    values = {"RH": (1., 2., 1., 6.), "revision_0": (1., 2., 1., 6.),
              "revision_fixed": (1., 2., .998, 6.), "fixed_005": (1., 2., .98, 6.),
              "adaptive_matched_005": (1.0003, 2.001, .97, 6.001),
              "fixed_010": (1.0008, 2., .96, 6.),
              "adaptive_matched_010": (1.0015, 2., .95, 6.)}
    return pd.DataFrame([
        {"seed": seed, "condition": condition, "strategy": strategy,
         **dict(zip(audit.FIELDS, values.get(strategy, (1.1, 2., .9, 6.)), strict=True))}
        for seed in SEEDS for condition in CONDITIONS for strategy in STRATEGIES
    ])


def test_primary_margin_and_sensitivity_do_not_promote():
    candidates, per_seed, grid, decision = audit.diagnose(matrix())
    assert decision["selected"] == "adaptive_matched_005"
    assert decision["meets_diagnostic_gate"]
    assert decision["advance"] is False and decision["val_run"] is False
    assert grid["ade_margin"].tolist() == list(audit.MARGINS)
    assert grid.iloc[0].selected == "fixed_005"
    assert len(candidates) == 6 and len(per_seed) == 18


@pytest.mark.parametrize("condition,field,value", [
    ("clean", "ade", 1.01), ("clean", "fde_p95", 6.1), ("clean", "fde", 2.2),
])
def test_clean_and_endpoint_guards(condition, field, value):
    frame = matrix()
    frame.loc[frame.strategy.eq("adaptive_matched_005") & frame.condition.eq(condition), field] = value
    assert audit.diagnose(frame)[-1]["selected"] == "fixed_005"


def test_pareto_dominance_equal_metrics_and_tie_are_explicit():
    frame = matrix()
    frame.loc[frame.strategy.eq("revision_fixed"), "jitter"] = .98
    candidates, _, _, _ = audit.diagnose(frame)
    rh = candidates.set_index("strategy").loc["RH"]
    assert not rh.pareto and "fixed_005" in rh.dominated_by
    assert candidates.set_index("strategy").loc["fixed_005", "pareto"]
    assert candidates.set_index("strategy").loc["revision_fixed", "pareto"]
    frame.loc[frame.strategy.eq("adaptive_matched_005"), list(audit.FIELDS)] = [1., 2., .98, 6.]
    assert audit.diagnose(frame)[-1]["selected"] == "adaptive_matched_005"


def test_no_feasible_new_arm_keeps_reference_without_promotion():
    frame = matrix()
    frame.loc[frame.strategy.ne("RH") & frame.strategy.ne("revision_0"), "ade"] = 2.
    decision = audit.diagnose(frame)[-1]
    assert decision["selected"] == "RH" and not decision["meets_diagnostic_gate"]


@pytest.mark.parametrize("change", ["missing", "duplicate", "nan", "negative"])
def test_rejects_invalid_experiment_matrix(change):
    frame = matrix()
    if change == "missing":
        frame = frame.iloc[:-1]
    elif change == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    else:
        frame.loc[0, "ade"] = np.nan if change == "nan" else -1.
    with pytest.raises(ValueError):
        audit.diagnose(frame)


def test_existing_output_is_not_overwritten(tmp_path):
    with pytest.raises(FileExistsError):
        audit.run(tmp_path / "nonexistent-input", tmp_path)


def test_pinned_parent_identity_is_checked_before_loading(tmp_path):
    (tmp_path / "selection.json").write_text("{}")
    with pytest.raises(ValueError, match="identity"):
        audit.load_parent(tmp_path)


def fixture_parent(tmp_path, monkeypatch):
    root = tmp_path / "selection"
    root.mkdir()
    data = matrix()
    data.to_csv(root / "selection_conditions.csv", index=False)
    references = {}
    for seed in SEEDS:
        folder = tmp_path / "evaluation" / "train_val" / f"seed-{seed}"
        folder.mkdir(parents=True)
        part = data[data.seed.eq(seed)]
        part.drop(columns="fde_p95").to_csv(folder / "by_condition.csv", index=False)
        tail = part[["seed", "strategy", "condition", "fde_p95"]].rename(columns={"fde_p95": "p95"})
        tail.assign(metric="fde").to_csv(folder / "quantiles.csv", index=False)
        summary = {"metadata": {"split": "train_val", "seed": seed}, "sha256": {
            name: sha256_file(folder / name) for name in ("by_condition.csv", "quantiles.csv")}}
        (folder / "summary.json").write_text(json.dumps(summary))
        references[str(seed)] = {"path": str((folder / "summary.json").resolve()),
                                 "sha256": sha256_file(folder / "summary.json")}
    parent = {"references": references}
    (root / "selection.json").write_text(json.dumps(parent))
    monkeypatch.setattr(audit, "PARENT_SELECTION_SHA256", sha256_file(root / "selection.json"))
    monkeypatch.setattr(audit, "load_strength_selection", lambda path: parent)
    return root


def test_rebuilds_train_val_tables_and_rejects_rehashed_numeric_tampering(tmp_path, monkeypatch):
    root = fixture_parent(tmp_path, monkeypatch)
    data, _ = audit.load_parent(root)
    pd.testing.assert_frame_equal(data, matrix(), check_dtype=False)
    table = root.parent / "evaluation" / "train_val" / "seed-2027" / "by_condition.csv"
    frame = pd.read_csv(table)
    frame.loc[0, "ade"] += 1.
    frame.to_csv(table, index=False)
    summary_path = table.parent / "summary.json"
    summary = json.loads(summary_path.read_text())
    summary["sha256"]["by_condition.csv"] = sha256_file(table)
    summary_path.write_text(json.dumps(summary))
    parent = json.loads((root / "selection.json").read_text())
    parent["references"]["2027"]["sha256"] = sha256_file(summary_path)
    monkeypatch.setattr(audit, "load_strength_selection", lambda path: parent)
    with pytest.raises(ValueError, match="reconstruction"):
        audit.load_parent(root)


def test_runner_never_reads_val_and_publishes_bound_outputs(tmp_path, monkeypatch):
    root = fixture_parent(tmp_path, monkeypatch)
    original = audit.Path.read_bytes
    original_text = audit.Path.read_text

    def check_bytes(path, *args, **kwargs):
        assert "val" not in path.parts
        return original(path, *args, **kwargs)

    def check_text(path, *args, **kwargs):
        assert "val" not in path.parts
        return original_text(path, *args, **kwargs)

    monkeypatch.setattr(audit.Path, "read_bytes", check_bytes)
    monkeypatch.setattr(audit.Path, "read_text", check_text)
    output = tmp_path / "new-output"
    result = audit.run(root, output)
    marker = json.loads((output / "completion.json").read_text())
    assert not result["decision"]["val_run"]
    for name, checksum in marker["sha256"].items():
        assert sha256_file(output / name) == checksum

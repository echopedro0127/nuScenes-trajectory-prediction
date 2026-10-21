import numpy as np
import pandas as pd
import pytest

from trajectory_prediction.analysis.refinement_report import (
    load_numeric_artifacts,
    paired_comparisons,
)
from trajectory_prediction.online.refinement_evaluation import STRATEGIES
from trajectory_prediction.utils.provenance import sha256_file


def test_numeric_artifacts_require_complete_shapes_dtypes_and_checksums(tmp_path):
    values = {s: np.zeros((6, 2, 11, 3), np.float64) for s in STRATEGIES}
    weights = {s: np.zeros((6, 2, 11), np.float32) for s in STRATEGIES}
    np.savez_compressed(tmp_path / "metrics.npz", **values)
    np.savez_compressed(tmp_path / "weights.npz", **weights)
    summary = {
        "sample_count": 2,
        "sha256": {n: sha256_file(tmp_path / n) for n in ("metrics.npz", "weights.npz")},
    }
    load_numeric_artifacts(tmp_path, summary)
    values.pop("original_gate")
    np.savez_compressed(tmp_path / "metrics.npz", **values)
    summary["sha256"]["metrics.npz"] = sha256_file(tmp_path / "metrics.npz")
    with pytest.raises(ValueError, match="strategies"):
        load_numeric_artifacts(tmp_path, summary)


def test_comparisons_average_fixed_seeds_then_weight_scenes():
    rows = []
    for seed in (2027, 2028, 2029):
        for strategy in STRATEGIES:
            for scene, count in (("a", 2), ("b", 6)):
                rows.append(
                    {
                        "seed": seed,
                        "condition": "clean",
                        "strategy": strategy,
                        "scene_name": scene,
                        "update_count": count,
                        **{
                            m + "_sum": count * (2 if strategy in ("p6_learned", "sliding") else 1)
                            for m in ("ade", "fde", "jitter")
                        },
                    }
                )
    result = paired_comparisons(
        pd.DataFrame(rows), {"accuracy": "original_gate", "stable": "time_age_gate"}
    )
    assert result and all(r["delta"] == -1 and r["ci95"] == [-1, -1] for r in result)
    with pytest.raises(ValueError, match="seed"):
        paired_comparisons(
            pd.DataFrame(rows).query("seed != 2029"),
            {"accuracy": "original_gate", "stable": "time_age_gate"},
        )


def test_stability_outcome_states_failure_and_does_not_reselect():
    from trajectory_prediction.analysis.refinement_report import stability_outcome

    rows = []
    for seed in (2027, 2028, 2029):
        for strategy, ade, jitter in (("p6_learned", 1.0, 1.0), ("picked", 1.02, 0.8)):
            for condition in ("clean", "missing"):
                rows.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "condition": condition,
                        "ade": ade,
                        "jitter": jitter,
                    }
                )
    result = stability_outcome(pd.DataFrame(rows), "picked")
    assert (
        not result["target_met"]
        and not result["within_ade_tolerance"]
        and result["jitter_improved"]
        and result["strategy"] == "picked"
    )


def test_existing_cv_control_is_not_claimed_as_new_refinement_success():
    from trajectory_prediction.analysis.refinement_report import stability_outcome

    rows = []
    for seed in (2027, 2028, 2029):
        for strategy, jitter in (("p6_learned", 1.0), ("sliding", 0.8)):
            for condition in ("clean", "missing"):
                rows.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "condition": condition,
                        "ade": 1.0,
                        "jitter": jitter,
                    }
                )
    result = stability_outcome(pd.DataFrame(rows), "sliding")
    assert result["selected_baseline"] and not result["target_met"]

import numpy as np
import pandas as pd
import pytest

from trajectory_prediction.analysis.horizon_report import factorial_effects, paired_intervals
from trajectory_prediction.online.horizon_artifacts import STRATEGIES
from trajectory_prediction.online.horizon_evaluation import CONDITIONS, SEEDS
from trajectory_prediction.online.horizon_gate import VARIANTS


def test_scene_intervals_pair_fixed_seeds_and_reject_missing_scene_seed():
    rows = []
    for seed, multiplier in zip(SEEDS, (1, 2, 3), strict=True):
        for condition in CONDITIONS:
            for strategy in STRATEGIES:
                for scene, count in (("small", 11), ("large", 33)):
                    delta = -.1*multiplier if strategy == "DRH" else 0.
                    rows.append({"seed": seed, "condition": condition, "strategy": strategy, "scene_name": scene,
                        "update_count": count, "ade_sum": (1+delta)*count, "fde_sum": (2+delta)*count,
                        "jitter_sum": (1+delta)*count})
    frame = pd.DataFrame(rows)
    result = paired_intervals(frame, {"accuracy": "DRH", "stable": "DRH"})
    row = next(x for x in result if x["condition"] == "clean" and x["contrast"] == "DRH-B" and x["metric"] == "ade")
    assert row["delta"] == pytest.approx(-.2)
    np.testing.assert_allclose(row["ci95"], [-.2, -.2])
    overall = next(x for x in result if x["condition"] == "overall" and x["contrast"] == "DRH-B" and x["metric"] == "ade")
    assert overall["delta"] == pytest.approx(-.2)
    assert len(result) == 18*7*3
    with pytest.raises(ValueError, match="paired"):
        paired_intervals(frame.drop(frame.query("seed == 2029 and strategy == 'DRH'").index[:1]), {"accuracy": "DRH", "stable": "DRH"})


def test_balanced_factorial_effects_average_all_four_pairs_and_fixed_seeds():
    frame = pd.DataFrame([{"seed": s, "strategy": name, "condition": c,
        "ade": 5+sum({"D": .1, "R": -.2, "H": -.3}[factor] for factor in name if factor in "DRH"),
        "fde": 2., "jitter": 1.} for s in SEEDS for c in CONDITIONS for name in VARIANTS])
    result = factorial_effects(frame)
    assert len(result) == 18
    for factor, expected in (("D", .1), ("R", -.2), ("H", -.3)):
        row = result.query("scope == 'overall' and factor == @factor and metric == 'ade'").iloc[0]
        assert row.delta == pytest.approx(expected)
        assert row.seed_std == pytest.approx(0)

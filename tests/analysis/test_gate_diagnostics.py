"""Observable contracts for frozen candidate/feature diagnostics."""
import json

import numpy as np
import pandas as pd
import pytest
import torch

from trajectory_prediction.analysis.gate_diagnostics import (
    FEATURE_GROUPS,
    combined_tables,
    diagnose,
    oracle_mix,
    permutation_donors,
    verify_outputs,
    write_outputs,
)
from trajectory_prediction.online.gate_refinement import RefinementDataset
from trajectory_prediction.online.quality_gate import CONDITIONS, QualityGate, packet_schedule
from trajectory_prediction.utils.provenance import sha256_file


def test_oracle_masks_released_labels_and_respects_invalid_packets():
    cv = torch.zeros(2, 12, 2)
    candidate = torch.full_like(cv, 2)
    target = torch.ones_like(cv)
    target[:, :7] = 900
    mask = torch.arange(12)[None].expand(2, -1) >= 7
    ade, weight = oracle_mix(cv, candidate, target, mask, torch.tensor([True, False]), torch.tensor([.13, 0.]))
    np.testing.assert_allclose(weight.numpy(), [.5, 0])
    np.testing.assert_allclose(ade.numpy(), [0, 2**.5])


def test_oracle_includes_off_grid_reference_and_rejects_bad_mask():
    cv = torch.zeros(1, 12, 2)
    candidate = torch.ones_like(cv)
    target = candidate * .137
    mask = torch.ones(1, 12, dtype=torch.bool)
    ade, weight = oracle_mix(cv, candidate, target, mask, torch.tensor([True]), torch.tensor([.137]))
    assert ade.item() == 0
    assert weight.item() == pytest.approx(.137)
    with pytest.raises(ValueError):
        oracle_mix(cv, candidate, target, mask & False, torch.tensor([True]), torch.tensor([.137]))


def test_donors_preserve_packet_strata_and_do_not_mutate_inputs():
    valid = np.ones((6, 20, 11), bool)
    valid[:, :2] = False
    ages = np.zeros(valid.shape, np.float32)
    ages[:, 10:] = .5
    donors = permutation_donors(valid, ages, 2)
    np.testing.assert_array_equal(donors, permutation_donors(valid, ages, 2))
    rows = np.broadcast_to(np.arange(20)[None, :, None], valid.shape)
    assert np.array_equal(donors[~valid], rows[~valid])
    for c in range(6):
        for k in range(11):
            np.testing.assert_array_equal(ages[c, donors[c, :, k], k], ages[c, :, k])
            np.testing.assert_array_equal(valid[c, donors[c, :, k], k], valid[c, :, k])
            assert sorted(donors[c, :, k]) == list(range(20))
    assert np.any(donors != rows)
    assert not valid[:, :2].any()


def synthetic_data():
    n = 6
    rng = np.random.default_rng(5)
    cv = np.zeros((n, 11, 12, 2), np.float32)
    original = np.full_like(cv, 2)
    maps = np.broadcast_to(original, (6, *original.shape)).copy()
    valid = np.stack([np.broadcast_to(packet_schedule(c)["age_valid"], (n, 11)) for c in CONDITIONS]).copy()
    ages = np.stack([np.broadcast_to(packet_schedule(c)["age_s"], (n, 11)) for c in CONDITIONS]).copy()
    quality = rng.random((6, n, 11, 8), dtype=np.float32)
    quality[..., 0] = 1
    quality[..., 1] = ages / 2
    quality[..., 6] = np.arange(11, 0, -1) / 12
    arrays = {"cv": cv, "map": maps, "original": original, "filtered": original * .4,
              "quality": quality, "valid": valid, "age_s": ages,
              "target": np.ones((n, 12, 2), np.float32),
              "scene_name": np.array(["a", "a", "b", "b", "c", "c"]),
              "instance_token": np.array([f"i{i}" for i in range(n)]),
              "sample_token": np.array([f"s{i}" for i in range(n)])}
    model = QualityGate().eval()
    with torch.no_grad():
        model.network[0].weight.fill_(.2)
        model.network[0].bias.zero_()
        model.network[-1].weight.fill_(.1)
    return RefinementDataset(arrays), {"models": {"p6_learned": model, "original_gate": model}}


def test_complete_diagnostics_preserve_quality_fallback_and_all_groups():
    data, suite = synthetic_data()
    quality = data.arrays["quality"].copy()
    arrays, tables = diagnose(data, suite, torch.device("cpu"), batch_size=19)
    np.testing.assert_array_equal(quality, data.arrays["quality"])
    assert len(FEATURE_GROUPS) == 11
    assert len([key for key in arrays if key.startswith("perm_")]) == 33
    cv = arrays["cv"]
    for value in arrays.values():
        if value.ndim == 4:
            assert value.shape == (6, 6, 11, 3)
            np.testing.assert_array_equal(value[4], cv[4])
    np.testing.assert_array_equal(arrays["oracle_map_weight"][4], 0)
    assert np.all(arrays["oracle_map_ade"] <= arrays["p6_learned"][..., 0] + 1e-6)
    changes = tables["feature_change.csv"]
    assert (changes.loc[changes.feature == "coverage", "changed_fraction"] == 0).all()
    scene = tables["scene_stats.csv"]
    assert scene[(scene.method == "p6_learned") & (scene.condition == "clean")]["count"].sum() == 66


def test_replay_rejects_npz_tampering_even_after_hash_is_updated(tmp_path):
    arrays = {"x": np.zeros((6, 2, 11, 3), np.float64)}
    tables = {"stats.csv": pd.DataFrame({"name": ["x"], "ade": [0.]})}
    meta = {"split": "train_val", "seed": 2027}
    write_outputs(tmp_path, arrays, tables, meta)
    verify_outputs(tmp_path, arrays, tables, meta)
    np.savez_compressed(tmp_path / "diagnostics.npz", x=np.ones_like(arrays["x"]))
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["sha256"]["diagnostics.npz"] = sha256_file(tmp_path / "diagnostics.npz")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="replay"):
        verify_outputs(tmp_path, arrays, tables, meta)


def test_replay_rejects_missing_table_column(tmp_path):
    arrays = {"x": np.zeros(2)}
    tables = {"stats.csv": pd.DataFrame({"name": ["x"], "ade": [0.]})}
    write_outputs(tmp_path, arrays, tables, {})
    (tmp_path / "stats.csv").write_text("name\nx\n")
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    manifest["sha256"]["stats.csv"] = sha256_file(tmp_path / "stats.csv")
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="replay"):
        verify_outputs(tmp_path, arrays, tables, {})


def test_filtered_oracle_reference_is_same_candidate_not_map_gate_forecast():
    data, suite = synthetic_data()
    _, tables = diagnose(data, suite, torch.device("cpu"))
    quality = torch.from_numpy(data.arrays["quality"][0].reshape(-1, 8))
    valid = torch.from_numpy(data.arrays["valid"][0].reshape(-1))
    with torch.inference_mode():
        weight = suite["models"]["p6_learned"](quality, valid).numpy().reshape(6, 11)
    forecast = data.arrays["cv"] + weight[..., None, None] * (data.arrays["filtered"]-data.arrays["cv"])
    mask = np.arange(12)[None] >= np.arange(1, 12)[:, None]
    distance = np.linalg.norm(forecast-data.arrays["target"][:, None], axis=-1)
    expected = ((distance*mask).sum(-1, dtype=np.float64)/mask.sum(-1)).mean()
    row = tables["oracle_summary.csv"].query("condition == 'clean' and candidate == 'filtered'").iloc[0]
    assert row.reference_ade == pytest.approx(expected)
    assert row.reference_gap >= -1e-6


def test_statistics_average_repeats_then_seeds_and_keep_constant_group_zero():
    data, suite = synthetic_data()
    _, tables = diagnose(data, suite, torch.device("cpu"))
    combined = combined_tables({seed: tables for seed in (2027, 2028, 2029)})
    importance = combined["feature_importance.csv"]
    assert len(importance) == 66
    zero = importance[importance.feature == "coverage"]
    np.testing.assert_array_equal(zero[["delta", "seed_std", "ci95_low", "ci95_high"]], 0)
    condition = tables["conditions.csv"]
    base = condition.query("condition == 'clean' and method == 'p6_learned'").ade.iloc[0]
    perm = condition[(condition.condition == "clean") & condition.method.isin([f"perm_all_r{r}" for r in range(3)])].ade.mean()
    row = importance.query("scope == 'clean' and feature == 'all' and metric == 'ade'").iloc[0]
    assert row.delta == pytest.approx(perm-base)
    assert row.seed_std == pytest.approx(0)


def test_scene_bootstrap_with_unequal_scene_counts_and_distinct_seeds_repeats():
    all_tables = {}
    for seed, factor in zip((2027, 2028, 2029), (1, 2, 3), strict=True):
        scenes = []
        for condition in CONDITIONS:
            for scene, count, effect in (("small", 11, .1), ("large", 33, .5)):
                scenes.append({"method": "p6_learned", "condition": condition, "scene": scene, "count": count,
                    "ade_sum": count, "fde_sum": 2*count, "jitter_sum": 3*count})
                for feature in FEATURE_GROUPS:
                    for repeat in range(3):
                        difference = factor * (repeat+1) * effect
                        scenes.append({"method": f"perm_{feature}_r{repeat}", "condition": condition, "scene": scene, "count": count,
                            "ade_sum": (1+difference)*count, "fde_sum": (2+.5*difference)*count,
                            "jitter_sum": (3-.25*difference)*count})
        all_tables[seed] = {
            "scene_stats.csv": pd.DataFrame(scenes),
            "conditions.csv": pd.DataFrame([{"method": "p6_learned", "condition": "clean", "ade": 1., "fde": 2., "jitter": 3.}]),
            "oracle_summary.csv": pd.DataFrame([{"candidate": "map", "condition": "clean", "ade": .5,
                "reference_gap": .5, "endpoint_better_cv_fraction": .5, "oracle_zero_fraction": .5,
                "weight_distance": .5, "valid_count": 44}]),
            "feature_change.csv": pd.DataFrame([{"feature": "all", "repeat": 0, "changed_fraction": 1.}]),
        }
    importance = combined_tables(all_tables)["feature_importance.csv"]
    row = importance.query("scope == 'overall' and feature == 'all' and metric == 'ade'").iloc[0]
    assert row.delta == pytest.approx(1.6)
    assert row.seed_std == pytest.approx(.8)
    assert row.ci95_low == pytest.approx(.4)
    assert row.ci95_high == pytest.approx(2.)
    row = importance.query("scope == 'clean' and feature == 'motion' and metric == 'jitter'").iloc[0]
    assert row.delta == pytest.approx(-.4)
    assert row.seed_std == pytest.approx(.2)
    assert row.ci95_low == pytest.approx(-.5)
    assert row.ci95_high == pytest.approx(-.1)

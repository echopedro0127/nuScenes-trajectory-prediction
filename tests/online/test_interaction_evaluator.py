import numpy as np
import torch
from torch import nn

from trajectory_prediction.online.corrector import ResidualCorrector
from trajectory_prediction.online.interaction_corrector import (
    InteractionCorrector,
    predict_interaction,
)
from trajectory_prediction.online.interaction_evaluator import evaluate_interactions
from trajectory_prediction.online.interaction_report import STRATEGIES, write_interaction_outputs
from trajectory_prediction.online.observations import ObservationView


def fixture():
    future = np.arange(24, dtype=np.float32).reshape(1, 12, 2)
    arrays = {"history_xy": np.zeros((1, 5, 2), np.float32), "future_xy": future,
              "instance_token": np.array(["target"]), "sample_token": np.array(["sample"]),
              "scene_name": np.array(["scene"])}
    maps = {"features": np.zeros((1, 11, 1, 2, 4), np.float32), "mask": np.zeros((1, 11, 1, 2), bool),
            "lane_tokens": np.full((1, 11, 1), "", dtype="U64"), "anchor_timestamp_us": np.zeros(1, np.int64)}
    neighbors = {"features": np.zeros((1, 11, 8, 5, 6), np.float32),
        "mask": np.zeros((1, 11, 8, 5), bool), "velocity_mask": np.zeros((1, 11, 8, 5), bool),
        "instance_tokens": np.full((1, 11, 8), "", dtype="U64"),
        "annotation_tokens": np.full((1, 11, 8, 5), "", dtype="U64"),
        "source_timestamp_us": np.zeros((1, 11, 8, 5), np.int64),
        "available_timestamp_us": np.zeros((1, 11, 8, 5), np.int64),
        "timestamp_us": np.arange(1, 12, dtype=np.int64)[None] * 500_000}
    from copy import deepcopy
    original = ResidualCorrector(hidden_dim=8)
    models = nn.ModuleDict({"original": original})
    for mode in ("map", "neighbors", "fusion"):
        models[mode] = InteractionCorrector(deepcopy(original), mode, 8)
        models["matched_" + mode] = ResidualCorrector(hidden_dim=16)
    return arrays, maps, neighbors, future + 1, models


def test_nine_strategies_have_same_rows_and_all_context_missing_is_real_fallback(tmp_path):
    torch.set_num_threads(1)
    arrays, maps, neighbors, base, models = fixture()
    frame = evaluate_interactions(arrays, maps, neighbors, base, models)
    assert len(frame) == 99 and set(frame.strategy) == set(STRATEGIES)
    original = frame[frame.strategy == "original"].reset_index(drop=True)
    for mode in ("map", "neighbors", "fusion"):
        group = frame[frame.strategy == mode].reset_index(drop=True)
        np.testing.assert_array_equal(group[["ade", "fde", "jitter"]], original[["ade", "fde", "jitter"]])
        assert group.fallback.all()
    summary = write_interaction_outputs(frame, tmp_path, {"purpose": "test"})
    assert summary["strategies"]["fusion"]["sample_count"] == 1
    assert summary["strategies"]["fusion"]["update_count"] == 11
    assert summary["coverage"]["neighbors"] == 0
    assert summary["fallback_updates"]["fusion"] == 11
    for name in ("by_update.csv", "by_horizon.csv", "by_maneuver.csv"):
        assert (tmp_path / name).exists()


def test_actual_fusion_predictor_never_receives_labels_or_unrevealed_future():
    arrays, maps, neighbors, base, models = fixture()
    received = []

    def spy(model, observation, base, map_view, neighbor_view, device):
        assert isinstance(observation, ObservationView)
        assert not hasattr(observation, "remaining_target")
        result = predict_interaction(model, observation, base, map_view, neighbor_view, device)
        received.append((observation.observed_history.copy(), result[0].copy()))
        return result

    evaluate_interactions(arrays, maps, neighbors, base, models, predictor=spy)
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["future_xy"][:, 1:] += 10000
    evaluate_interactions(changed, maps, neighbors, base, models, predictor=spy)
    assert len(received) == 66
    np.testing.assert_array_equal(received[0][0], received[33][0])
    np.testing.assert_array_equal(received[0][1], received[33][1])



def test_populated_context_single_missing_modalities_and_quality_reporting(tmp_path):
    arrays, maps, neighbors, base, models = fixture()
    for mode in ("map", "neighbors", "fusion"):
        torch.nn.init.constant_(models[mode].head[-1].weight, .1)
    # k=2: map absent, neighbors present; k=3: opposite; k=4: both present.
    for k in (2, 4):
        timestamp = int(neighbors["timestamp_us"][0, k])
        neighbors["instance_tokens"][0, k, 0] = "neighbor"
        neighbors["annotation_tokens"][0, k, 0, 3:] = ["past", "now"]
        neighbors["mask"][0, k, 0, 3:] = True
        neighbors["velocity_mask"][0, k, 0, -1] = True
        neighbors["features"][0, k, 0, 3:, 0] = [1, 3]
        neighbors["features"][0, k, 0, -1, 2] = 4
        neighbors["features"][0, k, 0, 3:, 4] = 1
        times = [timestamp - 600_000, timestamp - 100_000]
        neighbors["source_timestamp_us"][0, k, 0, 3:] = times
        neighbors["available_timestamp_us"][0, k, 0, 3:] = times
    for k in (3, 4):
        maps["mask"][0, k, 0, 0] = True
        maps["lane_tokens"][0, k, 0] = "lane"
        maps["features"][0, k, 0, 0] = [1, 0, 1, 0]
    frame = evaluate_interactions(arrays, maps, neighbors, base, models)
    for step, flags in ((3, {"map": True, "neighbors": False, "fusion": False}),
                        (4, {"map": False, "neighbors": True, "fusion": False}),
                        (5, {"map": False, "neighbors": False, "fusion": False})):
        for name, expected in flags.items():
            row = frame[(frame.update_step == step) & (frame.strategy == name)].iloc[0]
            assert bool(row.fallback) == expected
            original = frame[(frame.update_step == step) & (frame.strategy == "original")].iloc[0]
            if not expected:
                assert row.ade != original.ade
    summary = write_interaction_outputs(frame, tmp_path, {})
    assert summary["coverage"] == {"map": 2 / 11, "neighbors": 2 / 11, "either": 3 / 11}
    assert summary["fallback_updates"]["fusion"] == 8
    quality = summary["neighbor_quality"]
    assert quality["selected_neighbors"] == 2
    assert quality["valid_history_points"] == 4
    assert quality["valid_velocity_points"] == 2
    assert quality["history_valid_fraction"] == .4 and quality["velocity_valid_fraction"] == .5
    assert quality["frame_age_p50_us"] == quality["frame_age_p95_us"] == 100_000

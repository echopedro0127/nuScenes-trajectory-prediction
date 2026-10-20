import numpy as np
import torch

from trajectory_prediction.context.maps import MapView
from trajectory_prediction.online.context_corrector import MapCorrector, profiled_predict_offsets
from trajectory_prediction.online.context_evaluator import (
    evaluate_context_arrays,
    write_context_outputs,
)
from trajectory_prediction.online.corrector import ResidualCorrector
from trajectory_prediction.online.observations import ObservationView


def inputs():
    arrays = {"history_xy": np.zeros((1, 5, 2), np.float32),
              "future_xy": np.arange(24, dtype=np.float32).reshape(1, 12, 2),
              "instance_token": np.array(["i"]), "sample_token": np.array(["s"]),
              "scene_name": np.array(["scene"])}
    context = {"features": np.zeros((1, 11, 1, 2, 4), np.float32),
               "mask": np.zeros((1, 11, 1, 2), bool),
               "lane_tokens": np.full((1, 11, 1), "", dtype="U1"),
               "anchor_timestamp_us": np.zeros(1, np.int64)}
    return arrays, context, arrays["future_xy"].copy() + 1


def test_all_strategies_keep_rows_and_real_map_missing_fallback(tmp_path):
    torch.set_num_threads(1)
    arrays, context, base = inputs()
    model = MapCorrector(ResidualCorrector(hidden_dim=8), 8)
    frame = evaluate_context_arrays(arrays, context, base, model)
    assert len(frame) == 44
    reference = None
    for strategy, group in frame.groupby("strategy"):
        keys = list(zip(group.instance_token, group.sample_token, group.update_step, strict=True))
        if reference is None:
            reference = keys
        assert keys == reference
    original = frame[frame.strategy == "corrected"].reset_index(drop=True)
    mapped = frame[frame.strategy == "map_corrected"].reset_index(drop=True)
    np.testing.assert_array_equal(original[["ade", "fde", "jitter"]], mapped[["ade", "fde", "jitter"]])
    assert mapped.fallback.all()
    assert not mapped.map_covered.any()
    write_context_outputs(frame, tmp_path, {"purpose": "test"})
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "by_update.csv").exists()
    assert (tmp_path / "by_horizon.csv").exists()
    assert (tmp_path / "by_maneuver.csv").exists()


def test_actual_predictor_call_excludes_labels_and_future_mutation():
    arrays, context, base = inputs()
    model = MapCorrector(ResidualCorrector(hidden_dim=8), 8)
    received = []

    def spy(model, observation, base, map_view, device):
        assert isinstance(observation, ObservationView)
        assert isinstance(map_view, MapView)
        result = profiled_predict_offsets(model, observation, base, map_view, device)
        received.append((observation.observed_history.copy(), result[0].copy()))
        return result

    evaluate_context_arrays(arrays, context, base, model, predictor=spy)
    changed = {name: value.copy() for name, value in arrays.items()}
    changed["future_xy"][:, 1:] += 10000
    evaluate_context_arrays(changed, context, base, model, predictor=spy)
    np.testing.assert_array_equal(received[0][0], received[11][0])
    np.testing.assert_array_equal(received[0][1], received[11][1])

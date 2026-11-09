import json

import numpy as np
import pytest
from test_observation_stream import tracks


def test_sampling_is_log_capped_and_key_stable_under_row_reordering():
    from trajectory_prediction.robustness.benchmark import sample_indices
    arrays = {"instance_token": np.array(["i0", "i1", "i2", "i3"]),
              "sample_token": np.array(["s0", "s1", "s2", "s3"])}
    logs = np.array(["a", "a", "b", "b"])
    index = sample_indices(arrays, logs, "independent_fit", 1)
    assert len(index) == 2 and len(set(logs[index])) == 2
    order = np.array([3, 1, 0, 2])
    reordered = {k: v[order] for k, v in arrays.items()}
    other = sample_indices(reordered, logs[order], "independent_fit", 1)
    assert set(arrays["sample_token"][index]) == set(reordered["sample_token"][other])
    with pytest.raises(ValueError, match="development"):
        sample_indices(arrays, logs, "independent_holdout", 1)


def test_labels_score_but_cannot_modify_estimator_outputs():
    from trajectory_prediction.robustness.benchmark import score_result
    from trajectory_prediction.robustness.filters import run_packets
    from trajectory_prediction.robustness.packets import PROFILES, make_packets
    history, future = tracks()
    output = run_packets(make_packets(history, future[:11], "key", PROFILES["clean"], 2027), "kf_cv")
    before = output["forecasts"].copy()
    first = score_result(output, future)
    assert first["ade"] < 1e-10 and first["position_rmse"] < 1e-10
    changed = score_result(output, future + 100)
    assert changed["ade"] > 100
    np.testing.assert_array_equal(output["forecasts"], before)


def test_small_complete_benchmark_saves_replays_and_refuses_overwrite(tmp_path):
    from trajectory_prediction.data.cache import save_cache
    from trajectory_prediction.robustness.benchmark import ROLES, run, verify
    history, future = tracks()
    records = []
    for i, role in enumerate(ROLES):
        scene = "scene-" + str(i)
        records.append({"name": scene, "log_token": "log-" + str(i)})
        values = {"history_xy": history[None].astype(np.float32), "future_xy": future[None].astype(np.float32),
                  "history_features": np.zeros((1, 5, 6), np.float32), "history_mask": np.ones((1, 5), bool),
                  "anchor_xy": np.zeros((1, 2), np.float32), "anchor_yaw": np.zeros(1, np.float32),
                  "current_speed": np.ones(1, np.float32), "instance_token": np.array(["i" + str(i)]),
                  "sample_token": np.array(["s" + str(i)]), "scene_name": np.array([scene])}
        save_cache(tmp_path / "cache" / role, values, {"schema_version": 1, "split": role,
                   "preprocessing_config_sha256": "a" * 64, "partition_identity": "b" * 64})
    metadata = tmp_path / "scene.json"
    metadata.write_text(json.dumps(records))
    output = tmp_path / "experiment"
    result = run(tmp_path / "cache", metadata, output)
    assert result["rows"] == 216 and not result["holdout_evaluated"]
    assert verify(output)["prediction_spot_checks"] == 64
    with pytest.raises(FileExistsError):
        run(tmp_path / "cache", metadata, output)
    with (output / "per_trajectory.csv").open("a") as stream:
        stream.write("tampered\n")
    with pytest.raises(ValueError, match="checksum"):
        verify(output)

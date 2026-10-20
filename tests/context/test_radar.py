import json
import struct

import numpy as np
import pytest

from trajectory_prediction.context.radar import RadarConfig, aggregate_returns, transform_returns
from trajectory_prediction.context.radar_index import RadarIndex
from trajectory_prediction.online.observation_noise import CausalState


def test_velocity_uses_rotation_without_translations():
    points = np.zeros((18, 1))
    points[:3, 0], points[8:10, 0] = [1, 0, 0], [2, 0]
    points[10, 0] = 1
    sensor = {"rotation": [2 ** -.5, 0, 0, 2 ** -.5], "translation": [10, 0, 0]}
    pose = {"rotation": [1, 0, 0, 0], "translation": [100, 200, 0]}
    value = transform_returns(points, sensor, pose, np.array([100, 200]), np.pi / 2)
    np.testing.assert_allclose(value[0, :4], [10, 1, 0, 2], atol=1e-5)
    pose["translation"] = [1000, 2000, 0]
    again = transform_returns(points, sensor, pose, np.array([100, 200]), np.pi / 2)
    np.testing.assert_allclose(value[:, 2:4], again[:, 2:4], atol=1e-5)


def test_aggregation_gates_on_supplied_degraded_position_and_keeps_age():
    points = np.array([[0, 0, 1, 2, 4, 1, 1], [4, 0, 3, 4, 8, 0, 2], [10, 0, 9, 9, 6, 1, 1]], float)
    sources = np.array([1000000, 1500000, 2000000])
    channels = np.array([0, 1, 1])
    view = aggregate_returns(points, sources, channels, np.array([0, 0]), 2000000, RadarConfig())
    assert view.point_count == 2 and view.present
    np.testing.assert_allclose(view.features[3:5], [.2, .3])
    np.testing.assert_allclose(view.features[9:12], [.25, .375, .5])
    assert view.features[8] == .5
    changed = aggregate_returns(points, sources, channels, np.array([10, 0]), 2000000, RadarConfig())
    assert changed.point_count == 1
    empty = aggregate_returns(points, sources, channels, np.array([100, 0]), 2000000, RadarConfig())
    assert not empty.present and not empty.features.any()
    with pytest.raises(ValueError):
        aggregate_returns(points, sources + 2000000, channels, np.zeros(2), 2000000, RadarConfig())


def pcd(path, x=0):
    header = "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms\nSIZE 4 4 4 1 2 4 4 4 4 4 1 1 1 1 1 1 1 1\nTYPE F F F I I F F F F F I I I I I I I I\nCOUNT 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1\nWIDTH 1\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 1\nDATA binary\n"
    values = [x, 0, 0, 0, 1, 3, 1, 0, 1, 0, 1, 3, 0, 0, 0, 1, 0, 0]
    formats = ['f', 'f', 'f', 'b', 'h', 'f', 'f', 'f', 'f', 'f', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b']
    path.write_bytes(header.encode() + b''.join(struct.pack(fmt, value) for fmt, value in zip(formats, values, strict=True)) + b'\0')


def radar_fixture(root):
    version = root / "v1.0-mini"
    version.mkdir(parents=True)
    pcd(root / "old.pcd")
    pcd(root / "current.pcd")
    pose = {"rotation": [1, 0, 0, 0], "translation": [0, 0, 0]}
    tables = {"sample": [{"token": "sample", "scene_token": "sc", "timestamp": 2000000}],
        "scene": [{"token": "sc", "name": "scene"}],
        "sensor": [{"token": "radar", "modality": "radar", "channel": "RADAR_FRONT"}],
        "calibrated_sensor": [{"token": "cal", "sensor_token": "radar", **pose}],
        "ego_pose": [{"token": "p1", "timestamp": 1000000, **pose}, {"token": "p2", "timestamp": 2000000, **pose}, {"token": "p3", "timestamp": 4000000, **pose}],
        "sample_data": [{"token": name, "sample_token": "sample", "calibrated_sensor_token": "cal", "ego_pose_token": token, "timestamp": timestamp, "filename": filename}
                        for name, token, timestamp, filename in [("old", "p1", 1000000, "old.pcd"), ("current", "p2", 2000000, "current.pcd"), ("future", "p3", 4000000, "unread-future.pcd")]]}
    for name, rows in tables.items():
        (version / (name + ".json")).write_text(json.dumps(rows))
    return tables


def test_index_ignores_future_files_and_rejects_future_pose(tmp_path):
    radar_fixture(tmp_path)
    index = RadarIndex(tmp_path)
    state = CausalState(1, 2500000, 2500000, 2500000, np.zeros(2), np.zeros(2), True)
    view = index.query(state, "scene", np.zeros(2), np.pi / 2)
    assert view.frame_tokens == ("current", "old")
    assert view.point_count == 2
    assert set(index.used_files) == {"current.pcd", "old.pcd"}
    index.poses["p2"]["timestamp"] = 3000000
    with pytest.raises(ValueError, match="pose"):
        index.query(state, "scene", np.zeros(2), np.pi / 2)

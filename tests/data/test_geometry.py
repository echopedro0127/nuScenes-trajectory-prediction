from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from trajectory_prediction.data.geometry import (
    agent_to_global,
    build_motion_features,
    global_to_agent,
)
from trajectory_prediction.data.schema import TrajectorySample


def test_global_to_agent_uses_devkit_y_forward_convention() -> None:
    result = global_to_agent([[11, 5]], anchor_xy=[10, 5], anchor_yaw=0)

    np.testing.assert_allclose(result, [[0, 1]], atol=1e-6)


def test_agent_frame_transform_round_trip() -> None:
    points = np.array([[4, -2], [7, 8]], dtype=np.float64)
    anchor = np.array([2, 3], dtype=np.float64)

    agent_points = global_to_agent(points, anchor_xy=anchor, anchor_yaw=0.37)
    restored_points = agent_to_global(
        agent_points, anchor_xy=anchor, anchor_yaw=0.37
    )

    np.testing.assert_allclose(restored_points, points, atol=1e-5)


def test_build_motion_features_contains_xy_displacement_and_velocity() -> None:
    result = build_motion_features([[0, -2], [0, -1], [0, 0]], dt=0.5)

    assert result.shape == (3, 6)
    np.testing.assert_allclose(result[-1], [0, 0, 0, 1, 0, 2])


def test_trajectory_sample_is_frozen() -> None:
    sample = TrajectorySample(
        instance_token='instance',
        sample_token='sample',
        scene_name='scene',
        history_xy=np.zeros((3, 2), dtype=np.float32),
        history_features=np.zeros((3, 6), dtype=np.float32),
        history_mask=np.ones(3, dtype=np.bool_),
        future_xy=np.zeros((12, 2), dtype=np.float32),
        anchor_xy=np.zeros(2, dtype=np.float32),
        anchor_yaw=0.0,
        current_speed=0.0,
    )

    with pytest.raises(FrozenInstanceError):
        sample.current_speed = 1.0

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


@pytest.mark.parametrize(
    'history_xy',
    [[], np.empty((0, 2)), [0, 1], [[0, 1, 2]], [[[0, 1]]]],
)
def test_build_motion_features_rejects_invalid_history_shape(history_xy) -> None:
    with pytest.raises(
        ValueError, match=r'history_xy must have shape \[T, 2\] with T >= 1'
    ):
        build_motion_features(history_xy, dt=0.5)


@pytest.mark.parametrize('dt', [0, -0.5, np.nan, np.inf, -np.inf])
def test_build_motion_features_rejects_invalid_dt(dt: float) -> None:
    with pytest.raises(
        ValueError, match='dt must be finite and greater than zero'
    ):
        build_motion_features([[0, 0]], dt=dt)


@pytest.mark.parametrize('transform', [global_to_agent, agent_to_global])
def test_coordinate_transform_rejects_points_without_xy_dimension(transform) -> None:
    with pytest.raises(ValueError, match='points must have final dimension 2'):
        transform([[1, 2, 3]], anchor_xy=[0, 0], anchor_yaw=0)


@pytest.mark.parametrize('transform', [global_to_agent, agent_to_global])
def test_coordinate_transform_rejects_invalid_anchor_shape(transform) -> None:
    with pytest.raises(ValueError, match=r'anchor_xy must have shape \(2,\)'):
        transform([[1, 2]], anchor_xy=[[0, 0]], anchor_yaw=0)


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

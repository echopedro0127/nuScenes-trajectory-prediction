import numpy as np
import pytest

from trajectory_prediction.context.maps import LaneMap, MapConfig
from trajectory_prediction.online.observations import ObservationView


def observation(xy=(0, 0)):
    return ObservationView(1, 500_000, np.zeros((5, 2)), np.array(xy), ())


def test_anchor_rotation_and_direction_have_no_translation():
    lanes = LaneMap({"a": np.array([[11., 20., 0.], [13., 20., 0.]])}, MapConfig())
    view = lanes.query(observation((0, 1)), np.array([10., 20.]), 0.)
    np.testing.assert_allclose(view.features[0, :2], [[0, 1, 0, 1], [0, 3, 0, 1]], atol=1e-6)
    assert view.mask.sum() == 2
    assert view.lane_tokens[0] == "a"
    assert not view.mask[1:].any()


def test_query_is_causal_and_stably_sorted():
    config = MapConfig(radius_m=5., max_lanes=2, points_per_lane=3)
    lanes = {"z": np.array([[1, 0, 0]]), "a": np.array([[-1, 0, 0]]),
             "far": np.array([[100, 0, 0]])}
    first = LaneMap(lanes, config).query(observation(), np.zeros(2), np.pi / 2)
    other = LaneMap(dict(reversed(list(lanes.items()))), config).query(
        observation(), np.zeros(2), np.pi / 2)
    assert first.lane_tokens == ("a", "z")
    np.testing.assert_array_equal(first.features, other.features)
    with pytest.raises(ValueError, match="read-only"):
        first.features[0, 0, 0] = 1


def test_missing_lanes_and_contiguous_local_crop():
    config = MapConfig(radius_m=2., max_lanes=1, points_per_lane=3)
    lanes = LaneMap({"lane": np.column_stack((np.arange(10), np.zeros((10, 2))))}, config)
    view = lanes.query(observation((7, 0)), np.zeros(2), np.pi / 2)
    np.testing.assert_allclose(view.features[0, :, 0], [6, 7, 8])
    assert not lanes.query(observation((100, 100)), np.zeros(2), 0).mask.any()
    assert not LaneMap({}, config).query(observation(), np.zeros(2), 0).mask.any()


@pytest.mark.parametrize("kwargs", [{"radius_m": 0}, {"resolution_m": float("nan")},
                                   {"max_lanes": True}, {"points_per_lane": 0}])
def test_invalid_config(kwargs):
    with pytest.raises(ValueError):
        MapConfig(**kwargs)

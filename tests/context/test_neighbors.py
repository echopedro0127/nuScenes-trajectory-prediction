from dataclasses import replace

import numpy as np
import pytest

from trajectory_prediction.context.neighbors import NeighborConfig, NeighborIndex
from trajectory_prediction.online.observations import ObservationView


def observation(timestamp=2_000_000, position=(0, 0)):
    return ObservationView(1, timestamp, np.zeros((5, 2)), np.array(position), ())


def index(extra=(), config=None):
    samples = [{"token": str(k), "timestamp": k * 500_000, "scene_token": "sc"} for k in range(7)]
    annotations = []
    for k in range(7):
        for token, xy in (("self", (0, 0)), ("a", (k, 0)), ("b", (-k, 0)), ("cone", (0, 0))):
            annotations.append({"token": f"{token}-{k}", "sample_token": str(k),
                                "instance_token": token, "translation": [*xy, 0]})
    return NeighborIndex(samples, annotations + list(extra),
        {"self": "vehicle.car", "a": "vehicle.car", "b": "human.pedestrian.adult", "cone": "movable_object.trafficcone"},
        {"sc": "scene"}, config)


def test_target_exclusion_category_stable_sort_and_backward_velocity():
    view = index().query(observation(), "self", "scene", np.zeros(2), np.pi / 2)
    assert view.instance_tokens[:2] == ("a", "b")
    np.testing.assert_allclose(view.features[0, :, 0], np.arange(5))
    np.testing.assert_allclose(view.features[0, 1:, 2:4], [[2, 0]] * 4)
    assert not view.velocity_mask[0, 0]
    assert view.velocity_mask[0, 1:].all()
    np.testing.assert_array_equal(view.features[0, :, 4:], [[1, 0]] * 5)
    np.testing.assert_array_equal(view.features[1, :, 4:], [[0, 1]] * 5)
    assert not view.mask[2:].any()
    with pytest.raises(ValueError, match="read-only"):
        view.features[0, 0, 0] = 1


def test_rotation_relative_position_uses_current_target_and_no_velocity_translation():
    view = index().query(observation(position=(0, 1)), "self", "scene", np.zeros(2), 0.)
    np.testing.assert_allclose(view.features[0, -1, :4], [0, 3, 0, 2], atol=1e-6)


def test_future_annotation_perturbation_and_removal_cannot_change_input():
    first = index()
    other = index()
    for token, annotations in other.annotations.items():
        if int(token) > 4:
            annotations.clear()
    a = first.query(observation(), "self", "scene", np.zeros(2), np.pi / 2)
    b = other.query(observation(), "self", "scene", np.zeros(2), np.pi / 2)
    np.testing.assert_array_equal(a.features, b.features)
    np.testing.assert_array_equal(a.mask, b.mask)


def test_disappeared_agent_history_gap_and_latency_are_masked():
    data = index()
    del data.annotations["4"]["a"]
    view = data.query(observation(), "self", "scene", np.zeros(2), np.pi / 2)
    assert view.instance_tokens[0] == "b" and "a" not in view.instance_tokens
    data = index()
    del data.annotations["2"]["a"]
    view = data.query(observation(), "self", "scene", np.zeros(2), np.pi / 2)
    assert not view.mask[0, 2] and not view.velocity_mask[0, 3]
    delayed = index(config=NeighborConfig(latency_us=500_000)).query(
        observation(), "self", "scene", np.zeros(2), np.pi / 2)
    assert delayed.source_timestamp_us[0, -1] == 1_500_000
    assert delayed.available_timestamp_us[0, -1] == 2_000_000
    assert not index().query(observation(4_000_001), "self", "scene", np.zeros(2), 0).mask.any()


def test_velocity_uses_real_timestamp_difference_and_short_history_mask():
    data = index()
    data.times["scene"][1] = 400_000
    data.samples["1"]["timestamp"] = 400_000
    view = data.query(observation(), "self", "scene", np.zeros(2), np.pi / 2)
    assert view.mask[0].all()
    np.testing.assert_allclose(view.features[0, 1, 2], 2.5)
    np.testing.assert_allclose(view.features[0, 2, 2], 1 / 0.6)
    short = data.query(observation(0), "self", "scene", np.zeros(2), np.pi / 2)
    assert short.mask[:, -1].sum() == 2 and not short.mask[:, :-1].any()


@pytest.mark.parametrize("kwargs", [{"max_neighbors": 0}, {"max_neighbors": 9},
    {"radius_m": float("nan")}, {"latency_us": -1}, {"max_age_us": True}])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        NeighborConfig(**kwargs)


def test_view_rejects_future_source_or_invalid_velocity_mask():
    view = index().query(observation(), "self", "scene", np.zeros(2), np.pi / 2)
    times = view.source_timestamp_us.copy()
    times[0, -1] += 1
    with pytest.raises(ValueError):
        replace(view, source_timestamp_us=times)
    velocity_mask = view.velocity_mask.copy()
    velocity_mask[0, 0] = True
    with pytest.raises(ValueError):
        replace(view, velocity_mask=velocity_mask)

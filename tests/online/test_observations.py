from dataclasses import fields

import numpy as np
import pytest

from trajectory_prediction.context.schema import FrameRef
from trajectory_prediction.online.observations import (
    EvaluationLabel,
    ObservationView,
    replay_observations,
)
from trajectory_prediction.online.replay import replay_updates


def inputs():
    history = np.zeros((5, 2), dtype=np.float32)
    future = np.arange(24, dtype=np.float32).reshape(12, 2)
    return history, future


def test_future_mutation_cannot_change_current_observation():
    history, future = inputs()
    changed = future.copy()
    changed[1:] += 10000
    first, _ = next(replay_observations(history, future, anchor_timestamp_us=1_000_000))
    second, _ = next(replay_observations(history, changed, anchor_timestamp_us=1_000_000))
    np.testing.assert_array_equal(first.observed_history, second.observed_history)
    np.testing.assert_array_equal(first.observed_position, second.observed_position)
    assert first.timestamp_us == 1_500_000
    assert all("target" not in field.name and "future" not in field.name for field in fields(first))
    with pytest.raises(ValueError, match="read-only"):
        first.observed_history[0, 0] = 1


def test_adapter_preserves_old_replay_and_separates_labels():
    history, future = inputs()
    old = list(replay_updates(history, future))
    new = list(replay_observations(history, future, anchor_timestamp_us=1_000_000))
    assert len(old) == len(new) == 11
    for original, (observed, label) in zip(old, new, strict=True):
        assert observed.update_step == original.update_step
        np.testing.assert_array_equal(observed.observed_history, original.observed_history)
        np.testing.assert_array_equal(label.remaining_target, original.remaining_target)


def test_new_frames_are_released_only_when_available():
    history, future = inputs()
    frames = (
        FrameRef("old", "CAM_FRONT", 1_400_000, 1_400_000),
        FrameRef("delayed", "CAM_FRONT", 1_490_000, 1_600_000),
        FrameRef("future", "CAM_FRONT", 2_100_000, 2_100_000),
    )
    observed, _ = next(replay_observations(
        history, future, anchor_timestamp_us=1_000_000, frames=frames,
    ))
    assert tuple(item.token for item in observed.frames) == ("old",)



def test_every_update_hides_unrevealed_future_and_keeps_fixed_endpoint():
    history, future = inputs()
    for k, (observed, label) in enumerate(
        replay_observations(history, future, anchor_timestamp_us=0), start=1
    ):
        changed = future.copy()
        changed[k:] += 10000
        other, _ = list(replay_observations(history, changed, anchor_timestamp_us=0))[k - 1]
        np.testing.assert_array_equal(observed.observed_history, other.observed_history)
        np.testing.assert_array_equal(observed.observed_position, other.observed_position)
        np.testing.assert_array_equal(label.remaining_target, future[k:])
        assert label.remaining_target.shape == (12 - k, 2)
        assert observed.timestamp_us == k * 500_000


def test_direct_views_copy_inputs_and_expose_readonly_arrays():
    history, future = inputs()
    position = future[0].copy()
    original_history = history.copy()
    original_position = position.copy()
    original_target = future[1:].copy()
    frames = []
    observed = ObservationView(1, 500_000, history, position, frames)
    label = EvaluationLabel(future[1:])
    history[:] = -99
    position[:] = -99
    future[:] = -99
    frames.append(FrameRef("later", "CAM_FRONT", 600_000, 600_000))
    np.testing.assert_array_equal(observed.observed_history, original_history)
    np.testing.assert_array_equal(observed.observed_position, original_position)
    np.testing.assert_array_equal(label.remaining_target, original_target)
    assert observed.frames == ()
    for array in (observed.observed_history, observed.observed_position, label.remaining_target):
        assert array.dtype == np.float32
        with pytest.raises(ValueError, match="read-only"):
            array.flat[0] = 0


def test_replay_snapshots_survive_input_mutation():
    history, future = inputs()
    iterator = replay_observations(history, future, anchor_timestamp_us=0)
    observed, label = next(iterator)
    expected_history = observed.observed_history.copy()
    expected_target = label.remaining_target.copy()
    history[:] = -99
    future[:] = -99
    next(iterator)
    np.testing.assert_array_equal(observed.observed_history, expected_history)
    np.testing.assert_array_equal(label.remaining_target, expected_target)


def test_constructor_rejects_unavailable_frame():
    history, future = inputs()
    with pytest.raises(ValueError, match="unavailable"):
        ObservationView(1, 500_000, history, future[0], (FrameRef("late", "CAM", 0, 500_001),))


@pytest.mark.parametrize("step", [0, 12, True, 1.5])
def test_constructor_rejects_invalid_update_step(step):
    history, future = inputs()
    with pytest.raises(ValueError):
        ObservationView(step, 500_000, history, future[0], ())


@pytest.mark.parametrize("timestamp", [-1, True, 0.5])
def test_constructor_rejects_invalid_timestamp(timestamp):
    history, future = inputs()
    with pytest.raises(ValueError):
        ObservationView(1, timestamp, history, future[0], ())


@pytest.mark.parametrize("history", [np.zeros((0, 2)), np.zeros((2, 3)), np.zeros(2),
                                      np.full((5, 2), np.nan), np.full((5, 2), np.inf)])
def test_constructor_rejects_invalid_history(history):
    with pytest.raises(ValueError):
        ObservationView(1, 500_000, history, np.zeros(2), ())


@pytest.mark.parametrize("position", [np.zeros((1, 2)), np.zeros(3), [np.nan, 0], [0, np.inf]])
def test_constructor_rejects_invalid_position(position):
    with pytest.raises(ValueError):
        ObservationView(1, 500_000, np.zeros((5, 2)), position, ())


@pytest.mark.parametrize("target", [np.zeros((0, 2)), np.zeros((2, 3)), np.zeros(2),
                                     np.full((2, 2), np.nan), np.full((2, 2), np.inf)])
def test_label_rejects_invalid_target(target):
    with pytest.raises(ValueError):
        EvaluationLabel(target)


@pytest.mark.parametrize("steps", [2, 11, 13])
def test_adapter_rejects_nonstandard_horizon_but_legacy_replay_remains_general(steps):
    history, _ = inputs()
    future = np.ones((steps, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="12"):
        list(replay_observations(history, future, anchor_timestamp_us=0))
    assert len(list(replay_updates(history, future))) == steps - 1


@pytest.mark.parametrize("timestamp", [-1, True, 0.5])
def test_adapter_rejects_invalid_anchor(timestamp):
    history, future = inputs()
    with pytest.raises(ValueError):
        list(replay_observations(history, future, anchor_timestamp_us=timestamp))


def test_frame_removal_and_order_cannot_change_current_observation():
    history, future = inputs()
    current = (FrameRef("radar", "RADAR", 1_400_000, 1_400_000),
               FrameRef("camera", "CAM", 1_400_000, 1_500_000))
    hidden = (FrameRef("future", "CAM", 2_000_000, 2_000_000),
              FrameRef("delayed", "CAM", 1_490_000, 1_600_000))
    results = [next(replay_observations(history, future, anchor_timestamp_us=1_000_000,
                                      frames=frames))[0]
               for frames in (current, current + hidden, tuple(reversed(current + hidden)))]
    for observed in results:
        assert tuple(item.token for item in observed.frames) == ("camera", "radar")
        np.testing.assert_array_equal(observed.observed_history, results[0].observed_history)


def test_frames_release_at_later_updates_and_expire():
    history, future = inputs()
    frames = (FrameRef("a", "CAM", 0, 0), FrameRef("b", "CAM", 500_000, 1_000_000))
    updates = list(replay_observations(history, future, anchor_timestamp_us=0, frames=frames))
    assert updates[0][0].frames[0].token == "a"
    assert updates[1][0].frames[0].token == "b"
    assert updates[4][0].frames[0].token == "b"
    assert updates[5][0].frames == ()


def test_predictor_receives_only_observation_fields():
    received = []

    def predict(observed: ObservationView):
        assert isinstance(observed, ObservationView)
        received.append({field.name for field in fields(observed)})
        return observed.observed_position.copy()

    history, future = inputs()
    for observed, label in replay_observations(history, future, anchor_timestamp_us=0):
        result = predict(observed)
        assert isinstance(label, EvaluationLabel)
        assert result.shape == (2,)
    assert len(received) == 11
    assert all(names == {"update_step", "timestamp_us", "observed_history",
                         "observed_position", "frames"} for names in received)


def test_custom_history_window_preserves_legacy_observations():
    history, future = inputs()
    old = list(replay_updates(history, future, history_steps=3))
    new = list(replay_observations(history, future, anchor_timestamp_us=0, history_steps=3))
    for original, (observed, _) in zip(old, new, strict=True):
        np.testing.assert_array_equal(observed.observed_history, original.observed_history)

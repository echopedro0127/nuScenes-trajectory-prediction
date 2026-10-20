import numpy as np
import pytest

from trajectory_prediction.online.context_features import residual_features
from trajectory_prediction.online.observation_noise import (
    CONDITIONS,
    filtered_positions,
    replay_states,
    state_residual,
)
from trajectory_prediction.online.observations import replay_observations


def series():
    h = np.column_stack((np.arange(-4, 1, dtype=np.float32) * .5, np.zeros(5, np.float32)))
    f = np.column_stack((np.arange(1, 13, dtype=np.float32) * .5, np.zeros(12, np.float32)))
    return h, f


def test_delays_and_drops_extrapolate_using_unique_source_times():
    h, f = series()
    for condition in CONDITIONS:
        values = list(replay_states(h, f[:11], 2000000, "sample", condition))
        assert len(values) == 11
        if condition != "noise":
            for k, state in enumerate(values, 1):
                np.testing.assert_allclose(state.position, [k * .5, 0], atol=1e-6)
                np.testing.assert_allclose(state.velocity, [1, 0], atol=1e-6)
                assert state.source_us <= state.available_us <= state.timestamp_us
    delayed = list(replay_states(h, f[:11], 2000000, "sample", "delay_1000ms"))
    assert delayed[0].source_us == delayed[1].source_us == 2000000
    assert delayed[2].source_us == 2500000
    dropped = list(replay_states(h, f[:11], 2000000, "sample", "dropout"))
    assert [s.fresh for s in dropped[3:6]] == [False] * 3
    assert dropped[6].fresh


def test_noise_is_keyed_repeatable_and_prefix_causal():
    h, f = series()
    first = list(replay_states(h, f[:11], 2000000, "sample", "noise"))
    again = list(replay_states(h, f[:3], 2000000, "sample", "noise"))
    for a, b in zip(first, again, strict=False):
        np.testing.assert_array_equal(a.position, b.position)
        assert not a.position.flags.writeable
    changed = f.copy()
    changed[3:] += 10000
    second = list(replay_states(h, changed[:11], 2000000, "sample", "noise"))
    for a, b in zip(first[:3], second[:3], strict=True):
        np.testing.assert_array_equal(a.position, b.position)
    different = list(replay_states(h, f[:11], 2000000, "another", "noise"))
    assert not np.array_equal(first[0].position, different[0].position)


def test_clean_residual_and_alpha_beta_match_constant_velocity():
    h, f = series()
    states = list(replay_states(h, f[:11], 2000000, "sample", "clean"))
    observations = list(replay_observations(h, f, anchor_timestamp_us=2000000))
    for state, (observation, _) in zip(states, observations, strict=True):
        actual, mask = state_residual(state, f + .2)
        expected, expected_mask = residual_features(observation, f + .2)
        np.testing.assert_allclose(actual, expected, atol=1e-5)
        np.testing.assert_array_equal(mask, expected_mask)
    for condition in CONDITIONS:
        if condition == "noise":
            continue
        states = list(replay_states(h, f[:11], 2000000, "sample", condition))
        position, velocity = filtered_positions(h, 2000000, states)
        np.testing.assert_allclose(position, f[:11], atol=1e-5)
        np.testing.assert_allclose(velocity, np.tile([1, 0], (11, 1)), atol=1e-5)
    with pytest.raises(ValueError):
        list(replay_states(h, f[:11], 2000000, "sample", "bad"))

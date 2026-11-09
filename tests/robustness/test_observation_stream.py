import numpy as np
import pytest


def tracks():
    history = np.arange(-4, 1)[:, None] * np.array([[1., .5]])
    future = np.arange(1, 13)[:, None] * np.array([[1., .5]])
    return history, future


def test_noise_is_keyed_prefix_stable_and_applied_to_history():
    from trajectory_prediction.robustness.packets import PROFILES, make_packets
    history, future = tracks()
    packets = make_packets(history, future[:11], "key", PROFILES["noise_10"], 2027)
    again = make_packets(history, future[:5], "key", PROFILES["noise_10"], 2027)
    assert any(not np.array_equal(p.position, history[p.source_step + 4]) for p in packets if p.source_step <= 0)
    for a, b in zip(packets, again, strict=False):
        np.testing.assert_array_equal(a.position, b.position)
        assert not a.position.flags.writeable
    other = make_packets(history, future[:11], "other", PROFILES["noise_10"], 2027)
    assert not np.array_equal(other[0].position, packets[0].position)


@pytest.mark.parametrize("delay", [1, 2])
def test_delayed_history_also_arrives_late(delay):
    from trajectory_prediction.robustness.packets import Profile, make_packets
    history, future = tracks()
    packets = make_packets(history, future[:11], "key", Profile("test", 0., delay_steps=delay), 2027)
    assert max(p.source_step for p in packets if p.available_step <= 0) == -delay
    assert all(p.available_step == p.source_step + delay for p in packets)


@pytest.mark.parametrize("strategy", ["raw_cv", "alpha_beta", "kf_cv", "kf_nis"])
def test_future_observations_cannot_change_prefix_predictions(strategy):
    from trajectory_prediction.robustness.filters import run_packets
    from trajectory_prediction.robustness.packets import PROFILES, make_packets
    history, future = tracks()
    before = run_packets(make_packets(history, future[:11], "key", PROFILES["noise_delay_500ms"], 2027), strategy)
    future[5:] += 1000
    after = run_packets(make_packets(history, future[:11], "key", PROFILES["noise_delay_500ms"], 2027), strategy)
    np.testing.assert_array_equal(before["forecasts"][:5], after["forecasts"][:5])


def test_packets_reject_noncausal_availability():
    from trajectory_prediction.robustness.packets import Packet
    with pytest.raises(ValueError):
        Packet(2, 1, np.zeros(2))

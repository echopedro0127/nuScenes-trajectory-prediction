import numpy as np
import pytest
from test_observation_stream import tracks


@pytest.mark.parametrize("strategy", ["raw_cv", "alpha_beta", "kf_cv", "kf_nis"])
@pytest.mark.parametrize("delay", [0, 1, 2])
def test_clean_constant_velocity_and_delayed_source_clock_are_exact(strategy, delay):
    from trajectory_prediction.robustness.filters import run_packets
    from trajectory_prediction.robustness.packets import Profile, make_packets
    history, future = tracks()
    output = run_packets(make_packets(history, future[:11], "k", Profile("unit", 0., delay_steps=delay), 7), strategy)
    np.testing.assert_allclose(output["forecasts"], np.broadcast_to(future, (11, 12, 2)), atol=1e-10)


def test_missing_packets_do_not_repeat_measurement_updates():
    from trajectory_prediction.robustness.filters import run_packets
    from trajectory_prediction.robustness.packets import PROFILES, make_packets
    history, future = tracks()
    output = run_packets(make_packets(history, future[:11], "k", PROFILES["noise_dropout"], 7), "kf_cv")
    np.testing.assert_array_equal(output["fresh"][3:6], False)
    assert output["measurement_updates"] == 8
    assert all(np.linalg.eigvalsh(p).min() >= -1e-10 for p in output["position_covariances"])


def test_nis_inflation_reduces_single_outlier_displacement():
    from trajectory_prediction.robustness.filters import run_packets
    from trajectory_prediction.robustness.packets import Profile, make_packets
    history, future = tracks()
    packets = make_packets(history, future[:11], "k", Profile("outlier", 0., outlier_steps=(4,)), 7)
    standard, robust = (run_packets(packets, mode) for mode in ("kf_cv", "kf_nis"))
    assert robust["r_scales"][3] > 1
    assert np.linalg.norm(robust["positions"][3] - future[3]) < np.linalg.norm(standard["positions"][3] - future[3])


def test_source_reversal_is_rejected():
    from trajectory_prediction.robustness.filters import Estimator
    from trajectory_prediction.robustness.packets import Packet
    model = Estimator("kf_cv", Packet(-4, -4, np.zeros(2)), Packet(-3, -3, np.ones(2)))
    with pytest.raises(ValueError, match="source"):
        model.receive(Packet(-4, 1, np.zeros(2)))

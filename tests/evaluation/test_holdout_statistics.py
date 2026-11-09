import numpy as np
import pytest


def test_cluster_bootstrap_uses_pooled_counts_and_paired_log_draws():
    from trajectory_prediction.evaluation.holdout_statistics import paired_log_bootstrap
    values = np.array([-1., -1., -1., 2.])
    logs = np.array(["a", "a", "a", "b"])
    summary, draws = paired_log_bootstrap(values, logs, replicates=100, seed=11)
    sampled = np.random.Generator(np.random.PCG64(11)).integers(0, 2, size=(100, 2))
    expected = np.array([np.array([-3., 2.])[row].sum() / np.array([3, 1])[row].sum() for row in sampled])
    np.testing.assert_array_equal(draws, expected)
    assert summary["difference"] == -.25
    assert summary["log_count"] == 2
    assert summary["interval"] == np.quantile(expected, [.025, .975], method="linear").tolist()
    assert not summary["superior"]


def test_negative_cluster_interval_is_deterministic():
    from trajectory_prediction.evaluation.holdout_statistics import paired_log_bootstrap
    a, draws = paired_log_bootstrap(np.full(6, -.5), np.array(["a", "a", "b", "c", "c", "c"]))
    b, again = paired_log_bootstrap(np.full(6, -.5), np.array(["a", "a", "b", "c", "c", "c"]))
    assert a == b and a["superior"] and a["interval"] == [-.5, -.5]
    np.testing.assert_array_equal(draws, again)


@pytest.mark.parametrize("values,logs", [([1., np.nan], ["a", "b"]), ([1., 2.], ["a", "a"]), ([1., 2.], ["a", ""]), ([1.], ["a", "b"])])
def test_invalid_cluster_inputs_are_rejected(values, logs):
    from trajectory_prediction.evaluation.holdout_statistics import paired_log_bootstrap
    with pytest.raises(ValueError):
        paired_log_bootstrap(np.array(values), np.array(logs))

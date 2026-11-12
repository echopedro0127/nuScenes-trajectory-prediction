"""Same-target gains, explicit reduction units and tail/miss contracts."""
import math

import numpy as np
import pytest

from trajectory_prediction.evaluation.revision_metrics import (
    remaining_miss_flags,
    revision_gains,
    summarize_revisions,
    summarize_tail,
)


def test_gain_compares_same_target_in_previous_and_current_query():
    p, y = np.zeros((1, 11, 12, 2)), np.zeros((1, 12, 2))
    p[0, 0, 3, 0], p[0, 1, 3, 0], p[0, 2, 3, 0] = 3, 1, 80
    r = revision_gains(p, y, lead_steps=2)
    assert r['values'].shape == (1, 9) and r['values'][0, 0] == 2
    assert r['query_steps'].tolist() == list(range(2, 11))
    assert r['query_rows'][0] == 1 and r['previous_query_rows'][0] == 0
    assert r['target_columns'][0] == 3
    assert r['query_times_s'][0] == 1 and r['target_times_s'][0] == 2


@pytest.mark.parametrize('m,count', [(1, 10), (2, 9), (4, 7)])
def test_gain_matches_scalar_reference_for_every_valid_pair(m, count):
    rng = np.random.default_rng(66)
    p, y = rng.normal(size=(3, 11, 12, 2)), rng.normal(size=(3, 12, 2))
    r = revision_gains(p, y, lead_steps=m)
    expected = []
    for i in range(3):
        sample = []
        for k in range(2, 12):
            t = k * .5 + m * .5
            if t <= 6:
                j = round(t / .5) - 1
                a = math.hypot(*(float(p[i, k - 2, j, d]) - float(y[i, j, d]) for d in (0, 1)))
                b = math.hypot(*(float(p[i, k - 1, j, d]) - float(y[i, j, d]) for d in (0, 1)))
                sample.append(a - b)
        expected.append(sample)
    np.testing.assert_allclose(r['values'], expected, atol=1e-12, rtol=1e-12)
    assert r['values'].shape == (3, count)


def test_equal_thresholds_are_neutral_without_epsilon():
    p, y = np.zeros((1, 11, 12, 2)), np.zeros((1, 12, 2))
    # k=2..6, each pair targets a distinct column, so fixtures do not overlap.
    gains = [.05, -.05, 0., .0500000001, -.0500000001]
    for k, g in enumerate(gains, 2):
        j = k + 1
        p[0, k - 2 if g >= 0 else k - 1, j, 0] = abs(g)
    r = revision_gains(p, y, lead_steps=2)
    assert r['neutral'][0, :3].all()
    assert r['beneficial'][0, 3] and r['harmful'][0, 4]
    assert (r['beneficial'].astype(int) + r['harmful'] + r['neutral'] == 1).all()


def test_pooled_proportions_and_window_macro_are_distinct():
    gains = np.array([[1., 1.], [1., 1.], [-1., -1.]])
    result = summarize_revisions(gains, ['a', 'a', 'b'], [2027, 2028, 2027])
    assert result['pair_count'] == 6 and result['window_count'] == 2
    assert result['beneficial_fraction'] == 4 / 6
    assert result['window_macro_beneficial_fraction'] == .5
    assert result['mean_gain_m'] == 0  # Per-window repeated noises do not add weight.
    assert sum(result[k + '_count'] for k in ('beneficial', 'harmful', 'neutral')) == 6


def test_tail_quantile_level_and_miss_strictness():
    r = summarize_tail(np.array([[0., 0., 10.], [4., 4., 4.]]), ['a', 'b'], [2027, 2027])
    assert r['p95_query_error'] == 8.5
    assert r['p95_window_mean_error'] == pytest.approx(3.966666666666667)
    assert r['query_count'] == 6 and r['miss_count'] == 4
    boundary = summarize_tail(np.array([[2., 2.0000000001]]), ['a'], [2027])
    assert boundary['miss_fraction'] == .5


@pytest.mark.parametrize('function', [summarize_revisions, summarize_tail])
def test_empty_group_is_not_zero_success(function):
    r = function(np.empty((0, 9)), [], [])
    assert r['status'] == 'not_evaluable' and r['reason'] == 'empty_group'
    assert r['window_count'] == 0
    assert r.get('beneficial_fraction', r.get('miss_fraction')) is None


@pytest.mark.parametrize('function', [summarize_revisions, summarize_tail])
@pytest.mark.parametrize('kind', ['duplicate', 'nan', 'keys', 'integer_seed'])
def test_integrity_errors_are_not_discarded(function, kind):
    x, keys, seeds = np.ones((2, 2)), ['a', 'b'], [2027, 2027]
    if kind == 'duplicate':
        keys[1] = 'a'
    elif kind == 'nan':
        x[1, 0] = np.nan
    elif kind == 'keys':
        keys.pop()
    else:
        seeds[0] = True
    with pytest.raises(ValueError):
        function(x, keys, seeds)


def test_remaining_miss_excludes_already_observed_columns():
    p, y = np.zeros((1, 11, 12, 2)), np.zeros((1, 12, 2))
    p[0, 0, 0, 0] = 50  # query1, observed target at .5s, must not count.
    p[0, 1, 11, 0] = 2
    p[0, 2, 11, 0] = 2.000001
    r = remaining_miss_flags(p, y)
    assert r['miss'].tolist() == [[False, False, True] + [False] * 8]
    assert r['remaining_points'].tolist() == list(range(11, 0, -1))


@pytest.mark.parametrize('bad', [np.nan, np.inf, -.01, True])
def test_gain_tolerance_is_frozen(bad):
    with pytest.raises(ValueError):
        revision_gains(np.zeros((1, 11, 12, 2)), np.zeros((1, 12, 2)),
                       lead_steps=2, tolerance_m=bad)


@pytest.mark.parametrize('function', [summarize_revisions, summarize_tail])
def test_summary_never_exports_infinity_for_finite_extreme_inputs(function):
    with np.errstate(over='ignore', invalid='ignore'), pytest.raises(ValueError, match='range'):
        function(np.array([[1.7e308], [1.7e308]]), ['a', 'b'], [2027, 2027])


def test_negative_position_errors_are_rejected():
    with pytest.raises(ValueError, match='nonnegative'):
        summarize_tail(np.array([[-1.]]), ['a'], [2027])

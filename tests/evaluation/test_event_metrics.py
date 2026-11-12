"""Complete-pair identity, excess-error recovery semantics and denominators."""
import numpy as np
import pytest

from trajectory_prediction.evaluation.event_metrics import (
    pair_first_outlier,
    recovery_from_excess,
    summarize_recovery,
)


def paired_fixture():
    p = np.zeros((2, 11, 12, 2))
    y = np.zeros((2, 12, 2))
    records = [{'role': 'independent_select', 'instance': 'i', 'sample': 's', 'log': 'l',
                'scene': 'scene', 'row': 5, 'seed': 2027, 'condition': c}
               for c in ('noise_05', 'noise_outlier')]
    return p, y, records


def test_recovery_needs_sustained_confirmation_and_no_later_relapse():
    a = np.array([[1., .3, .2, .1], [1., .1, .1, .8], [.1, .8, .2, .1], [.1, .2, .1, .3]])
    r = recovery_from_excess(a, dt=.5, threshold_m=.5, consecutive_updates=2)
    assert r['recovered'].tolist() == [True, False, True, False]
    assert r['affected'].tolist() == [True, True, True, False]
    assert r['confirm_delay_s'][0] == 1 and r['confirm_delay_s'][2] == 1.5
    assert np.isnan(r['confirm_delay_s'][1]) and np.isnan(r['confirm_delay_s'][3])
    assert r['excess_auc'][0] == pytest.approx(.525)
    assert r['confirm_step'].tolist() == [6, -1, 7, -1]


@pytest.mark.parametrize('values,state,delay', [
    ([.5, .5, .5, .5], 'unaffected', None),
    ([1, 1, 1, .1], 'not_recovered', None),
    ([0, 0, .6, .1], 'not_recovered', None),
    ([1, .6, .1, .1], 'recovered', 1.5),
    ([1, .1, .6, .1], 'not_recovered', None),
    ([1, .5, .5, .5], 'recovered', 1.),
])
def test_recovery_boundary_states(values, state, delay):
    result = recovery_from_excess(np.array([values], float))
    assert result['status'][0] == state
    assert np.isnan(result['confirm_delay_s'][0]) if delay is None else result['confirm_delay_s'][0] == delay


def test_pairing_uses_identity_and_only_queries_four_to_seven():
    p, y, records = paired_fixture()
    p[1, 3:7, :, 0] = 2
    p[1, 7:, :, 0] = 1000
    r = pair_first_outlier(p, y, records, strategy_identity='model')
    np.testing.assert_array_equal(r['excess'], [[2., 2., 2., 2.]])
    assert r['query_steps'].tolist() == [4, 5, 6, 7]
    assert r['target_columns'].tolist() == [5, 6, 7, 8]
    assert r['reference_indices'] == [0] and r['outlier_indices'] == [1]
    flipped = pair_first_outlier(p[::-1], y[::-1], records[::-1], strategy_identity='model')
    assert np.array_equal(r['excess'], flipped['excess'])


def test_same_gt_error_does_not_prove_prediction_state_recovery():
    p, y, records = paired_fixture()
    p[0, ..., 0], p[1, ..., 1] = 3, 3
    r = pair_first_outlier(p, y, records, strategy_identity='model')
    assert not r['excess'].any()
    np.testing.assert_allclose(r['prediction_distance'], np.sqrt(18))
    np.testing.assert_array_equal(r['reference_errors'], 3.)
    np.testing.assert_array_equal(r['outlier_errors'], 3.)


@pytest.mark.parametrize('kind', ['missing', 'duplicate', 'gt', 'seed', 'role', 'log', 'row', 'model'])
def test_incomplete_or_mismatched_pairs_are_fatal(kind):
    p, y, records = paired_fixture()
    if kind == 'missing':
        p, y, records = p[:1], y[:1], records[:1]
    elif kind == 'duplicate':
        records[1] = dict(records[0])
    elif kind == 'gt':
        y[1, 0, 0] = 1
    elif kind == 'seed':
        records[1]['seed'] = 2028
    elif kind == 'role':
        records[1]['role'] = 'independent_fit'
    elif kind == 'model':
        records[1]['strategy_identity'] = 'another-model'
    else:
        records[1][kind] = 'wrong' if kind == 'log' else 9
    with pytest.raises(ValueError):
        pair_first_outlier(p, y, records, strategy_identity='model')


def test_auc_includes_unaffected_events_and_rates_use_affected_denominator():
    a = np.array([[1., .1, .1, .1], [1., 1., 1., 1.], [.1, .1, .1, .1]])
    result = summarize_recovery(a, ['a', 'b', 'c'], [2027] * 3)
    assert result['eligible'] == 3 and result['affected'] == 2
    assert result['recovered'] == 1 and result['not_recovered'] == 1
    assert result['recovered_fraction_of_affected'] == .5
    assert result['unaffected_fraction_of_eligible'] == 1 / 3
    assert result['mean_excess_auc_m_s'] == pytest.approx((.375 + 1.5 + .15) / 3)
    assert result['mean_confirm_delay_s_recovered_events'] == 1.


def test_unaffected_only_and_empty_groups_have_null_conditional_rates():
    for a, keys, seeds in [(np.zeros((2, 4)), ['a', 'b'], [2027, 2027]),
                           (np.empty((0, 4)), [], [])]:
        r = summarize_recovery(a, keys, seeds)
        assert r['recovered_fraction_of_affected'] is None
        assert r['mean_confirm_delay_s_recovered_events'] is None
        assert r['eligible'] == len(a)


def test_noise_repeats_do_not_add_auc_window_weight():
    a = np.array([[1., 1., 1., 1.], [1., 1., 1., 1.], [0., 0., 0., 0.]])
    r = summarize_recovery(a, ['a', 'a', 'b'], [2027, 2028, 2027])
    assert r['mean_excess_auc_m_s'] == .75
    with pytest.raises(ValueError, match='duplicate'):
        summarize_recovery(a, ['a', 'a', 'b'], [2027, 2027, 2027])


@pytest.mark.parametrize('kind', ['negative', 'nan', 'short', 'long'])
def test_bad_event_arrays_are_not_censored(kind):
    a = np.ones((1, 4))
    if kind == 'negative':
        a[0, 0] = -1
    elif kind == 'nan':
        a[0, 0] = np.nan
    else:
        a = np.ones((1, 3 if kind == 'short' else 5))
    with pytest.raises(ValueError):
        recovery_from_excess(a)


@pytest.mark.parametrize('kwargs', [{'dt': 1.}, {'threshold_m': .4}, {'consecutive_updates': 1}])
def test_v1_recovery_protocol_is_frozen(kwargs):
    with pytest.raises(ValueError):
        recovery_from_excess(np.ones((1, 4)), **kwargs)


def test_representable_auc_does_not_overflow_intermediate_addition():
    r = recovery_from_excess(np.array([[1.7e308, 1.7e308, 0., 0.]]))
    assert r['excess_auc'][0] == pytest.approx(1.275e308)

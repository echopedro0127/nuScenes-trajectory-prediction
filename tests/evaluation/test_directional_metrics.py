"""Common GT frame, past-only headings and independent validity contracts."""
import numpy as np
import pytest

from trajectory_prediction.evaluation.directional_metrics import (
    directional_errors,
    gt_reference_directions,
    summarize_directional,
)


def straight():
    history = np.stack((np.arange(-4, 1) * .5, np.zeros(5)), axis=-1)[None]
    future = np.stack((np.arange(1, 13) * .5, np.zeros(12)), axis=-1)[None]
    return history, future, np.ones((1, 5), bool)


def test_current_gt_frame_projects_known_error_and_preserves_norm():
    h, y, mask = straight()
    p = np.repeat(y[:, None], 11, axis=1) + [3., 4.]
    r = directional_errors(p, y, h, mask)
    assert r['valid_mask'].shape == (1, 10) and r['valid_mask'].all()
    np.testing.assert_array_equal(r['longitudinal_signed_m'], 3)
    np.testing.assert_array_equal(r['lateral_signed_m'], 4)
    np.testing.assert_allclose(np.hypot(r['longitudinal_signed_m'], r['lateral_signed_m']), 5)
    assert r['query_steps'].tolist() == list(range(1, 11))
    assert r['target_columns'].tolist() == list(range(2, 12))


def test_no_future_direction_is_used_to_fill_stationary_prefix():
    h, y, mask = np.zeros((1, 5, 2)), np.zeros((1, 12, 2)), np.ones((1, 5), bool)
    y[0, 7:, 0] = np.arange(1, 6)
    result = gt_reference_directions(h, y, mask)
    assert not result['valid_mask'][0, :7].any()
    assert result['valid_mask'][0, 7:].all()
    changed = y.copy()
    changed[0, 7:, 1] = 100
    other = gt_reference_directions(h, changed, mask)
    assert np.array_equal(result['directions'][:, :7], other['directions'][:, :7])


def test_low_speed_carries_only_past_last_reliable_heading():
    h, y, mask = straight()
    y[:] = 0
    r = gt_reference_directions(h, y, mask)
    assert r['valid_mask'].all()
    assert not r['fresh_mask'].any()
    np.testing.assert_array_equal(r['source_steps'], 0)
    np.testing.assert_array_equal(r['directions'], np.broadcast_to([1., 0.], (1, 11, 2)))


def test_padded_history_points_do_not_provide_heading():
    h, y, mask = straight()
    mask[:, :4] = False
    h[:, :4] = 100
    y[:] = 0
    r = gt_reference_directions(h, y, mask)
    assert not r['valid_mask'].any()


def test_exact_half_meter_per_second_speed_is_reliable():
    h, y, mask = straight()
    h *= .5
    y *= .5
    assert gt_reference_directions(h, y, mask)['fresh_mask'].all()


def test_stationary_invalid_does_not_change_non_directional_metrics():
    from trajectory_prediction.evaluation.streaming_metrics import fixed_lead_errors
    h, y, mask = np.zeros((1, 5, 2)), np.zeros((1, 12, 2)), np.ones((1, 5), bool)
    p = np.ones((1, 11, 12, 2))
    r = directional_errors(p, y, h, mask)
    assert not r['valid_mask'].any()
    assert np.isnan(r['longitudinal_signed_m']).all()
    assert np.isfinite(fixed_lead_errors(p, y, lead_steps=2)['values']).all()


def test_translation_rotation_and_model_choice_do_not_change_gt_frame():
    h, y, mask = straight()
    p = np.repeat(y[:, None], 11, axis=1) + [3., -4.]
    a = directional_errors(p, y, h, mask)
    theta = .9
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    b = directional_errors(p @ rotation.T + 17, y @ rotation.T + 17, h @ rotation.T + 17, mask)
    np.testing.assert_allclose(a['longitudinal_signed_m'], b['longitudinal_signed_m'], atol=1e-12)
    np.testing.assert_allclose(a['lateral_signed_m'], b['lateral_signed_m'], atol=1e-12)
    c = directional_errors(p * 3, y, h, mask)
    assert np.array_equal(a['reference_directions'], c['reference_directions'])


def test_direction_wrap_does_not_invert_normal_or_read_future_heading():
    # Moving almost west on either side of +/-pi remains a valid current frame.
    h, y, mask = straight()
    h *= -1
    y *= -1
    y[0, :, 1] = np.tile([1e-8, -1e-8], 6)
    p = np.repeat(y[:, None], 11, axis=1)
    p[..., 1] += 2
    r = directional_errors(p, y, h, mask)
    assert (r['lateral_signed_m'] < -1.99).all()


def test_summary_reduces_queries_repeats_then_windows_and_counts_invalid():
    long = np.array([[2., np.nan], [4., 4.], [-1., -1.], [np.nan, np.nan]])
    lateral = np.array([[3., np.nan], [1., 1.], [2., 2.], [np.nan, np.nan]])
    valid = np.isfinite(long)
    r = summarize_directional(long, lateral, valid, ['a', 'a', 'b', 'c'], [2027, 2028, 2027, 2027])
    assert r['mean_longitudinal_abs_m'] == 2  # window a=3, b=1; c excluded only here.
    assert r['mean_longitudinal_signed_m'] == 1
    assert r['mean_lateral_abs_m'] == 2
    assert r['valid_query_count'] == 5 and r['invalid_query_count'] == 3
    assert r['evaluable_window_count'] == 2 and r['window_count'] == 3


def test_no_evaluable_heading_returns_null_not_zero():
    r = summarize_directional(np.full((1, 10), np.nan), np.full((1, 10), np.nan),
                              np.zeros((1, 10), bool), ['a'], [2027])
    assert r['mean_longitudinal_abs_m'] is None and r['status'] == 'not_evaluable'
    assert r['reason'] == 'no_valid_heading'


@pytest.mark.parametrize('kind', ['history_shape', 'mask_dtype', 'nan', 'future_shape'])
def test_gt_contract_rejects_integrity_errors(kind):
    h, y, mask = straight()
    if kind == 'history_shape':
        h = h[:, :4]
    elif kind == 'mask_dtype':
        mask = mask.astype(int)
    elif kind == 'nan':
        h[0, 0, 0] = np.nan
    else:
        y = y[:, :11]
    with pytest.raises(ValueError):
        gt_reference_directions(h, y, mask)


@pytest.mark.parametrize('kind', ['duplicate', 'valid_nan', 'infinity', 'mask_dtype'])
def test_summary_does_not_hide_corruption_behind_invalid_mask(kind):
    x, mask, keys = np.ones((2, 10)), np.ones((2, 10), bool), ['a', 'b']
    if kind == 'duplicate':
        keys[1] = 'a'
    elif kind == 'valid_nan':
        x[0, 0] = np.nan
    elif kind == 'infinity':
        x[0, 0], mask[0, 0] = np.inf, False
    else:
        mask = mask.astype(int)
    with pytest.raises(ValueError):
        summarize_directional(x, x, mask, keys, [2027, 2027])


def test_last_future_point_cannot_affect_any_current_reference_direction():
    h, y, mask = straight()
    original = gt_reference_directions(h, y, mask)
    y[0, 11] = [-1000, 800]
    changed = gt_reference_directions(h, y, mask)
    for name in original:
        assert np.array_equal(original[name], changed[name])


def test_summary_empty_group_and_overflow_have_distinct_states():
    empty = np.empty((0, 10))
    r = summarize_directional(empty, empty, np.zeros((0, 10), bool), [], [])
    assert r['reason'] == 'empty_group' and r['heading_coverage'] is None
    x = np.full((2, 1), 1.7e308)
    with pytest.raises(ValueError, match='range'):
        summarize_directional(x, x, np.ones((2, 1), bool), ['a', 'b'], [2027, 2027])


def test_reverse_motion_uses_velocity_direction_rather_than_fixed_agent_yaw():
    h, y, mask = straight()
    y *= -1
    p = np.repeat(y[:, None], 11, axis=1) + [3., 4.]
    result = directional_errors(p, y, h, mask)
    np.testing.assert_allclose(result['longitudinal_signed_m'], -3.)
    np.testing.assert_allclose(result['lateral_signed_m'], -4.)

"""Independent timing and scalar references for fixed-lead errors."""
import math

import numpy as np
import pytest

from trajectory_prediction.evaluation.streaming_metrics import fixed_lead_errors


def scalar_reference(p, y, lead_steps):
    rows = []
    for sample in range(len(y)):
        errors = []
        for query_step in range(1, 12):
            target_time = query_step * .5 + lead_steps * .5
            if target_time > 6:
                continue
            target_column = round(target_time / .5) - 1
            x = float(p[sample, query_step - 1, target_column, 0])
            z = float(p[sample, query_step - 1, target_column, 1])
            errors.append(math.hypot(x - float(y[sample, target_column, 0]),
                                     z - float(y[sample, target_column, 1])))
        rows.append(errors)
    return np.array(rows, np.float64)


def test_one_second_lead_uses_absolute_target_column():
    p = np.zeros((1, 11, 12, 2), np.float32)
    y = np.zeros((1, 12, 2), np.float32)
    p[0, 0, 2] = [3, 4]
    p[0, 1, 2] = [90, 90]  # A later query must not be read.
    r = fixed_lead_errors(p, y, lead_steps=2)
    assert r['values'].shape == (1, 10)
    assert r['values'][0, 0] == 5
    assert r['query_steps'].tolist() == list(range(1, 11))
    assert r['query_rows'].tolist() == list(range(10))
    assert r['target_columns'].tolist() == list(range(2, 12))
    assert r['query_times_s'][0] == .5 and r['target_times_s'][0] == 1.5


@pytest.mark.parametrize('lead,count', [(1, 11), (2, 10), (4, 8)])
@pytest.mark.parametrize('dtype', [np.float32, np.float64])
def test_all_leads_match_independent_scalar_reference(lead, count, dtype):
    rng = np.random.default_rng(502)
    p = rng.normal(size=(7, 11, 12, 2)).astype(dtype)
    y = rng.normal(size=(7, 12, 2)).astype(dtype)
    before = p.copy()
    r = fixed_lead_errors(p, y, lead_steps=lead)
    np.testing.assert_allclose(r['values'], scalar_reference(p, y, lead), rtol=1e-14, atol=1e-14)
    assert r['values'].dtype == np.float64
    assert r['valid_mask'].shape == (7, count) and r['valid_mask'].all()
    assert np.array_equal(r['target_times_s'] - r['query_times_s'], lead * .5 * np.ones(count))
    assert np.array_equal(p, before)
    assert r['excluded_query_steps'].tolist() == list(range(count + 1, 12))


def test_excluded_tail_is_not_zero_padded_or_extrapolated():
    p = np.ones((1, 11, 12, 2), np.float32)
    y = np.zeros((1, 12, 2), np.float32)
    r = fixed_lead_errors(p, y, lead_steps=4)
    assert r['values'].shape == (1, 8)
    np.testing.assert_allclose(r['values'], math.sqrt(2))
    assert r['target_columns'][-1] == 11


@pytest.mark.parametrize('lead', [0, -1, 3, 12, 2., True, None])
def test_invalid_lead_is_rejected(lead):
    with pytest.raises(ValueError, match='lead'):
        fixed_lead_errors(np.zeros((1, 11, 12, 2)), np.zeros((1, 12, 2)), lead_steps=lead)


@pytest.mark.parametrize('kind', ['shape', 'target_shape', 'empty', 'nan', 'inf', 'integer'])
def test_invalid_forecast_or_gt_is_not_silently_dropped(kind):
    p = np.zeros((1, 11, 12, 2), np.float32)
    y = np.zeros((1, 12, 2), np.float32)
    if kind == 'shape':
        p = p[:, :10]
    elif kind == 'target_shape':
        y = y[:, :11]
    elif kind == 'empty':
        p, y = p[:0], y[:0]
    elif kind == 'nan':
        p[0, 10, 0, 0] = np.nan  # Even an unscored slot violates the input contract.
    elif kind == 'inf':
        y[0, 0, 0] = np.inf
    else:
        p = p.astype('int64')
    with pytest.raises(ValueError):
        fixed_lead_errors(p, y, lead_steps=2)


def test_float32_large_coordinates_are_subtracted_in_float64():
    p = np.full((1, 11, 12, 2), 2e38, np.float32)
    y = np.full((1, 12, 2), -2e38, np.float32)
    r = fixed_lead_errors(p, y, lead_steps=2)
    assert np.isfinite(r['values']).all()


def test_unrepresentable_float64_error_is_a_validation_failure():
    p = np.full((1, 11, 12, 2), 1.7e308)
    y = np.full((1, 12, 2), -1.7e308)
    with np.errstate(over='raise', invalid='raise'), pytest.raises(ValueError, match='range'):
        fixed_lead_errors(p, y, lead_steps=2)

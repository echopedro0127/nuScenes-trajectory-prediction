import copy

import numpy as np
import pytest
from test_physics_clean_metrics import metric_fixture

from trajectory_prediction.evaluation.physics_clean_metrics import summarize_errors


def records():
    windows,shared,errors=metric_fixture()
    rows=[]
    for seed,multiplier in zip((2027,2028,2029),(1.,2.,3.),strict=True):
        rows.append({'seed':seed,**summarize_errors(errors*multiplier,shared,windows)})
    return windows,shared,rows


def test_seed_then_window_reduction_and_not_pooled_tail():
    from trajectory_prediction.evaluation.gaussian_metrics import aggregate_scores
    windows,shared,rows=records();result=aggregate_scores(rows,shared,windows)
    group=next(g for g in result['groups'] if g['role']=='independent_fit' and g['log']=='__all__' and g['lead_us']==500000)
    assert group['mean_window_error_m']==pytest.approx(20/3)
    assert group['noise_seed_std_window_macro_m']==pytest.approx(10/3)
    assert group['query_p95_error_m_seed_mean']==pytest.approx(14)
    assert group['valid_queries_per_seed']==7 and group['repeated_query_evaluations']==21
    assert group['log_macro_mean_error_m']==5
    assert group['query_error_gt2m_count_over_repeats']==3
    empty=next(g for g in result['groups'] if g['role']=='independent_fit' and g['log']=='__all__' and g['lead_us']==2000000)
    assert empty['mean_window_error_m'] is None and empty['query_p95_error_m_seed_mean'] is None
    assert np.array_equal(result['arrays']['mean_error_over_noise_seeds'],aggregate_scores(rows[::-1],shared,windows)['arrays']['mean_error_over_noise_seeds'])


@pytest.mark.parametrize('kind',['missing','duplicate','mask','nan','query_count','group_missing'])
def test_bad_repeat_records_rejected(kind):
    from trajectory_prediction.evaluation.gaussian_metrics import aggregate_scores
    windows,shared,rows=records();rows=copy.deepcopy(rows)
    if kind=='missing':rows.pop()
    elif kind=='duplicate':rows[2]['seed']=2027
    elif kind=='mask':rows[1]['window_metrics']['valid_windows'][0,0]=False
    elif kind=='nan':rows[0]['window_metrics']['window_mean_error_m'][0,0]=np.nan
    elif kind=='query_count':rows[0]['window_metrics']['query_counts'][0,0]+=1
    else:rows[0]['groups'].pop()
    with pytest.raises(ValueError):aggregate_scores(rows,shared,windows)


def test_signed_clean_degradation_is_not_clipped():
    from trajectory_prediction.evaluation.gaussian_metrics import aggregate_differences
    windows,shared,_=records();valid=shared['valid'].any(axis=1)
    rows=[{'seed':seed,'values':np.where(valid,-i,0.).astype(np.float64)} for i,seed in enumerate((2027,2028,2029),1)]
    result=aggregate_differences(rows,shared,windows,'mean_window_extra_error_m')
    group=next(g for g in result['groups'] if g['role']=='independent_fit' and g['log']=='__all__' and g['lead_us']==500000)
    assert group['mean_window_extra_error_m']==-2
    assert group['noise_seed_std_window_macro_m']==1

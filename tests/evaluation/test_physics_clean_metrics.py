import numpy as np
import pytest


def metric_fixture():
    windows=[{'role':'independent_fit','log':log,'instance':str(i),'tokens':[str(i)+'t'+str(j) for j in range(17)]}
             for i,log in enumerate(('a','a','b'))]
    valid=np.zeros((3,11,3),bool);valid[0,0,:2]=True;valid[1:,:3,:2]=True
    errors=np.zeros(valid.shape,np.float64);errors[0,0,:2]=10
    shared={'valid':valid,'actual_lead_us':np.broadcast_to(np.array([500000,1000000,2000000]),valid.shape).copy(),
            'target_index':np.broadcast_to(np.array([6,7,9],np.int16),valid.shape).copy()}
    return windows,shared,errors


def test_window_log_query_reductions_and_empty_lead():
    from trajectory_prediction.evaluation.physics_clean_metrics import summarize_errors
    windows,shared,errors=metric_fixture()
    result=summarize_errors(errors,shared,windows)
    rows=result['groups'];main=next(g for g in rows if g['role']=='independent_fit' and g['log']=='__all__' and g['lead_us']==500000)
    assert main['mean_window_error_m']==pytest.approx(10/3)
    assert main['log_macro_mean_error_m']==2.5
    assert main['valid_queries']==7 and main['valid_windows']==3
    assert main['query_p50_error_m']==0 and main['query_p95_error_m']==pytest.approx(7)
    assert main['query_error_gt2m_fraction']==pytest.approx(1/7)
    empty=next(g for g in rows if g['role']=='independent_fit' and g['log']=='__all__' and g['lead_us']==2000000)
    assert empty['mean_window_error_m'] is None and empty['reason']=='no_valid_queries'
    assert not result['window_metrics']['valid_windows'][:,2].any()


def test_paired_gains_use_same_mask_and_positive_means_compared_better():
    from trajectory_prediction.evaluation.physics_clean_metrics import paired_gains
    windows,shared,errors=metric_fixture()
    results={'a':{'errors':errors,'valid':shared['valid']},'b':{'errors':errors*.5,'valid':shared['valid'].copy()}}
    result=paired_gains(results,shared,windows,[('comparison','a','b')])
    main=next(g for g in result['groups'] if g['role']=='independent_fit' and g['log']=='__all__' and g['lead_us']==500000)
    assert main['mean_window_gain_m']==pytest.approx(5/3)
    results['b']['valid'][0,0,0]=False
    with pytest.raises(ValueError):paired_gains(results,shared,windows,[('comparison','a','b')])


@pytest.mark.parametrize('kind',['nan','negative','invalid_nonzero','dtype'])
def test_invalid_error_arrays_fail(kind):
    from trajectory_prediction.evaluation.physics_clean_metrics import summarize_errors
    windows,shared,errors=metric_fixture()
    if kind=='nan':errors[0,0,0]=np.nan
    elif kind=='negative':errors[0,0,0]=-1
    elif kind=='invalid_nonzero':errors[0,10,0]=1
    else:errors=errors.astype(np.float32)
    with pytest.raises(ValueError):summarize_errors(errors,shared,windows)

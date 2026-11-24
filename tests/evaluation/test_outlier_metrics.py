import numpy as np


def sample():
    shared={'valid':np.ones((2,11,3),bool),'times_us':np.tile(1000000+np.arange(17)*500000,(2,1))}
    windows=[{'role':'independent_select','log':'a'},{'role':'independent_select','log':'b'}]
    spikes=np.zeros((2,16),bool);spikes[0,[0,5,6,15]]=True
    alarm=np.zeros((2,14),bool);alarm[0,[3,5]]=True
    errors=np.ones((2,11,3));niserrors=errors*.5
    return shared,windows,spikes,alarm,errors,niserrors


def test_detection_denominators_and_recovery_interruption():
    from trajectory_prediction.evaluation.outlier_metrics import diagnostics
    shared,w,spikes,alarm,e,n=sample();rows=diagnostics(alarm,spikes,e,n,shared,w)
    allrow=next(r for r in rows if r['log']=='__all__')
    assert [allrow[k] for k in ('tp','fp','fn','tn')]==[1,1,2,24]
    assert allrow['precision']==.5 and allrow['recall']==1/3
    assert allrow['updates']==28 and allrow['initialization_injected_observations']==1
    r0=allrow['recovery'][0];r1=allrow['recovery'][1]
    assert r0['events']==3 and r0['valid_windows']==1
    assert r0['nis_minus_kf_m']==-.5
    assert r1['events']==1 and r1['excluded_new_outlier']==1 and r1['excluded_truncated']==1
    assert r1['elapsed_seconds_mean']==.5
    assert len(allrow['exposure_intersections'])==8


def test_no_events_null_and_invalid_target_excluded():
    from trajectory_prediction.evaluation.outlier_metrics import diagnostics
    shared,w,spikes,alarm,e,n=sample();shared['valid'][0,0,1]=False
    rows=diagnostics(alarm,spikes,e,n,shared,w)
    assert next(r for r in rows if r['log']=='__all__')['recovery'][0]['excluded_invalid_gt']==1
    empty=next(r for r in rows if r['log']=='b')
    assert empty['precision'] is None and empty['recall'] is None
    assert empty['recovery'][0]['nis_minus_kf_m'] is None
    assert empty['recovery'][0]['reason']=='no_eligible_events'

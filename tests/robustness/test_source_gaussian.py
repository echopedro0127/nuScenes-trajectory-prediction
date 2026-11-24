import copy

import numpy as np
import pytest

from trajectory_prediction.analysis.track_timeline import digest


def packets():
    rows=[]
    for j in range(3):
        key={'schema':'realclock_source_packet_v1','dataset_release':'v1.0-trainval','role':'independent_fit',
             'scene':'scene','log':'log','instance':'i','annotation_token':'t'+str(j)}
        rows.append({**key,'key_digest':digest(key),'source_us':10000000+j*500000})
    return rows


def test_literal_golden_vector_and_global_rng_independence():
    from trajectory_prediction.robustness.source_gaussian import normal_vector
    expected=np.array([0.9757769166040525,0.36552841812889825],np.float64)
    assert np.array_equal(normal_vector('0'*64,2027),expected)
    previous=np.random.get_state()
    try:
        np.random.seed(1);first=normal_vector('0'*64,2027)
        np.random.seed(100);assert np.array_equal(first,normal_vector('0'*64,2027))
    finally:np.random.set_state(previous)
    assert not np.array_equal(first,normal_vector('0'*64,2028))
    assert not np.array_equal(first,normal_vector('1'*64,2027))


def test_manifest_sorting_and_duplicate_source_consistency():
    from trajectory_prediction.robustness.source_gaussian import (
        normal_table,
        validate_packets,
        window_indices,
    )
    p=packets();ordered=validate_packets(p)
    assert np.array_equal(normal_table(p,2027),normal_table(p[::-1],2027))
    w={'role':'independent_fit','scene':'scene','log':'log','instance':'i','tokens':['t0','t1','t2'],
       'time_us':[10000000,10500000,11000000]}
    indices=window_indices(p,[w,w])
    assert np.array_equal(indices[0],indices[1])
    table=normal_table(p,2027)
    for j,key in enumerate(w['tokens']):assert ordered[indices[0,j]]['annotation_token']==key
    assert np.array_equal(table[indices[0]],table[indices[1]])


@pytest.mark.parametrize('kind',['duplicate','digest','time','extra','role','blank','foreign_window'])
def test_manifest_integrity_rejected(kind):
    from trajectory_prediction.robustness.source_gaussian import validate_packets, window_indices
    p=packets()
    if kind=='duplicate':p.append(copy.deepcopy(p[0]))
    elif kind=='digest':p[0]['key_digest']='0'*64
    elif kind=='time':p[0]['source_us']=True
    elif kind=='extra':p[0]['window_id']='foreign'
    elif kind=='role':p[0]['role']='holdout'
    elif kind=='blank':p[0]['instance']=''
    else:
        with pytest.raises(ValueError):window_indices(p,[{'role':'independent_fit','tokens':['foreign'],'time_us':[1]}])
        return
    with pytest.raises(ValueError):validate_packets(p)


@pytest.mark.parametrize('key,seed',[('bad',2027),('A'*64,2027),('0'*64,True),('0'*64,2030)])
def test_strict_rng_key_and_seed(key,seed):
    from trajectory_prediction.robustness.source_gaussian import normal_vector
    with pytest.raises(ValueError):normal_vector(key,seed)

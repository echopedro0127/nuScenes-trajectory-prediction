import numpy as np
import pytest


@pytest.mark.parametrize('times,query,desired,index,status', [
    ([100000,600000,1100000],100000,600000,1,'matched'),
    ([100000,600000,1100000],100000,550000,1,'matched'),
    ([100000,600000,1100000],100000,549999,-1,'target_outside_tolerance'),
    ([100000,600000,1100000],100000,850000,-1,'ambiguous_nearest_target'),
    ([100000,600000,1100000],600000,1100001,-1,'beyond_window_horizon'),
    ([100000,600000],600000,600000,-1,'no_future_target'),
    ([100000],100000,200000,-1,'beyond_window_horizon'),
])
def test_integer_matching_boundaries(times,query,desired,index,status):
    from trajectory_prediction.evaluation.query_clock import match_target
    result=match_target(times,query,desired)
    assert result['target_index']==index and result['status']==status
    assert result['signed_offset_us']==(times[index]-desired if index>=0 else None)


@pytest.mark.parametrize('times,query,desired', [
    ([1,1],1,2),([2,1],1,2),([0,1],1,2),([1.,2.],1,2),
    ([True,2],1,2),([],1,2),([1,2],True,2),([1,2],1,2.),([1,2],2,1),
])
def test_bad_clocks_rejected(times,query,desired):
    from trajectory_prediction.evaluation.query_clock import match_target
    with pytest.raises(ValueError):match_target(times,query,desired)


def test_nominal_query_grid_and_tail_denominators():
    from trajectory_prediction.evaluation.query_clock import window_queries
    times=(10000000+np.arange(17)*500000).astype(np.int64)[None,:]
    result=window_queries(times)
    assert result['query_us'].shape==(1,11)
    assert result['target_index'].shape==(1,11,3)
    assert (result['status_code']==0).sum(axis=(0,1)).tolist()==[11,10,8]
    assert result['target_index'][0,0].tolist()==[6,7,9]
    assert result['query_us'][0,0]==12500000
    assert result['target_index'][0,-1].tolist()==[16,-1,-1]


def test_vector_matches_independent_exhaustive_scalar_reference():
    from trajectory_prediction.evaluation.query_clock import STATUS, window_queries
    rng=np.random.default_rng(123)
    dt=rng.integers(395000,650000,size=(7,16))
    times=np.column_stack([np.full(7,10000000),10000000+dt.cumsum(1)]).astype(np.int64)
    result=window_queries(times)
    for row in range(7):
        for k in range(11):
            query=int(times[row,5+k])
            for lead,h in enumerate((500000,1000000,2000000)):
                desired=query+h
                candidates=[(abs(int(times[row,j])-desired),j) for j in range(5,17) if times[row,j]>query]
                if desired>times[row,-1]: expected='beyond_window_horizon'; index=-1
                elif not candidates: expected='no_future_target';index=-1
                else:
                    best=min(d for d,_ in candidates); matches=[j for d,j in candidates if d==best]
                    expected=('ambiguous_nearest_target' if len(matches)>1 else 'target_outside_tolerance' if best>50000 else 'matched')
                    index=matches[0] if expected=='matched' else -1
                assert result['status_code'][row,k,lead]==STATUS.index(expected)
                assert result['target_index'][row,k,lead]==index


@pytest.mark.parametrize('kind',['float','shape','nonmonotonic','overflow'])
def test_invalid_window_array_rejected(kind):
    from trajectory_prediction.evaluation.query_clock import window_queries
    times=(10000000+np.arange(17)*500000).astype(np.int64)[None,:]
    if kind=='float':times=times.astype(float)
    elif kind=='shape':times=times[:,:16]
    elif kind=='nonmonotonic':times[0,4]=times[0,3]
    else:times=np.arange(17,dtype=np.int64)[None,:]+np.iinfo(np.int64).max-16
    with pytest.raises(ValueError):window_queries(times)

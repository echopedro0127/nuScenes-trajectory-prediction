import numpy as np
import pytest
from test_query_audit_data import audit_data as audit_data  # noqa: PLC0414


def shared_fixture():
    from trajectory_prediction.evaluation.query_clock import window_queries
    times=(10000000+np.arange(17)*500000).astype(np.int64)[None,:]
    elapsed=(times-times[:,:1])/1e6
    xy=np.stack([elapsed*3,elapsed*-2],axis=-1)
    mapping=window_queries(times)
    return {**mapping,'times_us':times,'positions_local':xy,'origin_xy':np.array([[100.,50.]])}


def test_continuous_forecast_uses_matched_target_only_after_state_replay():
    from trajectory_prediction.evaluation.physics_clean_data import evaluate_states
    from trajectory_prediction.robustness.realclock_physics import replay_states
    shared=shared_fixture()
    states=replay_states(shared['times_us'][:,:16],shared['positions_local'][:,:16],'cv_actual_dt')['states']
    score=evaluate_states(states,shared)
    assert np.max(score['errors'])<1e-12
    assert np.all(score['predictions'][~shared['valid']]==0)
    original_states=states.copy();shared['positions_local'][:,16,0]+=10
    changed=evaluate_states(states,shared)
    assert np.array_equal(states,original_states)
    assert changed['errors'][0,-1,0]==pytest.approx(10)


def test_independent_saved_prediction_rescore():
    from trajectory_prediction.evaluation.physics_clean_data import (
        evaluate_states,
        rescore_predictions,
    )
    from trajectory_prediction.robustness.realclock_physics import replay_states
    shared=shared_fixture();states=replay_states(shared['times_us'][:,:16],shared['positions_local'][:,:16],'kf_actual_dt')['states']
    states[:,:,:2]+=.1
    score=evaluate_states(states,shared)
    np.testing.assert_allclose(rescore_predictions(score['predictions'],shared),score['errors'],atol=1e-10,rtol=1e-10)


@pytest.mark.parametrize('kind',['past_target','target_clock','nonfinite','state_dtype','mask','index'])
def test_bad_mapping_or_states_rejected(kind):
    from trajectory_prediction.evaluation.physics_clean_data import evaluate_states
    from trajectory_prediction.robustness.realclock_physics import replay_states
    shared=shared_fixture();states=replay_states(shared['times_us'][:,:16],shared['positions_local'][:,:16],'cv_actual_dt')['states']
    if kind=='past_target':shared['target_index'][0,0,0]=1
    elif kind=='target_clock':shared['target_us'][0,0,0]+=1
    elif kind=='nonfinite':states[0,0,0]=np.nan
    elif kind=='state_dtype':states=states.astype(np.float32)
    elif kind=='mask':shared['valid']=shared['valid'].astype(np.int64)
    else:shared['target_index'][0,0,0]=99
    with pytest.raises(ValueError):evaluate_states(states,shared)

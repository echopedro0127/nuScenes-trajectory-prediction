import numpy as np
import pytest
from test_physics_clean_data import shared_fixture


def test_noisy_origin_and_unmodified_gt_channels():
    from trajectory_prediction.evaluation.gaussian_data import noisy_frame
    base=shared_fixture();before=base['positions_local'].copy()
    normals=np.arange(34,dtype=np.float64).reshape(1,17,2)/10
    observed,evaluation,context=noisy_frame(base,normals,.5)
    assert observed.shape==(1,16,2)
    assert np.array_equal(observed[:,0],np.zeros((1,2)))
    np.testing.assert_allclose(observed,before[:,:16]+.5*(normals[:,:16]-normals[:,:1]))
    np.testing.assert_allclose(evaluation['positions_local'],before-.5*normals[:,:1])
    np.testing.assert_allclose(evaluation['origin_xy'],base['origin_xy']+.5*normals[:,0])
    assert np.array_equal(base['positions_local'],before)
    assert np.array_equal(evaluation['valid'],base['valid'])
    assert np.array_equal(context['observations_local'],observed)


def test_sigma0_reproduces_parent_clean_states_and_errors():
    from trajectory_prediction.evaluation.gaussian_data import noisy_frame
    from trajectory_prediction.evaluation.physics_clean_data import evaluate_states
    from trajectory_prediction.robustness.realclock_physics import STRATEGIES, replay_states
    base=shared_fixture();normals=np.ones((1,17,2),np.float64)
    obs,evaluation,_=noisy_frame(base,normals,0.)
    for strategy in STRATEGIES:
        a=replay_states(base['times_us'][:,:16],base['positions_local'][:,:16],strategy)
        b=replay_states(base['times_us'][:,:16],obs,strategy)
        assert np.array_equal(a['states'],b['states'])
        assert np.array_equal(evaluate_states(a['states'],base)['errors'],evaluate_states(b['states'],evaluation)['errors'])


def test_shared_standard_normals_across_amplitudes_and_future_release():
    from trajectory_prediction.evaluation.gaussian_data import noisy_frame
    from trajectory_prediction.robustness.realclock_physics import replay_states
    base=shared_fixture();normals=np.arange(34,dtype=np.float64).reshape(1,17,2)/10
    a,_,_=noisy_frame(base,normals,.5);b,_,_=noisy_frame(base,normals,1.)
    np.testing.assert_allclose(b-base['positions_local'][:,:16],2*(a-base['positions_local'][:,:16]))
    before=replay_states(base['times_us'][:,:16],a,'kf_actual_dt')['states']
    normals[:,9:]+=10
    changed,_,_=noisy_frame(base,normals,.5)
    after=replay_states(base['times_us'][:,:16],changed,'kf_actual_dt')['states']
    assert np.array_equal(before[:,:4],after[:,:4])


@pytest.mark.parametrize('kind',['shape','dtype','nan','sigma','bool'])
def test_bad_noise_rejected(kind):
    from trajectory_prediction.evaluation.gaussian_data import noisy_frame
    base=shared_fixture();z=np.zeros((1,17,2),np.float64);sigma=.5
    if kind=='shape':z=z[:,:16]
    elif kind=='dtype':z=z.astype(np.float32)
    elif kind=='nan':z[0,0,0]=np.nan
    elif kind=='sigma':sigma=2.
    else:sigma=True
    with pytest.raises(ValueError):noisy_frame(base,z,sigma)

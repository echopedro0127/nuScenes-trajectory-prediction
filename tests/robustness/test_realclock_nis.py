import numpy as np
import pytest


def fixture():
    t=(1000000+np.cumsum(np.tile([500000,470000,530000,510000],4))).astype(np.int64)[None]
    sec=(t-t[:,0])/1e6
    z=np.stack([sec*2,sec*.3],axis=-1)
    return t,z


def test_threshold_cap_and_invalid_nis():
    from trajectory_prediction.robustness.realclock_nis import nis_scale
    np.testing.assert_array_equal(nis_scale(np.array([-1e-12,0.,9.21,18.42,1e6])),[1,1,1,2,25])
    for n in [-2e-12,np.inf,np.nan]:
        with pytest.raises(ValueError):nis_scale(np.array([n]))


@pytest.mark.parametrize('strategy,parent',[('kf_nis_actual_dt','kf_actual_dt'),('kf_nis_fixed_dt_state_control','kf_fixed_dt_state_control')])
def test_no_alarm_equal_parent_and_causal(strategy,parent):
    from trajectory_prediction.robustness.realclock_nis import replay_nis
    from trajectory_prediction.robustness.realclock_physics import replay_states
    t,z=fixture();a=replay_nis(t,z,strategy);b=replay_states(t,z,parent)
    assert not a['alarm'].any()
    for key in ('states','covariances','query_us'):np.testing.assert_array_equal(a[key],b[key])
    assert a['nis'].shape==(1,14) and a['update_us'].shape==(1,14)
    changed=z.copy();changed[:,12:]+=100
    c=replay_nis(t,changed,strategy)
    np.testing.assert_array_equal(a['states'][:,:7],c['states'][:,:7])
    np.testing.assert_array_equal(a['nis'][:,:10],c['nis'][:,:10])


@pytest.mark.parametrize('strategy',['kf_nis_actual_dt','kf_nis_fixed_dt_state_control'])
@pytest.mark.parametrize('motion',['accelerate','turn','initial','consecutive'])
def test_scalar_reference_and_psd(strategy,motion):
    from trajectory_prediction.evaluation.outlier_reference import scalar_nis
    from trajectory_prediction.robustness.realclock_nis import replay_nis
    t,z=fixture();s=(t-t[:,0])/1e6
    if motion=='accelerate':z[:,:,0]+=.3*s*s
    if motion=='turn':z=np.stack([8*np.sin(s/4),8*(1-np.cos(s/4))],axis=-1)
    if motion=='initial':z[:,0]+=[3.,4.]
    if motion=='consecutive':z[:,7:10]+=[5.,0.]
    a=replay_nis(t,z,strategy);b=scalar_nis(t[0],z[0],strategy)
    for key in ('all_states','all_covariances','nis','r_scale'):
        np.testing.assert_allclose(a[key][0],b[key],atol=1e-10,rtol=1e-10)
    assert np.linalg.eigvalsh(a['all_covariances']).min()>=-1e-10


def test_input_clock_and_nonfinite_rejected():
    from trajectory_prediction.robustness.realclock_nis import replay_nis
    t,z=fixture()
    for tt,zz,name in [(t.astype(float),z,'kf_nis_actual_dt'),(t,z.astype('float32'),'kf_nis_actual_dt'),
                       (t,z,'bad'),(t[:,::-1].copy(),z,'kf_nis_actual_dt')]:
        with pytest.raises(ValueError):replay_nis(tt,zz,name)

import numpy as np
import pytest


def stream(regular=False):
    dt=np.full(15,500000,np.int64) if regular else np.array([410000,590000,470000,550000,430000]*3,np.int64)
    t=np.concatenate([[1530000000000000],1530000000000000+dt.cumsum()])[None,:]
    elapsed=(t-t[:,:1])/1000000
    xy=np.stack([elapsed*3,elapsed*-2],axis=-1).astype(np.float64)
    return t,xy


def scalar_reference(times,xy,strategy):
    # Deliberately scalar, explicit observation matrix and inverse instead of batched solve.
    actual='actual_dt' in strategy;kalman=strategy.startswith('kf')
    dt=(int(times[1])-int(times[0]))/1e6 if actual else .5
    x=np.r_[xy[1],(xy[1]-xy[0])/dt]
    r=np.eye(2)*.25;h=np.array([[1,0,0,0],[0,1,0,0]],np.float64)
    p=np.block([[r,r/dt],[r/dt,2*r/dt**2]])
    states=[];cov=[]
    for j in range(2,16):
        dt=(int(times[j])-int(times[j-1]))/1e6 if actual else .5
        if not kalman:x=np.r_[xy[j],(xy[j]-xy[j-1])/dt]
        else:
            f=np.eye(4);f[0,2]=f[1,3]=dt
            q=np.zeros((4,4))
            for axis in range(2):
                q[axis,axis]=dt**3/3;q[axis,axis+2]=q[axis+2,axis]=dt**2/2;q[axis+2,axis+2]=dt
            x=f@x;p=f@p@f.T+q
            gain=p@h.T@np.linalg.inv(h@p@h.T+r)
            x=x+gain@(xy[j]-h@x)
            a=np.eye(4)-gain@h;p=a@p@a.T+gain@r@gain.T;p=(p+p.T)/2
        if j>=5:states.append(x.copy());cov.append(p.copy() if kalman else np.zeros((4,4)))
    return np.array(states),np.array(cov)


def test_irregular_actual_cv_recovers_velocity_and_fixed_dt_can_err():
    from trajectory_prediction.robustness.realclock_physics import replay_states
    t,xy=stream()
    a=replay_states(t,xy,'cv_actual_dt')['states']
    b=replay_states(t,xy,'cv_fixed_dt_state_control')['states']
    np.testing.assert_allclose(a[0,:,2:],np.tile([3.,-2.],(11,1)),atol=1e-12)
    assert np.max(np.abs(b[0,:,2:]-a[0,:,2:]))>.1


@pytest.mark.parametrize('algorithm',['cv','kf'])
def test_regular_dt_controls_equal_actual(algorithm):
    from trajectory_prediction.robustness.realclock_physics import replay_states
    t,xy=stream(True)
    a=replay_states(t,xy,algorithm+'_actual_dt');b=replay_states(t,xy,algorithm+'_fixed_dt_state_control')
    assert np.array_equal(a['states'],b['states'])
    assert np.array_equal(a['covariances'],b['covariances'])


@pytest.mark.parametrize('strategy',['cv_actual_dt','kf_actual_dt','cv_fixed_dt_state_control','kf_fixed_dt_state_control'])
def test_independent_scalar_reference_and_warmup(strategy):
    from trajectory_prediction.robustness.realclock_physics import replay_states
    t,xy=stream();xy[0,:,1]+=np.sin(np.arange(16))*.1
    actual=replay_states(t,xy,strategy);x,p=scalar_reference(t[0],xy[0],strategy)
    np.testing.assert_allclose(actual['states'][0],x,atol=1e-10,rtol=1e-10)
    np.testing.assert_allclose(actual['covariances'][0],p,atol=1e-10,rtol=1e-10)
    if strategy.startswith('kf'):
        assert np.all(np.linalg.eigvalsh(actual['covariances'])>=-1e-12)
        assert np.array_equal(actual['covariances'],actual['covariances'].swapaxes(-1,-2))


@pytest.mark.parametrize('strategy',['cv_actual_dt','kf_actual_dt'])
def test_future_observation_perturbation_preserves_prior_states(strategy):
    from trajectory_prediction.robustness.realclock_physics import replay_states
    t,xy=stream();first=replay_states(t,xy,strategy)['states']
    xy[:,9:]+=100
    second=replay_states(t,xy,strategy)['states']
    assert np.array_equal(first[:,:4],second[:,:4])
    assert not np.array_equal(first[:,4:],second[:,4:])


@pytest.mark.parametrize('kind',['float_time','bool_time','duplicate','reverse','float32','nan','inf','shape','target_leak','strategy'])
def test_invalid_packets_rejected(kind):
    from trajectory_prediction.robustness.realclock_physics import replay_states
    t,xy=stream();strategy='kf_actual_dt'
    if kind=='float_time':t=t.astype(float)
    elif kind=='bool_time':t=t.astype(bool)
    elif kind=='duplicate':t[0,5]=t[0,4]
    elif kind=='reverse':t[0,5]=t[0,4]-1
    elif kind=='float32':xy=xy.astype(np.float32)
    elif kind=='nan':xy[0,4,0]=np.nan
    elif kind=='inf':xy[0,4,0]=np.inf
    elif kind=='shape':xy=xy[:,:15]
    elif kind=='target_leak':t=np.c_[t,t[:,-1]+500000];xy=np.concatenate([xy,xy[:,-1:]],axis=1)
    else:strategy='kf_nis'
    with pytest.raises(ValueError):replay_states(t,xy,strategy)

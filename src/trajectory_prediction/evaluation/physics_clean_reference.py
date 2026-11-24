"""Independent scalar CV/KF cross-check for fixed-hash frozen source windows."""
import hashlib

import numpy as np


def sample_indices(windows):
    selected = []
    for role in ('independent_fit','independent_select'):
        indices = [i for i,w in enumerate(windows) if w['role'] == role]
        selected.extend(sorted(indices,key=lambda i:hashlib.sha256(
            (windows[i]['window_id']+':b3b1_scalar_v1').encode()).digest())[:16])
    return selected


def scalar_states(times,positions,strategy):
    actual = strategy.endswith('actual_dt');kalman = strategy.startswith('kf')
    dt = (int(times[1])-int(times[0]))/1e6 if actual else .5
    state = np.r_[positions[1],(positions[1]-positions[0])/dt]
    r = np.diag([.25,.25]);h = np.array([[1.,0,0,0],[0,1.,0,0]])
    covariance = np.block([[r,r/dt],[r/dt,2*r/dt**2]])
    states,covariances = [],[]
    for j in range(2,16):
        dt = (int(times[j])-int(times[j-1]))/1e6 if actual else .5
        if kalman:
            transition = np.eye(4);transition[0,2] = transition[1,3] = dt
            noise = np.zeros((4,4))
            for axis in (0,1):
                noise[axis,axis] = dt**3/3
                noise[axis,axis+2] = noise[axis+2,axis] = dt**2/2
                noise[axis+2,axis+2] = dt
            prior = transition@state
            prior_covariance = transition@covariance@transition.T+noise
            gain = prior_covariance@h.T@np.linalg.inv(h@prior_covariance@h.T+r)
            state = prior+gain@(positions[j]-h@prior)
            residual_operator = np.eye(4)-gain@h
            covariance = residual_operator@prior_covariance@residual_operator.T+gain@r@gain.T
            covariance = (covariance+covariance.T)/2
        else:
            state = np.r_[positions[j],(positions[j]-positions[j-1])/dt]
        if j >= 5:
            states.append(state.copy())
            covariances.append(covariance.copy() if kalman else np.zeros((4,4)))
    return np.array(states),np.array(covariances)


def check_samples(state,shared,windows,indices,strategy):
    for i in indices:
        expected,p = scalar_states(shared['times_us'][i,:16],shared['positions_local'][i,:16],strategy)
        if (not np.allclose(state['states'][i],expected,atol=1e-10,rtol=1e-10)
                or not np.allclose(state['covariances'][i],p,atol=1e-10,rtol=1e-10)):
            raise ValueError('independent scalar state/covariance mismatch: '+strategy+' '+windows[i]['window_id'])

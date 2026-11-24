"""Causal clean position states with explicit actual/fixed assimilation clocks."""
import numpy as np

STRATEGIES = ('cv_actual_dt','kf_actual_dt','cv_fixed_dt_state_control','kf_fixed_dt_state_control')
MEASUREMENT_VARIANCE = .25
ACCELERATION_SPECTRAL_DENSITY = 1.


def _replay(t,z,strategy):
    n = len(t)
    actual = strategy.endswith('actual_dt')
    kalman = strategy.startswith('kf')
    dt = (t[:,1]-t[:,0])/1000000 if actual else np.full(n,.5)
    x = np.concatenate((z[:,1],(z[:,1]-z[:,0])/dt[:,None]),axis=1)
    p = np.zeros((n,4,4),np.float64)
    for a in range(2):
        p[:,a,a] = MEASUREMENT_VARIANCE
        p[:,a,a+2] = p[:,a+2,a] = MEASUREMENT_VARIANCE/dt
        p[:,a+2,a+2] = 2*MEASUREMENT_VARIANCE/dt**2
    states = np.empty((n,11,4),np.float64)
    covariances = np.zeros((n,11,4,4),np.float64)
    eye = np.broadcast_to(np.eye(4),(n,4,4))
    for j in range(2,16):
        dt = (t[:,j]-t[:,j-1])/1000000 if actual else np.full(n,.5)
        if not kalman:
            x = np.concatenate((z[:,j],(z[:,j]-z[:,j-1])/dt[:,None]),axis=1)
        else:
            f = eye.copy()
            f[:,0,2] = f[:,1,3] = dt
            q = np.zeros_like(p)
            for a in range(2):
                q[:,a,a] = dt**3/3
                q[:,a,a+2] = q[:,a+2,a] = dt**2/2
                q[:,a+2,a+2] = dt
            q *= ACCELERATION_SPECTRAL_DENSITY
            x = np.einsum('nij,nj->ni',f,x)
            p = f@p@f.swapaxes(-1,-2)+q
            r = np.eye(2)*MEASUREMENT_VARIANCE
            gain = np.linalg.solve(p[:,:2,:2]+r,p[:,:2,:]).swapaxes(-1,-2)
            x += np.einsum('nij,nj->ni',gain,z[:,j]-x[:,:2])
            a = eye.copy()
            a[:,:,:2] -= gain
            p = a@p@a.swapaxes(-1,-2)+gain@r@gain.swapaxes(-1,-2)
            p = (p+p.swapaxes(-1,-2))/2
        if j >= 5:
            states[:,j-5] = x
            if kalman:
                covariances[:,j-5] = p
    if not np.isfinite(states).all() or not np.isfinite(covariances).all():
        raise ValueError('nonfinite physics state or covariance')
    return {'states':states,'covariances':covariances,'query_us':t[:,5:16].copy()}


def replay_states(times,positions,strategy):
    t,z = np.asarray(times),np.asarray(positions)
    if (strategy not in STRATEGIES or t.ndim != 2 or t.shape[1] != 16 or len(t) < 1
            or t.dtype != np.dtype('int64') or (t <= 0).any() or (np.diff(t,axis=1) <= 0).any()
            or z.shape != (*t.shape,2) or z.dtype != np.dtype('float64') or not np.isfinite(z).all()):
        raise ValueError('strict positive int64 [N,16] times and finite float64 [N,16,2] positions required')
    try:
        with np.errstate(over='raise',divide='raise',invalid='raise'):
            return _replay(t,z,strategy)
    except (FloatingPointError,np.linalg.LinAlgError) as error:
        raise ValueError('physics arithmetic/solve failed') from error

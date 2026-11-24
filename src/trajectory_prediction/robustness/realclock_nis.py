"""Causal real-clock NIS covariance inflation, without event labels or GT."""
import numpy as np

STRATEGIES=('kf_nis_actual_dt','kf_nis_fixed_dt_state_control')


def nis_scale(nis):
    value=np.asarray(nis,dtype=np.float64)
    if not np.isfinite(value).all() or (value < -1e-12).any():raise ValueError('invalid NIS')
    return np.minimum(25.,np.maximum(1.,np.maximum(0.,value)/9.21))


def _replay(t,z,strategy):
    n=len(t);actual=strategy=='kf_nis_actual_dt'
    dt=(t[:,1]-t[:,0])/1e6 if actual else np.full(n,.5)
    x=np.concatenate((z[:,1],(z[:,1]-z[:,0])/dt[:,None]),axis=1)
    p=np.zeros((n,4,4),np.float64)
    for a in range(2):
        p[:,a,a]=.25;p[:,a,a+2]=p[:,a+2,a]=.25/dt;p[:,a+2,a+2]=.5/dt**2
    states=np.empty((n,14,4));covs=np.empty((n,14,4,4))
    values=np.empty((n,14));scales=np.empty((n,14));eye=np.broadcast_to(np.eye(4),(n,4,4))
    for j in range(2,16):
        dt=(t[:,j]-t[:,j-1])/1e6 if actual else np.full(n,.5)
        f=eye.copy();f[:,0,2]=f[:,1,3]=dt;q=np.zeros_like(p)
        for a in range(2):
            q[:,a,a]=dt**3/3;q[:,a,a+2]=q[:,a+2,a]=dt**2/2;q[:,a+2,a+2]=dt
        x=np.einsum('nij,nj->ni',f,x);p=f@p@f.swapaxes(-1,-2)+q
        innovation=z[:,j]-x[:,:2]
        nis=np.einsum('ni,ni->n',innovation,np.linalg.solve(p[:,:2,:2]+np.eye(2)*.25,innovation[...,None])[...,0])
        scale=nis_scale(nis);nis=np.maximum(nis,0.)
        r=np.eye(2)[None]*.25*scale[:,None,None]
        gain=np.linalg.solve(p[:,:2,:2]+r,p[:,:2,:]).swapaxes(-1,-2)
        x+=np.einsum('nij,nj->ni',gain,innovation)
        a=eye.copy();a[:,:,:2]-=gain
        p=a@p@a.swapaxes(-1,-2)+gain@r@gain.swapaxes(-1,-2);p=(p+p.swapaxes(-1,-2))/2
        states[:,j-2]=x;covs[:,j-2]=p;values[:,j-2]=nis;scales[:,j-2]=scale
    if not np.isfinite(states).all() or not np.isfinite(covs).all():raise ValueError('nonfinite NIS state')
    if np.linalg.eigvalsh(covs).min() < -1e-10:raise ValueError('NIS covariance is not PSD')
    return {'states':states[:,3:].copy(),'covariances':covs[:,3:].copy(),'query_us':t[:,5:].copy(),
            'all_states':states,'all_covariances':covs,'nis':values,'r_scale':scales,
            'alarm':values>9.21,'update_us':t[:,2:].copy(),'diagnostic_valid':np.ones((n,14),bool)}


def replay_nis(times,positions,strategy):
    t,z=np.asarray(times),np.asarray(positions)
    if (strategy not in STRATEGIES or t.ndim!=2 or t.shape[1]!=16 or len(t)<1
            or t.dtype!=np.dtype('int64') or (t<=0).any() or (np.diff(t,axis=1)<=0).any()
            or z.shape!=(*t.shape,2) or z.dtype!=np.dtype('float64') or not np.isfinite(z).all()):
        raise ValueError('strict positive int64 [N,16] clocks and float64 [N,16,2] observations required')
    try:
        with np.errstate(over='raise',divide='raise',invalid='raise'):return _replay(t,z,strategy)
    except (FloatingPointError,np.linalg.LinAlgError) as error:raise ValueError('NIS arithmetic/solve failed') from error

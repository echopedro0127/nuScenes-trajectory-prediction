"""Independent per-window H/inverse/Joseph reference for the NIS kernel."""
import numpy as np


def scalar_nis(t,z,strategy):
    fixed=strategy=='kf_nis_fixed_dt_state_control'
    dt=.5 if fixed else int(t[1]-t[0])/1e6
    x=np.r_[z[1],(z[1]-z[0])/dt]
    p=np.block([[np.eye(2)*.25,np.eye(2)*(.25/dt)],
                [np.eye(2)*(.25/dt),np.eye(2)*(.5/dt**2)]])
    h=np.array([[1.,0.,0.,0.],[0.,1.,0.,0.]])
    states=[];covs=[];values=[];scales=[]
    for j in range(2,16):
        dt=.5 if fixed else int(t[j]-t[j-1])/1e6
        f=np.block([[np.eye(2),np.eye(2)*dt],[np.zeros((2,2)),np.eye(2)]])
        q=np.block([[np.eye(2)*(dt**3/3),np.eye(2)*(dt**2/2)],
                    [np.eye(2)*(dt**2/2),np.eye(2)*dt]])
        xp=f@x;pp=f@p@f.T+q;innovation=z[j]-h@xp
        nis=float(innovation@np.linalg.inv(h@pp@h.T+np.eye(2)*.25)@innovation)
        if not np.isfinite(nis) or nis < -1e-12:raise ValueError('scalar invalid NIS')
        nis=max(0.,nis);scale=min(25.,max(1.,nis/9.21));r=np.eye(2)*(.25*scale)
        k=pp@h.T@np.linalg.inv(h@pp@h.T+r)
        x=xp+k@innovation;a=np.eye(4)-k@h;p=a@pp@a.T+k@r@k.T;p=(p+p.T)/2
        states.append(x.copy());covs.append(p.copy());values.append(nis);scales.append(scale)
    return {'all_states':np.array(states),'all_covariances':np.array(covs),
            'nis':np.array(values),'r_scale':np.array(scales)}


def check_nis_samples(state,t,z,indices,strategy):
    for i in indices:
        reference=scalar_nis(t[i],z[i],strategy)
        for key,expected in reference.items():
            if not np.allclose(state[key][i],expected,atol=1e-10,rtol=1e-10):
                raise ValueError('NIS scalar reference mismatch: '+key)

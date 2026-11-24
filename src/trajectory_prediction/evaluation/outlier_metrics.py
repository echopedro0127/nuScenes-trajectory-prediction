"""Evaluation-only injection diagnostics and censored packet-count recovery."""
import numpy as np


def _recover(spikes,baseline,robust,shared,indices,k):
    n=len(spikes);counts=np.zeros(n,np.int64);sums=np.zeros((n,4),np.float64)
    excluded=dict(excluded_truncated=0,excluded_new_outlier=0,excluded_invalid_gt=0)
    for j in range(5,16):
        active=spikes[indices,j];at=indices[active]
        if j+k>15:excluded['excluded_truncated']+=len(at);continue
        blocked=spikes[at,j+1:j+k+1].any(axis=1)
        excluded['excluded_new_outlier']+=int(blocked.sum());at=at[~blocked]
        valid=shared['valid'][at,j+k-5,1]
        excluded['excluded_invalid_gt']+=int((~valid).sum());at=at[valid]
        b=baseline[at,j+k-5,1];r=robust[at,j+k-5,1]
        elapsed=(shared['times_us'][at,j+k]-shared['times_us'][at,j])/1e6
        counts[at]+=1;sums[at]+=np.stack((b,r,r-b,elapsed),axis=1)
    keep=counts[indices]>0;at=indices[keep]
    means=(sums[at]/counts[at,None]).mean(axis=0) if len(at) else [None]*4
    return {'following_packets':k,'events':int(counts.sum()),'valid_windows':int(keep.sum()),
            'kf_error_m':None if means[0] is None else float(means[0]),
            'nis_error_m':None if means[1] is None else float(means[1]),
            'nis_minus_kf_m':None if means[2] is None else float(means[2]),
            'elapsed_seconds_mean':None if means[3] is None else float(means[3]),
            **excluded,'reason':None if len(at) else 'no_eligible_events'}


def diagnostics(alarm,spikes,baseline,robust,shared,windows):
    n=len(windows)
    if (alarm.shape!=(n,14) or alarm.dtype!=np.dtype('bool') or spikes.shape!=(n,16)
            or spikes.dtype!=np.dtype('bool') or baseline.shape!=(n,11,3) or robust.shape!=baseline.shape
            or not np.isfinite(baseline).all() or not np.isfinite(robust).all()):
        raise ValueError('strict NIS update and evaluation arrays required')
    rows=[]
    for role in sorted({w['role'] for w in windows}):
        logs=['__all__',*sorted({w['log'] for w in windows if w['role']==role})]
        for log in logs:
            idx=np.array([i for i,w in enumerate(windows) if w['role']==role and (log=='__all__' or w['log']==log)])
            a=alarm[idx];truth=spikes[idx,2:];tp=int((a&truth).sum());fp=int((a&~truth).sum())
            fn=int((~a&truth).sum());tn=int((~a&~truth).sum())
            flags=np.stack((spikes[idx,:2].any(axis=1),spikes[idx,2:5].any(axis=1),spikes[idx,5:].any(axis=1)),axis=1)
            intersections=[]
            for bits in range(8):
                target=np.array([bool(bits&(1<<b)) for b in range(3)])
                intersections.append({'initial':bool(target[0]),'warmup':bool(target[1]),'query':bool(target[2]),
                                      'windows':int((flags==target).all(axis=1).sum())})
            rows.append({'role':role,'log':log,'unit':'window_update_exposure','windows':len(idx),
                'updates':int(a.size),'tp':tp,'fp':fp,'fn':fn,'tn':tn,
                'precision':tp/(tp+fp) if tp+fp else None,'precision_reason':None if tp+fp else 'no_alarms',
                'recall':tp/(tp+fn) if tp+fn else None,'recall_reason':None if tp+fn else 'no_injected_updates',
                'fpr':fp/(fp+tn) if fp+tn else None,'fpr_reason':None if fp+tn else 'no_noninjected_updates',
                'initialization_injected_observations':int(spikes[idx,:2].sum()),
                'observed_injected_exposures':int(spikes[idx].sum()),
                'consecutive_injected_pairs':int((spikes[idx,5:15]&spikes[idx,6:16]).sum()),
                'late_injected_events':int(spikes[idx,15].sum()),'exposure_intersections':intersections,
                'recovery':[_recover(spikes,baseline,robust,shared,idx,k) for k in range(4)]})
    return rows

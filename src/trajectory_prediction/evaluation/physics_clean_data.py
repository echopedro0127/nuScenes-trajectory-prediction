"""Shared clean observations and offline target evaluation; no GT in state replay."""
from pathlib import Path

import numpy as np

from trajectory_prediction.evaluation.event_sources import read_json


def load_shared(info):
    parent = Path(info['parent'])
    windows = read_json(parent/'windows.json')
    binding = read_json(parent/'run.json')['binding']
    with np.load(parent/'window_clocks.npz',allow_pickle=False) as z:
        times,tokens = z['actual_timestamp_us'],z['annotation_token']
    with np.load(parent/'query_matches.npz',allow_pickle=False) as z:
        shared = {k:z[k] for k in z.files}
    positions = np.empty((len(windows),17,2),np.float64)
    for role in sorted({w['role'] for w in windows}):
        path = Path(binding['tracks'])/(role+'-tracks.npz')
        with np.load(path,allow_pickle=False) as z:
            source_tokens,source_times,xy,instances = (z[k] for k in
                ('annotation_token','timestamp_us','world_xy','instance_token'))
        if xy.dtype != np.dtype('float64') or xy.shape != (len(source_tokens),2) or not np.isfinite(xy).all():
            raise ValueError('invalid clean source GT coordinates')
        lookup = {str(t):i for i,t in enumerate(source_tokens)}
        if len(lookup) != len(source_tokens):raise ValueError('duplicate clean source token')
        for i,w in enumerate(windows):
            if w['role'] != role:continue
            if tokens[i].tolist() != w['tokens'] or times[i].tolist() != w['time_us']:
                raise ValueError('parent window clock/token disagreement')
            try:indices = [lookup[t] for t in w['tokens']]
            except KeyError as error:raise ValueError('missing clean source token') from error
            if (not np.array_equal(source_times[indices],times[i])
                    or any(str(x) != w['instance'] for x in instances[indices])):
                raise ValueError('clean source token clock/instance mismatch')
            positions[i] = xy[indices]
    origin = positions[:,0].copy()
    shared.update(times_us=times,origin_xy=origin,positions_local=positions-origin[:,None,:])
    if not np.isfinite(shared['positions_local']).all():raise ValueError('nonfinite translated coordinates')
    return shared,windows


def evaluate_states(states,shared):
    x = np.asarray(states);valid = shared['valid'];t = shared['times_us'];index = shared['target_index']
    xy = shared['positions_local'];n = len(t)
    if (x.shape != (n,11,4) or x.dtype != np.dtype('float64') or not np.isfinite(x).all()
            or t.shape != (n,17) or t.dtype != np.dtype('int64') or (t <= 0).any()
            or (np.diff(t,axis=1) <= 0).any() or xy.shape != (n,17,2)
            or xy.dtype != np.dtype('float64') or not np.isfinite(xy).all()
            or valid.shape != (n,11,3) or valid.dtype != np.dtype('bool')
            or index.shape != valid.shape or index.dtype != np.dtype('int16')
            or np.any(index[~valid] != -1) or np.any(index[valid] < 0) or np.any(index[valid] >= 17)):
        raise ValueError('invalid state/shared mapping arrays')
    safe = np.where(valid,index,0)
    rows = np.arange(n)[:,None,None]
    target_us = t[rows,safe]
    lead_us = target_us-t[:,5:16,None]
    if (shared['target_us'].shape != valid.shape or shared['target_us'].dtype != np.dtype('int64')
            or np.any(target_us[valid] != shared['target_us'][valid]) or np.any(lead_us[valid] <= 0)
            or not np.array_equal(shared['query_us'],t[:,5:16])
            or not np.array_equal(shared['actual_lead_us'][valid],lead_us[valid])):
        raise ValueError('target/query timestamp or lead mismatch')
    with np.errstate(over='raise',invalid='raise'):
        predicted = x[:,:,:2][:,:,None,:]+lead_us[:,:,:,None]/1e6*x[:,:,2:][:,:,None,:]
        predicted = np.where(valid[:,:,:,None],predicted,0.)
        delta = predicted-xy[rows,safe]
        errors = np.where(valid,np.linalg.norm(delta,axis=-1),0.)
    if not np.isfinite(predicted).all() or not np.isfinite(errors).all():
        raise ValueError('nonfinite forecast/errors')
    return {'predictions':predicted,'errors':errors,'valid':valid.copy()}


def rescore_predictions(predictions,shared):
    p = np.asarray(predictions);v = shared['valid'];indices = shared['target_index']
    if (p.shape != (*v.shape,2) or p.dtype != np.dtype('float64') or not np.isfinite(p).all()
            or np.any(p[~v] != 0)):
        raise ValueError('finite saved predictions with explicit zero invalid slots required')
    rows = np.arange(len(p))[:,None,None]
    target = shared['positions_local'][rows,np.where(v,indices,0)]
    dx,dy = p[...,0]-target[...,0],p[...,1]-target[...,1]
    errors = np.where(v,np.hypot(dx,dy),0.)
    if not np.isfinite(errors).all():raise ValueError('nonfinite independent saved errors')
    return errors

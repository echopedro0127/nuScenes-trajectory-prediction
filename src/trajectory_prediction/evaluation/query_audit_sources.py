"""Explicit frozen B0b/B1a sources for real-clock metadata evaluation."""
from pathlib import Path

import numpy as np

from trajectory_prediction.evaluation.event_sources import parent_snapshot as tracks_snapshot
from trajectory_prediction.evaluation.event_sources import read_json
from trajectory_prediction.evaluation.review_sources import parent_snapshot as events_snapshot
from trajectory_prediction.evaluation.track_sources import add_reference, check_references


def snapshot(tracks, tracks_owner, events, events_owner, proofs=None):
    if proofs is not None and (not isinstance(proofs,dict) or set(proofs) != {'tracks','events'}):
        raise ValueError('exact two-parent proofs required')
    a = tracks_snapshot(tracks,tracks_owner,None if proofs is None else proofs['tracks'])
    b = events_snapshot(events,events_owner,None if proofs is None else proofs['events'])
    binding = read_json(Path(b['parent'])/'run.json')['binding']
    if (binding['parent_identity'] != a['identity'] or Path(binding['parent']).resolve() != Path(a['parent'])
            or Path(binding['parent_owner']).resolve() != Path(a['owner'])
            or a['dataset_release'] != b['dataset_release']):
        raise ValueError('B1a parent identity/owner does not match supplied B0b')
    refs = dict(a['references'])
    for path,expected in b['references'].items():
        add_reference(refs,path,expected)
    check_references(refs)
    return {'tracks':a['parent'],'tracks_owner':a['owner'],'tracks_identity':a['identity'],
            'events':b['parent'],'events_owner':b['owner'],'events_identity':b['identity'],
            'dataset_release':a['dataset_release'],'references':refs,
            'proofs':{'tracks':a['proof'],'events':b['proof']}}


def load_data(info):
    tracks,events = Path(info['tracks']),Path(info['events'])
    data = {'dataset_release':info['dataset_release'],'windows':read_json(tracks/'windows.json'),
            'tracks':read_json(tracks/'tracks.json'),'blocks':read_json(events/'blocks.json'),
            'events':read_json(events/'events.json'),'window_arrays':{},'track_arrays':{}}
    if len(data['windows']) != read_json(tracks/'summary.json')['window_count']:
        raise ValueError('frozen full-cache window count mismatch')
    for role in sorted({t['role'] for t in data['tracks']}):
        for suffix,key in [('windows','window_arrays'),('tracks','track_arrays')]:
            with np.load(tracks/(role+'-'+suffix+'.npz'),allow_pickle=False) as z:
                data[key][role] = {k:z[k] for k in z.files}
    return data

"""Publish and independently rebuild a real-clock metadata audit; no inference."""
import argparse
import io
import json
import platform
import sys
from pathlib import Path

import numpy as np

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation.event_sources import read_json
from trajectory_prediction.evaluation.query_audit_data import derive_audit
from trajectory_prediction.evaluation.query_audit_sources import load_data, snapshot
from trajectory_prediction.evaluation.query_clock import LEADS, STATUS, TOLERANCE
from trajectory_prediction.evaluation.track_sources import check_references, sources
from trajectory_prediction.evaluation.verdict_overlay import _files, _json, _write_bytes
from trajectory_prediction.utils.provenance import sha256_file

PROJECT = Path(__file__).resolve().parents[3]


def protocol():
    return {'schema':'clock_anchor_audit_b3a_v1','query_steps':list(range(1,12)),
            'anchor_index':4,'source_window_points':17,'target_source_indices':list(range(5,17)),
            'leads_us':list(LEADS),'tolerance_us':TOLERANCE,'time_unit':'integer_us',
            'status_codes':{s:i for i,s in enumerate(STATUS)},'invalid_target_index':-1,
            'invalid_target_us':-1,'invalid_actual_lead_us':-1,'invalid_offset_us':0,
            'invalid_offset_requires_valid_mask':True,'gt_interpolation':False,
            'arrival_policy':'clean_arrival_equals_source','inference_performed':False,
            'noise_sampling_performed':False,'label_source':'rule_candidate'}


def _binding(info):
    return {**{k:info[k] for k in ('tracks','tracks_owner','tracks_identity','events','events_owner',
                                  'events_identity','dataset_release','references')},
            'schema':'clock_anchor_audit_b3a_v1','protocol':protocol(),'sources':sources(PROJECT),
            'runtime':{'python':sys.version,'numpy':str(np.__version__),'platform':platform.platform()},
            'neural_training':False,'holdout_evaluated':False,'new_prediction_scores_computed':False}


def _documents(info):
    result = derive_audit(load_data(info))
    docs = {name:_json(value) for name,value in result['documents'].items()}
    for name,arrays in result['arrays'].items():
        out = io.BytesIO()
        np.savez_compressed(out,**arrays)
        docs[name] = out.getvalue()
    docs['protocol.json'] = _json(protocol())
    docs['parent_proofs.json'] = _json(info['proofs'])
    docs['progress.json'] = _json({'state':'complete'})
    return docs,result['documents']['summary.json']


def _check(binding):
    if sources(PROJECT) != binding['sources']:
        raise ValueError('query audit source changed')
    check_references(binding['references'])


def prepare_audit(tracks,tracks_owner,events,events_owner,output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('use a new query-audit output directory')
    before = sources(PROJECT)
    info = snapshot(tracks,tracks_owner,events,events_owner)
    binding = _binding(info)
    if binding['sources'] != before:
        raise ValueError('query audit source changed during parent validation')
    output.mkdir(parents=True,exist_ok=False)
    try:
        _write_bytes(output/'progress.json',_json({'state':'assembling'}))
        docs,summary = _documents(info)
        docs['run.json'] = _json({'identity':digest(binding),'binding':binding})
        for name,value in docs.items():
            _write_bytes(output/name,value)
            if (output/name).read_bytes() != value:
                raise ValueError('query audit write/read mismatch')
        _check(binding)
        hashes = {name:sha256_file(output/name) for name in docs}
        _files(output,hashes)
        _write_bytes(output/'completed.json',_json({'identity':digest(binding),'sha256':hashes}))
    except BaseException:
        (output/'progress.json').write_bytes(_json({'state':'failed'}))
        raise
    return summary


def verify_audit(output):
    output = Path(output).resolve()
    marker_path = output/'completed.json'
    if not marker_path.is_file():
        raise ValueError('incomplete real-clock query audit')
    marker_hash,marker = sha256_file(marker_path),read_json(marker_path)
    if set(marker) != {'identity','sha256'}:
        raise ValueError('strict immutable query audit marker required')
    _files(output,marker['sha256'],True)
    run = read_json(output/'run.json');binding = run['binding']
    if digest(binding) != run['identity'] or marker['identity'] != run['identity']:
        raise ValueError('query audit identity mismatch')
    _check(binding)
    info = snapshot(binding['tracks'],binding['tracks_owner'],binding['events'],binding['events_owner'],
                    read_json(output/'parent_proofs.json'))
    if digest(binding) != digest(_binding(info)):
        raise ValueError('query audit policy/runtime replay mismatch')
    docs,summary = _documents(info)
    docs['run.json'] = _json(run)
    if set(docs) != set(marker['sha256']):
        raise ValueError('query audit expected file coverage mismatch')
    for name,value in docs.items():
        if (output/name).read_bytes() != value:
            raise ValueError('query audit semantic replay mismatch: '+name)
    _check(binding)
    _files(output,marker['sha256'],True)
    if sha256_file(marker_path) != marker_hash:
        raise ValueError('query audit completed marker changed')
    return {**summary,'query_audit_semantically_replayed':True,'identity':run['identity']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('tracks','tracks-owner','events','events-owner'):
        parser.add_argument('--'+key,type=Path)
    parser.add_argument('--output-root',type=Path,required=True)
    parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        result = verify_audit(args.output_root)
    else:
        if any(getattr(args,k) is None for k in ('tracks','tracks_owner','events','events_owner')):
            parser.error('explicit tracks/events and both original owners required')
        result = prepare_audit(args.tracks,args.tracks_owner,args.events,args.events_owner,args.output_root)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)

"""Four causal clean CV/KF comparisons with immutable replayable scores."""
import argparse
import io
import json
import math
import os
import platform
import sys
from pathlib import Path

import numpy as np

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation.event_sources import read_json
from trajectory_prediction.evaluation.physics_clean_data import (
    evaluate_states,
    load_shared,
    rescore_predictions,
)
from trajectory_prediction.evaluation.physics_clean_metrics import paired_gains, summarize_errors
from trajectory_prediction.evaluation.physics_clean_reference import check_samples, sample_indices
from trajectory_prediction.evaluation.physics_clean_sources import snapshot
from trajectory_prediction.evaluation.track_sources import check_references, sources
from trajectory_prediction.evaluation.verdict_overlay import _files, _json, _write_bytes
from trajectory_prediction.robustness.realclock_physics import STRATEGIES, replay_states
from trajectory_prediction.utils.provenance import sha256_file

PROJECT = Path(__file__).resolve().parents[3]
COMPARISONS = [('cv_dt','cv_fixed_dt_state_control','cv_actual_dt'),
               ('kf_dt','kf_fixed_dt_state_control','kf_actual_dt'),
               ('actual_kf_vs_cv','cv_actual_dt','kf_actual_dt')]


def protocol():
    return {'schema':'realclock_physics_clean_b3b1_v1','strategies':list(STRATEGIES),
            'condition':'clean_annotation_positions','source_indices_assimilated':list(range(16)),
            'query_indices':list(range(5,16)),'history_initialization_indices':[0,1],
            'origin':'first_arrived_observation','compute_dtype':'float64','measurement_variance':.25,
            'acceleration_spectral_density':1.,'covariance_form':'Joseph',
            'cv_covariance_applicable':False,'cv_covariance_placeholder':0.,
            'forecast':'frozen_query_position_plus_velocity_times_actual_target_lead',
            'invalid_prediction_error_window_mean_placeholder':0.,'valid_mask_required':True,
            'mean_reduction':'query_then_window','log_macro_reduction':'window_then_log',
            'tail_unit':'pooled_query','quantile_method':'linear','error_threshold_m':2.,
            'comparisons':[list(x) for x in COMPARISONS],'scalar_and_rescore_atol':1e-10,
            'scalar_and_rescore_rtol':1e-10,'scalar_sample_per_role':16,
            'scalar_sample_hash_suffix':':b3b1_scalar_v1','noise_sampling':False,'training':False}


def _npz(arrays):
    out = io.BytesIO();np.savez_compressed(out,**arrays);return out.getvalue()


def _binding(info):
    return {'schema':'realclock_physics_clean_b3b1_v1','parent':info['parent'],'parent_owner':info['owner'],
            'parent_identity':info['identity'],'sources':sources(PROJECT),'references':dict(info['references']),
            'protocol':protocol(),'runtime':{'python':sys.version,'numpy':str(np.__version__),
                                            'platform':platform.platform(),
                                            'thread_policy':'environment_at_process_start',
                                            'thread_environment':{k:os.environ.get(k) for k in
                                                ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}},
            'neural_training':False,'holdout_evaluated':False,'new_prediction_scores_computed':True}


def _documents(info):
    shared,windows = load_shared(info)
    docs = {'shared.npz':_npz(shared),'windows.json':_json(windows),'protocol.json':_json(protocol()),
            'parent_proof.json':_json(info['proof']),'progress.json':_json({'state':'complete'})}
    results,groups = {},{}
    indices = sample_indices(windows)
    for strategy in STRATEGIES:
        state = replay_states(shared['times_us'][:,:16],shared['positions_local'][:,:16],strategy)
        check_samples(state,shared,windows,indices,strategy)
        if not np.array_equal(state['query_us'],shared['query_us']):raise ValueError('physics query clock mismatch')
        score = evaluate_states(state['states'],shared)
        metrics = summarize_errors(score['errors'],shared,windows)
        docs[strategy+'.npz'] = _npz({**state,**score,**metrics['window_metrics']})
        groups[strategy] = metrics['groups']
        results[strategy] = {'errors':score['errors'],'valid':score['valid']}
    paired = paired_gains(results,shared,windows,COMPARISONS)
    docs['scalar_reference.json'] = _json({'status':'verified','sample_per_role_cap':16,
        'sample_windows':[{'window_id':windows[i]['window_id'],'role':windows[i]['role']} for i in indices],
        'query_states_checked':len(indices)*len(STRATEGIES)*11,
        'kf_covariance_matrices_checked':len(indices)*2*11,'cv_zero_covariance_placeholders_checked':len(indices)*2*11,
        'independent_algorithm':'scalar_observation_matrix_inverse_Joseph','atol':1e-10,'rtol':1e-10})
    docs['paired_deltas.npz'] = _npz(paired['arrays'])
    summary = {'status':'realclock_physics_clean_complete','windows':len(windows),'queries_per_strategy':len(windows)*11,
               'matching_attempts_per_strategy':int(shared['valid'].size),
               'valid_pairs_per_strategy':int(shared['valid'].sum()),'strategy_groups':groups,
               'paired_groups':paired['groups'],'condition':'clean_annotation_positions',
               'human_labels_imported':False,'neural_training':False,'holdout_evaluated':False,
               'new_prediction_scores_computed':True,'official_challenge_metrics':False}
    docs['summary.json'] = _json(summary)
    return docs,summary


def _check(binding):
    if sources(PROJECT) != binding['sources']:raise ValueError('physics clean source changed')
    check_references(binding['references'])


def prepare_clean(parent,parent_owner,output):
    output = Path(output).resolve()
    if output.exists():raise FileExistsError('use a new physical clean output directory')
    before = sources(PROJECT);info = snapshot(parent,parent_owner);binding = _binding(info)
    if binding['sources'] != before:raise ValueError('physics source changed during parent verification')
    output.mkdir(parents=True,exist_ok=False)
    try:
        _write_bytes(output/'progress.json',_json({'state':'assembling'}))
        docs,summary = _documents(info)
        docs['run.json'] = _json({'identity':digest(binding),'binding':binding})
        for name,value in docs.items():
            _write_bytes(output/name,value)
            if (output/name).read_bytes() != value:raise ValueError('physics clean write/read mismatch')
        _check(binding);hashes = {name:sha256_file(output/name) for name in docs};_files(output,hashes)
        _write_bytes(output/'completed.json',_json({'identity':digest(binding),'sha256':hashes}))
    except BaseException:
        (output/'progress.json').write_bytes(_json({'state':'failed'}));raise
    return summary


def _close(actual,expected):
    if type(actual) is not type(expected):return False
    if isinstance(actual,dict):return set(actual) == set(expected) and all(_close(actual[k],expected[k]) for k in actual)
    if isinstance(actual,list):return len(actual) == len(expected) and all(_close(a,b) for a,b in zip(actual,expected,strict=True))
    if isinstance(actual,float):return math.isclose(actual,expected,abs_tol=1e-10,rel_tol=1e-10)
    return actual == expected


def _rescore_saved(output,summary):
    with np.load(output/'shared.npz',allow_pickle=False) as z:shared = {k:z[k] for k in z.files}
    windows = read_json(output/'windows.json');results = {}
    for strategy in STRATEGIES:
        with np.load(output/(strategy+'.npz'),allow_pickle=False) as z:
            predictions,errors,valid = z['predictions'],z['errors'],z['valid']
        if not np.array_equal(valid,shared['valid']):raise ValueError('saved strategy mask differs')
        rescored = rescore_predictions(predictions,shared)
        if not np.allclose(rescored,errors,atol=1e-10,rtol=1e-10):raise ValueError('saved prediction error mismatch')
        if not _close(summarize_errors(rescored,shared,windows)['groups'],summary['strategy_groups'][strategy]):
            raise ValueError('saved prediction metric summary mismatch')
        results[strategy] = {'errors':rescored,'valid':valid}
    if not _close(paired_gains(results,shared,windows,COMPARISONS)['groups'],summary['paired_groups']):
        raise ValueError('saved prediction paired gains mismatch')


def verify_clean(output):
    output = Path(output).resolve();marker_path = output/'completed.json'
    if not marker_path.is_file():raise ValueError('incomplete clean physics output')
    marker_hash,marker = sha256_file(marker_path),read_json(marker_path)
    if set(marker) != {'identity','sha256'}:raise ValueError('strict immutable physics marker required')
    _files(output,marker['sha256'],True);run = read_json(output/'run.json');binding = run['binding']
    if digest(binding) != run['identity'] or run['identity'] != marker['identity']:raise ValueError('physics clean identity mismatch')
    _check(binding);info = snapshot(binding['parent'],binding['parent_owner'],read_json(output/'parent_proof.json'))
    if digest(binding) != digest(_binding(info)):raise ValueError('physics policy/runtime replay mismatch')
    docs,summary = _documents(info);docs['run.json'] = _json(run)
    if set(docs) != set(marker['sha256']):raise ValueError('physics expected file coverage mismatch')
    for name,value in docs.items():
        if (output/name).read_bytes() != value:raise ValueError('physics semantic replay mismatch: '+name)
    _rescore_saved(output,summary)
    _check(binding);_files(output,marker['sha256'],True)
    if sha256_file(marker_path) != marker_hash:raise ValueError('physics marker changed during replay')
    return {**summary,'identity':run['identity'],'physics_clean_semantically_replayed':True,
            'saved_predictions_independently_rescored':True,'scalar_reference_verified':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path);parser.add_argument('--parent-owner',type=Path)
    parser.add_argument('--output-root',type=Path,required=True);parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    if args.verify_only:result = verify_clean(args.output_root)
    else:
        if args.parent is None or args.parent_owner is None:parser.error('explicit B3a parent and original owner required')
        result = prepare_clean(args.parent,args.parent_owner,args.output_root)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)

"""Stream source-consistent Gaussian CV/KF shards with immutable replay."""
import argparse
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation.event_sources import read_json
from trajectory_prediction.evaluation.gaussian_data import noisy_frame
from trajectory_prediction.evaluation.gaussian_metrics import (
    aggregate_differences,
    aggregate_scores,
)
from trajectory_prediction.evaluation.gaussian_sources import snapshot
from trajectory_prediction.evaluation.physics_clean import COMPARISONS, _close, _npz
from trajectory_prediction.evaluation.physics_clean_data import evaluate_states, rescore_predictions
from trajectory_prediction.evaluation.physics_clean_metrics import paired_gains, summarize_errors
from trajectory_prediction.evaluation.physics_clean_reference import check_samples, sample_indices
from trajectory_prediction.evaluation.track_sources import check_references, sources
from trajectory_prediction.evaluation.verdict_overlay import _files, _json, _write_bytes
from trajectory_prediction.robustness.realclock_physics import STRATEGIES, replay_states
from trajectory_prediction.robustness.source_gaussian import (
    SEEDS,
    normal_table,
    validate_packets,
    window_indices,
)
from trajectory_prediction.utils.provenance import sha256_file

PROJECT = Path(__file__).resolve().parents[3]
CONDITIONS = (('gaussian_05',.5),('gaussian_10',1.))


def protocol():
    return {'schema':'realclock_physics_gaussian_b3b2a_v1','conditions':dict(CONDITIONS),'noise_seeds':list(SEEDS),
            'strategies':list(STRATEGIES),'measurement_variance':.25,'acceleration_spectral_density':1.,
            'rng':{'key_schema':'realclock_gaussian_packet_b3b2a_v1','channel':'world_xy_standard_normal',
                   'json_sort_keys':True,'json_separators':[',',':'],'json_ensure_ascii':False,
                   'hash':'SHA256','seed_bytes':8,'seed_endian':'big','bit_generator':'PCG64',
                   'draw':'standard_normal(size=2,dtype=float64)','sigma_in_key':False},
            'origin':'first_noisy_arrived_observation','gt_channel':'original_truth_translated_only',
            'observed_indices':list(range(16)),'noise_reduction':'seed_then_window',
            'sigma0_parent_regression':'all_four_frozen_states_covariances_predictions_errors',
            'tail_reduction':'mean_and_std_of_per_seed_query_quantiles','scalar_sample_per_role':16,
            'scalar_hash_suffix':':b3b1_scalar_v1','atol':1e-10,'rtol':1e-10,
            'delay':False,'dropout':False,'outliers':False,'noise_in_timestamps':False,'training':False}


def _binding(info):
    return {'schema':'realclock_physics_gaussian_b3b2a_v1','parent':info['parent'],'parent_owner':info['owner'],
            'parent_identity':info['identity'],'packets_path':info['packets_path'],'protocol':protocol(),
            'sources':sources(PROJECT),'references':dict(info['references']),
            'runtime':{'python':sys.version,'numpy':str(np.__version__),'platform':platform.platform(),
                       'thread_policy':'environment_at_process_start','thread_environment':{k:os.environ.get(k) for k in
                            ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}},
            'neural_training':False,'holdout_evaluated':False,'new_prediction_scores_computed':True}


def _load(info):
    parent = Path(info['parent'])
    with np.load(parent/'shared.npz',allow_pickle=False) as z:base = {k:z[k] for k in z.files}
    windows = read_json(parent/'windows.json');packets = validate_packets(read_json(info['packets_path']))
    point_indices = window_indices(packets,windows)
    if point_indices.shape != (len(windows),17) or len(np.unique(point_indices)) != len(packets):
        raise ValueError('source manifest/window coverage mismatch')
    clean = {}
    for strategy in STRATEGIES:
        with np.load(parent/(strategy+'.npz'),allow_pickle=False) as z:
            clean['clean_'+strategy] = {'errors':z['errors'],'valid':z['valid']}
        if not np.array_equal(clean['clean_'+strategy]['valid'],base['valid']):raise ValueError('clean shared mask mismatch')
    return base,windows,packets,point_indices,clean


def _stream(info):
    base,windows,packets,point_indices,clean = _load(info)
    yield 'clean_regression.json',_json(verify_clean_regression(info,base))
    yield 'protocol.json',_json(protocol())
    yield 'parent_proof.json',_json(info['proof'])
    yield 'windows.json',_json(windows)
    yield 'shared_clean.npz',_npz(base)
    yield 'source_packets.json',_json(packets)
    yield 'source_point_indices.npz',_npz({'indices':point_indices})
    tables = {}
    for seed in SEEDS:
        tables[seed] = normal_table(packets,seed)
        yield f'normals/{seed}.npz',_npz({'normals':tables[seed]})
    selected = sample_indices(windows)
    seed_results,seed_differences = [],[]
    score_records,gain_records,extra_records = defaultdict(list),defaultdict(list),defaultdict(list)
    comparisons = [*COMPARISONS,*[('extra_'+s,s,'clean_'+s) for s in STRATEGIES]]
    for condition,sigma in CONDITIONS:
        for seed in SEEDS:
            prefix = f'{condition}/seed_{seed}'
            normals = tables[seed][point_indices]
            observed,evaluation,context = noisy_frame(base,normals,sigma)
            yield prefix+'/observations.npz',_npz(context)
            reference = {'times_us':base['times_us'],'positions_local':observed}
            results = {}
            for strategy in STRATEGIES:
                state = replay_states(base['times_us'][:,:16],observed,strategy)
                check_samples(state,reference,windows,selected,strategy)
                if not np.array_equal(state['query_us'],base['query_us']):raise ValueError('noisy query clock mismatch')
                score = evaluate_states(state['states'],evaluation)
                metrics = summarize_errors(score['errors'],evaluation,windows)
                yield prefix+'/'+strategy+'.npz',_npz({**state,**score,**metrics['window_metrics']})
                score_records[condition,strategy].append({'seed':seed,**metrics})
                seed_results.append({'condition':condition,'sigma_m':sigma,'noise_seed':seed,'strategy':strategy,'groups':metrics['groups']})
                results[strategy] = {'errors':score['errors'],'valid':score['valid']}
            paired = paired_gains({**results,**clean},base,windows,comparisons)
            yield prefix+'/paired_deltas.npz',_npz(paired['arrays'])
            for row in paired['groups']:
                if row['comparison'].startswith('extra_'):
                    row['mean_window_extra_error_m'] = row.pop('mean_window_gain_m')
                    row['positive_means'] = 'noisy_worse_than_clean'
                else:row['positive_means'] = 'compared_strategy_better'
            seed_differences.append({'condition':condition,'sigma_m':sigma,'noise_seed':seed,'groups':paired['groups']})
            for name,ref,compared in COMPARISONS:
                gain_records[condition,name].append({'seed':seed,'values':paired['arrays'][name]})
            for strategy in STRATEGIES:
                extra_records[condition,strategy].append({'seed':seed,'values':paired['arrays']['extra_'+strategy]})
    macro,gains,extras = [],[],[]
    for condition,sigma in CONDITIONS:
        for strategy in STRATEGIES:
            result = aggregate_scores(score_records[condition,strategy],base,windows)
            yield f'aggregates/{condition}/{strategy}.npz',_npz(result['arrays'])
            macro.append({'condition':condition,'sigma_m':sigma,'strategy':strategy,'groups':result['groups']})
            delta = aggregate_differences(extra_records[condition,strategy],base,windows,'mean_window_extra_error_m')
            yield f'aggregates/{condition}/extra_{strategy}.npz',_npz(delta['arrays'])
            extras.append({'condition':condition,'sigma_m':sigma,'strategy':strategy,'groups':delta['groups']})
        for name,ref,compared in COMPARISONS:
            result = aggregate_differences(gain_records[condition,name],base,windows,'mean_window_gain_m')
            yield f'aggregates/{condition}/paired_{name}.npz',_npz(result['arrays'])
            gains.append({'condition':condition,'sigma_m':sigma,'comparison':name,'reference_strategy':ref,
                          'compared_strategy':compared,'groups':result['groups']})
    yield 'scalar_reference.json',_json({'status':'verified','sample_windows':[{'window_id':windows[i]['window_id'],'role':windows[i]['role']} for i in selected],
        'combinations':24,'query_states_checked':len(selected)*24*11,'kf_covariance_matrices_checked':len(selected)*12*11,
        'atol':1e-10,'rtol':1e-10,'reference':'independent_scalar_H_inverse_Joseph_on_noisy_observations'})
    yield 'summary.json',_json({'status':'source_gaussian_physics_complete','windows':len(windows),
        'queries_per_combination':len(windows)*11,'valid_pairs_per_combination':int(base['valid'].sum()),
        'combinations':24,'unique_source_packets':len(packets),'noise_seeds':list(SEEDS),
        'seed_results':seed_results,'seed_differences':seed_differences,'seed_macro_results':macro,
        'paired_seed_macro_results':gains,'clean_degradation_seed_macro_results':extras,
        'clean_reference_groups':read_json(Path(info['parent'])/'summary.json')['strategy_groups'],
        'human_labels_imported':False,'neural_training':False,'holdout_evaluated':False,
        'new_prediction_scores_computed':True,'synthetic_observation_noise':True,'official_challenge_metrics':False})
    yield 'progress.json',_json({'state':'complete'})


def _check(binding):
    if sources(PROJECT) != binding['sources']:raise ValueError('Gaussian source changed')
    check_references(binding['references'])


def verify_clean_regression(info,base):
    observed,evaluation,_context = noisy_frame(base,np.zeros_like(base['positions_local']),0.)
    for strategy in STRATEGIES:
        state = replay_states(base['times_us'][:,:16],observed,strategy)
        score = evaluate_states(state['states'],evaluation)
        actual = {**state,**score}
        with np.load(Path(info['parent'])/(strategy+'.npz'),allow_pickle=False) as z:
            if any(not np.array_equal(actual[k],z[k]) for k in
                   ('states','covariances','query_us','predictions','errors','valid')):
                raise ValueError('clean parent regression mismatch: '+strategy)
    return {'status':'verified','sigma':0.,'strategies':list(STRATEGIES),
            'query_states_checked':len(observed)*11*4,'counted_as_new_noise_repeats':False}


def prepare_gaussian(parent,parent_owner,output):
    output = Path(output).resolve()
    if output.exists():raise FileExistsError('use a new Gaussian output directory')
    before = sources(PROJECT);info = snapshot(parent,parent_owner);binding = _binding(info)
    if binding['sources'] != before:raise ValueError('Gaussian source changed during parent verification')
    output.mkdir(parents=True,exist_ok=False)
    try:
        _write_bytes(output/'progress.json',_json({'state':'assembling'}));hashes = {}
        for name,value in _stream(info):
            path = output/name;path.parent.mkdir(parents=True,exist_ok=True)
            _write_bytes(path,value)
            if path.read_bytes() != value:raise ValueError('Gaussian write/read mismatch')
            hashes[name] = sha256_file(path)
        _write_bytes(output/'run.json',_json({'identity':digest(binding),'binding':binding}))
        hashes['run.json'] = sha256_file(output/'run.json')
        _check(binding);_files(output,hashes)
        _write_bytes(output/'completed.json',_json({'identity':digest(binding),'sha256':hashes}))
    except BaseException:
        (output/'progress.json').write_bytes(_json({'state':'failed'}));raise
    return read_json(output/'summary.json')


def _rescore(output,info,summary):
    base,windows,_packets,_indices,clean = _load(info)
    score_records,gain_records,extra_records = defaultdict(list),defaultdict(list),defaultdict(list)
    for condition,sigma in CONDITIONS:
        for seed in SEEDS:
            prefix = output/condition/f'seed_{seed}'
            with np.load(prefix/'observations.npz',allow_pickle=False) as z:offset = z['origin_offset_xy']
            evaluation = {**base,'positions_local':base['positions_local']-offset[:,None,:],'origin_xy':base['origin_xy']+offset}
            results = {}
            for strategy in STRATEGIES:
                with np.load(prefix/(strategy+'.npz'),allow_pickle=False) as z:
                    pred,errors,valid = z['predictions'],z['errors'],z['valid']
                if not np.array_equal(valid,base['valid']):raise ValueError('saved noisy mask mismatch')
                rescored = rescore_predictions(pred,evaluation)
                if not np.allclose(rescored,errors,atol=1e-10,rtol=1e-10):raise ValueError('saved noisy error mismatch')
                expected = next(r['groups'] for r in summary['seed_results'] if
                    (r['condition'],r['noise_seed'],r['strategy']) == (condition,seed,strategy))
                metrics = summarize_errors(rescored,evaluation,windows)
                if not _close(metrics['groups'],expected):
                    raise ValueError('saved noisy metric reconstruction mismatch')
                score_records[condition,strategy].append({'seed':seed,**metrics})
                results[strategy] = {'errors':rescored,'valid':valid}
            pairs = paired_gains({**results,**clean},base,windows,[*COMPARISONS,*[('extra_'+s,s,'clean_'+s) for s in STRATEGIES]])
            for row in pairs['groups']:
                if row['comparison'].startswith('extra_'):
                    row['mean_window_extra_error_m'] = row.pop('mean_window_gain_m');row['positive_means'] = 'noisy_worse_than_clean'
                else:row['positive_means'] = 'compared_strategy_better'
            expected = next(r['groups'] for r in summary['seed_differences'] if (r['condition'],r['noise_seed']) == (condition,seed))
            if not _close(pairs['groups'],expected):raise ValueError('saved noisy paired reconstruction mismatch')
            for name,ref,compared in COMPARISONS:
                gain_records[condition,name].append({'seed':seed,'values':pairs['arrays'][name]})
            for strategy in STRATEGIES:
                extra_records[condition,strategy].append({'seed':seed,'values':pairs['arrays']['extra_'+strategy]})
    for condition,sigma in CONDITIONS:
        for strategy in STRATEGIES:
            actual = aggregate_scores(score_records[condition,strategy],base,windows)['groups']
            expected = next(r['groups'] for r in summary['seed_macro_results'] if (r['condition'],r['strategy']) == (condition,strategy))
            if not _close(actual,expected):raise ValueError('saved prediction seed macro mismatch')
            actual = aggregate_differences(extra_records[condition,strategy],base,windows,'mean_window_extra_error_m')['groups']
            expected = next(r['groups'] for r in summary['clean_degradation_seed_macro_results'] if (r['condition'],r['strategy']) == (condition,strategy))
            if not _close(actual,expected):raise ValueError('saved prediction clean macro mismatch')
        for name,ref,compared in COMPARISONS:
            actual = aggregate_differences(gain_records[condition,name],base,windows,'mean_window_gain_m')['groups']
            expected = next(r['groups'] for r in summary['paired_seed_macro_results'] if (r['condition'],r['comparison']) == (condition,name))
            if not _close(actual,expected):raise ValueError('saved prediction paired macro mismatch')


def verify_gaussian(output):
    output = Path(output).resolve();marker_path = output/'completed.json'
    if not marker_path.is_file():raise ValueError('incomplete Gaussian output')
    marker_hash,marker = sha256_file(marker_path),read_json(marker_path)
    if set(marker) != {'identity','sha256'}:raise ValueError('strict immutable Gaussian marker required')
    _files(output,marker['sha256'],True);run = read_json(output/'run.json');binding = run['binding']
    if digest(binding) != run['identity'] or run['identity'] != marker['identity']:raise ValueError('Gaussian identity mismatch')
    _check(binding);info = snapshot(binding['parent'],binding['parent_owner'],read_json(output/'parent_proof.json'))
    if digest(binding) != digest(_binding(info)):raise ValueError('Gaussian policy/runtime replay mismatch')
    seen = set()
    for name,value in _stream(info):
        seen.add(name)
        if (output/name).read_bytes() != value:raise ValueError('Gaussian semantic replay mismatch: '+name)
    seen.add('run.json')
    if (output/'run.json').read_bytes() != _json(run) or seen != set(marker['sha256']):raise ValueError('Gaussian expected file coverage mismatch')
    summary = read_json(output/'summary.json');_rescore(output,info,summary)
    _check(binding);_files(output,marker['sha256'],True)
    if sha256_file(marker_path) != marker_hash:raise ValueError('Gaussian marker changed during replay')
    return {**summary,'identity':run['identity'],'gaussian_semantically_replayed':True,
            'saved_predictions_independently_rescored':True,'scalar_reference_verified':True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path);parser.add_argument('--parent-owner',type=Path)
    parser.add_argument('--output-root',type=Path,required=True);parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args()
    if args.verify_only:result = verify_gaussian(args.output_root)
    else:
        if args.parent is None or args.parent_owner is None:parser.error('explicit B3b1 clean parent and original owner required')
        result = prepare_gaussian(args.parent,args.parent_owner,args.output_root)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)

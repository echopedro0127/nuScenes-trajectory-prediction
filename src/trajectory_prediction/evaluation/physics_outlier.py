"""Immutable source-keyed outlier/NIS replay, paired scores and diagnostics."""
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
from trajectory_prediction.evaluation.gaussian_metrics import (
    aggregate_differences,
    aggregate_scores,
)
from trajectory_prediction.evaluation.outlier_data import outlier_frame
from trajectory_prediction.evaluation.outlier_metrics import diagnostics
from trajectory_prediction.evaluation.outlier_reference import check_nis_samples
from trajectory_prediction.evaluation.outlier_sources import snapshot
from trajectory_prediction.evaluation.physics_clean import _close, _npz
from trajectory_prediction.evaluation.physics_clean_data import evaluate_states, rescore_predictions
from trajectory_prediction.evaluation.physics_clean_metrics import paired_gains, summarize_errors
from trajectory_prediction.evaluation.physics_clean_reference import check_samples, sample_indices
from trajectory_prediction.evaluation.track_sources import check_references, sources
from trajectory_prediction.evaluation.verdict_overlay import _files, _json, _write_bytes
from trajectory_prediction.robustness.realclock_nis import STRATEGIES as NIS_STRATEGIES
from trajectory_prediction.robustness.realclock_nis import replay_nis
from trajectory_prediction.robustness.realclock_physics import replay_states
from trajectory_prediction.robustness.source_gaussian import (
    SEEDS,
    normal_table,
    validate_packets,
    window_indices,
)
from trajectory_prediction.robustness.source_outlier import outlier_table
from trajectory_prediction.utils.provenance import sha256_file

PROJECT=Path(__file__).resolve().parents[3]
STRATEGIES=('cv_actual_dt','kf_actual_dt',*NIS_STRATEGIES)
CONDITIONS=('clean','gaussian_05','gaussian_outlier')
PAIRS=(('nis_minus_kf','kf_nis_actual_dt','kf_actual_dt'),
       ('nis_actual_minus_fixed','kf_nis_actual_dt','kf_nis_fixed_dt_state_control'))


def protocol():
    return {'schema':'realclock_physics_outlier_b3b2b_v1','conditions':list(CONDITIONS),'strategies':list(STRATEGIES),
        'noise_seeds':list(SEEDS),'clean_repeats':1,'combinations':28,'sigma_m':.5,'outlier_norm_m':5.,
        'outlier_hash_schema':'realclock_outlier_packet_b3b2b_v1','channels':['occurrence','direction'],
        'occurrence_integer_cutoff':2**64//20,'hash':'SHA256_first8_big_endian',
        'json_sort_keys':True,'json_separators':[',',':'],'json_ensure_ascii':False,
        'direction':'2*pi*(uint64/2**64)','gaussian_rng':'unchanged_B3b2a_source_PCG64',
        'nis_threshold':9.21,'r_scale_cap':25.,'negative_nis_tolerance':1e-12,
        'r0':.25,'q':1.,'joseph_covariance':True,'observed_indices':list(range(16)),
        'diagnostic_indices':list(range(2,16)),'initialization_diagnostic':'not_available',
        'origin':'first_noisy_observation','gt':'original_world_truth','source_equals_arrival':True,
        'delay':False,'dropout':False,'neural_training':False,'holdout_evaluated':False,
        'main':'independent_select_1s_gaussian_outlier_nis_minus_kf',
        'recovery':'query_events_k0_to3_no_intervening_outlier_window_first_1s_valid_gt',
        'seed_reduction':'seed_then_window','tail_reduction':'mean_std_of_seed_quantiles',
        'scalar_atol':1e-10,'scalar_rtol':1e-10,'scalar_per_role':16,'pairs':[list(x) for x in PAIRS]}


def _binding(info):
    return {'schema':'realclock_physics_outlier_b3b2b_v1','parent':info['parent'],'parent_owner':info['owner'],
        'parent_identity':info['identity'],'clean':info['clean'],'protocol':protocol(),'sources':sources(PROJECT),
        'references':dict(info['references']),'runtime':{'python':sys.version,'numpy':str(np.__version__),
        'platform':platform.platform(),'threads':{k:os.environ.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}}}


def _load(info):
    parent=Path(info['parent'])
    with np.load(parent/'shared_clean.npz',allow_pickle=False) as f:base={k:f[k] for k in f.files}
    windows=read_json(parent/'windows.json');packets=validate_packets(read_json(parent/'source_packets.json'))
    indices=window_indices(packets,windows)
    if indices.shape!=(len(windows),17) or len(np.unique(indices))!=len(packets):raise ValueError('outlier source coverage mismatch')
    return base,windows,packets,indices


def _cases():
    return [('clean',None),*[(c,s) for c in CONDITIONS[1:] for s in SEEDS]]


def _prefix(condition,seed):return 'clean' if seed is None else f'{condition}/seed_{seed}'


def _summary(info,base,windows,packets,results,alarms,flags,regressions):
    scores={};seed_rows=[];diagnostic_rows=[];paired_rows=[];pair_records=defaultdict(list)
    for condition,seed in _cases():
        res=results[condition,seed]
        for strategy in STRATEGIES:
            metric=summarize_errors(res[strategy]['errors'],base,windows);scores[condition,seed,strategy]=metric
            seed_rows.append({'condition':condition,'noise_seed':seed,'strategy':strategy,'groups':metric['groups']})
        pairs=list(PAIRS);pool=dict(res)
        if seed is not None:
            for strategy in STRATEGIES:
                pool['clean_'+strategy]=results['clean',None][strategy]
                pairs.append(('extra_clean_'+strategy,strategy,'clean_'+strategy))
        if condition=='gaussian_outlier':
            for strategy in STRATEGIES:
                pool['gaussian_'+strategy]=results['gaussian_05',seed][strategy]
                pairs.append(('extra_outlier_'+strategy,strategy,'gaussian_'+strategy))
        paired=paired_gains(pool,base,windows,pairs)
        # All comparisons are deliberately reference-minus-compared, positive = more error.
        paired_rows.append({'condition':condition,'noise_seed':seed,'positive_means':'first_named_strategy_or_condition_worse',
                            'groups':paired['groups']})
        for name,first,second in pairs:
            if seed is not None:pair_records[condition,name].append({'seed':seed,'values':paired['arrays'][name]})
        for strategy in NIS_STRATEGIES:
            rows=diagnostics(alarms[condition,seed,strategy],flags[condition,seed],res['kf_actual_dt']['errors'],
                             res[strategy]['errors'],base,windows)
            diagnostic_rows.append({'condition':condition,'noise_seed':seed,'strategy':strategy,'groups':rows})
    macros=[];aggregates={};delta_macros=[]
    for condition in CONDITIONS[1:]:
        for strategy in STRATEGIES:
            records=[{'seed':s,**scores[condition,s,strategy]} for s in SEEDS]
            result=aggregate_scores(records,base,windows)
            aggregates[f'aggregates/{condition}/{strategy}.npz']=result['arrays']
            macros.append({'condition':condition,'strategy':strategy,'groups':result['groups']})
    for (condition,name),records in sorted(pair_records.items()):
        result=aggregate_differences(records,base,windows,'mean_window_extra_error_m')
        aggregates[f'aggregates/{condition}/{name}.npz']=result['arrays']
        delta_macros.append({'condition':condition,'comparison':name,'positive_means':'first_named_strategy_or_condition_worse',
                             'groups':result['groups']})
    source_stats=[]
    for seed in SEEDS:
        table=outlier_table(packets,seed)
        for role in sorted({p['role'] for p in packets}):
            mask=np.array([p['role']==role for p in packets]);total=int(mask.sum());num=int(table['injected'][mask].sum())
            source_stats.append({'noise_seed':seed,'role':role,'unique_sources':total,'injected_sources':num,
                                 'injected_fraction':num/total,'unit':'unique_source_per_seed'})
    summary={'status':'source_outlier_nis_complete','windows':len(windows),'combinations':28,
        'queries_per_combination':len(windows)*11,'valid_pairs_per_combination':int(base['valid'].sum()),
        'unique_source_packets':len(packets),'noise_seeds':list(SEEDS),'seed_results':seed_rows,
        'seed_macro_results':macros,'seed_differences':paired_rows,'paired_seed_macro_results':delta_macros,
        'diagnostics':diagnostic_rows,'source_injection':source_stats,'regressions':regressions,
        'human_labels_imported':False,'neural_training':False,'holdout_evaluated':False,
        'official_challenge_metrics':False,'synthetic_observation_noise':True,'new_prediction_scores_computed':True}
    return summary,aggregates


def _regression(state,score,info,condition,seed,strategy):
    if strategy not in STRATEGIES[:2] or condition=='gaussian_outlier':return None
    root=Path(info['clean']) if condition=='clean' else Path(info['parent'])/_prefix(condition,seed)
    with np.load(root/(strategy+'.npz'),allow_pickle=False) as f:
        actual={**state,**score}
        if any(not np.array_equal(actual[k],f[k]) for k in ('states','covariances','query_us','predictions','errors','valid')):
            raise ValueError('frozen parent regression mismatch: '+condition+'/'+strategy)
    return {'condition':condition,'noise_seed':seed,'strategy':strategy,'status':'exact_parent_match'}


def _stream(info):
    base,windows,packets,indices=_load(info);selected=sample_indices(windows)
    yield 'protocol.json',_json(protocol());yield 'parent_proof.json',_json(info['proof'])
    yield 'shared_clean.npz',_npz(base);yield 'windows.json',_json(windows);yield 'source_packets.json',_json(packets)
    yield 'source_point_indices.npz',_npz({'indices':indices})
    tables={}
    for seed in SEEDS:
        tables[seed]={'normals':normal_table(packets,seed),**outlier_table(packets,seed)}
        yield f'source_noise/{seed}.npz',_npz(tables[seed])
    results={};alarms={};flags={};regressions=[]
    for condition,seed in _cases():
        prefix=_prefix(condition,seed)
        normal=np.zeros_like(base['positions_local']) if seed is None else tables[seed]['normals'][indices]
        spike=np.zeros_like(normal) if seed is None else tables[seed]['outliers'][indices]
        injected=np.zeros((len(windows),16),bool) if condition!='gaussian_outlier' else tables[seed]['injected'][indices[:,:16]]
        observed,evaluation,context=outlier_frame(base,normal,spike,condition)
        yield prefix+'/observations.npz',_npz({**context,'injected':injected})
        results[condition,seed]={};flags[condition,seed]=injected
        for strategy in STRATEGIES:
            if strategy in NIS_STRATEGIES:
                state=replay_nis(base['times_us'][:,:16],observed,strategy)
                check_nis_samples(state,base['times_us'][:,:16],observed,selected,strategy)
                alarms[condition,seed,strategy]=state['alarm']
            else:
                state=replay_states(base['times_us'][:,:16],observed,strategy)
                check_samples(state,{'times_us':base['times_us'],'positions_local':observed},windows,selected,strategy)
            if not np.array_equal(state['query_us'],base['query_us']):raise ValueError('outlier query clock mismatch')
            score=evaluate_states(state['states'],evaluation)
            regression=_regression(state,score,info,condition,seed,strategy)
            if regression is not None:regressions.append(regression)
            metrics=summarize_errors(score['errors'],evaluation,windows)
            yield prefix+'/'+strategy+'.npz',_npz({**state,**score,**metrics['window_metrics']})
            results[condition,seed][strategy]={'errors':score['errors'],'valid':score['valid']}
    summary,aggregates=_summary(info,base,windows,packets,results,alarms,flags,regressions)
    for name,value in aggregates.items():yield name,_npz(value)
    yield 'scalar_reference.json',_json({'status':'verified','sample_windows':[windows[i]['window_id'] for i in selected],
        'nis_all_update_states_checked':len(selected)*14*14,'nis_covariances_checked':len(selected)*14*14,
        'baseline_query_states_checked':len(selected)*14*11,'atol':1e-10,'rtol':1e-10})
    yield 'summary.json',_json(summary);yield 'progress.json',_json({'state':'complete'})


def _check(binding):
    if sources(PROJECT)!=binding['sources']:raise ValueError('outlier source changed')
    check_references(binding['references'])


def prepare_outlier(parent,parent_owner,output):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('use a new outlier output directory')
    before=sources(PROJECT);info=snapshot(parent,parent_owner);binding=_binding(info)
    if before!=binding['sources']:raise ValueError('source changed during original-owner verification')
    output.mkdir(parents=True,exist_ok=False)
    try:
        _write_bytes(output/'progress.json',_json({'state':'assembling'}));hashes={}
        for name,value in _stream(info):
            path=output/name;path.parent.mkdir(parents=True,exist_ok=True);_write_bytes(path,value)
            if path.read_bytes()!=value:raise ValueError('outlier write/read mismatch')
            hashes[name]=sha256_file(path)
        _write_bytes(output/'run.json',_json({'identity':digest(binding),'binding':binding}));hashes['run.json']=sha256_file(output/'run.json')
        _check(binding);_files(output,hashes)
        _write_bytes(output/'completed.json',_json({'identity':digest(binding),'sha256':hashes}))
    except BaseException:
        (output/'progress.json').write_bytes(_json({'state':'failed'}));raise
    return read_json(output/'summary.json')


def rescore(output,info,summary):
    base,windows,packets,indices=_load(info);results={};alarms={};flags={};regressions=[]
    for condition,seed in _cases():
        prefix=output/_prefix(condition,seed)
        with np.load(prefix/'observations.npz',allow_pickle=False) as f:offset=f['origin_offset_xy']
        evaluation={**base,'positions_local':base['positions_local']-offset[:,None,:]}
        flags[condition,seed]=(outlier_table(packets,seed)['injected'][indices[:,:16]]
                               if condition=='gaussian_outlier' else np.zeros((len(windows),16),bool))
        results[condition,seed]={}
        for strategy in STRATEGIES:
            with np.load(prefix/(strategy+'.npz'),allow_pickle=False) as f:
                state={k:f[k] for k in ('states','covariances','query_us')}
                score={'predictions':f['predictions'],'errors':f['errors'],'valid':f['valid']}
                if strategy in NIS_STRATEGIES:alarms[condition,seed,strategy]=f['alarm']
            if not np.array_equal(score['valid'],base['valid']):raise ValueError('saved outlier mask mismatch')
            errors=rescore_predictions(score['predictions'],evaluation)
            if not np.allclose(errors,score['errors'],atol=1e-10,rtol=1e-10):raise ValueError('saved outlier rescore mismatch')
            regression=_regression(state,score,info,condition,seed,strategy)
            if regression is not None:regressions.append(regression)
            results[condition,seed][strategy]={'errors':errors,'valid':score['valid']}
    actual,_aggregates=_summary(info,base,windows,packets,results,alarms,flags,regressions)
    if not _close(actual,summary):raise ValueError('saved prediction/diagnostic aggregate rescore mismatch')


def verify_outlier(output):
    output=Path(output).resolve();marker_path=output/'completed.json'
    if not marker_path.is_file():raise ValueError('incomplete outlier output')
    marker_hash=sha256_file(marker_path);marker=read_json(marker_path)
    if set(marker)!={'identity','sha256'}:raise ValueError('strict outlier marker required')
    _files(output,marker['sha256'],True);run=read_json(output/'run.json');binding=run['binding']
    if digest(binding)!=run['identity'] or marker['identity']!=run['identity']:raise ValueError('outlier identity mismatch')
    _check(binding);info=snapshot(binding['parent'],binding['parent_owner'],read_json(output/'parent_proof.json'))
    if digest(binding)!=digest(_binding(info)):raise ValueError('outlier protocol/runtime replay mismatch')
    seen={'run.json'}
    for name,value in _stream(info):
        seen.add(name)
        if (output/name).read_bytes()!=value:raise ValueError('outlier semantic replay mismatch: '+name)
    if seen!=set(marker['sha256']) or (output/'run.json').read_bytes()!=_json(run):raise ValueError('outlier expected coverage mismatch')
    summary=read_json(output/'summary.json');rescore(output,info,summary)
    _check(binding);_files(output,marker['sha256'],True)
    if sha256_file(marker_path)!=marker_hash:raise ValueError('outlier marker changed')
    return {**summary,'identity':run['identity'],'outlier_semantically_replayed':True,
            'saved_predictions_independently_rescored':True,'scalar_reference_verified':True}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path);parser.add_argument('--parent-owner',type=Path)
    parser.add_argument('--output-root',type=Path,required=True);parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:result=verify_outlier(args.output_root)
    else:
        if args.parent is None or args.parent_owner is None:parser.error('explicit original Gaussian parent/owner required')
        result=prepare_outlier(args.parent,args.parent_owner,args.output_root)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)

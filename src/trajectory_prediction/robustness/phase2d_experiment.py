"""Paired multi-initialization development trials with a fixed causal CA guard."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.data.cache import load_cache
from trajectory_prediction.online.batched_evaluation import forecast_metrics
from trajectory_prediction.robustness.benchmark import PROJECT, ROLES
from trajectory_prediction.robustness.maneuver_diagnostics import build_tags, grouped_metrics
from trajectory_prediction.robustness.motion_guard import GUARD_SETTINGS, build_guard
from trajectory_prediction.robustness.motion_learning import MotionGate, fit_motion, predict_motion
from trajectory_prediction.robustness.packets import PROFILES, make_packets
from trajectory_prediction.robustness.phase2_experiment import summaries
from trajectory_prediction.robustness.reliability_learning import TRAINING
from trajectory_prediction.robustness.seed_statistics import seed_statistics
from trajectory_prediction.utils.provenance import sha256_file

VARIANTS = ('baseline7', 'plus_ca', 'guarded_ca')
MODEL_SEEDS = (2027, 2028, 2029)


def sources():
    paths = sorted((PROJECT/'src/trajectory_prediction').rglob('*.py'))
    paths += [PROJECT/'scripts/run_noise_aware_stability.py', PROJECT/'configs/analysis/maneuvers.yaml',
              PROJECT/'docs/superpowers/specs/2026-11-10-noise-aware-phase2d-design.md']
    return {p.relative_to(PROJECT).as_posix(): sha256_file(p) for p in paths}


def load_inputs(parent):
    parent = Path(parent).resolve()
    project = parent.parents[2]
    process = subprocess.run([sys.executable, str(project/'scripts/run_noise_aware_motion.py'),
                              '--output-root', str(parent), '--verify-only'], cwd=project,
                             env={**os.environ, 'PYTHONPATH': str(project/'src'), 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'},
                             check=True, capture_output=True, text=True)
    proof = json.loads(process.stdout.splitlines()[-1])
    if not proof.get('all_metrics_groups_weights_replayed') or proof.get('holdout_evaluated'):
        raise ValueError('verified development-only parent required')
    marker = json.loads((parent/'completed.json').read_text())
    old_binding = json.loads((parent/'run.json').read_text())['binding']
    cache = Path(old_binding['source_cache_root'])
    references = {name: {'path': str(parent/name), 'sha256': digest} for name, digest in marker['sha256'].items()}
    references['parent_completion'] = {'path': str(parent/'completed.json'), 'sha256': sha256_file(parent/'completed.json')}
    datasets, original = {}, {}
    for role in ROLES:
        with np.load(parent/f'{role}.npz', allow_pickle=False) as z:
            datasets[role] = {k: z[k] for k in z.files}
        original[role], manifest = load_cache(cache/role, role)
        for path in (cache/role/'manifest.json', cache/role/manifest['npz_file']):
            references[str(path)] = {'path': str(path), 'sha256': sha256_file(path)}
    records = json.loads((parent/'records.json').read_text())
    baseline = pd.read_csv(parent/'per_trajectory.csv', float_precision='round_trip')
    return datasets, records, baseline, {'parent_path': str(parent), 'parent_identity': proof['identity'],
                                        'source_cache_root': str(cache), 'references': references}, original


def rebuild_guard(original, row):
    arrays, i = original[row['role']], int(row['row'])
    if not arrays['history_mask'][i].all():
        raise ValueError('guard requires complete real history')
    key = str(arrays['instance_token'][i])+':'+str(arrays['sample_token'][i])
    packets = make_packets(arrays['history_xy'][i], arrays['future_xy'][i, :11], key,
                           PROFILES[row['condition']], int(row['seed']))
    return build_guard(packets)


def attach_guards(datasets, records, original):
    result = {}
    for role in ROLES:
        values = {'ca_allowed': [], 'guard_yaw': [], 'guard_min_speed': []}
        for i, row in enumerate(records[role]):
            if not np.array_equal(datasets[role]['target'][i], original[role]['future_xy'][row['row']]):
                raise ValueError('guard trajectory key/target mismatch')
            guard = rebuild_guard(original, row)
            for name, items in values.items():
                items.append(guard[name])
        result[role] = {**datasets[role], **{k: np.array(v) for k, v in values.items()}}
        print(json.dumps({'prepared_guard_role': role, 'streams': len(records[role])}), flush=True)
    return result


def model_name(seed, variant):
    return f's{seed}_{variant}'


def _evaluate(output, datasets, records, baseline, binding, *, check_stats=False):
    rows, updates, weight_rows, coverage = [], {}, [], []
    tags = build_tags(datasets, records, binding['thresholds'])
    lookup = {(t['role'], t['row']): t['tags'] for t in tags}
    fit_flat = datasets[ROLES[0]]['features'].reshape(-1, 18)
    metadata = {model_name(s, v): {'train_seed': s, 'variant': v} for s in binding['model_seeds'] for v in VARIANTS}
    for role in ROLES:
        data = datasets[role]
        cohorts = []
        for condition in sorted({r['condition'] for r in records[role]}):
            for tag in ['all', *sorted({t for r in records[role] for t in lookup[role, r['row']]})]:
                indices = [i for i, r in enumerate(records[role]) if r['condition'] == condition
                           and (tag == 'all' or tag in lookup[role, r['row']])]
                if indices:
                    context = {'role': role, 'condition': condition, 'tag': tag}
                    cohorts.append((context, indices))
                    coverage.append({**context, 'blocked_fraction': float((~data['ca_allowed'][indices]).mean()),
                                     'unique_trajectories': len({records[role][i]['row'] for i in indices}),
                                     'updates': len(indices)*11})
        for name, meta in metadata.items():
            payload = torch.load(output/f'{name}.pt', map_location='cpu', weights_only=True)
            if payload['train_seed'] != meta['train_seed'] or payload['variant'] != meta['variant']:
                raise ValueError('training seed/control identity mismatch')
            if check_stats:
                for k, value in [('mean', fit_flat.mean(0, dtype=np.float64)),
                                 ('std', np.maximum(fit_flat.std(0, dtype=np.float64), 1e-4))]:
                    if not np.array_equal(payload['stats'][k], value):
                        raise ValueError('seeded normalization must use fit only')
            model = MotionGate(seed=meta['train_seed'])
            model.load_state_dict(payload['model'], strict=True)
            model.eval()
            forecast, weight = predict_motion(model, payload['stats'], data['features'], data['candidates'], meta['variant'],
                                               ca_allowed=data['ca_allowed'] if meta['variant'] == 'guarded_ca' else None)
            if meta['variant'] == 'guarded_ca' and weight[:, :, 7][~data['ca_allowed']].any():
                raise ValueError('guarded CA must have exact zero weight when blocked')
            metrics = forecast_metrics(forecast, data['target'])
            updates[role+'__'+name] = metrics
            rows.extend([{**r, 'strategy': name, **meta,
                          **dict(zip(('ade', 'fde', 'jitter'), v.tolist(), strict=True))}
                         for r, v in zip(records[role], metrics.mean(1), strict=True)])
            for context, indices in cohorts:
                weight_rows.append({**context, **meta, 'ca_weight': float(weight[indices, :, 7].mean()),
                                    'cv_weight': float(weight[indices, :, 0].mean()),
                                    'kf_nis_weight': float(weight[indices, :, 3].mean())})
    frame = pd.DataFrame(rows)
    if len(baseline):
        frame = pd.concat((baseline, frame), ignore_index=True)
    groups = grouped_metrics(frame, tags)
    for row in groups:
        if row['strategy'] in metadata:
            row.update(metadata[row['strategy']])
    statistics = {'conditions': seed_statistics(frame),
                  'maneuvers': seed_statistics(pd.DataFrame(groups), extra=('tag',))}
    return frame, updates, tags, groups, weight_rows, coverage, statistics


def publish(output, datasets, records, baseline, binding, thresholds, *, epochs=20, seeds=MODEL_SEEDS):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if len(set(seeds)) != len(seeds) or not seeds:
        raise ValueError('unique nonempty model seeds required')
    if {r['log'] for r in records[ROLES[0]]} & {r['log'] for r in records[ROLES[1]]}:
        raise ValueError('seeded fit/select log overlap')
    training = {k: v for k, v in TRAINING.items() if k != 'seed'}
    binding = {**binding, 'sources': sources(), 'thresholds': thresholds, 'guard_settings': GUARD_SETTINGS,
               'model_seeds': list(seeds), 'noise_seeds': sorted({int(r['seed']) for rows in records.values() for r in rows}),
               'training': {**training, 'epochs': epochs}, 'variants': list(VARIANTS),
               'runtime': {'python': sys.version, 'numpy': str(np.__version__), 'torch': str(torch.__version__),
                           'threads': torch.get_num_threads(), 'device': 'cpu'},
               'git_commit': subprocess.run(['git', '-c', 'safe.directory=*', 'rev-parse', 'HEAD'],
                                            check=True, capture_output=True, text=True).stdout.strip()}
    identity = digest_json(binding)
    (output/'run.json').write_text(json.dumps({'identity': identity, 'binding': binding}, indent=2)+'\n')
    (output/'records.json').write_text(json.dumps(records, indent=2)+'\n')
    baseline.to_csv(output/'baseline.csv', index=False)
    for role in ROLES:
        np.savez_compressed(output/f'{role}.npz', **datasets[role])
    start, logs = time.perf_counter(), {}
    fit = datasets[ROLES[0]]
    for seed in seeds:
        for variant in VARIANTS:
            name = model_name(seed, variant)
            model, stats, log = fit_motion(fit['features'], fit['candidates'], fit['target'], variant, epochs=epochs, seed=seed,
                                           ca_allowed=fit['ca_allowed'] if variant == 'guarded_ca' else None)
            torch.save({'train_seed': seed, 'variant': variant, 'model': model.state_dict(), 'stats': stats}, output/f'{name}.pt')
            logs[name] = log
            print(json.dumps({'train_seed': seed, 'variant': variant, 'final_fit_loss': log[-1]['fit_ade_loss']}), flush=True)
    frame, updates, tags, groups, weights, coverage, stats = _evaluate(output, datasets, records, baseline, binding)
    frame.to_csv(output/'per_trajectory.csv', index=False)
    np.savez_compressed(output/'per_update.npz', **updates)
    summary = {**summaries(frame), 'identity': identity, 'model_seeds': list(seeds), 'seconds': time.perf_counter()-start}
    for name, value in [('summary', summary), ('training', logs), ('tags', tags), ('maneuvers', groups),
                        ('weights', weights), ('guard_coverage', coverage), ('seed_summary', stats)]:
        (output/f'{name}.json').write_text(json.dumps(value, indent=2)+'\n')
    if sources() != binding['sources']:
        raise ValueError('seeded source changed during run')
    names = ['run.json', 'records.json', 'baseline.csv', 'per_trajectory.csv', 'per_update.npz',
             *[f'{r}.npz' for r in ROLES], *[f'{model_name(s,v)}.pt' for s in seeds for v in VARIANTS],
             *[f'{k}.json' for k in ('summary', 'training', 'tags', 'maneuvers', 'weights', 'guard_coverage', 'seed_summary')]]
    (output/'completed.json').write_text(json.dumps({'identity': identity, 'sha256': {
        name: sha256_file(output/name) for name in names}}, indent=2)+'\n')
    return summary


def verify_output(output):
    output = Path(output)
    marker = json.loads((output/'completed.json').read_text())
    for name, digest in marker['sha256'].items():
        if sha256_file(output/name) != digest:
            raise ValueError('seeded output checksum mismatch')
    run = json.loads((output/'run.json').read_text())
    binding = run['binding']
    if digest_json(binding) != marker['identity'] or run['identity'] != marker['identity'] or sources() != binding['sources']:
        raise ValueError('seeded source/run identity mismatch')
    for ref in binding.get('references', {}).values():
        if sha256_file(Path(ref['path'])) != ref['sha256']:
            raise ValueError('seeded parent checksum mismatch')
    datasets = {}
    for role in ROLES:
        with np.load(output/f'{role}.npz', allow_pickle=False) as z:
            datasets[role] = {k: z[k] for k in z.files}
    records = json.loads((output/'records.json').read_text())
    if {r['log'] for r in records[ROLES[0]]} & {r['log'] for r in records[ROLES[1]]}:
        raise ValueError('seeded fit/select log overlap')
    baseline = pd.read_csv(output/'baseline.csv', float_precision='round_trip')
    frame, updates, tags, groups, weights, coverage, stats = _evaluate(output, datasets, records, baseline, binding, check_stats=True)
    saved = pd.read_csv(output/'per_trajectory.csv', float_precision='round_trip')
    if not np.array_equal(frame[['ade', 'fde', 'jitter']].to_numpy(), saved[['ade', 'fde', 'jitter']].to_numpy()):
        raise ValueError('seeded metric replay mismatch')
    with np.load(output/'per_update.npz', allow_pickle=False) as z:
        if set(z.files) != set(updates) or any(not np.array_equal(z[k], updates[k]) for k in updates):
            raise ValueError('seeded per-update replay mismatch')
    summary = json.loads((output/'summary.json').read_text())
    for name, value in summaries(frame).items():
        if summary[name] != value:
            raise ValueError('seeded summary replay mismatch')
    for name, value in [('tags', tags), ('maneuvers', groups), ('weights', weights), ('guard_coverage', coverage), ('seed_summary', stats)]:
        if json.loads((output/f'{name}.json').read_text()) != value:
            raise ValueError('seeded group/statistic replay mismatch')
    return {'identity': marker['identity'], 'all_seed_metrics_groups_weights_replayed': True,
            'fit_only_normalization_verified': True, 'holdout_evaluated': False}


def verify_lineage(output, parent):
    data, records, _, _, original = load_inputs(parent)
    output, parent = Path(output), Path(parent)
    for variant in ('baseline7', 'plus_ca'):
        a = torch.load(output/f'{model_name(2027,variant)}.pt', map_location='cpu', weights_only=True)
        b = torch.load(parent/f'{variant}.pt', map_location='cpu', weights_only=True)
        if a['stats'] != b['stats'] or any(not torch.equal(v, b['model'][k]) for k, v in a['model'].items()):
            raise ValueError('2027 controls must reproduce parent weights')
    cached = {}
    for role in ROLES:
        with np.load(output/f'{role}.npz', allow_pickle=False) as z:
            cached[role] = {k: z[k] for k in z.files}
        for key in ('features', 'candidates', 'target'):
            if not np.array_equal(data[role][key], cached[role][key]):
                raise ValueError('seeded parent input changed')
    items = [(r,i,row) for r in ROLES for i,row in enumerate(records[r])]
    for k in np.linspace(0, len(items)-1, min(64,len(items)), dtype=int):
        role, i, row = items[k]
        for key, value in rebuild_guard(original,row).items():
            if not np.array_equal(cached[role][key][i],value):
                raise ValueError('causal guard replay mismatch')
    frame = pd.read_csv(output/'per_trajectory.csv', float_precision='round_trip')
    keys = ['role','row','condition','seed']
    for variant in ('baseline7','plus_ca'):
        a,b = (frame[frame.strategy==name].sort_values(keys) for name in (model_name(2027,variant),variant))
        if (not np.array_equal(a[keys].to_numpy(),b[keys].to_numpy())
                or not np.array_equal(a[['ade','fde','jitter']].to_numpy(),b[['ade','fde','jitter']].to_numpy())):
            raise ValueError('2027 metric identity mismatch')
    return {'seed2027_two_controls_match_parent': True, 'matched_streams_per_control': len(a), 'guard_stream_spot_checks': min(64,len(items))}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,default=Path('../noise-aware-motion/outputs/noise-aware/phase2c'))
    parser.add_argument('--output-root',type=Path,default=Path('outputs/noise-aware/phase2d'))
    parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args(argv)
    torch.set_num_threads(2)
    if args.verify_only:
        result=verify_output(args.output_root)
        parent=Path(json.loads((args.output_root/'run.json').read_text())['binding']['parent_path'])
        print(json.dumps({**result,**verify_lineage(args.output_root,parent)}),flush=True)
    else:
        if args.output_root.exists():
            raise FileExistsError('use a new seeded experiment directory')
        before=sources()
        datasets,records,baseline,binding,original=load_inputs(args.parent)
        datasets=attach_guards(datasets,records,original)
        if before!=sources():
            raise ValueError('seeded source changed during preparation')
        result=publish(args.output_root,datasets,records,baseline,binding,load_yaml(PROJECT/'configs/analysis/maneuvers.yaml'))
        print(json.dumps({k:result[k] for k in ('identity','rows','seconds')}),flush=True)


if __name__=='__main__':
    main()

"""Frozen-budget bounded-motion candidate experiments and source-chain replay."""
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
from trajectory_prediction.robustness.motion_experts import (
    EXPERTS,
    MOTION_SETTINGS,
    build_motion_experts,
)
from trajectory_prediction.robustness.motion_learning import (
    CONTROLS,
    MotionGate,
    fit_motion,
    predict_motion,
)
from trajectory_prediction.robustness.noisy_candidates import CANDIDATES
from trajectory_prediction.robustness.packets import PROFILES, make_packets
from trajectory_prediction.robustness.phase2_experiment import summaries
from trajectory_prediction.robustness.reliability_learning import TRAINING
from trajectory_prediction.utils.provenance import sha256_file


def sources():
    paths = sorted((PROJECT/'src/trajectory_prediction').rglob('*.py'))
    paths += [PROJECT/'scripts/run_noise_aware_motion.py', PROJECT/'configs/analysis/maneuvers.yaml',
              PROJECT/'docs/superpowers/specs/2026-11-10-noise-aware-phase2c-design.md']
    return {p.relative_to(PROJECT).as_posix(): sha256_file(p) for p in paths}


def load_inputs(parent):
    parent = Path(parent).resolve()
    project = parent.parents[2]
    result = subprocess.run([sys.executable, str(project/'scripts/run_noise_aware_diagnostics.py'),
                             '--output-root', str(parent), '--verify-only'], cwd=project,
                            env={**os.environ, 'PYTHONPATH': str(project/'src'), 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'},
                            check=True, capture_output=True, text=True)
    proof = json.loads(result.stdout.splitlines()[-1])
    if not proof.get('all_metrics_and_groups_replayed') or proof.get('holdout_evaluated'):
        raise ValueError('verified development-only parent required')
    marker = json.loads((parent/'completed.json').read_text())
    binding_b = json.loads((parent/'run.json').read_text())['binding']
    parent_a = Path(binding_b['parent_path'])
    binding_a = json.loads((parent_a/'run.json').read_text())['binding']
    cache = Path(binding_a['model_root'])/'cache'
    references = {name: {'path': str(parent/name), 'sha256': digest} for name, digest in marker['sha256'].items()}
    references['parent_completion'] = {'path': str(parent/'completed.json'), 'sha256': sha256_file(parent/'completed.json')}
    datasets, original = {}, {}
    for role in ROLES:
        with np.load(parent/f'{role}.npz', allow_pickle=False) as z:
            datasets[role] = {k: z[k] for k in z.files}
        original[role], manifest = load_cache(cache/role, role)
        if manifest['partition_identity'] != binding_a['partition_identity']:
            raise ValueError('raw motion history partition mismatch')
        for path in (cache/role/'manifest.json', cache/role/manifest['npz_file']):
            references[str(path)] = {'path': str(path), 'sha256': sha256_file(path)}
    records = json.loads((parent/'records.json').read_text())
    baseline = pd.read_csv(parent/'per_trajectory.csv', float_precision='round_trip')
    binding = {'parent_path': str(parent), 'parent_identity': proof['identity'], 'references': references,
               'source_cache_root': str(cache), 'partition_identity': binding_a['partition_identity']}
    return datasets, records, baseline, binding, original


def rebuild(original, row):
    arrays = original[row['role']]
    i = int(row['row'])
    if not arrays['history_mask'][i].all():
        raise ValueError('motion priors require complete real history')
    key = str(arrays['instance_token'][i])+':'+str(arrays['sample_token'][i])
    return build_motion_experts(make_packets(arrays['history_xy'][i], arrays['future_xy'][i, :11], key,
                                             PROFILES[row['condition']], int(row['seed'])))


def expand_inputs(datasets, records, original):
    expanded = {}
    for role in ROLES:
        data = datasets[role]
        candidates = np.empty((len(records[role]), 11, 10, 12, 2), np.float32)
        candidates[:, :, :7] = data['candidates']
        for i, row in enumerate(records[role]):
            if not np.array_equal(data['target'][i], original[role]['future_xy'][row['row']]):
                raise ValueError('raw motion trajectory key/label mismatch')
            for name in ('instance', 'sample'):
                if row[name] != str(original[role][name+'_token'][row['row']]):
                    raise ValueError('raw motion identifier mismatch')
            candidates[i, :, 7:] = rebuild(original, row)
        expanded[role] = {**data, 'candidates': candidates}
        print(json.dumps({'prepared_motion_role': role, 'streams': len(records[role])}), flush=True)
    return expanded


def _evaluate(output, datasets, records, baseline, thresholds, *, check_stats=False):
    rows, updates, weight_rows = [], {}, []
    tags = build_tags(datasets, records, thresholds)
    tag_lookup = {(t['role'], t['row']): t['tags'] for t in tags}
    fit_flat = datasets[ROLES[0]]['features'].reshape(-1, 18)
    for role in ROLES:
        data = datasets[role]
        forecasts = {name: data['candidates'][:, :, 7+j] for j, name in enumerate(EXPERTS)}
        for variant in CONTROLS:
            payload = torch.load(output/f'{variant}.pt', map_location='cpu', weights_only=True)
            if payload['variant'] != variant:
                raise ValueError('motion control identity mismatch')
            stats = payload['stats']
            if check_stats:
                for field, value in [('mean', fit_flat.mean(0, dtype=np.float64)),
                                     ('std', np.maximum(fit_flat.std(0, dtype=np.float64), 1e-4))]:
                    if not np.array_equal(stats[field], value):
                        raise ValueError('motion normalization must use fit only')
            model = MotionGate()
            model.load_state_dict(payload['model'], strict=True)
            model.eval()
            forecasts[variant], weights = predict_motion(model, stats, data['features'], data['candidates'], variant)
            for condition in sorted({r['condition'] for r in records[role]}):
                for tag in ['all', *sorted({t for r in records[role] for t in tag_lookup[role, r['row']]})]:
                    selected = [i for i, r in enumerate(records[role]) if r['condition'] == condition
                                and (tag == 'all' or tag in tag_lookup[role, r['row']])]
                    if selected:
                        weight_rows.append({'role': role, 'condition': condition, 'tag': tag, 'strategy': variant,
                                            'weights': dict(zip((*CANDIDATES, *EXPERTS),
                                                weights[selected].mean((0, 1)).tolist(), strict=True))})
        for strategy, forecast in forecasts.items():
            metrics = forecast_metrics(forecast, data['target'])
            updates[role+'__'+strategy] = metrics
            rows.extend([{**r, 'strategy': strategy, **dict(zip(('ade', 'fde', 'jitter'), value.tolist(), strict=True))}
                         for r, value in zip(records[role], metrics.mean(1), strict=True)])
    frame = pd.DataFrame(rows)
    if len(baseline):
        frame = pd.concat((baseline, frame), ignore_index=True)
    return frame, updates, tags, grouped_metrics(frame, tags), weight_rows


def publish(output, datasets, records, baseline, binding, thresholds, *, epochs=20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if {r['log'] for r in records[ROLES[0]]} & {r['log'] for r in records[ROLES[1]]}:
        raise ValueError('motion fit/select log overlap')
    binding = {**binding, 'sources': sources(), 'thresholds': thresholds, 'motion_settings': MOTION_SETTINGS,
               'training': {**TRAINING, 'epochs': epochs}, 'controls': list(CONTROLS),
               'initialization': {'extra_weight': 0., 'extra_bias': -2., 'warmstart_trained': False},
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
    started, training = time.perf_counter(), {}
    fit = datasets[ROLES[0]]
    for variant in CONTROLS:
        model, stats, log = fit_motion(fit['features'], fit['candidates'], fit['target'], variant, epochs=epochs)
        torch.save({'variant': variant, 'model': model.state_dict(), 'stats': stats}, output/f'{variant}.pt')
        training[variant] = log
        print(json.dumps({'trained_motion_control': variant, 'final_fit_loss': log[-1]['fit_ade_loss']}), flush=True)
    frame, updates, tags, groups, weights = _evaluate(output, datasets, records, baseline, thresholds)
    frame.to_csv(output/'per_trajectory.csv', index=False)
    np.savez_compressed(output/'per_update.npz', **updates)
    summary = {**summaries(frame), 'identity': identity, 'seconds': time.perf_counter()-started,
               'unique_trajectories_by_role': {r: len({v['row'] for v in records[r]}) for r in ROLES}}
    for name, value in [('summary', summary), ('training', training), ('tags', tags), ('maneuvers', groups), ('weights', weights)]:
        (output/f'{name}.json').write_text(json.dumps(value, indent=2)+'\n')
    if sources() != binding['sources']:
        raise ValueError('motion source changed during experiment')
    names = ['run.json', 'records.json', 'baseline.csv', 'per_trajectory.csv', 'per_update.npz',
             *[f'{r}.npz' for r in ROLES], *[f'{v}.pt' for v in CONTROLS],
             *[f'{v}.json' for v in ('summary', 'training', 'tags', 'maneuvers', 'weights')]]
    (output/'completed.json').write_text(json.dumps({'identity': identity, 'sha256': {
        name: sha256_file(output/name) for name in names}}, indent=2)+'\n')
    return summary


def verify_output(output):
    output = Path(output)
    marker = json.loads((output/'completed.json').read_text())
    for name, digest in marker['sha256'].items():
        if sha256_file(output/name) != digest:
            raise ValueError('motion output checksum mismatch')
    run = json.loads((output/'run.json').read_text())
    binding = run['binding']
    if digest_json(binding) != marker['identity'] or run['identity'] != marker['identity'] or sources() != binding['sources']:
        raise ValueError('motion source/run identity mismatch')
    for ref in binding.get('references', {}).values():
        if sha256_file(Path(ref['path'])) != ref['sha256']:
            raise ValueError('motion parent checksum mismatch')
    datasets = {}
    for role in ROLES:
        with np.load(output/f'{role}.npz', allow_pickle=False) as z:
            datasets[role] = {k: z[k] for k in z.files}
    records = json.loads((output/'records.json').read_text())
    if {r['log'] for r in records[ROLES[0]]} & {r['log'] for r in records[ROLES[1]]}:
        raise ValueError('motion fit/select log overlap')
    baseline = pd.read_csv(output/'baseline.csv', float_precision='round_trip')
    frame, updates, tags, groups, weights = _evaluate(output, datasets, records, baseline, binding['thresholds'], check_stats=True)
    saved = pd.read_csv(output/'per_trajectory.csv', float_precision='round_trip')
    if not np.array_equal(frame[['ade', 'fde', 'jitter']].to_numpy(), saved[['ade', 'fde', 'jitter']].to_numpy()):
        raise ValueError('motion metric replay mismatch')
    with np.load(output/'per_update.npz', allow_pickle=False) as z:
        if set(z.files) != set(updates) or any(not np.array_equal(z[k], updates[k]) for k in updates):
            raise ValueError('motion per-update replay mismatch')
    summary = json.loads((output/'summary.json').read_text())
    for name, value in summaries(frame).items():
        if summary[name] != value:
            raise ValueError('motion summary replay mismatch')
    for name, value in [('tags', tags), ('maneuvers', groups), ('weights', weights)]:
        if json.loads((output/f'{name}.json').read_text()) != value:
            raise ValueError('motion group/weight replay mismatch')
    return {'identity': marker['identity'], 'all_metrics_groups_weights_replayed': True,
            'fit_only_normalization_verified': True, 'holdout_evaluated': False}


def verify_lineage(output, parent):
    datasets, records, _, _, original = load_inputs(parent)
    output, parent = Path(output), Path(parent)
    new = torch.load(output/'baseline7.pt', map_location='cpu', weights_only=True)
    old = torch.load(parent/'full.pt', map_location='cpu', weights_only=True)
    if new['stats'] != old['stats'] or any(not torch.equal(new['model']['legacy.'+k], v) for k, v in old['model'].items()):
        raise ValueError('motion baseline must reproduce parent weights')
    cached = {}
    for role in ROLES:
        with np.load(output/f'{role}.npz', allow_pickle=False) as z:
            cached[role] = {k: z[k] for k in z.files}
        for key in ('features', 'target'):
            if not np.array_equal(cached[role][key], datasets[role][key]):
                raise ValueError('motion parent input changed')
        if not np.array_equal(cached[role]['candidates'][:, :, :7], datasets[role]['candidates']):
            raise ValueError('motion parent candidates changed')
    items = [(role, i, row) for role in ROLES for i, row in enumerate(records[role])]
    for k in np.linspace(0, len(items)-1, min(64, len(items)), dtype=int):
        role, i, row = items[k]
        if not np.array_equal(cached[role]['candidates'][i, :, 7:], rebuild(original, row)):
            raise ValueError('motion candidate deterministic replay mismatch')
    frame = pd.read_csv(output/'per_trajectory.csv', float_precision='round_trip')
    keys = ['role', 'row', 'condition', 'seed']
    a, b = (frame[frame.strategy == name].sort_values(keys) for name in ('baseline7', 'full'))
    if (not np.array_equal(a[keys].to_numpy(), b[keys].to_numpy())
            or not np.array_equal(a[['ade', 'fde', 'jitter']].to_numpy(), b[['ade', 'fde', 'jitter']].to_numpy())):
        raise ValueError('motion baseline metric identity mismatch')
    return {'baseline_weights_and_metrics_match_parent': True, 'matched_streams': len(a), 'motion_stream_spot_checks': min(64, len(items))}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=Path('../noise-aware-diagnostics/outputs/noise-aware/phase2b'))
    parser.add_argument('--output-root', type=Path, default=Path('outputs/noise-aware/phase2c'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    if args.verify_only:
        result = verify_output(args.output_root)
        parent = Path(json.loads((args.output_root/'run.json').read_text())['binding']['parent_path'])
        print(json.dumps({**result, **verify_lineage(args.output_root, parent)}), flush=True)
    else:
        if args.output_root.exists():
            raise FileExistsError('use a new motion experiment directory')
        before = sources()
        datasets, records, baseline, binding, original = load_inputs(args.parent)
        expanded = expand_inputs(datasets, records, original)
        if sources() != before:
            raise ValueError('motion source changed during candidate preparation')
        summary = publish(args.output_root, expanded, records, baseline, binding,
                          load_yaml(PROJECT/'configs/analysis/maneuvers.yaml'))
        print(json.dumps({k: summary[k] for k in ('identity', 'rows', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()

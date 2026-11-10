"""Development-only time/expert controls and offline maneuver diagnostics."""
import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.online.batched_evaluation import forecast_metrics
from trajectory_prediction.robustness.benchmark import PROJECT, ROLES
from trajectory_prediction.robustness.diagnostic_controls import (
    CONTROLS,
    covariance_lookup,
    fit_control,
    predict_control,
    transform_features,
)
from trajectory_prediction.robustness.maneuver_diagnostics import (
    build_tags,
    event_diagnostics,
    grouped_metrics,
)
from trajectory_prediction.robustness.phase2_experiment import summaries
from trajectory_prediction.robustness.reliability_learning import TRAINING, ReliabilityGate
from trajectory_prediction.utils.provenance import sha256_file


def sources():
    paths = sorted((PROJECT/'src/trajectory_prediction').rglob('*.py'))
    paths += [PROJECT/'scripts/run_noise_aware_diagnostics.py', PROJECT/'configs/analysis/maneuvers.yaml',
              PROJECT/'docs/superpowers/specs/2026-11-10-noise-aware-phase2b-design.md']
    return {p.relative_to(PROJECT).as_posix(): sha256_file(p) for p in paths}


def load_parent(parent):
    parent = Path(parent).resolve()
    project = parent.parents[2]
    process = subprocess.run([sys.executable, str(project/'scripts/run_noise_aware_gate.py'),
                              '--output-root', str(parent), '--verify-only'], cwd=project,
                             env={**os.environ, 'PYTHONPATH': str(project/'src'),
                                  'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'},
                             check=True, capture_output=True, text=True)
    proof = json.loads(process.stdout.splitlines()[-1])
    if not proof.get('all_metrics_replayed') or proof.get('holdout_evaluated'):
        raise ValueError('fully verified development-only parent required')
    marker = json.loads((parent/'completed.json').read_text())
    datasets = {}
    for role in ROLES:
        with np.load(parent/f'{role}.npz', allow_pickle=False) as z:
            datasets[role] = {k: z[k] for k in z.files}
        features = datasets[role]['features'][:, :, 6:12]
        if not np.array_equal(features, np.broadcast_to(covariance_lookup(), features.shape)):
            raise ValueError('parent covariance differs from observation-free step table')
    records = json.loads((parent/'records.json').read_text())
    frame = pd.read_csv(parent/'per_trajectory.csv', float_precision='round_trip')
    references = {name: {'path': str(parent/name), 'sha256': digest} for name, digest in marker['sha256'].items()}
    references['parent_completion'] = {'path': str(parent/'completed.json'), 'sha256': sha256_file(parent/'completed.json')}
    binding = {'parent_path': str(parent), 'parent_identity': proof['identity'], 'references': references,
               'parent_verified_in_original_worktree': True, 'covariance_matches_step_table': True}
    return datasets, records, frame, binding


def _replay(output, datasets, records, baseline, thresholds, *, check_stats=False):
    frames, updates, full_weights = [], {}, {}
    for role in ROLES:
        data = datasets[role]
        for variant in CONTROLS:
            payload = torch.load(output/f'{variant}.pt', map_location='cpu', weights_only=True)
            if payload['variant'] != variant:
                raise ValueError('diagnostic control identity mismatch')
            if check_stats:
                flat = transform_features(datasets[ROLES[0]]['features'], variant).reshape(-1, 18)
                expected = {'mean': flat.mean(0, dtype=np.float64),
                            'std': np.maximum(flat.std(0, dtype=np.float64), 1e-4)}
                for name, value in expected.items():
                    if not np.array_equal(payload['stats'][name], value):
                        raise ValueError('diagnostic normalization must use fit only')
            model = ReliabilityGate()
            model.load_state_dict(payload['model'], strict=True)
            model.eval()
            forecast, weight = predict_control(model, payload['stats'], data['features'], data['candidates'], variant)
            metrics = forecast_metrics(forecast, data['target'])
            updates[role+'__'+variant] = metrics
            frames.extend([{**r, 'strategy': variant,
                            **dict(zip(('ade', 'fde', 'jitter'), value.tolist(), strict=True))}
                           for r, value in zip(records[role], metrics.mean(1), strict=True)])
            if variant == 'full':
                full_weights[role] = weight
    frame = pd.DataFrame(frames)
    if len(baseline):
        frame = pd.concat((baseline, frame), ignore_index=True)
    tags = build_tags(datasets, records, thresholds)
    return frame, updates, tags, grouped_metrics(frame, tags), event_diagnostics(datasets, records, tags, full_weights)


def publish(output, datasets, records, baseline, binding, thresholds, *, epochs=20):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    if {r['log'] for r in records[ROLES[0]]} & {r['log'] for r in records[ROLES[1]]}:
        raise ValueError('diagnostic fit/select log overlap')
    binding = {**binding, 'sources': sources(), 'thresholds': thresholds, 'training': {**TRAINING, 'epochs': epochs},
               'controls': list(CONTROLS), 'runtime': {'python': platform.python_version(), 'numpy': str(np.__version__),
               'torch': str(torch.__version__), 'threads': torch.get_num_threads(), 'device': 'cpu'},
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
    for variant in CONTROLS:
        model, stats, log = fit_control(fit['features'], fit['candidates'], fit['target'], variant, epochs=epochs)
        torch.save({'variant': variant, 'model': model.state_dict(), 'stats': stats}, output/f'{variant}.pt')
        logs[variant] = log
        print(json.dumps({'trained_control': variant, 'final_fit_loss': log[-1]['fit_ade_loss']}), flush=True)
    frame, updates, tags, groups, events = _replay(output, datasets, records, baseline, thresholds)
    frame.to_csv(output/'per_trajectory.csv', index=False)
    np.savez_compressed(output/'per_update.npz', **updates)
    summary = {**summaries(frame), 'identity': identity, 'controls': list(CONTROLS),
               'seconds': time.perf_counter()-start, 'holdout_evaluated': False,
               'unique_trajectories_by_role': {r: len({v['row'] for v in records[r]}) for r in ROLES}}
    for name, value in [('summary', summary), ('training', logs), ('tags', tags), ('maneuvers', groups), ('events', events)]:
        (output/f'{name}.json').write_text(json.dumps(value, indent=2)+'\n')
    if sources() != binding['sources']:
        raise ValueError('diagnostic source changed during run')
    names = ['run.json', 'records.json', 'baseline.csv', 'per_trajectory.csv', 'per_update.npz',
             *[f'{r}.npz' for r in ROLES], *[f'{v}.pt' for v in CONTROLS],
             *[f'{v}.json' for v in ('summary', 'training', 'tags', 'maneuvers', 'events')]]
    (output/'completed.json').write_text(json.dumps({'identity': identity, 'sha256': {
        name: sha256_file(output/name) for name in names}}, indent=2)+'\n')
    return summary


def verify_output(output):
    output = Path(output)
    marker = json.loads((output/'completed.json').read_text())
    for name, digest in marker['sha256'].items():
        if sha256_file(output/name) != digest:
            raise ValueError('diagnostic output checksum mismatch')
    run = json.loads((output/'run.json').read_text())
    binding = run['binding']
    if (digest_json(binding) != marker['identity'] or run['identity'] != marker['identity']
            or sources() != binding['sources']):
        raise ValueError('diagnostic run/source identity mismatch')
    for ref in binding.get('references', {}).values():
        if sha256_file(Path(ref['path'])) != ref['sha256']:
            raise ValueError('diagnostic parent checksum mismatch')
    datasets = {}
    for role in ROLES:
        with np.load(output/f'{role}.npz', allow_pickle=False) as z:
            datasets[role] = {k: z[k] for k in z.files}
    records = json.loads((output/'records.json').read_text())
    if {r['log'] for r in records[ROLES[0]]} & {r['log'] for r in records[ROLES[1]]}:
        raise ValueError('diagnostic fit/select log overlap')
    baseline = pd.read_csv(output/'baseline.csv', float_precision='round_trip')
    frame, updates, tags, groups, events = _replay(output, datasets, records, baseline, binding['thresholds'], check_stats=True)
    saved = pd.read_csv(output/'per_trajectory.csv', float_precision='round_trip')
    if not np.array_equal(frame[['ade', 'fde', 'jitter']].to_numpy(), saved[['ade', 'fde', 'jitter']].to_numpy()):
        raise ValueError('diagnostic metric replay mismatch')
    with np.load(output/'per_update.npz', allow_pickle=False) as z:
        if set(z.files) != set(updates) or any(not np.array_equal(z[k], updates[k]) for k in updates):
            raise ValueError('diagnostic per-update replay mismatch')
    summary = json.loads((output/'summary.json').read_text())
    for name, value in summaries(frame).items():
        if summary[name] != value:
            raise ValueError('diagnostic summary replay mismatch')
    for name, value in [('tags', tags), ('maneuvers', groups), ('events', events)]:
        if json.loads((output/f'{name}.json').read_text()) != value:
            raise ValueError('diagnostic grouped replay mismatch')
    return {'identity': marker['identity'], 'all_metrics_and_groups_replayed': True,
            'fit_only_normalization_verified': True, 'holdout_evaluated': False}


def verify_parent_reproduction(output, parent):
    output, parent = Path(output), Path(parent)
    new = torch.load(output/'full.pt', map_location='cpu', weights_only=True)
    old = torch.load(parent/'innovation.pt', map_location='cpu', weights_only=True)
    if new['stats'] != old['stats'] or any(not torch.equal(v, old['model'][k]) for k, v in new['model'].items()):
        raise ValueError('full control must exactly reproduce parent gate')
    frame = pd.read_csv(output/'per_trajectory.csv', float_precision='round_trip')
    keys = ['role', 'row', 'condition', 'seed']
    a = frame[frame.strategy == 'full'].sort_values(keys)
    b = frame[frame.strategy == 'gate_innovation'].sort_values(keys)
    if (not np.array_equal(a[keys].to_numpy(), b[keys].to_numpy())
            or not np.array_equal(a[['ade', 'fde', 'jitter']].to_numpy(), b[['ade', 'fde', 'jitter']].to_numpy())):
        raise ValueError('full control metric baseline mismatch')
    return {'full_weights_and_metrics_match_parent': True, 'matched_streams': len(a)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=Path('../noise-aware-online/outputs/noise-aware/phase2a-v1'))
    parser.add_argument('--output-root', type=Path, default=Path('outputs/noise-aware/phase2b'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    if args.verify_only:
        result = verify_output(args.output_root)
        parent = Path(json.loads((args.output_root/'run.json').read_text())['binding']['parent_path'])
        load_parent(parent)
        print(json.dumps({**result, **verify_parent_reproduction(args.output_root, parent)}), flush=True)
    else:
        if args.output_root.exists():
            raise FileExistsError('use a new diagnostic experiment directory')
        datasets, records, baseline, binding = load_parent(args.parent)
        summary = publish(args.output_root, datasets, records, baseline, binding,
                          load_yaml(PROJECT/'configs/analysis/maneuvers.yaml'))
        verify_parent_reproduction(args.output_root, args.parent)
        print(json.dumps({k: summary[k] for k in ('identity', 'rows', 'seconds')}), flush=True)


if __name__ == '__main__':
    main()

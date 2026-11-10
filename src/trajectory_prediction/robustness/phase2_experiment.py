"""Fixed-budget development ablations and independent metrics/model replay."""
import argparse
import json
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.online.batched_evaluation import forecast_metrics
from trajectory_prediction.robustness.benchmark import PROJECT, ROLES
from trajectory_prediction.robustness.noisy_candidates import CANDIDATES, FEATURE_GROUPS
from trajectory_prediction.robustness.reliability_learning import (
    TRAINING,
    ReliabilityGate,
    fit_gate,
    predict_gate,
)
from trajectory_prediction.utils.provenance import sha256_file


def sources():
    files = sorted((PROJECT/'src/trajectory_prediction').rglob('*.py'))
    files += [PROJECT/'scripts/run_noise_aware_gate.py',
              PROJECT/'docs/superpowers/specs/2026-11-10-noise-aware-phase2a-design.md']
    return {p.relative_to(PROJECT).as_posix(): sha256_file(p) for p in files}


def _metrics(data, forecasts, strategy, records):
    values = forecast_metrics(forecasts, data['target']).mean(1)
    return [{**row, 'strategy': strategy, **dict(zip(('ade', 'fde', 'jitter'), value.tolist(), strict=True))}
            for row, value in zip(records, values, strict=True)]


def summaries(frame):
    result = {'development_only': True, 'holdout_evaluated': False, 'rows': len(frame)}
    for name, columns in (('by_condition', ['role', 'condition', 'strategy']),
                          ('by_log', ['role', 'log', 'condition', 'strategy']),
                          ('by_seed', ['role', 'seed', 'condition', 'strategy'])):
        result[name] = frame.groupby(columns, sort=True)[['ade', 'fde', 'jitter']].mean().reset_index().to_dict('records')
    return result


def _evaluate(output, datasets, records, *, replay_stats=False):
    rows, weight_means = [], []
    fit = datasets[ROLES[0]]['features'].reshape(-1, 18)
    for role in ROLES:
        data = datasets[role]
        for j, name in enumerate(CANDIDATES):
            rows.extend(_metrics(data, data['candidates'][:, :, j], name, records[role]))
        rows.extend(_metrics(data, data['candidates'].mean(2), 'equal', records[role]))
        for variant in FEATURE_GROUPS:
            payload = torch.load(output/f'{variant}.pt', map_location='cpu', weights_only=True)
            if payload['variant'] != variant:
                raise ValueError('gate ablation identity mismatch')
            stats = payload['stats']
            if replay_stats:
                for field, expected in [('mean', fit.mean(0, dtype=np.float64)),
                                        ('std', np.maximum(fit.std(0, dtype=np.float64), 1e-4))]:
                    if not np.array_equal(np.array(stats[field]), expected):
                        raise ValueError('normalization must use fit only')
            model = ReliabilityGate()
            model.load_state_dict(payload['model'], strict=True)
            model.eval()
            forecast, weight = predict_gate(model, stats, data['features'], data['candidates'], variant)
            rows.extend(_metrics(data, forecast, 'gate_'+variant, records[role]))
            for condition in sorted({r['condition'] for r in records[role]}):
                selected = [i for i, row in enumerate(records[role]) if row['condition'] == condition]
                weight_means.append({'role': role, 'condition': condition, 'variant': variant,
                                     'weights': dict(zip(CANDIDATES, weight[selected].mean((0, 1)).tolist(), strict=True))})
    return pd.DataFrame(rows), weight_means


def publish_learning(output, datasets, records, binding, *, epochs=20, create=True):
    output = Path(output)
    if create:
        output.mkdir(parents=True, exist_ok=False)
    for role in ROLES:
        np.savez_compressed(output/f'{role}.npz', **datasets[role])
    if {r['log'] for r in records[ROLES[0]]} & {r['log'] for r in records[ROLES[1]]}:
        raise ValueError('development log overlap')
    binding = {**binding, 'sources': sources(), 'training': {**TRAINING, 'epochs': epochs},
               'runtime': {'python': platform.python_version(), 'numpy': str(np.__version__),
                           'torch': str(torch.__version__), 'platform': platform.platform(),
                           'device': 'cpu', 'torch_threads': torch.get_num_threads()},
               'git_commit': subprocess.run(['git', '-c', 'safe.directory=*', 'rev-parse', 'HEAD'],
                                            check=True, capture_output=True, text=True).stdout.strip()}
    identity = digest_json(binding)
    (output/'run.json').write_text(json.dumps({'identity': identity, 'binding': binding}, indent=2)+'\n')
    (output/'records.json').write_text(json.dumps(records, indent=2)+'\n')
    started = time.perf_counter()
    fit = datasets[ROLES[0]]
    training = {}
    for variant in FEATURE_GROUPS:
        model, stats, log = fit_gate(fit['features'], fit['candidates'], fit['target'], variant, epochs=epochs)
        torch.save({'variant': variant, 'model': model.state_dict(), 'stats': stats}, output/f'{variant}.pt')
        training[variant] = log
        print(json.dumps({'trained': variant, 'final_fit_loss': log[-1]['fit_ade_loss']}), flush=True)
    frame, weights = _evaluate(output, datasets, records)
    frame.to_csv(output/'per_trajectory.csv', index=False)
    result = {**summaries(frame), 'identity': identity, 'gate_weights': weights,
              'learning_evaluation_seconds': time.perf_counter()-started,
              'streams_by_role': {r: len(records[r]) for r in ROLES}}
    (output/'summary.json').write_text(json.dumps(result, indent=2)+'\n')
    (output/'training.json').write_text(json.dumps(training, indent=2)+'\n')
    if sources() != binding['sources']:
        raise ValueError('phase2 executable changed during learning')
    names = ['run.json', 'records.json', 'summary.json', 'training.json', 'per_trajectory.csv',
             *[f'{r}.npz' for r in ROLES], *[f'{v}.pt' for v in FEATURE_GROUPS]]
    if (output/'preparation.json').is_file():
        names.append('preparation.json')
    (output/'completed.json').write_text(json.dumps({'identity': identity, 'sha256': {
        name: sha256_file(output/name) for name in names}}, indent=2)+'\n')
    return result


def verify_learning(output):
    output = Path(output)
    marker = json.loads((output/'completed.json').read_text())
    for name, digest in marker['sha256'].items():
        if sha256_file(output/name) != digest:
            raise ValueError('phase2 output checksum mismatch')
    run = json.loads((output/'run.json').read_text())
    if (digest_json(run['binding']) != marker['identity'] or run['identity'] != marker['identity']
            or run['binding']['sources'] != sources()):
        raise ValueError('phase2 source/run identity mismatch')
    for ref in run['binding']['references'].values():
        if sha256_file(Path(ref['path'])) != ref['sha256']:
            raise ValueError('phase2 parent checksum mismatch')
    records = json.loads((output/'records.json').read_text())
    if {r['log'] for r in records[ROLES[0]]} & {r['log'] for r in records[ROLES[1]]}:
        raise ValueError('development log overlap')
    datasets = {}
    for role in ROLES:
        with np.load(output/f'{role}.npz', allow_pickle=False) as z:
            datasets[role] = {k: z[k] for k in z.files}
    frame, weights = _evaluate(output, datasets, records, replay_stats=True)
    saved = pd.read_csv(output/'per_trajectory.csv', float_precision='round_trip')
    for key in ('ade', 'fde', 'jitter'):
        if not np.allclose(frame[key], saved[key], atol=1e-10, rtol=0):
            raise ValueError('phase2 prediction metric replay mismatch')
    summary = json.loads((output/'summary.json').read_text())
    for key, value in summaries(frame).items():
        if summary[key] != value:
            raise ValueError('phase2 summary replay mismatch')
    if summary['gate_weights'] != weights:
        raise ValueError('phase2 mean weight replay mismatch')
    return {'identity': marker['identity'], 'all_metrics_replayed': True, 'fit_only_normalization_verified': True,
            'holdout_evaluated': False}


def main(argv=None):
    from trajectory_prediction.robustness.phase2_data import (
        generate_development,
        prepare_inputs,
        rebuild_one,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('../causal-context-online/outputs/context/p66-independent'))
    parser.add_argument('--phase1', type=Path, default=Path('outputs/noise-aware/phase1'))
    parser.add_argument('--dataroot', type=Path, default=Path('../../data/nuscenes'))
    parser.add_argument('--model-config', type=Path, default=Path('configs/models/gru.yaml'))
    parser.add_argument('--output-root', type=Path, default=Path('outputs/noise-aware/phase2a'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    if args.verify_only:
        result = verify_learning(args.output_root)
        binding = json.loads((args.output_root/'run.json').read_text())['binding']
        _, context = prepare_inputs(binding['model_root'], binding['phase1_root'],
                                    binding['model_config'], binding['dataroot'])
        records = json.loads((args.output_root/'records.json').read_text())
        checked = 0
        for role in ROLES:
            row = next(r for r in records[role] if r['condition'] == 'noise_outlier')
            i = records[role].index(row)
            rebuilt = rebuild_one(context, row, row['condition'], row['seed'])
            with np.load(args.output_root/f'{role}.npz', allow_pickle=False) as data:
                for name in ('features', 'candidates'):
                    if not np.allclose(data[name][i], rebuilt[name], atol=1e-5, rtol=1e-5):
                        raise ValueError('phase2 noisy map/GRU candidate replay mismatch')
            checked += 1
        print(json.dumps({**result, 'rebuilt_stream_spot_checks': checked}), flush=True)
    else:
        if args.output_root.exists():
            raise FileExistsError('use a new phase2 experiment directory')
        binding, context = prepare_inputs(args.root, args.phase1, args.model_config, args.dataroot)
        args.output_root.mkdir(parents=True, exist_ok=False)
        (args.output_root/'preparation.json').write_text(json.dumps({'binding': binding, 'sources': sources()}, indent=2)+'\n')
        datasets, records = generate_development(context)
        if json.loads((args.output_root/'preparation.json').read_text())['sources'] != sources():
            raise ValueError('phase2 source changed during candidate preparation')
        result = publish_learning(args.output_root, datasets, records, binding, create=False)
        print(json.dumps({k: result[k] for k in ('identity', 'rows', 'streams_by_role', 'learning_evaluation_seconds')}), flush=True)


if __name__ == '__main__':
    main()

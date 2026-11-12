"""S5 immutable raw prediction publication, without Phase 3A scoring."""
import argparse
import json
import platform
import sys
from pathlib import Path

import numpy as np
import torch

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.evaluation.scenario_inputs import assemble_inputs
from trajectory_prediction.evaluation.scenario_preflight import (
    PROJECT,
    _check_references,
    _sources,
)
from trajectory_prediction.evaluation.scenario_replay import (
    PHYSICAL_COLUMNS,
    check_parent_proof,
    replay_streams,
    time_grid,
    validate_protocol,
)
from trajectory_prediction.utils.provenance import sha256_file


def _read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _source_hashes():
    return {**_sources(), **{n: sha256_file(PROJECT / n) for n in (
        'src/trajectory_prediction/evaluation/scenario_publication.py',
        'scripts/publish_scenario_predictions.py')}}


def _artifact_hashes(root, expected):
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(expected) | {'completed.json'}:
        raise ValueError('publication file coverage mismatch')
    for name, digest in expected.items():
        path = (root / name).resolve()
        if not path.is_relative_to(root) or sha256_file(path) != digest:
            raise ValueError('publication artifact checksum mismatch')


def _prepare(preflight):
    preflight = Path(preflight).resolve()
    source_hashes = _source_hashes()
    marker_path = preflight / 'preflight_completed.json'
    marker_hash = sha256_file(marker_path)
    marker = _read(marker_path)
    for name, digest in marker['sha256'].items():
        path = (preflight / name).resolve()
        if not path.is_relative_to(preflight) or sha256_file(path) != digest:
            raise ValueError('preflight checksum mismatch')
    run = _read(preflight / 'run.json')
    old = run['binding']
    if (run['identity'] != marker['identity'] or digest_json(old) != marker['identity']
            or old['sources'] != _sources()):
        raise ValueError('preflight source identity mismatch')
    config = validate_protocol(old['protocol'])
    inputs = assemble_inputs(old['parent_path'], old['parent_project_root'])
    if inputs.parent_identity != old['parent_identity'] or inputs.references != old['references']:
        raise ValueError('parent provenance changed')
    check_parent_proof(inputs, _read(preflight / 'parent_proof.json'))
    strategies = [{'strategy': f's{s}_{v}', 'train_seed': s, 'variant': v}
                  for s in config['model_seeds'] for v in config['gate_variants']]
    strategies += [{'strategy': s, 'train_seed': None, 'candidate_column': c}
                   for s, c in PHYSICAL_COLUMNS.items()]
    binding = {'schema': 'scenario_prediction_publication_v1',
               'preflight': str(preflight), 'preflight_sha256': marker_hash,
               'preflight_identity': marker['identity'], 'sources': source_hashes,
               'protocol': config, 'strategies': strategies,
               'neural_training': False, 'holdout_evaluated': False,
               'new_metrics_computed': False,
               'preflight_runtime': old['runtime'], 'parent_git_commit': old['git_commit'],
               'runtime': {'python': sys.version, 'numpy': str(np.__version__),
                           'torch': str(torch.__version__), 'platform': platform.platform(),
                           'device': 'cpu', 'threads': config['inference']['threads']}}
    _recheck(binding, inputs)
    return binding, inputs


def _recheck(binding, inputs):
    _check_references(inputs.references)
    if (_source_hashes() != binding['sources'] or sha256_file(
            Path(binding['preflight']) / 'preflight_completed.json') != binding['preflight_sha256']):
        raise ValueError('source/preflight changed during publication')
    marker = _read(Path(binding['preflight']) / 'preflight_completed.json')
    for name, digest in marker['sha256'].items():
        if sha256_file(Path(binding['preflight']) / name) != digest:
            raise ValueError('preflight artifact changed during publication')


def _shared(inputs, role):
    unique, lookup, keys = [], {}, []
    for i, (row, gt) in enumerate(zip(inputs.records[role], inputs.gt_windows[role], strict=True)):
        key = tuple(gt['window_key'])
        if key not in lookup:
            lookup[key] = len(unique)
            unique.append(gt)
        keys.append({**{k: v for k, v in row.items() if k != 'seed'},
                     'noise_seed': row['seed'], 'stream_index': i, 'gt_index': lookup[key]})
    arrays = {name: np.asarray([g[name] for g in unique]) for name in (
        'history_xy', 'history_mask', 'future_xy', 'anchor_xy', 'anchor_yaw', 'source_row',
        'history_times_s', 'future_times_s', 'coordinate_frame', 'units')}
    arrays['window_key'] = np.asarray([g['window_key'] for g in unique])
    return keys, arrays


def _summary(binding, inputs):
    roles = binding['protocol']['roles']
    counts = {r: len(inputs.records[r]) for r in roles}
    return {'shards': len(roles) * len(binding['strategies']), 'observation_streams': counts,
            'unique_windows': {r: len({tuple(g['window_key']) for g in inputs.gt_windows[r]})
                               for r in roles},
            'forecast_bytes': sum(counts.values()) * len(binding['strategies']) * 11 * 12 * 2 * 4,
            'new_metrics_computed': False, 'neural_training': False, 'holdout_evaluated': False,
            'scope': 'raw_prediction_publication_not_full_phase3a_evaluation'}


def plan_publication(preflight):
    """Validate provenance and count resources; never call prediction inference."""
    binding, inputs = _prepare(preflight)
    return _summary(binding, inputs)


def _payload(result, keys):
    return {'forecast': result['forecast'], 'legacy_metrics': result['metrics'],
            'stream_index': np.arange(len(keys), dtype=np.int64),
            'query_step': np.arange(1, 12), 'query_row': np.arange(11),
            'target_column': np.arange(12),
            'valid_target_mask': np.arange(12)[None, :] >= np.arange(1, 12)[:, None],
            'stream_keys_digest': np.asarray(digest_json(keys))}


def publish_predictions(preflight, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('use a new publication output directory')
    binding, inputs = _prepare(preflight)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'predictions').mkdir()
    (output / 'gt').mkdir()
    summary = _summary(binding, inputs)
    progress = {'state': 'writing', 'completed_shards': 0, 'expected_shards': summary['shards']}
    hashes = {}
    def remember(name):
        hashes[name] = sha256_file(output / name)
    _write(output / 'progress.json', progress)
    try:
        _write(output / 'run.json', {'identity': digest_json(binding), 'binding': binding})
        _write(output / 'time_grid.json', time_grid(binding['protocol']))
        remember('run.json')
        remember('time_grid.json')
        for role in binding['protocol']['roles']:
            keys, gt = _shared(inputs, role)
            _write(output / f'{role}-keys.json', keys)
            np.savez_compressed(output / 'gt' / f'{role}.npz', **gt)
            _compare_npz(output / 'gt' / f'{role}.npz', gt)
            remember(f'{role}-keys.json')
            remember(f'gt/{role}.npz')
            indices = list(range(len(keys)))
            for strategy in binding['strategies']:
                result = replay_streams(inputs, role, strategy['strategy'], indices)
                name = f"predictions/{role}__{strategy['strategy']}.npz"
                payload = _payload(result, keys)
                np.savez_compressed(output / name, **payload)
                _compare_npz(output / name, payload)
                remember(name)
                progress['completed_shards'] += 1
                _write(output / 'progress.json', progress)
        _recheck(binding, inputs)
        _write(output / 'summary.json', summary)
        progress['state'] = 'complete'
        _write(output / 'progress.json', progress)
        remember('summary.json')
        remember('progress.json')
        actual = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
        if actual != set(hashes) or progress['completed_shards'] != summary['shards']:
            raise ValueError('publication coverage mismatch before completion')
        for name, digest in hashes.items():
            if sha256_file(output / name) != digest:
                raise ValueError('publication artifact changed before completion')
        _write(output / 'completed.json', {'identity': digest_json(binding), 'sha256': hashes})
    except BaseException:
        progress['state'] = 'failed'
        _write(output / 'progress.json', progress)
        raise
    return summary


def verify_publication(output):
    """Reload each frozen strategy and compare raw predictions, keys and GT exactly."""
    output = Path(output).resolve()
    if not (output / 'completed.json').is_file():
        raise ValueError('incomplete prediction publication')
    marker_hash = sha256_file(output / 'completed.json')
    marker = _read(output / 'completed.json')
    _artifact_hashes(output, marker['sha256'])
    run = _read(output / 'run.json')
    binding, inputs = _prepare(run['binding']['preflight'])
    if run != {'identity': digest_json(binding), 'binding': binding} or marker['identity'] != run['identity']:
        raise ValueError('publication identity mismatch')
    expected_files = {'run.json', 'summary.json', 'progress.json', 'time_grid.json'}
    for role in binding['protocol']['roles']:
        keys, gt = _shared(inputs, role)
        expected_files.update({f'{role}-keys.json', f'gt/{role}.npz'})
        if _read(output / f'{role}-keys.json') != keys:
            raise ValueError('stream key/order mismatch')
        _compare_npz(output / 'gt' / f'{role}.npz', gt)
        for strategy in binding['strategies']:
            name = f"predictions/{role}__{strategy['strategy']}.npz"
            expected_files.add(name)
            result = replay_streams(inputs, role, strategy['strategy'], list(range(len(keys))))
            _compare_npz(output / name, _payload(result, keys))
    summary = _summary(binding, inputs)
    if (set(marker['sha256']) != expected_files or _read(output / 'summary.json') != summary
            or digest_json(_read(output / 'time_grid.json')) != digest_json(time_grid(binding['protocol']))
            or _read(output / 'progress.json') != {
                'state': 'complete', 'completed_shards': summary['shards'],
                'expected_shards': summary['shards']}):
        raise ValueError('publication coverage/summary mismatch')
    _recheck(binding, inputs)
    _artifact_hashes(output, marker['sha256'])
    if sha256_file(output / 'completed.json') != marker_hash:
        raise ValueError('publication marker changed during verification')
    return {**summary, 'frozen_replay_verified': True, 'identity': marker['identity']}


def _compare_npz(path, expected):
    with np.load(path, allow_pickle=False) as saved:
        if set(saved.files) != set(expected):
            raise ValueError('archive field coverage mismatch')
        for name, value in expected.items():
            if saved[name].dtype != value.dtype or not np.array_equal(saved[name], value):
                raise ValueError(f'frozen replay/GT mismatch: {name}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, default=Path('outputs/scenario-evaluation/preflight-v2'))
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.plan_only and args.verify_only:
        parser.error('choose one mode')
    if not args.plan_only and args.output_root is None:
        parser.error('--output-root required for publication/verification')
    result = (plan_publication(args.preflight) if args.plan_only else
              verify_publication(args.output_root) if args.verify_only else
              publish_predictions(args.preflight, args.output_root))
    print(json.dumps(result, indent=2, allow_nan=False))

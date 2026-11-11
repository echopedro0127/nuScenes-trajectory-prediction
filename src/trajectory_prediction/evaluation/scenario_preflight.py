"""S0-S2 readiness receipts; deliberately separate from Phase 3A evaluation."""
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.evaluation.scenario_inputs import assemble_inputs
from trajectory_prediction.evaluation.scenario_replay import (
    PHYSICAL_COLUMNS,
    check_parent_proof,
    replay_streams,
    time_grid,
    validate_protocol,
    verify_parent,
)
from trajectory_prediction.utils.provenance import sha256_file

PROJECT = Path(__file__).resolve().parents[3]


def _sources():
    names = [f'src/trajectory_prediction/evaluation/{n}.py'
             for n in ('scenario_inputs', 'scenario_replay', 'scenario_preflight')]
    names += ['scripts/run_scenario_preflight.py', 'configs/analysis/scenario_evaluation.yaml',
              'src/trajectory_prediction/robustness/motion_learning.py',
              'src/trajectory_prediction/robustness/reliability_learning.py',
              'src/trajectory_prediction/robustness/noisy_candidates.py',
              'src/trajectory_prediction/online/batched_evaluation.py',
              'src/trajectory_prediction/data/cache.py',
              'src/trajectory_prediction/context/cache.py',
              'src/trajectory_prediction/utils/provenance.py']
    return {name: sha256_file(PROJECT / name) for name in names}


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _check_references(references):
    for ref in references.values():
        if sha256_file(Path(ref['path'])) != ref['sha256']:
            raise ValueError(f"input/source checksum changed: {ref['path']}")


def _selection(inputs, role):
    rows = inputs.records[role]
    eligible = [i for i, r in enumerate(rows)
                if r['condition'] == 'noise_outlier' and r['seed'] == 2027]
    chosen = min(eligible, key=lambda i: hashlib.sha256(
        (rows[i]['instance'] + ':' + rows[i]['sample']).encode()).hexdigest())
    # Include all windows overlapping the original flat batches of this anchor.
    first, last = (chosen * 11) // 4096, (chosen * 11 + 10) // 4096
    start = (first * 4096) // 11
    end = min(len(rows), ((last + 1) * 4096 + 10) // 11)
    return {'anchor_stream_index': chosen, 'flat_batch_ids': list(range(first, last + 1)),
            'indices': list(range(start, end))}


def _report(identity, inputs, config, rows):
    return {
        'identity': identity, 'status': 's0_s2_verified', 'models_checked': 9,
        'physical_strategies_checked': 7, 'new_metrics_computed': False,
        'neural_training': False, 'holdout_evaluated': False,
        'scope': 'selected_window_and_original_batch_neighborhoods_not_full_phase3a',
        'unique_windows': {r: len({g['window_key'] for g in inputs.gt_windows[r]})
                           for r in config['roles']},
        'observation_streams': {r: len(inputs.records[r]) for r in config['roles']},
        'checks': rows,
    }


def run_preflight(parent, parent_project_root, config, output):
    """Verify parent once, replay frozen batch neighborhoods, publish only S0-S2."""
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('use a new scenario preflight output directory')
    config = validate_protocol(config)
    source_hashes = _sources()
    torch.set_num_threads(config['inference']['threads'])
    inputs = assemble_inputs(parent, parent_project_root)
    print(json.dumps({'stage': 'S1_assembled', 'parent_identity': inputs.parent_identity,
                      'streams': {r: len(v) for r, v in inputs.records.items()}}), flush=True)
    proof = verify_parent(inputs)
    print(json.dumps({'stage': 'parent_verified_in_original_worktree'}), flush=True)
    selection = {r: _selection(inputs, r) for r in config['roles']}
    strategies = [f's{s}_{v}' for s in config['model_seeds'] for v in config['gate_variants']]
    strategies += list(PHYSICAL_COLUMNS)
    binding = {
        'schema': 'scenario_readiness_s0_s2_v1', 'protocol': config,
        'parent_path': str(inputs.parent), 'parent_project_root': str(inputs.parent_project_root),
        'parent_identity': inputs.parent_identity, 'owner_roots': inputs.owner_roots,
        'references': inputs.references, 'sources': source_hashes, 'selection': selection,
        'model_input_fields': ['features', 'candidates', 'ca_allowed'],
        'gt_source': {'cache_root': inputs.binding['source_cache_root'],
                      'scene_reference': inputs.scene_reference,
                      'coordinate_frame': 'anchor_agent_y_forward', 'units': 'm',
                      'time_basis': 'window_relative'},
        'neural_training': False, 'holdout_evaluated': False, 'new_metrics_computed': False,
        'runtime': {'python': sys.version, 'numpy': str(np.__version__),
                    'torch': str(torch.__version__), 'device': 'cpu', 'threads': 2,
                    'platform': platform.platform()},
        'git_commit': subprocess.run(
            ['git', '-c', 'safe.directory=*', 'rev-parse', 'HEAD'], cwd=PROJECT,
            check=True, capture_output=True, text=True).stdout.strip(),
    }
    identity = digest_json(binding)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'run.json', {'identity': identity, 'binding': binding})
    _write(output / 'protocol.json', {'config': config, 'time_grid': time_grid(config)})
    _write(output / 'parent_proof.json', proof)
    rows, artifacts = [], []
    for role in config['roles']:
        indices = selection[role]['indices']
        gt = inputs.gt_windows[role]
        _write(output / (role + '-keys.json'), [
            {**inputs.records[role][i], 'stream_index': i,
             'window_key': gt[i]['window_key'], 'gt_source_row': gt[i]['source_row'],
             'source_manifest': gt[i]['source_manifest'], 'source_archive': gt[i]['source_archive']}
            for i in indices])
        for strategy in strategies:
            result = replay_streams(inputs, role, strategy, indices)
            name = role + '__' + strategy + '.npz'
            np.savez_compressed(output / name, forecast=result['forecast'],
                                legacy_metrics=result['metrics'], indices=np.array(indices))
            artifacts.append(name)
            rows.append({'role': role, 'strategy': strategy, 'checked_streams': len(indices),
                         'exact_parent_metrics': result['exact_parent_metrics'],
                         'comparison_unit': result['legacy_comparison_unit'],
                         'forecast_shape': list(result['forecast'].shape),
                         'dtype': str(result['forecast'].dtype)})
        print(json.dumps({'stage': 'S2_role_compatible', 'role': role,
                          'strategies': len(strategies), 'streams': len(indices)}), flush=True)
    _check_references(inputs.references)
    if _sources() != source_hashes:
        raise ValueError('scenario code/config changed during preflight')
    report = _report(identity, inputs, config, rows)
    _write(output / 'report.json', report)
    names = ['run.json', 'protocol.json', 'parent_proof.json', 'report.json',
             *[r + '-keys.json' for r in config['roles']], *artifacts]
    _write(output / 'preflight_completed.json', {'identity': identity,
           'sha256': {name: sha256_file(output / name) for name in names}})
    return report


def verify_preflight(output):
    """Independently reload frozen models and compare saved P and old metrics."""
    output = Path(output).resolve()
    marker = json.loads((output / 'preflight_completed.json').read_text(encoding='utf-8'))
    for name, digest in marker['sha256'].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or sha256_file(path) != digest:
            raise ValueError('preflight artifact checksum mismatch')
    run = json.loads((output / 'run.json').read_text(encoding='utf-8'))
    binding = run['binding']
    if (digest_json(binding) != marker['identity'] or run['identity'] != marker['identity']
            or binding['sources'] != _sources()):
        raise ValueError('preflight source/run identity mismatch')
    config = validate_protocol(binding['protocol'])
    _check_references(binding['references'])
    inputs = assemble_inputs(binding['parent_path'], binding['parent_project_root'])
    if inputs.parent_identity != binding['parent_identity']:
        raise ValueError('preflight parent identity changed')
    saved_protocol = json.loads((output / 'protocol.json').read_text(encoding='utf-8'))
    if digest_json(saved_protocol) != digest_json({'config': config, 'time_grid': time_grid(config)}):
        raise ValueError('saved protocol/time grid mismatch')
    proof = json.loads((output / 'parent_proof.json').read_text(encoding='utf-8'))
    check_parent_proof(inputs, proof)
    rows = []
    for role in config['roles']:
        if binding['selection'][role] != _selection(inputs, role):
            raise ValueError('deterministic preflight selection mismatch')
        indices = binding['selection'][role]['indices']
        gt = inputs.gt_windows[role]
        expected_keys = [
            {**inputs.records[role][i], 'stream_index': i,
             'window_key': gt[i]['window_key'], 'gt_source_row': gt[i]['source_row'],
             'source_manifest': gt[i]['source_manifest'], 'source_archive': gt[i]['source_archive']}
            for i in indices]
        saved_keys = json.loads((output / (role + '-keys.json')).read_text(encoding='utf-8'))
        if digest_json(saved_keys) != digest_json(expected_keys):
            raise ValueError('saved window/GT provenance keys mismatch')
        for strategy in [*[f's{s}_{v}' for s in config['model_seeds']
                            for v in config['gate_variants']], *PHYSICAL_COLUMNS]:
            result = replay_streams(inputs, role, strategy, indices)
            with np.load(output / (role + '__' + strategy + '.npz'), allow_pickle=False) as z:
                if (not np.array_equal(z['indices'], indices)
                        or not np.array_equal(z['forecast'], result['forecast'])
                        or not np.array_equal(z['legacy_metrics'], result['metrics'])):
                    raise ValueError('saved frozen forecast/metric replay mismatch')
            rows.append({'role': role, 'strategy': strategy, 'checked_streams': len(indices),
                         'exact_parent_metrics': result['exact_parent_metrics'],
                         'comparison_unit': result['legacy_comparison_unit'],
                         'forecast_shape': list(result['forecast'].shape),
                         'dtype': str(result['forecast'].dtype)})
    saved_report = json.loads((output / 'report.json').read_text(encoding='utf-8'))
    if saved_report != _report(marker['identity'], inputs, config, rows):
        raise ValueError('preflight report/summary replay mismatch')
    return {'identity': marker['identity'], 'saved_forecasts_and_legacy_metrics_replayed': True,
            'neural_training': False, 'holdout_evaluated': False, 'new_metrics_computed': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path,
                        default=Path('../noise-aware-stability/outputs/noise-aware/phase2d'))
    parser.add_argument('--parent-project-root', type=Path, default=Path('../noise-aware-stability'))
    parser.add_argument('--config', type=Path,
                        default=Path('configs/analysis/scenario_evaluation.yaml'))
    parser.add_argument('--output-root', type=Path,
                        default=Path('outputs/scenario-evaluation/preflight-v1'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args(argv)
    result = (verify_preflight(args.output_root) if args.verify_only else
              run_preflight(args.parent, args.parent_project_root,
                            load_yaml(args.config), args.output_root))
    print(json.dumps(result, allow_nan=False), flush=True)

"""Versioned scenario protocol and frozen Phase 2D inference adapters."""
import copy
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

from trajectory_prediction.online.batched_evaluation import forecast_metrics
from trajectory_prediction.robustness.motion_learning import MotionGate, predict_motion
from trajectory_prediction.utils.provenance import sha256_file

PROTOCOL_V1 = {
    'schema': 'scenario_evaluation_v1', 'dt': .5,
    'query_steps': list(range(1, 12)), 'lead_steps': [1, 2, 4],
    'primary_lead_steps': 2, 'revision_tolerance_m': .05,
    'miss_distance_m': 2., 'heading_min_speed_mps': .5,
    'quantiles': [.5, .95, .99], 'quantile_method': 'linear',
    'tail_unit': 'query_error', 'mean_reduction': 'query_then_noise_then_window',
    'time_basis': 'window_relative', 'event_scope': 'window',
    'missing_expected_pair': 'error', 'nonfinite_prediction': 'error',
    'json_nonfinite': 'reject', 'noise_seeds': [2027, 2028, 2029],
    'inference': {'dtype': 'float32', 'device': 'cpu', 'threads': 2,
                  'flat_batch_size': 4096, 'compatibility': 'exact'},
    'recovery': {'condition': 'noise_outlier', 'paired_reference': 'noise_05',
                 'source_step': 4, 'last_query_step': 7, 'excess_threshold_m': .5,
                 'consecutive_updates': 2, 'require_no_relapse': True},
    'roles': ['independent_fit', 'independent_select'],
    'conditions': ['clean', 'noise_05', 'noise_outlier'],
    'model_seeds': [2027, 2028, 2029],
    'gate_variants': ['baseline7', 'plus_ca', 'guarded_ca'],
    'physical_candidates': ['raw_cv', 'alpha_beta', 'kf_cv', 'kf_nis',
                            'bounded_ca', 'brake_stop', 'hold_position'],
    'event_labels': {
        'speed_median_window': 3, 'braking_acceleration_mps2': -2.,
        'braking_duration_steps': 2, 'braking_prior_speed_mps': 2.,
        'stop_speed_mps': .3, 'stop_duration_steps': 2, 'stop_prior_speed_mps': 1.,
        'prior_window_steps': 4, 'restart_speed_mps': 1., 'restart_duration_steps': 2,
        'turn_min_speed_mps': .5, 's_turn_absolute_yaw_deg': 45.,
        's_turn_max_net_yaw_deg': 15., 's_turn_each_direction_deg': 20.,
        'u_turn_net_yaw_deg': 135., 'manual_review_cap_per_log_tag': 20,
    },
}


def _match_contract(value, expected, field):
    if isinstance(expected, dict):
        if not isinstance(value, dict):
            raise ValueError(f'{field} must be a mapping')  # noqa: TRY004
        difference = set(value) ^ set(expected)
        if difference:
            raise ValueError(f'{field} missing/unknown fields: {sorted(difference)}')
        for key, item in expected.items():
            _match_contract(value[key], item, f'{field}.{key}')
    elif isinstance(expected, list):
        if not isinstance(value, list) or len(value) != len(expected):
            raise ValueError(f'{field} must equal frozen v1 list')
        for index, (actual, item) in enumerate(zip(value, expected, strict=True)):
            _match_contract(actual, item, f'{field}[{index}]')
    elif type(expected) is float:
        if (type(value) not in (float, int) or not math.isfinite(value)
                or value != expected):
            raise ValueError(f'{field} must equal {expected}')
    elif type(value) is not type(expected) or value != expected:
        raise ValueError(f'{field} must equal frozen v1 value {expected!r}')


def validate_protocol(config):
    """Reject unversioned changes, including silent skips and backend changes."""
    _match_contract(config, PROTOCOL_V1, 'protocol')
    return copy.deepcopy(config)


def time_grid(config):
    """Return explicit zero-based array indices and anchor-relative times."""
    config = validate_protocol(config)
    result = {}
    for m in config['lead_steps']:
        steps = [k for k in config['query_steps'] if k + m <= 12]
        result[m] = {
            'query_steps': steps, 'query_rows': [k - 1 for k in steps],
            'target_columns': [k + m - 1 for k in steps],
            'query_times_s': [k * config['dt'] for k in steps],
            'target_times_s': [(k + m) * config['dt'] for k in steps],
            'revision_query_steps': [k for k in steps if k >= 2],
            'excluded_query_steps': [k for k in config['query_steps'] if k + m > 12],
        }
    return result


def parent_command(inputs):
    return [sys.executable,
            str(inputs.parent_project_root / 'scripts/run_noise_aware_stability.py'),
            '--output-root', str(inputs.parent), '--verify-only']


def check_parent_proof(inputs, proof):
    """Use the same proof semantics at initial execution and independent audit."""
    flags = ('all_seed_metrics_groups_weights_replayed', 'fit_only_normalization_verified',
             'seed2027_two_controls_match_parent')
    if (proof.get('identity') != inputs.parent_identity
            or any(proof.get(k) is not True for k in flags)
            or proof.get('holdout_evaluated') is not False
            or proof.get('owner_project_root') != str(inputs.parent_project_root)
            or proof.get('command') != parent_command(inputs)):
        raise ValueError('native parent verification proof mismatch')
    try:
        native = json.loads(proof['stdout'].splitlines()[-1])
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError('native parent proof stdout invalid') from exc
    metadata = {'owner_project_root', 'command', 'stdout', 'stderr'}
    if native != {k: v for k, v in proof.items() if k not in metadata}:
        raise ValueError('native parent proof disagrees with stdout')


def verify_parent(inputs):
    """Execute the native verifier once per preflight, in the original worktree."""
    command = parent_command(inputs)
    process = subprocess.run(command, cwd=inputs.parent_project_root,
                             env={**os.environ,
                                  'PYTHONPATH': str(inputs.parent_project_root / 'src'),
                                  'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'},
                             check=True, capture_output=True, text=True)
    proof = json.loads(process.stdout.splitlines()[-1])
    proof = {**proof, 'owner_project_root': str(inputs.parent_project_root),
             'command': command, 'stdout': process.stdout, 'stderr': process.stderr}
    check_parent_proof(inputs, proof)
    return proof


PHYSICAL_COLUMNS = {'raw_cv': 0, 'alpha_beta': 1, 'kf_cv': 2, 'kf_nis': 3,
                    'bounded_ca': 7, 'brake_stop': 8, 'hold_position': 9}


def replay_streams(inputs, role, strategy, indices, *, compare_parent=True):
    """Keep the original complete-role flat batching, then extract requested streams.

    This is a frozen inference adapter. GT is used only by forecast_metrics after
    prediction, and original full-role batches avoid changing parent arithmetic.
    """
    if role not in PROTOCOL_V1['roles']:
        raise ValueError('only development roles may be replayed')
    data = inputs.datasets[role]
    if (not isinstance(indices, list) or not indices
            or any(type(i) is not int or i < 0 or i >= len(data['features']) for i in indices)
            or len(set(indices)) != len(indices)):
        raise ValueError('unique nonempty in-range stream indices required')
    torch.set_num_threads(2)
    expected_names = {f's{s}_{v}': (s, v) for s in PROTOCOL_V1['model_seeds']
                      for v in PROTOCOL_V1['gate_variants']}
    if strategy in PHYSICAL_COLUMNS:
        forecast = data['candidates'][indices, :, PHYSICAL_COLUMNS[strategy]].copy()
        weights = None
    elif strategy in expected_names:
        seed, variant = expected_names[strategy]
        path = inputs.parent / (strategy + '.pt')
        if sha256_file(path) != inputs.references[str(path)]['sha256']:
            raise ValueError('checkpoint checksum changed after preflight assembly')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        if payload['train_seed'] != seed or payload['variant'] != variant:
            raise ValueError('checkpoint strategy identity mismatch')
        if not all(torch.isfinite(t).all() for t in payload['model'].values()):
            raise ValueError('nonfinite checkpoint parameters')
        model = MotionGate(seed=seed)
        model.load_state_dict(payload['model'], strict=True)
        model.eval().requires_grad_(False)
        # Deliberately use the legacy function, including its 4096 flat batches.
        all_forecast, all_weights = predict_motion(
            model, payload['stats'], data['features'], data['candidates'], variant,
            ca_allowed=data['ca_allowed'] if variant == 'guarded_ca' else None)
        forecast, weights = all_forecast[indices], all_weights[indices]
    else:
        raise ValueError('unknown frozen strategy')
    if forecast.dtype != np.float32 or not np.isfinite(forecast).all():
        raise ValueError('finite float32 frozen forecast required')
    metrics = forecast_metrics(forecast, data['target'][indices])
    if compare_parent:
        if strategy in expected_names:
            key = role + '__' + strategy
            expected = inputs.legacy_updates.get(key)
            if (expected is None or expected.shape != (len(data['features']), 11, 3)
                    or not np.array_equal(metrics, expected[indices])):
                raise ValueError('legacy per-update metric replay mismatch')
        else:
            for i, value in zip(indices, metrics.mean(1), strict=True):
                row = inputs.records[role][i]
                key = (role, row['instance'], row['sample'], row['condition'],
                       row['seed'], strategy)
                expected = inputs.legacy_means.get(key)
                if expected is None or not np.array_equal(value, expected):
                    raise ValueError('legacy physical mean metric replay mismatch')
    return {'forecast': forecast, 'weights': weights, 'metrics': metrics,
            'indices': indices, 'strategy': strategy, 'role': role,
            'exact_parent_metrics': bool(compare_parent),
            'legacy_comparison_unit': 'window_mean' if strategy in PHYSICAL_COLUMNS
                                      else 'per_update'}

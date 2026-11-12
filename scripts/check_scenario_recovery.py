"""S4c complete-pair pilot: frozen replay, excess AUC and confirmed recovery."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.evaluation.event_metrics import (
    pair_first_outlier,
    recovery_from_excess,
    summarize_recovery,
)
from trajectory_prediction.evaluation.scenario_inputs import assemble_inputs
from trajectory_prediction.evaluation.scenario_preflight import verify_preflight
from trajectory_prediction.evaluation.scenario_replay import PHYSICAL_COLUMNS, replay_streams
from trajectory_prediction.robustness.packets import PROFILES, make_packets
from trajectory_prediction.utils.provenance import sha256_file


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _selection(records, role):
    keys = sorted({(r['instance'], r['sample']) for r in records}, key=lambda k: hashlib.sha256(
        (f's4c-pilot-v1:{role}:' + ':'.join(k)).encode()).hexdigest())[:8]
    indices = [i for i, r in enumerate(records) if (r['instance'], r['sample']) in keys
               and r['condition'] in ('noise_05', 'noise_outlier')]
    if len(indices) != len(keys) * 3 * 2 or not keys:
        raise ValueError('complete three-noise reference/outlier cohort required')
    return {'window_keys': keys, 'indices': indices}


def _verify_packet_pairs(inputs, selection):
    count = 0
    for role, chosen in selection.items():
        lookup = {(r['instance'], r['sample'], r['seed'], r['condition']): i
                  for i, r in enumerate(inputs.records[role])}
        for instance, sample in chosen['window_keys']:
            for seed in (2027, 2028, 2029):
                i = lookup[instance, sample, seed, 'noise_05']
                gt = inputs.gt_windows[role][i]
                streams = [make_packets(gt['history_xy'], gt['future_xy'][:11],
                                        instance + ':' + sample, PROFILES[condition], seed)
                           for condition in ('noise_05', 'noise_outlier')]
                for ref, out in zip(*streams, strict=True):
                    if (ref.source_step != out.source_step or ref.available_step != ref.source_step
                            or out.available_step != out.source_step):
                        raise ValueError('paired packet source/arrival mismatch')
                    if ref.source_step in (4, 8):
                        if not math.isclose(float(np.linalg.norm(out.position - ref.position)),
                                            5., rel_tol=0., abs_tol=1e-10):
                            raise ValueError('paired outlier must have declared 5m injection')
                    elif not np.array_equal(ref.position, out.position):
                        raise ValueError('paired base Gaussian noise must be shared')
                count += 1
    return count


def _scalar_check(p, y, rows, strategy, paired, states):
    lookup = {(r['role'], r['instance'], r['sample'], r['seed'], r['condition']): i
              for i, r in enumerate(rows)}
    expected_keys = sorted({(r['role'], r['instance'], r['sample'], r['seed'], strategy)
                            for r in rows if r['condition'] == 'noise_outlier'})
    if paired['pair_keys'] != expected_keys:
        raise ValueError('independent pair identity mismatch')
    maximum, scalar_aucs = 0., []
    for event, key in enumerate(expected_keys):
        a = lookup[(*key[:4], 'noise_05')]
        b = lookup[(*key[:4], 'noise_outlier')]
        er, eo, distance = [], [], []
        for step in (4, 5, 6, 7):
            column = round((step * .5 + 1.) / .5) - 1
            er.append(math.hypot(*(float(p[a, step - 1, column, d]) - float(y[a, column, d])
                                   for d in (0, 1))))
            eo.append(math.hypot(*(float(p[b, step - 1, column, d]) - float(y[b, column, d])
                                   for d in (0, 1))))
            distance.append(math.hypot(*(float(p[b, step - 1, column, d]) - float(p[a, step - 1, column, d])
                                         for d in (0, 1))))
        excess = [max(0., v - u) for u, v in zip(er, eo, strict=True)]
        for field, values in [('reference_errors', er), ('outlier_errors', eo),
                               ('prediction_distance', distance), ('excess', excess)]:
            if not np.allclose(paired[field][event], values, rtol=1e-12, atol=1e-12):
                raise ValueError('scalar paired error/excess mismatch')
            maximum = max(maximum, float(np.abs(paired[field][event] - values).max()))
        bad = [i for i, v in enumerate(excess) if v > .5]
        status = 'unaffected' if not bad else ('recovered' if max(bad) <= 1 else 'not_recovered')
        confirm = max(bad) + 2 if status == 'recovered' else None
        auc = sum((excess[i] + excess[i + 1]) * .25 for i in range(3))
        scalar_aucs.append(auc)
        if (str(states['status'][event]) != status
                or not math.isclose(float(states['excess_auc'][event]), auc,
                                    rel_tol=1e-12, abs_tol=1e-12)
                or (confirm is not None and (states['confirm_step'][event] != confirm + 4
                                             or states['confirm_delay_s'][event] != confirm * .5))):
            raise ValueError('scalar recovery status/AUC/confirmation mismatch')
    return maximum, scalar_aucs


def check_recovery(preflight, output, *, verify_only=False):
    output = Path(output).resolve()
    if output.exists() and not verify_only:
        raise FileExistsError('use a new S4c output directory')
    saved_run, saved_marker = None, None
    if verify_only:
        output_marker_hash = sha256_file(output / 'completed.json')
        saved_marker = json.loads((output / 'completed.json').read_text(encoding='utf-8'))
        for name, digest in saved_marker['sha256'].items():
            path = (output / name).resolve()
            if not path.is_relative_to(output) or sha256_file(path) != digest:
                raise ValueError('recovery output checksum mismatch')
        saved_run = json.loads((output / 'run.json').read_text(encoding='utf-8'))
        if (saved_run['identity'] != saved_marker['identity']
                or digest_json(saved_run['binding']) != saved_marker['identity']):
            raise ValueError('recovery run identity mismatch')
        preflight = saved_run['binding']['preflight_root']
    preflight = Path(preflight).resolve()
    script = Path(__file__).resolve()
    root = script.parents[1]
    paths = [script, root / 'src/trajectory_prediction/evaluation/event_metrics.py',
             root / 'src/trajectory_prediction/evaluation/streaming_metrics.py',
             root / 'src/trajectory_prediction/robustness/packets.py']
    sources = {str(p): sha256_file(p) for p in paths}
    marker_path = preflight / 'preflight_completed.json'
    marker_hash = sha256_file(marker_path)
    marker = json.loads(marker_path.read_text(encoding='utf-8'))
    if sha256_file(preflight / 'run.json') != marker['sha256']['run.json']:
        raise ValueError('S2 run snapshot checksum mismatch')
    old_run = json.loads((preflight / 'run.json').read_text(encoding='utf-8'))
    old = old_run['binding']
    if old_run['identity'] != marker['identity'] or digest_json(old) != marker['identity']:
        raise ValueError('S2 identity mismatch')
    proof = verify_preflight(preflight)
    if any(sha256_file(Path(p)) != h for p, h in sources.items()):
        raise ValueError('S4c source changed during verification')
    if proof['identity'] != marker['identity'] or sha256_file(marker_path) != marker_hash:
        raise ValueError('S2 proof identity/snapshot changed')
    inputs = assemble_inputs(old['parent_path'], old['parent_project_root'])
    if inputs.parent_identity != old['parent_identity'] or inputs.references != old['references']:
        raise ValueError('GT/parent references changed')
    selection = {role: _selection(rows, role) for role, rows in inputs.records.items()}
    packet_pairs = _verify_packet_pairs(inputs, selection)
    binding = {'schema': 'recovery_s4c_complete_pair_pilot_v1',
               'preflight_root': str(preflight), 'preflight_identity': proof['identity'],
               'preflight_marker_sha256': marker_hash, 'preflight_files': marker['sha256'],
               'sources': sources, 'selection': selection, 'windows_per_role_cap': 8,
               'rank_salt': 's4c-pilot-v1', 'noise_seeds': [2027, 2028, 2029],
               'query_steps': [4, 5, 6, 7], 'source_step': 4, 'lead_steps': 2,
               'dt': .5, 'excess_threshold_m': .5, 'consecutive_updates': 2,
               'require_no_relapse': True, 'scalar_atol': 1e-12, 'scalar_rtol': 1e-12,
               'packet_injection_atol_m': 1e-10,
               'scope': 'new_complete_pairs_from_parent_not_S2_subset_not_full_phase3a'}
    identity = digest_json(binding)
    if verify_only:
        if digest_json(binding) != saved_marker['identity']:
            raise ValueError('recovery pilot configuration/source/selection changed')
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write(output / 'run.json', {'identity': identity, 'binding': binding})
    strategies = [f's{s}_{v}' for s in (2027, 2028, 2029)
                  for v in ('baseline7', 'plus_ca', 'guarded_ca')] + list(PHYSICAL_COLUMNS)
    events, groups, curves, forecasts, common = [], [], {}, {}, {}
    maximum = 0.
    for role, chosen in selection.items():
        indices = chosen['indices']
        rows = [inputs.records[role][i] for i in indices]
        target = inputs.datasets[role]['target'][indices]
        for strategy in strategies:
            prediction = replay_streams(inputs, role, strategy, indices)['forecast']
            forecasts[role + '__' + strategy] = prediction
            paired = pair_first_outlier(prediction, target, rows, strategy_identity=strategy)
            states = recovery_from_excess(paired['excess'])
            difference, scalar_aucs = _scalar_check(prediction, target, rows, strategy, paired, states)
            maximum = max(maximum, difference)
            eligible_keys = [k[:4] for k in paired['pair_keys']]
            if role not in common:
                common[role] = eligible_keys
            elif eligible_keys != common[role]:
                raise ValueError('strategies must use identical eligible event keys')
            window_keys = [k[:3] for k in paired['pair_keys']]
            noise_seeds = [k[3] for k in paired['pair_keys']]
            summary = summarize_recovery(paired['excess'], window_keys, noise_seeds)
            if not math.isclose(summary['mean_excess_auc_m_s'], float(np.mean(scalar_aucs)),
                                rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError('scalar full-cohort AUC mean mismatch')
            context = {'role': role, 'strategy': strategy,
                       'train_seed': int(strategy.split('_')[0][1:]) if strategy.startswith('s20') else None}
            groups.append({**context, **summary,
                           'mean_reference_error_m': float(paired['reference_errors'].mean()),
                           'mean_outlier_error_m': float(paired['outlier_errors'].mean()),
                           'mean_prediction_distance_m_diagnostic': float(paired['prediction_distance'].mean()),
                           'log_count': len({r['log'] for r in rows})})
            prefix = role + '__' + strategy + '__'
            for name in ('excess', 'reference_errors', 'outlier_errors', 'prediction_distance'):
                curves[prefix + name] = paired[name]
            for name in ('affected', 'recovered', 'confirm_delay_s', 'first_affected_step',
                         'confirm_step', 'excess_auc', 'status'):
                curves[prefix + name] = states[name]
            for i, key in enumerate(paired['pair_keys']):
                a, b = paired['reference_indices'][i], paired['outlier_indices'][i]
                delay = states['confirm_delay_s'][i]
                events.append({**context, 'pair_key': key, 'log': rows[a]['log'], 'scene': rows[a]['scene'],
                               'source_row': rows[a]['row'], 'reference_stream_index': indices[a],
                               'outlier_stream_index': indices[b], 'status': str(states['status'][i]),
                               'excess_auc_m_s': float(states['excess_auc'][i]),
                               'confirm_delay_s': float(delay) if np.isfinite(delay) else None,
                               'confirm_step': int(states['confirm_step'][i]),
                               **{name: paired[name][i].tolist() for name in
                                  ('excess', 'reference_errors', 'outlier_errors', 'prediction_distance')}})
    for name, digest in marker['sha256'].items():
        if sha256_file(preflight / name) != digest:
            raise ValueError('S2 input changed during S4c')
    for ref in inputs.references.values():
        if sha256_file(Path(ref['path'])) != ref['sha256']:
            raise ValueError('GT/parent reference changed during S4c')
    if (sha256_file(marker_path) != marker_hash
            or any(sha256_file(Path(p)) != h for p, h in sources.items())):
        raise ValueError('source or S2 snapshot changed during S4c')
    report = {'identity': identity, 'strategy_event_records': len(events), 'group_rows': len(groups),
              'unique_window_noise_pairs': sum(len(keys) for keys in common.values()),
              'packet_pairs_checked': packet_pairs, 'max_scalar_abs_difference_m': maximum,
              'common_eligible_keys_verified': True, 'full_phase3a_evaluation': False,
              'neural_training': False, 'holdout_evaluated': False}
    if verify_only:
        for name, value in [('summary', report), ('groups', groups), ('events', events)]:
            if digest_json(json.loads((output / (name + '.json')).read_text(encoding='utf-8'))) != digest_json(value):
                raise ValueError('saved recovery status/summary replay mismatch')
        for name, values in [('forecasts.npz', forecasts), ('curves.npz', curves)]:
            with np.load(output / name, allow_pickle=False) as z:
                if set(z.files) != set(values) or any(not np.array_equal(z[k], v, equal_nan=v.dtype.kind == 'f')
                                                    for k, v in values.items()):
                    raise ValueError('saved frozen forecast/curve replay mismatch')
        for name, digest in saved_marker['sha256'].items():
            if sha256_file(output / name) != digest:
                raise ValueError('recovery output checksum changed during replay')
        if sha256_file(output / 'completed.json') != output_marker_hash:
            raise ValueError('recovery output completion snapshot changed during replay')
        return {**report, 'independent_replay_verified': True}
    for name, value in [('summary', report), ('groups', groups), ('events', events)]:
        _write(output / (name + '.json'), value)
    np.savez_compressed(output / 'forecasts.npz', **forecasts)
    np.savez_compressed(output / 'curves.npz', **curves)
    names = ['run.json', 'summary.json', 'groups.json', 'events.json', 'forecasts.npz', 'curves.npz']
    _write(output / 'completed.json', {'identity': identity,
           'sha256': {name: sha256_file(output / name) for name in names}})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path,
                        default=Path('outputs/scenario-evaluation/preflight-v2'))
    parser.add_argument('--output-root', type=Path,
                        default=Path('outputs/scenario-evaluation/s4c-pilot-v1'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check_recovery(args.preflight, args.output_root, verify_only=args.verify_only)), flush=True)

"""S6b paired recovery events and offline GT maneuver groups."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from trajectory_prediction.analysis.maneuvers import describe_motion, tag_maneuvers
from trajectory_prediction.analysis.scenario_events import (
    EVENT_RULES,
    TAGS,
    annotate_window,
    build_review_manifest,
)
from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.evaluation.event_metrics import (
    pair_first_outlier,
    recovery_from_excess,
    summarize_recovery,
)
from trajectory_prediction.evaluation.scenario_preflight import PROJECT
from trajectory_prediction.evaluation.scenario_publication import _artifact_hashes, _read, _write
from trajectory_prediction.evaluation.scenario_report import (
    _compare_npz_nan,
    _scalar_mean,
    summarize_queries,
    verify_report,
)
from trajectory_prediction.evaluation.scenario_report import (
    _sources as query_sources,
)
from trajectory_prediction.utils.provenance import sha256_file

LEGACY_TAGS = ('low_speed', 'straight', 'turn', 'high_curvature', 'acceleration', 'deceleration')


def _sources():
    names = ['scripts/build_scenario_event_report.py', 'configs/analysis/maneuvers.yaml',
             'src/trajectory_prediction/analysis/maneuvers.py',
             'src/trajectory_prediction/analysis/scenario_events.py',
             'src/trajectory_prediction/evaluation/event_metrics.py',
             'src/trajectory_prediction/evaluation/scenario_event_report.py']
    return {**query_sources(), **{n: sha256_file(PROJECT / n) for n in names}}


def _close(actual, expected):
    if actual is None or expected is None:
        equal = actual is expected
    elif isinstance(expected, float):
        equal = math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-12)
    else:
        equal = actual == expected
    if not equal:
        raise ValueError('scalar event summary disagreement')


def _scalar_states(curves):
    states, delays, firsts, confirms, aucs = [], [], [], [], []
    for curve in curves:
        bad = [i for i, v in enumerate(curve) if v > .5]
        confirm = None
        if bad:
            for j in range(bad[0] + 2, 4):
                if all(v <= .5 for v in curve[j - 1:]):
                    confirm = j
                    break
        states.append('unaffected' if not bad else 'recovered' if confirm is not None else 'not_recovered')
        delays.append(confirm * .5 if confirm is not None else math.nan)
        firsts.append(bad[0] + 4 if bad else -1)
        confirms.append(confirm + 4 if confirm is not None else -1)
        aucs.append(math.fsum(float(curve[j]) * .25 + float(curve[j + 1]) * .25 for j in range(3)))
    return {'status': np.asarray(states, dtype='U13'), 'confirm_delay_s': np.asarray(delays),
            'first_affected_step': np.asarray(firsts, dtype=np.int64),
            'confirm_step': np.asarray(confirms, dtype=np.int64), 'excess_auc': np.asarray(aucs)}


def compute_events(predictions, target, keys, strategy_identity):
    """Pair only identical role/window/noise/model; clean remains outside M3."""
    indices = [i for i, k in enumerate(keys) if k['condition'] in ('noise_05', 'noise_outlier')]
    records = [{**keys[i], 'seed': keys[i]['noise_seed']} for i in indices]
    p, y = np.asarray(predictions), np.asarray(target)
    paired = pair_first_outlier(p[indices], y[indices], records, strategy_identity=strategy_identity)
    expected_pairs = {}
    for i, row in enumerate(records):
        key = row['role'], row['instance'], row['sample'], row['seed'], strategy_identity
        entry = expected_pairs.setdefault(key, {})
        if row['condition'] in entry:
            raise ValueError('scalar duplicate pair key')
        entry[row['condition']] = i
    ordered = sorted(expected_pairs)
    if any(set(v) != {'noise_05', 'noise_outlier'} for v in expected_pairs.values()):
        raise ValueError('scalar missing pair')
    expected_ref = [expected_pairs[k]['noise_05'] for k in ordered]
    expected_out = [expected_pairs[k]['noise_outlier'] for k in ordered]
    if (paired['reference_indices'] != expected_ref or paired['outlier_indices'] != expected_out
            or paired['pair_keys'] != ordered):
        raise ValueError('scalar pairing identity/index disagreement')
    for a, b in zip(expected_ref, expected_out, strict=True):
        if (any(records[a][k] != records[b][k] for k in ('scene', 'log', 'row'))
                or not np.array_equal(y[indices[a]], y[indices[b]])):
            raise ValueError('scalar paired GT/metadata disagreement')
    rows, cohorts = [], {}
    for a, b in zip(paired['reference_indices'], paired['outlier_indices'], strict=True):
        row = {**keys[indices[a]], 'reference_stream_index': indices[a],
               'outlier_stream_index': indices[b], 'strategy_identity': strategy_identity}
        row.pop('condition')
        rows.append(row)
        cohorts.setdefault((row['role'], row['instance'], row['sample']), set()).add(row['noise_seed'])
    if not rows or any(s != {2027, 2028, 2029} for s in cohorts.values()):
        raise ValueError('complete paired noise cohort required')
    arrays = {k: paired[k] for k in ('reference_errors', 'outlier_errors', 'excess', 'prediction_distance')}
    for i, row in enumerate(rows):
        a, b = row['reference_stream_index'], row['outlier_stream_index']
        for j, k in enumerate(range(4, 8)):
            col = k + 1
            def distance(left, right):
                return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))
            ref = distance(p[a, k - 1, col], y[a, col])
            out = distance(p[b, k - 1, col], y[b, col])
            for name, value in [('reference_errors', ref), ('outlier_errors', out),
                                ('excess', max(0., out - ref)),
                                ('prediction_distance', distance(p[a, k - 1, col], p[b, k - 1, col]))]:
                _close(float(arrays[name][i, j]), value)
    state = recovery_from_excess(arrays['excess'])
    reference = _scalar_states(arrays['excess'])
    for name, values in reference.items():
        if name == 'status':
            equal = np.array_equal(state[name], values)
        else:
            equal = np.allclose(state[name], values, rtol=1e-12, atol=1e-12, equal_nan=True)
        if not equal:
            raise ValueError('scalar recovery state disagreement')
    arrays.update({name: state[name] for name in reference})
    arrays['reference_stream_index'] = np.array([r['reference_stream_index'] for r in rows], dtype=np.int64)
    arrays['outlier_stream_index'] = np.array([r['outlier_stream_index'] for r in rows], dtype=np.int64)
    return arrays, rows


def summarize_events(arrays, rows):
    windows = [(r['instance'], r['sample']) for r in rows]
    seeds = [r['noise_seed'] for r in rows]
    result = summarize_recovery(arrays['excess'], windows, seeds)
    scalar = _scalar_states(arrays['excess'])
    statuses = list(scalar['status'])
    eligible, unaffected, recovered = len(rows), statuses.count('unaffected'), statuses.count('recovered')
    affected = eligible - unaffected
    reduced = {}
    for i, key in enumerate(windows):
        reduced.setdefault(key, []).append(float(scalar['excess_auc'][i]))
    auc = math.fsum(math.fsum(v) / len(v) for v in reduced.values()) / len(reduced) if reduced else None
    delays = [float(scalar['confirm_delay_s'][i]) for i, s in enumerate(statuses) if s == 'recovered']
    expected = {'eligible': eligible, 'affected': affected, 'unaffected': unaffected,
                'recovered': recovered, 'not_recovered': affected - recovered,
                'window_count': len(set(windows)), 'noise_seed_count': len(set(seeds)),
                'mean_excess_auc_m_s': auc,
                'recovered_fraction_of_affected': recovered / affected if affected else None,
                'not_recovered_fraction_of_affected': (affected - recovered) / affected if affected else None,
                'unaffected_fraction_of_eligible': unaffected / eligible if eligible else None,
                'mean_confirm_delay_s_recovered_events': math.fsum(delays) / len(delays) if delays else None,
                'status': 'ok' if eligible else 'not_evaluable', 'reason': None if eligible else 'empty_group'}
    for name, value in expected.items():
        _close(result[name], value)
    for name in ('reference_errors', 'outlier_errors', 'prediction_distance'):
        result['mean_' + {'reference_errors': 'reference_error_m', 'outlier_errors': 'outlier_error_m',
                          'prediction_distance': 'prediction_distance_m'}[name]] = (
            _scalar_mean(arrays[name], rows) if rows else None)
    return result


def summarize_event_logs(arrays, rows):
    """Window-weighted main result plus separately named equal-log means."""
    pooled = {'log': None, **summarize_events(arrays, rows)}
    log_groups = []
    for log in sorted({r['log'] for r in rows}):
        indices = [i for i, r in enumerate(rows) if r['log'] == log]
        log_groups.append({'log': log, **summarize_events(
            {k: v[indices] for k, v in arrays.items()}, [rows[i] for i in indices])})
    pooled['valid_log_count'] = len(log_groups)
    pooled['affected_log_count'] = sum(g['affected'] > 0 for g in log_groups)
    pooled['recovered_log_count'] = sum(g['recovered'] > 0 for g in log_groups)
    fields = ('mean_excess_auc_m_s', 'mean_reference_error_m', 'mean_outlier_error_m',
              'mean_prediction_distance_m', 'recovered_fraction_of_affected',
              'mean_confirm_delay_s_recovered_events')
    for field in fields:
        values = [g[field] for g in log_groups if g[field] is not None]
        pooled['log_macro_' + field] = math.fsum(values) / len(values) if values else None
    return [pooled, *log_groups]


def _prepare(query_root):
    query_root = Path(query_root).resolve()
    sources, marker_hash = _sources(), sha256_file(query_root / 'completed.json')
    verify_report(query_root)
    query_run = _read(query_root / 'run.json')
    publication = Path(query_run['binding']['publication'])
    pub_run = _read(publication / 'run.json')['binding']
    if pub_run['protocol']['event_labels'] != EVENT_RULES:
        raise ValueError('event label protocol mismatch')
    thresholds = load_yaml(PROJECT / 'configs/analysis/maneuvers.yaml')
    preflight = _read(Path(pub_run['preflight']) / 'run.json')['binding']
    parent = _read(Path(preflight['parent_path']) / 'run.json')['binding']
    if parent.get('thresholds', thresholds) != thresholds:
        raise ValueError('legacy thresholds differ from parent')
    binding = {'schema': 'scenario_event_report_s6b_v1', 'query_root': str(query_root),
               'query_marker_sha256': marker_hash, 'query_identity': query_run['identity'],
               'publication': str(publication), 'sources': sources, 'legacy_thresholds': thresholds,
               'preflight': pub_run['preflight'], 'preflight_sha256': pub_run['preflight_sha256'],
               'input_references': preflight['references'],
               'full_s6_complete': False, 'neural_training': False, 'holdout_evaluated': False,
               'review_scope': 'fresh_rule_candidates_pending_no_manual_review_import',
               'pending': ['training_seed_ddof1', 'deterministic_cases', 's7_formal_full_run']}
    _recheck(binding)
    return binding


def _recheck(binding):
    query = Path(binding['query_root'])
    if _sources() != binding['sources'] or sha256_file(query / 'completed.json') != binding['query_marker_sha256']:
        raise ValueError('event report source/query changed')
    _artifact_hashes(query, _read(query / 'completed.json')['sha256'])
    publication = Path(binding['publication'])
    query_binding = _read(query / 'run.json')['binding']
    if sha256_file(publication / 'completed.json') != query_binding['publication_sha256']:
        raise ValueError('publication marker changed')
    _artifact_hashes(publication, _read(publication / 'completed.json')['sha256'])
    preflight = Path(binding['preflight'])
    if sha256_file(preflight / 'preflight_completed.json') != binding['preflight_sha256']:
        raise ValueError('preflight reference checksum changed')
    for name, digest in _read(preflight / 'preflight_completed.json')['sha256'].items():
        path = (preflight / name).resolve()
        if not path.is_relative_to(preflight) or sha256_file(path) != digest:
            raise ValueError('preflight reference checksum changed')
    for ref in binding['input_references'].values():
        if sha256_file(Path(ref['path'])) != ref['sha256']:
            raise ValueError('input reference checksum changed')


def _labels(publication, role, thresholds):
    keys = _read(publication / f'{role}-keys.json')
    by_gt = {k['gt_index']: k for k in keys}
    with np.load(publication / 'gt' / f'{role}.npz', allow_pickle=False) as z:
        rows = []
        for i in range(len(z['future_xy'])):
            k = by_gt[i]
            identity = role, k['instance'], k['sample']
            annotation = annotate_window(z['history_xy'][i], z['future_xy'][i], z['history_mask'][i])
            legacy = describe_motion(z['future_xy'][i], .5)
            rows.append({**annotation, 'role': role, 'instance': k['instance'], 'sample': k['sample'],
                         'scene': k['scene'], 'log': k['log'], 'source_row': k['row'],
                         'window_id': hashlib.sha256(':'.join(identity).encode()).hexdigest(),
                         'legacy_motion': legacy, 'legacy_tags': tag_maneuvers(legacy, thresholds)})
    return rows


def _calculate(binding):
    publication, query = Path(binding['publication']), Path(binding['query_root'])
    registry = _read(publication / 'run.json')['binding']
    labels, coverage, paired_keys, event_groups, query_groups = [], [], {}, [], []
    for role in registry['protocol']['roles']:
        windows = _labels(publication, role, binding['legacy_thresholds'])
        labels.extend(windows)
        for log in [None, *sorted({w['log'] for w in windows})]:
            selected = [w for w in windows if log is None or w['log'] == log]
            for tag in TAGS:
                counts = {s: sum(w['tag_status'][tag]['status'] == s for w in selected)
                          for s in ('candidate', 'absent', 'not_evaluable')}
                if sum(counts.values()) != len(selected):
                    raise ValueError('label status conservation failed')
                coverage.append({'role': role, 'log': log, 'tag': tag, 'window_count': len(selected), **counts})
        keys = _read(publication / f'{role}-keys.json')
        with np.load(publication / 'gt' / f'{role}.npz', allow_pickle=False) as z:
            target = z['future_xy'][[k['gt_index'] for k in keys]]
        selectors = [('all', 'all', {(w['instance'], w['sample']) for w in windows})]
        selectors += [('legacy', tag, {(w['instance'], w['sample']) for w in windows if tag in w['legacy_tags']})
                      for tag in LEGACY_TAGS]
        selectors += [('rule_candidate', tag, {(w['instance'], w['sample']) for w in windows if tag in w['labels']})
                      for tag in TAGS]
        for strategy in registry['strategies']:
            name = f"{role}__{strategy['strategy']}.npz"
            identity = digest_json({'publication_identity': _read(publication / 'run.json')['identity'],
                                    'strategy': strategy})
            with np.load(publication / 'predictions' / name, allow_pickle=False) as z:
                arrays, rows = compute_events(z['forecast'], target, keys, identity)
            paired_keys[name] = rows
            with np.load(query / 'queries' / name, allow_pickle=False) as z:
                metrics = {k: z[k] for k in z.files}
            meta = {'role': role, 'strategy': strategy['strategy'], 'train_seed': strategy['train_seed']}
            for namespace, tag, members in selectors:
                group_meta = {**meta, 'group_namespace': namespace, 'tag': tag}
                selected = [i for i, r in enumerate(rows) if (r['instance'], r['sample']) in members]
                summaries = summarize_event_logs({k: v[selected] for k, v in arrays.items()},
                                                  [rows[i] for i in selected])
                event_groups.extend([{**group_meta, **summary} for summary in summaries])
                selected = [i for i, k in enumerate(keys) if (k['instance'], k['sample']) in members]
                if selected:
                    sliced = {k: v[selected] if v.ndim == 2 else v for k, v in metrics.items()}
                    query_groups.extend([{**group_meta, **g} for g in summarize_queries(sliced, [keys[i] for i in selected])])
                else:
                    query_groups.append({**group_meta, 'status': 'not_evaluable',
                                         'reason': 'empty_gt_window_group', 'window_count': 0})
            yield name, arrays, labels, coverage, paired_keys, event_groups, query_groups


def _documents(binding, labels, coverage, pairs, event_groups, query_groups):
    review = build_review_manifest(labels)
    summary = {'event_shards': len(pairs), 'paired_observation_events': sum(len(r) for r in pairs.values()),
               'window_count': len(labels), 'review_assignments': len(review), 'manual_completed': 0,
               'full_s6_complete': False, 'neural_training': False, 'holdout_evaluated': False,
               'scope': 'window_events_not_independent_physical_episodes'}
    return {'windows.json': labels, 'label_coverage.json': coverage, 'pair_keys.json': pairs,
            'event_groups.json': event_groups, 'maneuver_query_groups.json': query_groups,
            'event_review.json': review, 'summary.json': summary,
            'run.json': {'identity': digest_json(binding), 'binding': binding},
            'event_time_grid.json': {'query_steps': [4, 5, 6, 7], 'target_columns': [5, 6, 7, 8],
                                     'lead_s': 1., 'relative_times_s': [0., .5, 1., 1.5]}}


def build_event_report(query_root, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('use a new event report directory')
    binding = _prepare(query_root)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'events').mkdir()
    hashes = {}
    for name, arrays, labels, coverage, pairs, eg, qg in _calculate(binding):
        path = output / 'events' / name
        np.savez_compressed(path, **arrays)
        _compare_npz_nan(path, arrays)
        hashes['events/' + name] = sha256_file(path)
    documents = _documents(binding, labels, coverage, pairs, eg, qg)
    for name, value in documents.items():
        _write(output / name, value)
        hashes[name] = sha256_file(output / name)
    _recheck(binding)
    actual = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    if actual != set(hashes) or any(sha256_file(output / n) != h for n, h in hashes.items()):
        raise ValueError('event artifact coverage/contents changed')
    _write(output / 'completed.json', {'identity': digest_json(binding), 'sha256': hashes})
    return documents['summary.json']


def verify_event_report(output):
    output = Path(output).resolve()
    marker_path = output / 'completed.json'
    if not marker_path.is_file():
        raise ValueError('incomplete event report')
    marker_hash, marker = sha256_file(marker_path), _read(marker_path)
    _artifact_hashes(output, marker['sha256'])
    run = _read(output / 'run.json')
    binding = _prepare(run['binding']['query_root'])
    if run != {'identity': digest_json(binding), 'binding': binding} or marker['identity'] != run['identity']:
        raise ValueError('event identity mismatch')
    names = set()
    for name, arrays, labels, coverage, pairs, eg, qg in _calculate(binding):
        names.add('events/' + name)
        _compare_npz_nan(output / 'events' / name, arrays)
    documents = _documents(binding, labels, coverage, pairs, eg, qg)
    for name, expected in documents.items():
        names.add(name)
        if _read(output / name) != expected:
            raise ValueError(f'event semantic replay mismatch: {name}')
    if set(marker['sha256']) != names:
        raise ValueError('event file coverage mismatch')
    _recheck(binding)
    _artifact_hashes(output, marker['sha256'])
    if sha256_file(marker_path) != marker_hash:
        raise ValueError('event completion marker changed')
    return {**documents['summary.json'], 'scalar_reference_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--query-report', type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only and args.query_report is None:
        parser.error('--query-report required for build')
    result = verify_event_report(args.output_root) if args.verify_only else build_event_report(args.query_report, args.output_root)
    print(json.dumps(result, indent=2, allow_nan=False))

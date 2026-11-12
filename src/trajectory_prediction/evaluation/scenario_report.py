"""S6a query metrics and reductions; event/seed synthesis and case plots follow."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.evaluation.directional_metrics import (
    directional_errors,
    summarize_directional,
)
from trajectory_prediction.evaluation.revision_metrics import (
    remaining_miss_flags,
    revision_gains,
    summarize_revisions,
    summarize_tail,
)
from trajectory_prediction.evaluation.scenario_preflight import PROJECT
from trajectory_prediction.evaluation.scenario_publication import (
    _artifact_hashes,
    _read,
    _source_hashes,
    _write,
    verify_publication,
)
from trajectory_prediction.evaluation.streaming_metrics import fixed_lead_errors
from trajectory_prediction.utils.provenance import sha256_file


def _sources():
    return {**_source_hashes(), **{n: sha256_file(PROJECT / n) for n in [
        'scripts/build_scenario_query_report.py',
        *[f'src/trajectory_prediction/evaluation/{m}.py' for m in (
            'scenario_report', 'streaming_metrics', 'revision_metrics', 'directional_metrics')]]}}


def _distance(a, b):
    return math.hypot(float(a[0]) - float(b[0]), float(a[1]) - float(b[1]))


def _scalar_queries(p, y, history, mask):
    """Independent Python loops; no calls to vector metric implementations."""
    result = {}
    for m in (1, 2, 4):
        errors, gains = [], []
        for i in range(len(p)):
            e, g = [], []
            for k in range(1, 13 - m):
                j = k + m - 1
                value = _distance(p[i, k - 1, j], y[i, j])
                e.append(value)
                if k > 1:
                    g.append(_distance(p[i, k - 2, j], y[i, j]) - value)
            errors.append(e)
            gains.append(g)
        result[f'lead{m}_error_m'] = np.asarray(errors)
        result[f'lead{m}_gain_m'] = np.asarray(gains)
    longitudinal, lateral, valid, source, fresh, remaining = [], [], [], [], [], []
    for i in range(len(p)):
        points = list(history[i]) + list(y[i])
        available = list(mask[i]) + [True] * 12
        direction, last_step = None, -999
        longs, lats, oks, origins, news = [], [], [], [], []
        for end in range(1, 15):
            d = _distance(points[end], points[end - 1])
            reliable = bool(available[end] and available[end - 1] and d >= .25)
            if reliable:
                direction = ((float(points[end][0]) - float(points[end - 1][0])) / d,
                             (float(points[end][1]) - float(points[end - 1][1])) / d)
                last_step = end - 4
            if end >= 5:
                k = end - 4
                j = k + 1
                dx = float(p[i, k - 1, j, 0]) - float(y[i, j, 0])
                dy = float(p[i, k - 1, j, 1]) - float(y[i, j, 1])
                oks.append(direction is not None)
                origins.append(last_step)
                news.append(reliable)
                longs.append(dx * direction[0] + dy * direction[1] if direction else math.nan)
                lats.append(-dx * direction[1] + dy * direction[0] if direction else math.nan)
        longitudinal.append(longs)
        lateral.append(lats)
        valid.append(oks)
        source.append(origins)
        fresh.append(news)
        remaining.append([any(_distance(p[i, k - 1, j], y[i, j]) > 2
                              for j in range(k, 12)) for k in range(1, 12)])
    result.update(direction_longitudinal_m=np.asarray(longitudinal),
                  direction_lateral_m=np.asarray(lateral), direction_valid=np.asarray(valid),
                  direction_source_step=np.asarray(source), direction_fresh=np.asarray(fresh),
                  remaining_miss=np.asarray(remaining))
    return result


def compute_queries(predictions, target, history, history_mask):
    """Compute and scalar-audit M1/M2/M4/M5, preserving all query indices."""
    arrays = {'stream_index': np.arange(len(predictions), dtype=np.int64)}
    for m in (1, 2, 4):
        e = fixed_lead_errors(predictions, target, lead_steps=m)
        g = revision_gains(predictions, target, lead_steps=m)
        arrays.update({f'lead{m}_error_m': e['values'], f'lead{m}_gain_m': g['values'],
                       f'lead{m}_query_step': e['query_steps'],
                       f'lead{m}_query_row': e['query_rows'],
                       f'lead{m}_target_column': e['target_columns'],
                       f'lead{m}_query_time_s': e['query_times_s'],
                       f'lead{m}_target_time_s': e['target_times_s'],
                       f'lead{m}_revision_query_step': g['query_steps'],
                       f'lead{m}_revision_previous_row': g['previous_query_rows'],
                       f'lead{m}_revision_target_column': g['target_columns']})
    d = directional_errors(predictions, target, history, history_mask)
    arrays.update(direction_longitudinal_m=d['longitudinal_signed_m'],
                  direction_lateral_m=d['lateral_signed_m'], direction_valid=d['valid_mask'],
                  direction_source_step=d['heading_source_steps'],
                  direction_fresh=d['fresh_heading_mask'], direction_reason=d['reason'],
                  direction_query_step=d['query_steps'],
                  direction_target_column=d['target_columns'],
                  remaining_query_step=np.arange(1, 12),
                  remaining_point_count=np.arange(11, 0, -1),
                  remaining_miss=remaining_miss_flags(predictions, target)['miss'])
    reference = _scalar_queries(predictions, target, history, history_mask)
    checked, max_error = 0, 0.
    for name, expected in reference.items():
        actual = arrays[name]
        if actual.shape != expected.shape:
            raise ValueError(f'scalar shape disagreement: {name}')
        if expected.dtype.kind in 'biu':
            matches = np.array_equal(actual, expected)
        else:
            matches = np.allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)
            finite = np.isfinite(expected)
            if finite.any():
                max_error = max(max_error, float(np.max(np.abs(actual[finite] - expected[finite]))))
        if not matches:
            raise ValueError(f'scalar metric disagreement: {name}')
        checked += expected.size
    if not np.array_equal(arrays['direction_reason'], np.where(
            reference['direction_valid'], 'ok', 'no_past_reliable_heading')):
        raise ValueError('scalar heading reason disagreement')
    return arrays, {'scalar_checked_values': checked, 'max_scalar_abs_error': max_error}


def _scalar_mean(x, keys, valid=None):
    groups = {}
    for i, key in enumerate(keys):
        values = [float(v) for j, v in enumerate(x[i]) if valid is None or valid[i, j]]
        if values:
            groups.setdefault((key['instance'], key['sample']), []).append(math.fsum(values) / len(values))
    windows = [math.fsum(v) / len(v) for v in groups.values()]
    return math.fsum(windows) / len(windows) if windows else None


def _assert_scalar(value, expected):
    if (value is None) != (expected is None) or (expected is not None and not math.isclose(
            value, expected, rel_tol=1e-12, abs_tol=1e-12)):
        raise ValueError('scalar reduction disagreement')


def _quantile(values, q):
    ordered = sorted(values)
    at = (len(ordered) - 1) * q
    lo, hi = math.floor(at), math.ceil(at)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def _scalar_window_values(x, keys):
    grouped = {}
    for row, key in zip(x, keys, strict=True):
        grouped.setdefault((key['instance'], key['sample']), []).append(
            math.fsum(float(v) for v in row) / len(row))
    return [math.fsum(v) / len(v) for v in grouped.values()]


def _audit_reductions(tail, revisions, errors, gains, keys):
    counts = {'stream_count': len(keys), 'window_count': len({(k['instance'], k['sample']) for k in keys}),
              'noise_seed_count': len({k['noise_seed'] for k in keys}), 'noise_repeat_count': len(keys)}
    for name, value in counts.items():
        if tail[name] != value or revisions[name] != value:
            raise ValueError('scalar cohort count disagreement')
    error_values = [float(v) for row in errors for v in row]
    gain_values = [float(v) for row in gains for v in row]
    if tail['query_count'] != len(error_values) or revisions['pair_count'] != len(gain_values):
        raise ValueError('scalar denominator disagreement')
    for item in (tail, revisions):
        if item['status'] != 'ok' or item['reason'] is not None:
            raise ValueError('scalar status disagreement')
    _assert_scalar(tail['miss_fraction'], sum(v > 2 for v in error_values) / len(error_values))
    _assert_scalar(tail['p95_window_mean_error'], _quantile(_scalar_window_values(errors, keys), .95))
    _assert_scalar(revisions['median_pair_gain_m'], _quantile(gain_values, .5))
    for label, predicate in [('beneficial', lambda v: v > .05),
                             ('harmful', lambda v: v < -.05),
                             ('neutral', lambda v: -.05 <= v <= .05)]:
        indicators = [[int(predicate(float(v))) for v in row] for row in gains]
        windows = _scalar_window_values(indicators, keys)
        _assert_scalar(revisions['window_macro_' + label + '_fraction'], math.fsum(windows) / len(windows))


def summarize_queries(arrays, keys):
    """Keep conditions separate; main means weight windows, not noise repeats."""
    seen, cohorts = set(), {}
    for key in keys:
        cohort = key['instance'], key['sample'], key['condition']
        pair = (*cohort, key['noise_seed'])
        if pair in seen:
            raise ValueError('duplicate window/condition/noise key')
        seen.add(pair)
        cohorts.setdefault(cohort, set()).add(key['noise_seed'])
    if not keys or any(seeds != {2027, 2028, 2029} for seeds in cohorts.values()):
        raise ValueError('complete v1 noise cohort required')
    if any(a.shape[0] != len(keys) for a in arrays.values() if a.ndim == 2):
        raise ValueError('query arrays and keys must align')
    result = []
    for condition in sorted({k['condition'] for k in keys}):
        logs = sorted({k['log'] for k in keys if k['condition'] == condition})
        for log in [None, *logs]:
            indices = [i for i, k in enumerate(keys) if k['condition'] == condition
                       and (log is None or k['log'] == log)]
            selected = [keys[i] for i in indices]
            windows = [(k['instance'], k['sample']) for k in selected]
            seeds = [k['noise_seed'] for k in selected]
            group = {'condition': condition, 'log': log, 'm1_m5': {}, 'm2': {}}
            for m in (1, 2, 4):
                errors = arrays[f'lead{m}_error_m'][indices]
                gains = arrays[f'lead{m}_gain_m'][indices]
                tail = summarize_tail(errors, windows, seeds)
                revisions = summarize_revisions(gains, windows, seeds)
                _assert_scalar(tail['mean_error_m'], _scalar_mean(errors, selected))
                _assert_scalar(revisions['mean_gain_m'], _scalar_mean(gains, selected))
                flat = sorted(float(v) for row in errors for v in row)
                for q, name in [(.5, 'p50_query_error'), (.95, 'p95_query_error'), (.99, 'p99_query_error')]:
                    at = (len(flat) - 1) * q
                    lo, hi = math.floor(at), math.ceil(at)
                    _assert_scalar(tail[name], flat[lo] + (flat[hi] - flat[lo]) * (at - lo))
                if tail['miss_count'] != sum(v > 2 for v in flat):
                    raise ValueError('scalar miss count disagreement')
                _audit_reductions(tail, revisions, errors, gains, selected)
                for label, predicate in [('beneficial', lambda v: v > .05),
                                         ('harmful', lambda v: v < -.05),
                                         ('neutral', lambda v: -.05 <= v <= .05)]:
                    count = sum(predicate(float(v)) for row in gains for v in row)
                    if revisions[label + '_count'] != count:
                        raise ValueError('scalar revision category disagreement')
                    _assert_scalar(revisions[label + '_fraction'], count / gains.size)
                group['m1_m5'][str(m)], group['m2'][str(m)] = tail, revisions
            x, y, valid = (arrays[n][indices] for n in (
                'direction_longitudinal_m', 'direction_lateral_m', 'direction_valid'))
            group['m4'] = summarize_directional(x, y, valid, windows, seeds)
            valid_count = sum(bool(v) for row in valid for v in row)
            evaluable_streams = sum(any(row) for row in valid)
            evaluable_windows = len({windows[i] for i, row in enumerate(valid) if any(row)})
            for name, expected in {
                'query_count': valid.size, 'valid_query_count': valid_count,
                'invalid_query_count': valid.size - valid_count, 'stream_count': len(selected),
                'evaluable_stream_count': evaluable_streams, 'window_count': len(set(windows)),
                'evaluable_window_count': evaluable_windows, 'noise_seed_count': len(set(seeds)),
                'status': 'ok' if valid_count else 'not_evaluable',
                'reason': None if valid_count else 'no_valid_heading',
            }.items():
                if group['m4'][name] != expected:
                    raise ValueError('scalar directional count/status disagreement')
            _assert_scalar(group['m4']['heading_coverage'], valid_count / valid.size)
            for name, value in [('mean_longitudinal_signed_m', x),
                                ('mean_longitudinal_abs_m', np.abs(x)),
                                ('mean_lateral_abs_m', np.abs(y))]:
                _assert_scalar(group['m4'][name], _scalar_mean(value, selected, valid))
            flags = arrays['remaining_miss'][indices]
            group['remaining_miss'] = {'query_steps': list(range(1, 12)),
                                       'remaining_points': list(range(11, 0, -1)),
                                       'miss_update_counts': flags.sum(axis=0).astype(int).tolist(),
                                       'query_denominators': [len(flags)] * 11,
                                       'online_MR2_remaining': flags.mean(axis=0).tolist(),
                                       'official_nuscenes_metric': False}
            result.append(group)
        pooled = next(g for g in result if g['condition'] == condition and g['log'] is None)
        log_groups = [g for g in result if g['condition'] == condition and g['log'] is not None]
        pooled['valid_log_count'] = len(log_groups)
        pooled['log_macro_mean_error_m'] = {
            str(m): math.fsum(g['m1_m5'][str(m)]['mean_error_m'] for g in log_groups) / len(log_groups)
            for m in (1, 2, 4)}
    json.dumps(result, allow_nan=False)
    return result


def _prepare(publication):
    publication = Path(publication).resolve()
    sources = _sources()
    marker_hash = sha256_file(publication / 'completed.json')
    proof = verify_publication(publication)
    binding = {'schema': 'scenario_query_report_s6a_v1', 'publication': str(publication),
               'publication_sha256': marker_hash, 'publication_identity': proof['identity'],
               'sources': sources, 'full_s6_complete': False, 'holdout_evaluated': False,
               'neural_training': False, 'scope': 'm1_m2_m4_m5_query_metrics_development',
               'pending': ['m3_event_pairing', 'maneuver_groups', 'training_seed_ddof1',
                           'deterministic_cases', 's7_formal_full_run']}
    _recheck(binding)
    return publication, binding


def _recheck(binding):
    root = Path(binding['publication'])
    if _sources() != binding['sources'] or sha256_file(root / 'completed.json') != binding['publication_sha256']:
        raise ValueError('report source/publication changed')
    _artifact_hashes(root, _read(root / 'completed.json')['sha256'])


def _calculate(publication, output=None):
    run = _read(publication / 'run.json')['binding']
    groups, audits = [], []
    artifacts = {}
    for role in run['protocol']['roles']:
        keys = _read(publication / f'{role}-keys.json')
        with np.load(publication / 'gt' / f'{role}.npz', allow_pickle=False) as z:
            gt = {k: z[k] for k in ('future_xy', 'history_xy', 'history_mask')}
        indices = [k['gt_index'] for k in keys]
        for strategy in run['strategies']:
            name = f"{role}__{strategy['strategy']}.npz"
            with np.load(publication / 'predictions' / name, allow_pickle=False) as z:
                arrays, audit = compute_queries(z['forecast'], gt['future_xy'][indices],
                                                gt['history_xy'][indices], gt['history_mask'][indices])
            metadata = {'role': role, 'strategy': strategy['strategy'],
                        'train_seed': strategy['train_seed']}
            groups.extend([{**metadata, **g} for g in summarize_queries(arrays, keys)])
            audits.append({**metadata, **audit})
            if output is not None:
                path = output / 'queries' / name
                np.savez_compressed(path, **arrays)
                _compare_npz_nan(path, arrays)
                artifacts[f'queries/{name}'] = sha256_file(path)
            yield name, arrays, groups, audits, artifacts


def _compare_npz_nan(path, expected):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != set(expected):
            raise ValueError('query archive field coverage mismatch')
        for name, value in expected.items():
            actual = z[name]
            equal = (np.array_equal(actual, value, equal_nan=True) if value.dtype.kind == 'f'
                     else np.array_equal(actual, value))
            if actual.dtype != value.dtype or not equal:
                raise ValueError(f'query scalar replay mismatch: {name}')


def build_report(publication, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('use a new query report directory')
    publication, binding = _prepare(publication)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'queries').mkdir()
    _write(output / 'run.json', {'identity': digest_json(binding), 'binding': binding})
    run_hash = sha256_file(output / 'run.json')
    shards = 0
    for _, _, groups, audits, hashes in _calculate(publication, output):
        shards += 1
    summary = {'metric_shards': shards, 'full_s6_complete': False,
               'scalar_checked_values': sum(a['scalar_checked_values'] for a in audits),
               'holdout_evaluated': False, 'neural_training': False,
               'event_review_status': 'not_assembled_in_s6a'}
    for name, value in [('groups.json', groups), ('audit.json', audits), ('summary.json', summary)]:
        _write(output / name, value)
        hashes[name] = sha256_file(output / name)
    hashes['run.json'] = run_hash
    _recheck(binding)
    actual = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    if actual != set(hashes) or any(sha256_file(output / n) != h for n, h in hashes.items()):
        raise ValueError('query report coverage/artifact changed')
    _write(output / 'completed.json', {'identity': digest_json(binding), 'sha256': hashes})
    return summary


def verify_report(output):
    output = Path(output).resolve()
    marker_path = output / 'completed.json'
    if not marker_path.is_file():
        raise ValueError('incomplete query report')
    marker_hash = sha256_file(marker_path)
    marker = _read(marker_path)
    _artifact_hashes(output, marker['sha256'])
    run = _read(output / 'run.json')
    publication, binding = _prepare(run['binding']['publication'])
    if run != {'identity': digest_json(binding), 'binding': binding} or marker['identity'] != run['identity']:
        raise ValueError('query report identity mismatch')
    names = {'run.json', 'groups.json', 'audit.json', 'summary.json'}
    shards = 0
    for name, arrays, groups, audits, _ in _calculate(publication):
        names.add('queries/' + name)
        _compare_npz_nan(output / 'queries' / name, arrays)
        shards += 1
    expected_summary = {'metric_shards': shards, 'full_s6_complete': False,
                        'scalar_checked_values': sum(a['scalar_checked_values'] for a in audits),
                        'holdout_evaluated': False, 'neural_training': False,
                        'event_review_status': 'not_assembled_in_s6a'}
    if (set(marker['sha256']) != names or _read(output / 'groups.json') != groups
            or _read(output / 'audit.json') != audits or _read(output / 'summary.json') != expected_summary):
        raise ValueError('query report group/summary replay mismatch')
    _recheck(binding)
    _artifact_hashes(output, marker['sha256'])
    if sha256_file(marker_path) != marker_hash:
        raise ValueError('query report marker changed')
    return {**expected_summary, 'scalar_reference_verified': True, 'frozen_publication_replayed': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--publication', type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only and args.publication is None:
        parser.error('--publication required when building')
    result = verify_report(args.output_root) if args.verify_only else build_report(args.publication, args.output_root)
    print(json.dumps(result, indent=2, allow_nan=False))

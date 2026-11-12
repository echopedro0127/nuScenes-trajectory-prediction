"""S4a subset integration: revision gains, tails and per-query remaining miss."""
import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.evaluation.revision_metrics import (
    remaining_miss_flags,
    revision_gains,
    summarize_revisions,
    summarize_tail,
)
from trajectory_prediction.evaluation.scenario_inputs import assemble_inputs
from trajectory_prediction.evaluation.scenario_preflight import verify_preflight
from trajectory_prediction.evaluation.streaming_metrics import fixed_lead_errors
from trajectory_prediction.utils.provenance import sha256_file


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _reference(p, y, m):
    errors, gains = [], []
    for sample in range(len(y)):
        e, g = [], []
        for k in range(1, 12):
            time = k * .5 + m * .5
            if time > 6:
                continue
            j = round(time / .5) - 1
            current = math.hypot(*(float(p[sample, k - 1, j, d]) - float(y[sample, j, d])
                                   for d in (0, 1)))
            e.append(current)
            if k >= 2:
                previous = math.hypot(*(float(p[sample, k - 2, j, d]) - float(y[sample, j, d])
                                        for d in (0, 1)))
                g.append(previous - current)
        errors.append(e)
        gains.append(g)
    return np.array(errors), np.array(gains)


def _quantile(values, q):
    ordered = sorted(float(v) for v in np.asarray(values).flat)
    rank = (len(ordered) - 1) * q
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _macro(values, keys):
    by_window = {}
    for row, key in zip(values, keys, strict=True):
        by_window.setdefault(key, []).append(statistics.fmean(row))
    return statistics.fmean(statistics.fmean(rows) for rows in by_window.values())


def check_revisions(preflight, output):
    preflight, output = Path(preflight).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('use a new S4a output directory')
    source = Path(__file__).resolve()
    root = source.parents[1]
    paths = [source, root / 'src/trajectory_prediction/evaluation/revision_metrics.py',
             root / 'src/trajectory_prediction/evaluation/streaming_metrics.py']
    source_hashes = {str(p): sha256_file(p) for p in paths}
    marker_path = preflight / 'preflight_completed.json'
    marker_hash = sha256_file(marker_path)
    marker = json.loads(marker_path.read_text(encoding='utf-8'))
    run_path = preflight / 'run.json'
    if sha256_file(run_path) != marker['sha256']['run.json']:
        raise ValueError('S2 run snapshot checksum mismatch')
    run = json.loads(run_path.read_text(encoding='utf-8'))
    old = run['binding']
    if run['identity'] != marker['identity'] or digest_json(old) != marker['identity']:
        raise ValueError('S2 snapshot identity mismatch')
    proof = verify_preflight(preflight)
    if any(sha256_file(Path(p)) != digest for p, digest in source_hashes.items()):
        raise ValueError('S4a source changed during parent verification')
    if proof['identity'] != marker['identity'] or sha256_file(marker_path) != marker_hash:
        raise ValueError('verified S2 identity/snapshot changed')
    inputs = assemble_inputs(old['parent_path'], old['parent_project_root'])
    if inputs.parent_identity != old['parent_identity'] or inputs.references != old['references']:
        raise ValueError('verified GT/parent references changed')
    binding = {'schema': 'revision_s4a_subset_v1', 'preflight_identity': proof['identity'],
               'preflight_root': str(preflight), 'preflight_marker_sha256': marker_hash,
               'preflight_files': marker['sha256'], 'lead_steps': [1, 2, 4],
               'revision_tolerance_m': .05, 'miss_distance_m': 2., 'quantile_method': 'linear',
               'scalar_atol': 1e-12, 'scalar_rtol': 1e-12,
               'scope': 'S2_batch_subset_only_not_full_phase3a',
               'sources': source_hashes}
    identity = digest_json(binding)
    groups, remaining_groups, arrays = [], [], {}
    gain_count, error_count, flag_count, max_difference, shards = 0, 0, 0, 0., 0
    for name in sorted(marker['sha256']):
        if not name.endswith('.npz'):
            continue
        role, strategy = name[:-4].split('__')
        with np.load(preflight / name, allow_pickle=False) as z:
            p, indices = z['forecast'], z['indices']
        records = [inputs.records[role][int(i)] for i in indices]
        y = inputs.datasets[role]['target'][indices]
        remaining = remaining_miss_flags(p, y)['miss']
        # Independent remaining-horizon check uses scalar distances and explicit future times.
        scalar_remaining = np.array([
            [any(math.hypot(*(float(p[i, k - 1, j, d]) - float(y[i, j, d]) for d in (0, 1))) > 2
                 for j in range(k, 12)) for k in range(1, 12)] for i in range(len(y))])
        if not np.array_equal(remaining, scalar_remaining):
            raise ValueError('remaining miss scalar reference mismatch')
        arrays[name[:-4] + '__remaining_miss'] = remaining
        flag_count += remaining.size
        for m in binding['lead_steps']:
            gains = revision_gains(p, y, lead_steps=m)
            errors = fixed_lead_errors(p, y, lead_steps=m)['values']
            ref_e, ref_g = _reference(p, y, m)
            for actual, expected in [(errors, ref_e), (gains['values'], ref_g)]:
                if not np.allclose(actual, expected, rtol=1e-12, atol=1e-12):
                    raise ValueError('S4a scalar distance/gain mismatch')
                max_difference = max(max_difference, float(np.abs(actual - expected).max()))
            for key, expected in [('beneficial', ref_g > .05), ('harmful', ref_g < -.05),
                                  ('neutral', (ref_g >= -.05) & (ref_g <= .05))]:
                if not np.array_equal(gains[key], expected):
                    raise ValueError('S4a gain classification mismatch')
            arrays[name[:-4] + '__g' + str(m)] = gains['values']
            arrays[name[:-4] + '__e' + str(m)] = errors
            gain_count += gains['values'].size
            error_count += errors.size
            for condition in sorted({r['condition'] for r in records}):
                selected = [i for i, r in enumerate(records) if r['condition'] == condition]
                keys = [(records[i]['instance'], records[i]['sample']) for i in selected]
                seeds = [records[i]['seed'] for i in selected]
                rev = summarize_revisions(gains['values'][selected], keys, seeds)
                tail = summarize_tail(errors[selected], keys, seeds)
                # Independently verify query quantiles, strict miss and weighted mean.
                for q, label in [(.5, 'p50_query_error'), (.95, 'p95_query_error'),
                                 (.99, 'p99_query_error')]:
                    if not math.isclose(tail[label], _quantile(ref_e[selected], q),
                                        rel_tol=1e-12, abs_tol=1e-12):
                        raise ValueError('tail scalar quantile mismatch')
                if (tail['miss_count'] != int((ref_e[selected] > 2).sum())
                        or not math.isclose(rev['mean_gain_m'], _macro(ref_g[selected], keys),
                                            rel_tol=1e-12, abs_tol=1e-12)):
                    raise ValueError('S4a miss/weighted gain mismatch')
                context = {'role': role, 'strategy': strategy, 'condition': condition,
                           'train_seed': int(strategy.split('_')[0][1:])
                                         if strategy.startswith('s20') else None,
                           'log_count': len({records[i]['log'] for i in selected})}
                groups.append({**context, 'lead_steps': m, 'revision': rev, 'tail': tail})
                if m == 2:
                    remaining_groups.append({**context, 'query_steps': list(range(1, 12)),
                                             'query_denominators': [len(selected)] * 11,
                                             'online_MR2_remaining': remaining[selected].mean(0).tolist()})
        shards += 1
    if shards != 32:
        raise ValueError('expected complete two-role sixteen-strategy S2 subset')
    for name, digest in marker['sha256'].items():
        if sha256_file(preflight / name) != digest:
            raise ValueError('S2 files changed during S4a')
    for ref in inputs.references.values():
        if sha256_file(Path(ref['path'])) != ref['sha256']:
            raise ValueError('GT/parent reference changed during S4a')
    if (sha256_file(marker_path) != marker_hash
            or any(sha256_file(Path(p)) != digest for p, digest in binding['sources'].items())):
        raise ValueError('S4a source or S2 snapshot changed')
    summary = {'identity': identity, 'prediction_shards': shards,
               'gain_values_checked': gain_count, 'error_values_checked': error_count,
               'remaining_flags_checked': flag_count, 'max_scalar_abs_difference_m': max_difference,
               'group_rows': len(groups), 'full_phase3a_evaluation': False,
               'neural_training': False, 'holdout_evaluated': False}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('run', {'identity': identity, 'binding': binding}), ('summary', summary),
                         ('groups', groups), ('remaining_groups', remaining_groups)]:
        _write(output / (name + '.json'), value)
    np.savez_compressed(output / 'values.npz', **arrays)
    names = ['run.json', 'summary.json', 'groups.json', 'remaining_groups.json', 'values.npz']
    _write(output / 'completed.json', {'identity': identity,
           'sha256': {n: sha256_file(output / n) for n in names}})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path,
                        default=Path('outputs/scenario-evaluation/preflight-v2'))
    parser.add_argument('--output-root', type=Path,
                        default=Path('outputs/scenario-evaluation/s4a-integration-v1'))
    args = parser.parse_args()
    print(json.dumps(check_revisions(args.preflight, args.output_root)), flush=True)

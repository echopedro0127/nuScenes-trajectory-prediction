"""S4b: common past-GT directional errors on the saved S2 subset."""
import argparse
import json
import math
import statistics
from pathlib import Path

import numpy as np

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.evaluation.directional_metrics import (
    directional_errors,
    summarize_directional,
)
from trajectory_prediction.evaluation.scenario_inputs import assemble_inputs
from trajectory_prediction.evaluation.scenario_preflight import verify_preflight
from trajectory_prediction.evaluation.streaming_metrics import fixed_lead_errors
from trajectory_prediction.utils.provenance import sha256_file


def _scalar(p, y, history, history_mask):
    longitudinal = np.full((len(y), 10), np.nan)
    lateral = np.full((len(y), 10), np.nan)
    valid = np.zeros((len(y), 10), bool)
    source = np.full((len(y), 10), -999, np.int64)
    for i in range(len(y)):
        for k in range(1, 11):
            # Build only the prefix visible at this query, not the whole future.
            points = list(history[i]) + list(y[i, :k])
            available = list(history_mask[i]) + [True] * k
            direction = None
            for end in range(1, len(points)):
                if not available[end - 1] or not available[end]:
                    continue
                dx = float(points[end][0]) - float(points[end - 1][0])
                dy = float(points[end][1]) - float(points[end - 1][1])
                length = math.hypot(dx, dy)
                if length / .5 >= .5:
                    direction = dx / length, dy / length
                    source[i, k - 1] = end - 4
            if direction is not None:
                valid[i, k - 1] = True
                column = round((k * .5 + 1.) / .5) - 1
                ex = float(p[i, k - 1, column, 0]) - float(y[i, column, 0])
                ey = float(p[i, k - 1, column, 1]) - float(y[i, column, 1])
                longitudinal[i, k - 1] = ex * direction[0] + ey * direction[1]
                lateral[i, k - 1] = -ex * direction[1] + ey * direction[0]
    return longitudinal, lateral, valid, source


def _macro(values, valid, keys):
    groups = {}
    for row, mask, key in zip(values, valid, keys, strict=True):
        if any(mask):
            groups.setdefault(key, []).append(statistics.fmean(float(v) for v in row[mask]))
    return statistics.fmean(statistics.fmean(v) for v in groups.values()) if groups else None


def check_directions(preflight, output):
    preflight, output = Path(preflight).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError('use a new S4b output directory')
    script = Path(__file__).resolve()
    root = script.parents[1]
    paths = [script, root / 'src/trajectory_prediction/evaluation/directional_metrics.py',
             root / 'src/trajectory_prediction/evaluation/streaming_metrics.py']
    sources = {str(p): sha256_file(p) for p in paths}
    marker_path = preflight / 'preflight_completed.json'
    marker_hash = sha256_file(marker_path)
    marker = json.loads(marker_path.read_text(encoding='utf-8'))
    if sha256_file(preflight / 'run.json') != marker['sha256']['run.json']:
        raise ValueError('S2 run snapshot mismatch')
    run = json.loads((preflight / 'run.json').read_text(encoding='utf-8'))
    old = run['binding']
    if run['identity'] != marker['identity'] or digest_json(old) != marker['identity']:
        raise ValueError('S2 run/marker identity mismatch')
    proof = verify_preflight(preflight)
    if any(sha256_file(Path(p)) != h for p, h in sources.items()):
        raise ValueError('S4b source changed during verification')
    if proof['identity'] != marker['identity'] or sha256_file(marker_path) != marker_hash:
        raise ValueError('S2 proof identity/snapshot changed')
    inputs = assemble_inputs(old['parent_path'], old['parent_project_root'])
    if inputs.parent_identity != old['parent_identity'] or inputs.references != old['references']:
        raise ValueError('verified GT/parent references changed')
    binding = {'schema': 'directional_s4b_subset_v1', 'preflight_identity': proof['identity'],
               'preflight_root': str(preflight), 'preflight_marker_sha256': marker_hash,
               'preflight_files': marker['sha256'], 'sources': sources,
               'lead_steps': 2, 'dt': .5, 'heading_min_speed_mps': .5,
               'normal': 'left=(-dy,dx)', 'error_sign': 'prediction_minus_GT',
               'scalar_atol': 1e-12, 'scalar_rtol': 1e-12,
               'reduction': 'valid_queries_then_valid_noise_repeats_then_valid_windows',
               'scope': 'S2_batch_subset_only_not_full_phase3a'}
    identity = digest_json(binding)
    arrays, common, groups = {}, {}, []
    shards, checked, valid_count, fresh_count, maximum = 0, 0, 0, 0, 0.
    for name in sorted(marker['sha256']):
        if not name.endswith('.npz'):
            continue
        role, strategy = name[:-4].split('__')
        with np.load(preflight / name, allow_pickle=False) as z:
            p, indices = z['forecast'], z['indices']
        gt = [inputs.gt_windows[role][int(i)] for i in indices]
        records = [inputs.records[role][int(i)] for i in indices]
        history = np.stack([g['history_xy'] for g in gt])
        history_mask = np.stack([g['history_mask'] for g in gt])
        y = np.stack([g['future_xy'] for g in gt])
        r = directional_errors(p, y, history, history_mask)
        sx, sy, valid, step = _scalar(p, y, history, history_mask)
        if (not np.array_equal(valid, r['valid_mask'])
                or not np.array_equal(step, r['heading_source_steps'])):
            raise ValueError('scalar heading validity/source mismatch')
        for actual, expected in [(r['longitudinal_signed_m'], sx), (r['lateral_signed_m'], sy)]:
            if not np.allclose(actual, expected, equal_nan=True, atol=1e-12, rtol=1e-12):
                raise ValueError('directional scalar projection mismatch')
            if valid.any():
                maximum = max(maximum, float(np.abs(actual[valid] - expected[valid]).max()))
        norm = np.hypot(r['longitudinal_signed_m'][valid], r['lateral_signed_m'][valid])
        if not np.allclose(norm, fixed_lead_errors(p, y, lead_steps=2)['values'][valid],
                           atol=1e-12, rtol=1e-12):
            raise ValueError('longitudinal/lateral norm does not preserve L2 error')
        frame_keys = ('reference_directions', 'valid_mask', 'fresh_heading_mask', 'heading_source_steps')
        if role not in common:
            common[role] = {k: r[k].copy() for k in frame_keys}
            common[role]['indices'] = indices.copy()
            for k, value in common[role].items():
                arrays[role + '__' + k] = value
        elif (not np.array_equal(indices, common[role]['indices'])
              or any(not np.array_equal(r[k], common[role][k]) for k in frame_keys)):
            raise ValueError('models must share exactly the same GT reference frame and mask')
        arrays[name[:-4] + '__longitudinal_signed_m'] = r['longitudinal_signed_m']
        arrays[name[:-4] + '__lateral_signed_m'] = r['lateral_signed_m']
        checked += valid.size
        valid_count += int(valid.sum())
        fresh_count += int((valid & r['fresh_heading_mask']).sum())
        for condition in sorted({row['condition'] for row in records}):
            selected = [i for i, row in enumerate(records) if row['condition'] == condition]
            keys = [(records[i]['instance'], records[i]['sample']) for i in selected]
            seeds = [records[i]['seed'] for i in selected]
            summary = summarize_directional(r['longitudinal_signed_m'][selected],
                                            r['lateral_signed_m'][selected], valid[selected], keys, seeds)
            for field, values in [('mean_longitudinal_abs_m', np.abs(sx)),
                                  ('mean_longitudinal_signed_m', sx), ('mean_lateral_abs_m', np.abs(sy))]:
                reference = _macro(values[selected], valid[selected], keys)
                actual = summary[field]
                if ((reference is None) != (actual is None)
                        or (actual is not None and not math.isclose(actual, reference,
                                                                    rel_tol=1e-12, abs_tol=1e-12))):
                    raise ValueError('directional scalar weighted summary mismatch')
            groups.append({'role': role, 'strategy': strategy, 'condition': condition,
                           'train_seed': int(strategy.split('_')[0][1:])
                                         if strategy.startswith('s20') else None,
                           'log_count': len({records[i]['log'] for i in selected}),
                           'directional': summary})
        shards += 1
    if shards != 32:
        raise ValueError('complete two-role sixteen-strategy S2 subset required')
    for name, digest in marker['sha256'].items():
        if sha256_file(preflight / name) != digest:
            raise ValueError('S2 file changed during S4b')
    for ref in inputs.references.values():
        if sha256_file(Path(ref['path'])) != ref['sha256']:
            raise ValueError('GT/parent source changed during S4b')
    if (sha256_file(marker_path) != marker_hash
            or any(sha256_file(Path(p)) != h for p, h in sources.items())):
        raise ValueError('source or S2 completion snapshot changed')
    report = {'identity': identity, 'prediction_shards': shards,
              'query_pairs_checked': checked, 'valid_query_pairs': valid_count,
              'invalid_query_pairs': checked - valid_count, 'fresh_heading_pairs': fresh_count,
              'carried_heading_pairs': valid_count - fresh_count,
              'max_scalar_abs_difference_m': maximum, 'group_rows': len(groups),
              'common_gt_masks_verified': True, 'full_phase3a_evaluation': False,
              'neural_training': False, 'holdout_evaluated': False}
    output.mkdir(parents=True, exist_ok=False)
    for name, value in [('run', {'identity': identity, 'binding': binding}),
                         ('summary', report), ('groups', groups)]:
        (output / (name + '.json')).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n',
                                               encoding='utf-8')
    np.savez_compressed(output / 'projections.npz', **arrays)
    names = ['run.json', 'summary.json', 'groups.json', 'projections.npz']
    (output / 'completed.json').write_text(json.dumps({'identity': identity,
        'sha256': {n: sha256_file(output / n) for n in names}}, indent=2) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path,
                        default=Path('outputs/scenario-evaluation/preflight-v2'))
    parser.add_argument('--output-root', type=Path,
                        default=Path('outputs/scenario-evaluation/s4b-integration-v1'))
    args = parser.parse_args()
    print(json.dumps(check_directions(args.preflight, args.output_root)), flush=True)

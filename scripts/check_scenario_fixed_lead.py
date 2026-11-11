"""Validate S3 on saved S2 predictions; not a full Phase 3A benchmark."""
import argparse
import json
import math
from pathlib import Path

import numpy as np

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.evaluation.scenario_inputs import assemble_inputs
from trajectory_prediction.evaluation.scenario_preflight import verify_preflight
from trajectory_prediction.evaluation.streaming_metrics import fixed_lead_errors
from trajectory_prediction.utils.provenance import sha256_file


def _scalar(p, y, m):
    """Time-based scalar implementation independent of vectorized indexing."""
    output = []
    for sample in range(len(y)):
        values = []
        for step in range(1, 12):
            target_time = .5 * step + .5 * m
            if target_time <= 6:
                column = round(target_time / .5) - 1
                dx = float(p[sample, step - 1, column, 0]) - float(y[sample, column, 0])
                dy = float(p[sample, step - 1, column, 1]) - float(y[sample, column, 1])
                values.append(math.hypot(dx, dy))
        output.append(values)
    return np.array(output, np.float64)


def check_fixed_leads(preflight, output):
    output, preflight = Path(output).resolve(), Path(preflight).resolve()
    if output.exists():
        raise FileExistsError('use a new S3 integration output directory')
    marker_path = preflight / 'preflight_completed.json'
    marker_hash = sha256_file(marker_path)
    marker = json.loads(marker_path.read_text(encoding='utf-8'))
    run_path = preflight / 'run.json'
    if sha256_file(run_path) != marker['sha256']['run.json']:
        raise ValueError('preflight run snapshot checksum mismatch')
    run = json.loads(run_path.read_text(encoding='utf-8'))
    old = run['binding']
    if run['identity'] != marker['identity'] or digest_json(old) != marker['identity']:
        raise ValueError('preflight run/marker identity mismatch')
    proof = verify_preflight(preflight)
    if proof['identity'] != marker['identity'] or sha256_file(marker_path) != marker_hash:
        raise ValueError('verified preflight identity/snapshot changed')
    inputs = assemble_inputs(old['parent_path'], old['parent_project_root'])
    source = Path(__file__).resolve()
    metric_source = source.parents[1] / 'src/trajectory_prediction/evaluation/streaming_metrics.py'
    if (inputs.parent_identity != old['parent_identity']
            or inputs.references != old['references']):
        raise ValueError('GT/parent identity or verified references changed')
    binding = {
        'schema': 'fixed_lead_s3_integration_v1', 'preflight_identity': proof['identity'],
        'preflight_root': str(preflight), 'lead_steps': [1, 2, 4], 'dt': .5,
        'metric_dtype': 'float64', 'scalar_atol': 1e-12, 'scalar_rtol': 1e-12,
        'sources': {str(p): sha256_file(p) for p in (source, metric_source)},
        'preflight_files': marker['sha256'],
        'preflight_marker_sha256': marker_hash,
        'scope': 'S2_batch_subset_integration_only',
    }
    identity = digest_json(binding)
    arrays, grids = {}, {}
    max_difference, checked, shards = 0., 0, 0
    for name in sorted(marker['sha256']):
        if not name.endswith('.npz'):
            continue
        role = name.split('__', 1)[0]
        with np.load(preflight / name, allow_pickle=False) as z:
            p, indices = z['forecast'], z['indices']
        y = inputs.datasets[role]['target'][indices]
        for m in binding['lead_steps']:
            r = fixed_lead_errors(p, y, lead_steps=m)
            reference = _scalar(p, y, m)
            if not np.allclose(r['values'], reference, atol=1e-12, rtol=1e-12):
                raise ValueError('independent scalar fixed-lead mismatch')
            max_difference = max(max_difference, float(np.abs(r['values'] - reference).max()))
            checked += r['values'].size
            arrays[name[:-4] + '__lead' + str(m)] = r['values']
            grids[str(m)] = {k: v.tolist() for k, v in r.items() if k not in ('values', 'valid_mask')}
        shards += 1
    if shards != 32:
        raise ValueError('complete two-role sixteen-strategy preflight required')
    # Reject inputs or source changes during the audit before publishing completion.
    for name, digest in marker['sha256'].items():
        if sha256_file(preflight / name) != digest:
            raise ValueError('preflight changed during fixed-lead check')
    for ref in inputs.references.values():
        if sha256_file(Path(ref['path'])) != ref['sha256']:
            raise ValueError('GT/source reference changed during fixed-lead check')
    if any(sha256_file(Path(p)) != digest for p, digest in binding['sources'].items()):
        raise ValueError('S3 source changed during fixed-lead check')
    if sha256_file(marker_path) != marker_hash:
        raise ValueError('preflight completion snapshot changed during fixed-lead check')
    report = {'identity': identity, 'prediction_shards': shards,
              'lead_queries_per_stream': {'1': 11, '2': 10, '4': 8},
              'checked_error_values': checked, 'max_scalar_abs_difference_m': max_difference,
              'all_scalar_comparisons_within_tolerance': True,
              'full_phase3a_evaluation': False, 'neural_training': False,
              'holdout_evaluated': False}
    output.mkdir(parents=True, exist_ok=False)
    for name, data in [('run', {'identity': identity, 'binding': binding}),
                       ('summary', report), ('time_grid', grids)]:
        (output / (name + '.json')).write_text(
            json.dumps(data, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    np.savez_compressed(output / 'fixed_lead_errors.npz', **arrays)
    names = ['run.json', 'summary.json', 'time_grid.json', 'fixed_lead_errors.npz']
    (output / 'completed.json').write_text(json.dumps({
        'identity': identity, 'sha256': {n: sha256_file(output / n) for n in names}},
        indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path,
                        default=Path('outputs/scenario-evaluation/preflight-v2'))
    parser.add_argument('--output-root', type=Path,
                        default=Path('outputs/scenario-evaluation/s3-integration-v1'))
    args = parser.parse_args()
    print(json.dumps(check_fixed_leads(args.preflight, args.output_root)), flush=True)

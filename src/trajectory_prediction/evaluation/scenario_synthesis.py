"""S6c independent training-seed summaries and shared deterministic cases."""
import argparse
import hashlib
import html
import io
import json
import math
import platform
import statistics
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.evaluation.scenario_event_report import _recheck as recheck_events
from trajectory_prediction.evaluation.scenario_event_report import _sources as event_sources
from trajectory_prediction.evaluation.scenario_event_report import verify_event_report
from trajectory_prediction.evaluation.scenario_preflight import PROJECT
from trajectory_prediction.evaluation.scenario_publication import _artifact_hashes, _read, _write
from trajectory_prediction.evaluation.scenario_report import _compare_npz_nan
from trajectory_prediction.utils.provenance import sha256_file

DISPLAY = ('raw_cv', 'kf_nis', 's2027_baseline7', 's2027_plus_ca', 's2027_guarded_ca')
QUERY_STEPS = (3, 4, 6)
GROUP_FIELDS = ('role', 'condition', 'log', 'group_namespace', 'tag')


def _sources():
    return {**event_sources(), **{n: sha256_file(PROJECT / n) for n in (
        'src/trajectory_prediction/evaluation/scenario_synthesis.py',
        'scripts/build_scenario_synthesis.py')}}


def _metric_values(row):
    """Keep metric nulls; denominators remain in the referenced source groups."""
    output = {}
    def walk(value, prefix):
        if isinstance(value, dict):
            for k, v in sorted(value.items()):
                if k not in ('status', 'reason'):
                    walk(v, prefix + '.' + k if prefix else k)
        elif isinstance(value, list):
            for i, v in enumerate(value):
                walk(v, prefix + '.' + str(i))
        elif value is None or type(value) is float:
            if value is not None and not math.isfinite(value):
                raise ValueError('nonfinite seed metric')
            output[prefix] = value
    if 'eligible' in row:
        selected = {k: v for k, v in row.items() if k.startswith(('mean_', 'log_macro_')) or
                    k.endswith('_fraction_of_affected') or k == 'unaffected_fraction_of_eligible'}
    else:
        selected = {k: row[k] for k in ('m1_m5', 'm2', 'm4', 'remaining_miss', 'log_macro_mean_error_m') if k in row}
    walk(selected, '')
    return output


def aggregate_seeds(rows, registry):
    definitions = {r['strategy']: r for r in registry}
    if len(definitions) != len(registry):
        raise ValueError('duplicate strategy registry')
    families, grouped = {}, {}
    for name, definition in definitions.items():
        seed = definition['train_seed']
        if seed is not None and (type(seed) is not int or seed not in (2027, 2028, 2029)):
            raise ValueError('invalid training seed')
        family = definition.get('variant') if seed is not None else name
        families.setdefault(family, set()).add(name)
    for family, names in families.items():
        seeds = [definitions[n]['train_seed'] for n in names]
        if seeds != [None] and (len(seeds) != 3 or set(seeds) != {2027, 2028, 2029}):
            raise ValueError('complete three-seed model family required')
    for i, row in enumerate(rows):
        name = row['strategy']
        if name not in definitions or row['train_seed'] != definitions[name]['train_seed']:
            raise ValueError('strategy/training seed identity mismatch')
        family = definitions[name].get('variant') if row['train_seed'] is not None else name
        key = (family, *(row.get(k) for k in GROUP_FIELDS))
        members = grouped.setdefault(key, {})
        if name in members:
            raise ValueError('duplicate model/group')
        members[name] = (i, row)
    result = []
    for key in sorted(grouped, key=lambda k: json.dumps(k)):
        family, *values = key
        members = grouped[key]
        if set(members) != families[family]:
            raise ValueError('missing training model/group; complete family required')
        names = sorted(members, key=lambda n: (-1 if definitions[n]['train_seed'] is None else definitions[n]['train_seed']))
        metrics = [_metric_values(members[n][1]) for n in names]
        if any(set(m) != set(metrics[0]) for m in metrics):
            raise ValueError('inconsistent model metric schema')
        stats = {}
        for metric in metrics[0]:
            per_model = [m[metric] for m in metrics]
            valid = [v for v in per_model if v is not None]
            complete = len(valid) == len(per_model)
            mean = float(np.mean(valid)) if complete else None
            deviation = float(np.std(valid, ddof=1)) if complete and len(valid) > 1 else None
            if mean is not None and (not math.isfinite(mean) or not math.isclose(
                    mean, statistics.mean(valid), rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError('scalar seed mean disagreement')
            if deviation is not None and (not math.isfinite(deviation) or not math.isclose(
                    deviation, statistics.stdev(valid), rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError('scalar sample deviation disagreement')
            stats[metric] = {'values_by_model': per_model, 'valid_model_count': len(valid),
                             'mean': mean, 'sample_std_ddof1': deviation,
                             'status': 'ok' if complete else 'not_evaluable',
                             'reason': None if complete else 'metric_unavailable_for_some_models',
                             'std_reason': 'single_physical_strategy' if len(names) == 1 else
                                           None if complete else 'incomplete_evaluable_models'}
        result.append({'family': family, **dict(zip(GROUP_FIELDS, values, strict=True)),
                       'members': [{'strategy': n, 'train_seed': definitions[n]['train_seed'],
                                    'source_group_index': members[n][0]} for n in names],
                       'status': 'ok' if stats else 'not_evaluable',
                       'reason': None if stats else 'empty_gt_window_group', 'metrics': stats})
    return result


def _case_id(role, condition, instance, sample):
    return hashlib.sha256(f's6c-case-v1:{role}:{condition}:{instance}:{sample}'.encode()).hexdigest()


def select_cases(windows, errors):
    selected = {}
    def add(role, condition, instance, sample, reason):
        identity = _case_id(role, condition, instance, sample)
        row = selected.setdefault(identity, {'case_id': identity, 'role': role, 'condition': condition,
                                             'instance': instance, 'sample': sample, 'noise_seed': 2027,
                                             'selection_reasons': []})
        row['selection_reasons'].append(reason)
    ranked = {}
    seen = set()
    for row in errors:
        if not math.isfinite(row['mean_error_m']) or row['mean_error_m'] < 0:
            raise ValueError('finite nonnegative case ranking error required')
        key = row['role'], row['condition'], row['strategy']
        identity = (*key, row['instance'], row['sample'])
        if identity in seen:
            raise ValueError('duplicate case ranking window')
        seen.add(identity)
        ranked.setdefault(key, []).append(row)
    for key in sorted(ranked):
        row = min(ranked[key], key=lambda r: (-r['mean_error_m'], _case_id(
            r['role'], r['condition'], r['instance'], r['sample'])))
        add(row['role'], row['condition'], row['instance'], row['sample'], 'top_error:' + row['strategy'])
    labels = {}
    for row in windows:
        for tag in row['labels']:
            labels.setdefault((row['role'], tag), []).append(row)
    for (role, tag), candidates in sorted(labels.items()):
        row = min(candidates, key=lambda r: hashlib.sha256(
            ':'.join(('s6c-label-v1', role, tag, r['instance'], r['sample'])).encode()).hexdigest())
        add(role, 'noise_outlier', row['instance'], row['sample'], 'rule_candidate:' + tag)
    for row in selected.values():
        row['selection_reasons'] = sorted(set(row['selection_reasons']))
    return [selected[k] for k in sorted(selected)]


def _prepare(event_root):
    root = Path(event_root).resolve()
    sources, marker = _sources(), sha256_file(root / 'completed.json')
    verify_event_report(root)
    binding = {'schema': 'scenario_synthesis_s6c_v1', 'event_root': str(root),
               'event_marker_sha256': marker, 'event_identity': _read(root / 'run.json')['identity'],
               'sources': sources, 'display_strategies': list(DISPLAY), 'query_steps': list(QUERY_STEPS),
               'case_policy': 'top1_each_role_condition_strategy_noise_mean_union_plus_hash1_each_role_rule_tag',
               'display_noise_seed': 2027, 'runtime': {'numpy': str(np.__version__),
               'matplotlib': matplotlib.__version__, 'platform': platform.platform()},
               'formal_full_run': False, 'neural_training': False, 'holdout_evaluated': False}
    _recheck(binding)
    return binding


def _recheck(binding):
    root = Path(binding['event_root'])
    if _sources() != binding['sources'] or sha256_file(root / 'completed.json') != binding['event_marker_sha256']:
        raise ValueError('synthesis source/event changed')
    _artifact_hashes(root, _read(root / 'completed.json')['sha256'])
    recheck_events(_read(root / 'run.json')['binding'])


def _documents(binding):
    events = Path(binding['event_root'])
    event_binding = _read(events / 'run.json')['binding']
    publication, query = Path(event_binding['publication']), Path(event_binding['query_root'])
    registry = _read(publication / 'run.json')['binding']
    error_rows = []
    for role in registry['protocol']['roles']:
        keys = _read(publication / f'{role}-keys.json')
        for strategy in registry['strategies']:
            with np.load(query / 'queries' / f"{role}__{strategy['strategy']}.npz", allow_pickle=False) as z:
                errors = z['lead2_error_m']
            groups = {}
            for i, key in enumerate(keys):
                groups.setdefault((key['condition'], key['instance'], key['sample']), []).append(i)
            for (condition, instance, sample), indices in sorted(groups.items()):
                if {keys[i]['noise_seed'] for i in indices} != {2027, 2028, 2029} or len(indices) != 3:
                    raise ValueError('complete case noise cohort required')
                value = math.fsum(math.fsum(float(v) for v in errors[i]) / 10 for i in indices) / 3
                error_rows.append({'role': role, 'condition': condition, 'strategy': strategy['strategy'],
                                   'instance': instance, 'sample': sample, 'mean_error_m': value})
    cases = select_cases(_read(events / 'windows.json'), error_rows)
    summary = {'case_count': len(cases), 'seed_model_count': 9, 'physical_strategy_count': 7,
               'formal_full_run': False, 'neural_training': False, 'holdout_evaluated': False,
               'manual_completed': 0, 's6_development_complete': True,
               'scope': 'development_fixture_acceptance_not_phase3a_formal_results'}
    return {'query_seed_stats.json': aggregate_seeds(_read(events / 'maneuver_query_groups.json'), registry['strategies']),
            'event_seed_stats.json': aggregate_seeds(_read(events / 'event_groups.json'), registry['strategies']),
            'case_ranking.json': error_rows, 'case_manifest.json': cases, 'summary.json': summary,
            'run.json': {'identity': digest_json(binding), 'binding': binding}}


def _case_arrays(binding, case):
    events = Path(binding['event_root'])
    publication = Path(_read(events / 'run.json')['binding']['publication'])
    role = case['role']
    keys = _read(publication / f'{role}-keys.json')
    indices = [i for i, k in enumerate(keys) if all(k[f] == case[f] for f in (
        'instance', 'sample', 'condition', 'noise_seed'))]
    if len(indices) != 1:
        raise ValueError('case source stream missing/duplicate')
    index = indices[0]
    with np.load(publication / 'gt' / f'{role}.npz', allow_pickle=False) as z:
        gt_index = keys[index]['gt_index']
        arrays = {'history_xy': z['history_xy'][gt_index], 'future_xy': z['future_xy'][gt_index]}
    forecasts = []
    for name in DISPLAY:
        with np.load(publication / 'predictions' / f'{role}__{name}.npz', allow_pickle=False) as z:
            forecasts.append(z['forecast'][index])
    forecasts = np.asarray(forecasts)
    # Zero lead of the saved raw-CV forecast is the arrived position in this protocol.
    observed = forecasts[0, np.arange(11), np.arange(11)]
    arrays.update(forecasts=forecasts, arrived_raw_cv_anchors=observed,
                  arrival_steps=np.arange(1, 12), arrival_times_s=np.arange(1, 12) * .5,
                  injected_steps=np.array([4, 8]) if case['condition'] == 'noise_outlier' else np.array([], np.int64))
    return arrays


def _plot(case, arrays):
    fig = Figure(figsize=(16, 10), dpi=95)
    FigureCanvasAgg(fig)
    axes = fig.subplots(3, 5, sharex=True, sharey=True)
    gt = np.vstack((arrays['history_xy'], arrays['future_xy']))
    points = [gt, arrays['arrived_raw_cv_anchors']]
    points += [arrays['forecasts'][i, k - 1, k:] for i in range(5) for k in QUERY_STEPS]
    joined = np.vstack(points)
    lower, upper = joined.min(axis=0), joined.max(axis=0)
    center = (lower + upper) / 2
    radius = max(float((upper - lower).max()) * .55, 2.)
    for row, k in enumerate(QUERY_STEPS):
        for column, name in enumerate(DISPLAY):
            ax = axes[row, column]
            ax.plot(gt[:, 0], gt[:, 1], color='0.65', label='Offline GT')
            observed = arrays['arrived_raw_cv_anchors'][:k]
            ax.scatter(observed[:, 0], observed[:, 1], c=np.arange(1, k + 1) * .5,
                       vmin=.5, vmax=5.5, cmap='viridis', s=13, label='Arrived raw-CV anchors')
            for step in arrays['injected_steps']:
                if step <= k:
                    point = observed[step - 1]
                    ax.scatter(*point, color='red', marker='x', s=60)
                    ax.annotate(f'injected @{step * .5:g}s', point, fontsize=6)
            p = arrays['forecasts'][column, k - 1]
            ax.plot(p[k:, 0], p[k:, 1], color='tab:blue', label='Remaining prediction')
            target_column = k + 1
            target = arrays['future_xy'][target_column]
            ax.scatter(*target, color='black', marker='*', s=45, label='GT target (+1s)')
            ax.scatter(*p[target_column], facecolors='none', edgecolors='tab:blue', s=40)
            ax.set_title(f'{name}\nk={k}, arrival={k * .5:g}s, target={(k + 2) * .5:g}s', fontsize=8)
            ax.set_xlim(center[0] - radius, center[0] + radius)
            ax.set_ylim(center[1] - radius, center[1] + radius)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(alpha=.2)
            if row == 2:
                ax.set_xlabel('anchor x (m)')
            if column == 0:
                ax.set_ylabel('anchor y (m)')
    axes[0, 0].legend(fontsize=5, loc='best')
    fig.suptitle(f"{case['role']} / {case['condition']} / noise=2027 / {case['instance'][:12]}\n"
                 'Shared cases; displayed neural training seed=2027; GT is evaluation-only; anchors from saved raw CV', fontsize=10)
    fig.tight_layout(rect=(0, .06, 1, .94))
    color_axis = fig.add_axes((.38, .05, .25, .012))
    colorbar = fig.colorbar(matplotlib.cm.ScalarMappable(
        norm=matplotlib.colors.Normalize(.5, 5.5), cmap='viridis'), cax=color_axis, orientation='horizontal')
    colorbar.set_label('Observation arrival time since anchor (s)', fontsize=7)
    colorbar.ax.tick_params(labelsize=6)
    stream = io.BytesIO()
    fig.savefig(stream, format='png', metadata={'Software': 'trajectory-s6c'})
    fig.clear()
    return stream.getvalue()


def _gallery(cases):
    rows = ['<!doctype html><meta charset="utf-8"><title>S6c shared cases</title>',
            '<h1>Development shared cases; manual review pending</h1>',
            ('<p>16 strategies select cases; five fixed strategies shown. Neural display seed=2027. '
            'Colors indicate arrival time (0.5–5.5s); red X marks arrived injected outliers. '
             'All panels of each case share axes. Not formal Phase 3A results.</p>')]
    for case in cases:
        rows.append(f"<h2>{html.escape(case['role'] + ' / ' + case['condition'])}</h2>"
                    f"<p>{html.escape(', '.join(case['selection_reasons']))}</p>"
                    f"<img style='width:100%' src='cases/{case['case_id']}.png'>")
    return '\n'.join(rows)


def build_synthesis(event_root, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('use a new synthesis directory')
    binding = _prepare(event_root)
    documents = _documents(binding)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'cases').mkdir()
    hashes = {}
    for name, value in documents.items():
        _write(output / name, value)
        hashes[name] = sha256_file(output / name)
    for case in documents['case_manifest.json']:
        arrays = _case_arrays(binding, case)
        name = 'cases/' + case['case_id']
        np.savez_compressed(output / (name + '.npz'), **arrays)
        _compare_npz_nan(output / (name + '.npz'), arrays)
        (output / (name + '.png')).write_bytes(_plot(case, arrays))
        for suffix in ('.npz', '.png'):
            hashes[name + suffix] = sha256_file(output / (name + suffix))
    (output / 'gallery.html').write_text(_gallery(documents['case_manifest.json']), encoding='utf-8')
    hashes['gallery.html'] = sha256_file(output / 'gallery.html')
    _recheck(binding)
    actual = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    if actual != set(hashes) or any(sha256_file(output / n) != h for n, h in hashes.items()):
        raise ValueError('synthesis file coverage/contents changed')
    _write(output / 'completed.json', {'identity': digest_json(binding), 'sha256': hashes})
    return documents['summary.json']


def verify_synthesis(output):
    output = Path(output).resolve()
    marker_path = output / 'completed.json'
    if not marker_path.is_file():
        raise ValueError('incomplete synthesis')
    marker_hash, marker = sha256_file(marker_path), _read(marker_path)
    _artifact_hashes(output, marker['sha256'])
    binding = _prepare(_read(output / 'run.json')['binding']['event_root'])
    if marker['identity'] != digest_json(binding):
        raise ValueError('synthesis identity mismatch')
    documents = _documents(binding)
    names = {'gallery.html', *documents}
    for name, value in documents.items():
        if _read(output / name) != value:
            raise ValueError('synthesis semantic replay mismatch: ' + name)
    for case in documents['case_manifest.json']:
        arrays = _case_arrays(binding, case)
        name = 'cases/' + case['case_id']
        _compare_npz_nan(output / (name + '.npz'), arrays)
        if (output / (name + '.png')).read_bytes() != _plot(case, arrays):
            raise ValueError('case rendering semantic replay mismatch')
        names.update((name + '.npz', name + '.png'))
    if (output / 'gallery.html').read_text(encoding='utf-8') != _gallery(documents['case_manifest.json']):
        raise ValueError('gallery semantic replay mismatch')
    if names != set(marker['sha256']):
        raise ValueError('synthesis coverage mismatch')
    _recheck(binding)
    _artifact_hashes(output, marker['sha256'])
    if sha256_file(marker_path) != marker_hash:
        raise ValueError('synthesis marker changed')
    return {**documents['summary.json'], 'semantic_replay_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--event-report', type=Path)
    parser.add_argument('--output-root', required=True, type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only and args.event_report is None:
        parser.error('--event-report required for build')
    result = verify_synthesis(args.output_root) if args.verify_only else build_synthesis(args.event_report, args.output_root)
    print(json.dumps(result, indent=2, allow_nan=False))

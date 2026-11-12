"""Offline window labels and pending review gallery; no label enters a predictor."""
import argparse
import csv
import hashlib
import html
import io
import json
from pathlib import Path

import matplotlib
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
from trajectory_prediction.evaluation.scenario_inputs import assemble_inputs
from trajectory_prediction.evaluation.scenario_preflight import verify_preflight
from trajectory_prediction.evaluation.scenario_replay import validate_protocol
from trajectory_prediction.utils.provenance import sha256_file


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def _check_bound_inputs(preflight, marker, marker_hash, references, sources):
    for name, digest in marker['sha256'].items():
        if sha256_file(preflight / name) != digest:
            raise ValueError('S2 source changed during labels')
    for ref in references.values():
        if sha256_file(Path(ref['path'])) != ref['sha256']:
            raise ValueError('GT/parent reference changed during labels')
    if (sha256_file(preflight / 'preflight_completed.json') != marker_hash
            or any(sha256_file(Path(p)) != h for p, h in sources.items())):
        raise ValueError('S4d source/S2 completion snapshot changed')


def plot_gt_window(window, path):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(12, 3.7))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 3)
    trace = window['trace']
    xy = np.asarray(trace['positions_xy'], float)
    xy[~np.asarray(trace['point_valid'], bool)] = np.nan
    axes[0].plot(xy[:5, 0], xy[:5, 1], 'o--', color='gray', markersize=3, label='GT history')
    axes[0].plot(xy[4:, 0], xy[4:, 1], 'o-', color='tab:blue', markersize=3, label='GT future')
    axes[0].plot(*xy[4], 'ro', markersize=4, label='anchor')
    axes[0].set(xlabel='x / m', ylabel='y / m', title='GT trajectory (equal scale)')
    axes[0].set_aspect('equal', adjustable='datalim')
    times = trace['interval_start_s']
    axes[1].plot(times, np.asarray(trace['raw_speed_mps'], float), color='gray', label='raw speed')
    axes[1].plot(times, np.asarray(trace['smoothed_speed_mps'], float), color='tab:blue', label='median speed')
    axes[1].axhline(.3, color='orange', linestyle=':', label='stop 0.3 m/s')
    axes[1].axhline(1., color='green', linestyle=':', label='restart 1 m/s')
    axes[1].set(xlabel='relative time / s', ylabel='speed / m/s', title='GT speed intervals', xlim=(-2, 6))
    axes[2].plot(times, np.asarray(trace['acceleration_mps2'], float), color='tab:blue', label='median-speed acceleration')
    axes[2].axhline(-2., color='red', linestyle=':', label='braking -2 m/s^2')
    axes[2].set(xlabel='relative time / s', ylabel='acceleration / m/s^2', title='GT acceleration intervals', xlim=(-2, 6))
    for event in window['events']:
        if event['status'] == 'candidate':
            for ax in axes[1:]:
                ax.axvspan(event['onset_s'], event['end_s'], color='orange', alpha=.06)
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=6, loc='best')
    for ax in axes[1:]:
        ax.axvline(0, color='black', linewidth=.7)
    figure.suptitle('GT offline rule candidates - manual review pending\n'
                   + window['window_id'][:12] + ' | ' + ', '.join(window['labels']), fontsize=9)
    figure.tight_layout(rect=(0, 0, 1, .87))
    figure.savefig(path, dpi=120, metadata={'Software': 'trajectory-prediction'})


def _derive(inputs, thresholds):
    unique = {}
    for role in sorted(inputs.gt_windows):
        for gt in inputs.gt_windows[role]:
            key = gt['window_key']
            if key in unique:
                continue
            role, instance, sample = key
            annotation = annotate_window(gt['history_xy'], gt['future_xy'], gt['history_mask'])
            legacy = describe_motion(gt['future_xy'], .5)
            row = {**annotation, 'role': role, 'instance': instance, 'sample': sample,
                   'log': gt['log'], 'scene': gt['scene'], 'source_row': gt['source_row'],
                   'source_manifest': gt['source_manifest'], 'source_archive': gt['source_archive'],
                   'window_id': hashlib.sha256(':'.join(key).encode()).hexdigest(),
                   'legacy_tags': tag_maneuvers(legacy, thresholds), 'legacy_motion': legacy}
            row['evidence_sha256'] = digest_json(row)
            unique[key] = row
    windows = [unique[key] for key in sorted(unique)]
    manifest = build_review_manifest(windows)
    lookup = {w['window_id']: w for w in windows}
    for row in manifest:
        row['evidence_sha256'] = lookup[row['window_id']]['evidence_sha256']
        row['image_path'] = 'figures/' + row['window_id'] + '.png'
    events = [{**{k: w[k] for k in ('window_id', 'role', 'instance', 'sample', 'scene', 'log')}, **e}
              for w in windows for e in w['events']]
    coverage = []
    for role in sorted(inputs.gt_windows):
        selected = [w for w in windows if w['role'] == role]
        for tag in TAGS:
            positive = [w for w in selected if tag in w['labels']]
            coverage.append({'role': role, 'tag': tag, 'window_count': len(selected),
                             'candidate_windows': len(positive),
                             'absent_windows': sum(w['tag_status'][tag]['status'] == 'absent' for w in selected),
                             'not_evaluable_windows': sum(w['tag_status'][tag]['status'] == 'not_evaluable' for w in selected),
                             'candidate_instances': len({w['instance'] for w in positive}),
                             'candidate_logs': len({w['log'] for w in positive})})
    return windows, manifest, events, coverage


def _csv_text(manifest):
    fields = ['window_id', 'role', 'instance', 'sample', 'log', 'rule_version', 'rule_label',
              'review_status', 'reviewer', 'review_reason', 'manual_label', 'review_history',
              'selection_hash', 'evidence_sha256', 'image_path']
    stream = io.StringIO(newline='')
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in manifest:
        writer.writerow({**row, 'review_history': json.dumps(row['review_history'], ensure_ascii=False)})
    return stream.getvalue()


def _gallery(windows, manifest):
    lookup = {w['window_id']: w for w in windows}
    parts = ['<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>GT窗口事件抽查</title>',
             ('<style>body{font:16px sans-serif;max-width:1450px;margin:30px auto;padding:0 20px}'
              'article{border-top:1px solid #ccc;padding:20px 0}img{max-width:100%}pre{white-space:pre-wrap}</style>'),
             ('<h1>GT离线规则候选：待人工复核</h1><p>规则标签不等于人工真值。所有案例按固定哈希抽样，'
              '未按模型排名选择。人工完成数：0。时间均相对窗口anchor；不计独立episode。</p>'),
             ('<p>图示GT轨迹、原始/平滑速度和加速度；橙色区间为规则候选。请将人工修改另存版本，'
              '保留原rule_label、规则版本、复核人及原因，不覆盖冻结证据。</p>')]
    if not manifest:
        parts.append('<p>当前窗口没有规则阳性候选，抽查清单为空。</p>')
    for row in manifest:
        window = lookup[row['window_id']]
        evidence = [e for e in window['events'] if e['tag'] == row['rule_label']]
        parts.extend(['<article><h2>' + html.escape(row['role'] + ' / ' + row['rule_label']) + '</h2>',
                      '<p>' + html.escape(row['log'] + ' | ' + row['window_id']) + ' | pending</p>',
                      '<p>GT离线标签（待人工复核）</p><img alt="GT offline candidate" src="'
                      + html.escape(row['image_path'], quote=True) + '">',
                      '<details><summary>规则时间与依据</summary><pre>'
                      + html.escape(json.dumps(evidence, indent=2, ensure_ascii=False)) + '</pre></details></article>'])
    return '\n'.join(parts) + '</html>\n'


def check_labels(preflight, output, *, verify_only=False):
    output = Path(output).resolve()
    if output.exists() and not verify_only:
        raise FileExistsError('use a new S4d output directory')
    saved = None
    if verify_only:
        saved_marker_hash = sha256_file(output / 'completed.json')
        saved = json.loads((output / 'completed.json').read_text(encoding='utf-8'))
        for name, digest in saved['sha256'].items():
            path = (output / name).resolve()
            if not path.is_relative_to(output) or sha256_file(path) != digest:
                raise ValueError('label output checksum mismatch')
        old_run = json.loads((output / 'run.json').read_text(encoding='utf-8'))
        if old_run['identity'] != saved['identity'] or digest_json(old_run['binding']) != saved['identity']:
            raise ValueError('label run identity mismatch')
        preflight = old_run['binding']['preflight_root']
    preflight = Path(preflight).resolve()
    script = Path(__file__).resolve()
    root = script.parents[1]
    paths = [script, root / 'src/trajectory_prediction/analysis/scenario_events.py',
             root / 'src/trajectory_prediction/analysis/maneuvers.py',
             root / 'configs/analysis/scenario_evaluation.yaml', root / 'configs/analysis/maneuvers.yaml']
    sources = {str(p): sha256_file(p) for p in paths}
    config = validate_protocol(load_yaml(root / 'configs/analysis/scenario_evaluation.yaml'))
    if config['event_labels'] != EVENT_RULES:
        raise ValueError('event rule implementation/config mismatch')
    thresholds = load_yaml(root / 'configs/analysis/maneuvers.yaml')
    marker_path = preflight / 'preflight_completed.json'
    marker_hash = sha256_file(marker_path)
    marker = json.loads(marker_path.read_text(encoding='utf-8'))
    if sha256_file(preflight / 'run.json') != marker['sha256']['run.json']:
        raise ValueError('S2 run checksum mismatch')
    parent = json.loads((preflight / 'run.json').read_text(encoding='utf-8'))
    old = parent['binding']
    if parent['identity'] != marker['identity'] or digest_json(old) != marker['identity']:
        raise ValueError('S2 identity mismatch')
    proof = verify_preflight(preflight)
    if any(sha256_file(Path(p)) != h for p, h in sources.items()):
        raise ValueError('S4d source changed during verification')
    if proof['identity'] != marker['identity'] or sha256_file(marker_path) != marker_hash:
        raise ValueError('S2 snapshot/identity changed')
    inputs = assemble_inputs(old['parent_path'], old['parent_project_root'])
    if inputs.parent_identity != old['parent_identity'] or inputs.references != old['references']:
        raise ValueError('verified GT/parent source changed')
    if inputs.binding.get('thresholds', thresholds) != thresholds:
        raise ValueError('legacy maneuver thresholds differ from parent')
    binding = {'schema': 'scenario_s4d_window_labels_v1', 'preflight_root': str(preflight),
               'preflight_identity': proof['identity'], 'preflight_marker_sha256': marker_hash,
               'preflight_files': marker['sha256'], 'sources': sources, 'rules': EVENT_RULES,
               'legacy_thresholds': thresholds, 'scope': 'window_only',
               'event_onset_domain_s': [0., 6.], 'history_is_context_only': True,
               'smooth_speed': 'centered_median3_nearest_endpoints_invalid_support_propagates',
               'turn_scope': 'maximal_contiguous_reliable_future_segments',
               'review_sampling': 'sha256_s4d-review-v1_per_role_log_tag_cap20',
               'runtime': {'numpy': str(np.__version__), 'matplotlib': matplotlib.__version__}}
    identity = digest_json(binding)
    if verify_only and identity != saved['identity']:
        raise ValueError('S4d source/config/environment changed')
    if not verify_only:
        output.mkdir(parents=True, exist_ok=False)
        _write(output / 'run.json', {'identity': identity, 'binding': binding})
    windows, manifest, events, coverage = _derive(inputs, thresholds)
    figure_ids = sorted({r['window_id'] for r in manifest})
    summary = {'identity': identity, 'unique_windows': len(windows),
               'windows_by_role': {r: sum(w['role'] == r for w in windows) for r in sorted(inputs.records)},
               'windows_with_candidate': sum(bool(w['labels']) for w in windows),
               'candidate_tag_assignments': sum(len(w['labels']) for w in windows),
               'event_records': len(events), 'review_cases': len(manifest), 'review_figures': len(figure_ids),
               'manual_review_completed': 0, 'manual_review_pending': len(manifest),
               'scope': 'window_only', 'labels_enter_model': False,
               'new_prediction_scores_computed': False, 'neural_training': False, 'holdout_evaluated': False}
    text = _csv_text(manifest)
    gallery = _gallery(windows, manifest)
    _check_bound_inputs(preflight, marker, marker_hash, inputs.references, sources)
    data = [('windows', windows), ('events', events), ('coverage', coverage), ('review', manifest), ('summary', summary)]
    if verify_only:
        for name, value in data:
            if digest_json(json.loads((output / (name + '.json')).read_text(encoding='utf-8'))) != digest_json(value):
                raise ValueError('label/manifest/summary semantic replay mismatch')
        if ((output / 'event_review.csv').read_text(encoding='utf-8-sig') != text
                or (output / 'review.html').read_text(encoding='utf-8') != gallery):
            raise ValueError('review CSV/gallery semantic replay mismatch')
        for name, digest in saved['sha256'].items():
            if sha256_file(output / name) != digest:
                raise ValueError('label output changed during verification')
        if sha256_file(output / 'completed.json') != saved_marker_hash:
            raise ValueError('label output snapshot changed during verification')
        return {**summary, 'labels_and_manifest_recomputed': True}
    for name, value in data:
        _write(output / (name + '.json'), value)
    (output / 'event_review.csv').write_text(text, encoding='utf-8-sig', newline='')
    (output / 'review.html').write_text(gallery, encoding='utf-8')
    lookup = {w['window_id']: w for w in windows}
    if figure_ids:
        (output / 'figures').mkdir()
        for window_id in figure_ids:
            plot_gt_window(lookup[window_id], output / 'figures' / (window_id + '.png'))
    _check_bound_inputs(preflight, marker, marker_hash, inputs.references, sources)
    names = ['run.json', *[name + '.json' for name, _ in data], 'event_review.csv', 'review.html',
             *['figures/' + i + '.png' for i in figure_ids]]
    _write(output / 'completed.json', {'identity': identity,
           'sha256': {name: sha256_file(output / name) for name in names}})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path,
                        default=Path('outputs/scenario-evaluation/preflight-v2'))
    parser.add_argument('--output-root', type=Path,
                        default=Path('outputs/scenario-evaluation/s4d-labels-v1'))
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check_labels(args.preflight, args.output_root, verify_only=args.verify_only)), flush=True)

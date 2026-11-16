"""Frozen B1a offline GT rule-candidate extraction and semantic replay."""
import argparse
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

from trajectory_prediction.analysis.realclock_motion import build_motion
from trajectory_prediction.analysis.stop_restart_events import extract_events
from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation.event_sources import load_config, parent_snapshot, read_json
from trajectory_prediction.evaluation.event_windows import TYPES, coverage_summary, link_windows
from trajectory_prediction.evaluation.track_sources import add_reference, check_references, sources
from trajectory_prediction.utils.provenance import sha256_file

PROJECT = Path(__file__).resolve().parents[3]
ROLES = ('independent_fit', 'independent_select')


def _write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def _save_npz(path, arrays):
    np.savez_compressed(path, **arrays)


def _compare_npz(path, arrays):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != set(arrays) or any(z[k].dtype != v.dtype or not np.array_equal(z[k], v)
                                              for k, v in arrays.items()):
            raise ValueError('event kinematics semantic replay mismatch')


def _interval_arrays(rows):
    fields = {'interval_id': 'U64', 'segment_id': 'U64', 'block_id': 'U64',
              'left_token': 'U32', 'right_token': 'U32', 'left_sample': 'U32', 'right_sample': 'U32',
              'start_us': np.int64, 'end_us': np.int64, 'dt_us': np.int64,
              'speed_mps': np.float64, 'valid': bool, 'reason': 'U40'}
    return {k: np.array([r[k] if r[k] is not None else '' for r in rows], dtype=dtype)
            for k, dtype in fields.items()}


def _event_counts(events):
    return {kind: sum(e['event_type'] == kind for e in events) for kind in TYPES}


def _censor_counts(events, diagnostics):
    return {'right_censored_stops': sum(e['event_type'] == 'stop_candidate' and e['right_censored'] for e in events),
            'low_end_censored_stops': sum(e['event_type'] == 'stop_candidate' and e['low_end_censored'] is True
                                          for e in events),
            'left_censored_stop_runs': sum(d['reason'] == 'insufficient_prior_history' for d in diagnostics),
            'initially_low_speed_runs': sum(d.get('initially_low_speed') is True for d in diagnostics),
            'unpaired_confirmed_restart_diagnostics': sum(d['reason'] == 'restart_without_confirmed_stop'
                                                        for d in diagnostics),
            'insufficient_followup_runs': sum(d['reason'].startswith('insufficient_followup') for d in diagnostics)}


def _analyse(info, config):
    parent = Path(info['parent'])
    tracks, segments = read_json(parent/'tracks.json'), read_json(parent/'segments.json')
    windows = read_json(parent/'windows.json')
    grouped = defaultdict(list)
    for segment in segments:
        grouped[segment['role'], segment['instance']].append(segment)
    arrays, all_blocks, events, diagnostics, trace, role_stats, clocks = {}, [], [], [], [], {}, {}
    for role in ROLES:  # frozen fit first; identical rules then applied to select
        print(json.dumps({'stage': 'offline_events', 'role': role}), flush=True)
        with np.load(parent/(role+'-tracks.npz'), allow_pickle=False) as z:
            points = {k: z[k] for k in ('timestamp_us', 'world_xy', 'annotation_token', 'sample_token')}
        with np.load(parent/(role+'-windows.npz'), allow_pickle=False) as z:
            clocks[role] = {'actual_timestamp_us': z['actual_timestamp_us']}
        rows = []
        for track in (t for t in tracks if t['role'] == role):
            for segment in grouped[role, track['instance']]:
                motion = build_motion(track, segment, points, config, info['dataset_release'])
                rows.extend(motion['intervals']); all_blocks.extend(motion['blocks'])
                diagnostics.extend(motion['diagnostics'])
                by_block = defaultdict(list)
                for row in motion['intervals']:
                    if row['valid']:
                        by_block[row['block_id']].append(row)
                    else:
                        trace.append({'role': role, 'instance': track['instance'], 'scene': track['scene'],
                                      'log': track['log'], 'category': track['category'], 'block_id': None,
                                      'interval_id': row['interval_id'], 'start_us': row['start_us'],
                                      'state_before': 'BARRIER', 'state_after': 'UNKNOWN', 'pending_us': 0,
                                      'pending_intervals': 0, 'context_stop_id': None, 'unpaired_origin': None,
                                      'emitted_keys': [], 'reason': 'temporal_resolution_gap'})
                for block in motion['blocks']:
                    block_rows = by_block[block['block_id']]
                    result = extract_events(block, block_rows, config)
                    events.extend(result['events']); diagnostics.extend(result['diagnostics'])
                    starts = {r['interval_id']: r['start_us'] for r in block_rows}
                    for step in result['state_trace']:
                        trace.append({**{k: block[k] for k in ('role', 'instance', 'scene', 'log', 'category')},
                                      **step, 'start_us': starts[step['interval_id']], 'reason': None})
        arrays[role+'-kinematics.npz'] = _interval_arrays(rows)
        role_tracks = [t for t in tracks if t['role'] == role]
        role_stats[role] = {'tracks': len(role_tracks),
                            'annotations': sum(t['role_row_stop']-t['role_row_start'] for t in role_tracks),
                            'segments': sum(s['role'] == role for s in segments), 'intervals': len(rows),
                            'valid_intervals': sum(r['valid'] for r in rows),
                            'temporal_resolution_gap_intervals': sum(not r['valid'] for r in rows)}
        cats = sorted({t['category'] for t in role_tracks})
        role_stats[role]['category_counts'] = {}
        for cat in cats:
            ts = [t for t in role_tracks if t['category'] == cat]
            instances = {t['instance'] for t in ts}
            rs = [r for r in rows if r['instance'] in instances]
            role_stats[role]['category_counts'][cat] = {
                'tracks': len(ts), 'input_annotations': sum(t['role_row_stop']-t['role_row_start'] for t in ts),
                'input_segments': sum(len(grouped[role, i]) for i in instances),
                'input_intervals': len(rs), 'valid_intervals': sum(r['valid'] for r in rs),
                'temporal_resolution_gap_intervals': sum(not r['valid'] for r in rs),
                'analysis_blocks': sum(b['role'] == role and b['category'] == cat for b in all_blocks)}
        del points, rows
    event_key = lambda e: (e['role'], e['log'], e['scene'], e['instance'], e['block_start_us'],
                           e['onset_us'], e['event_type'], e['event_id'])
    events.sort(key=event_key)
    for d in diagnostics:
        if 'diagnostic_id' not in d:
            d['diagnostic_id'] = digest({'schema': 'motion_diagnostic_v1', **d})
    diagnostics.sort(key=lambda d: (d['role'], d['log'], d['scene'], d['instance'],
                                   d['start_us'], d['reason'], d['diagnostic_id']))
    trace.sort(key=lambda r: (r['role'], r['log'], r['scene'], r['instance'], r['start_us'], r['interval_id']))
    all_blocks.sort(key=lambda b: (b['role'], b['log'], b['scene'], b['instance'], b['start_us'], b['block_id']))
    links = link_windows(events, windows, clocks)
    categories = {role: {t['category'] for t in tracks if t['role'] == role} for role in ROLES}
    for role in ROLES:
        es = [e for e in events if e['role'] == role]
        ds = [d for d in diagnostics if d['role'] == role]
        role_stats[role].update(event_counts=_event_counts(es),
            diagnostic_counts=dict(sorted(Counter(d['reason'] for d in ds).items())),
            analysis_blocks=sum(b['role'] == role for b in all_blocks), **_censor_counts(es, ds))
        cats = role_stats[role]['category_counts']
        for cat, counts in cats.items():
            ces, cds = [e for e in es if e['category'] == cat], [d for d in ds if d['category'] == cat]
            counts.update(event_counts=_event_counts(ces),
                          diagnostic_counts=dict(sorted(Counter(d['reason'] for d in cds).items())),
                          **_censor_counts(ces, cds))
        mapping = {'tracks': 'tracks', 'input_annotations': 'annotations', 'input_segments': 'segments',
                   'input_intervals': 'intervals', 'valid_intervals': 'valid_intervals',
                   'temporal_resolution_gap_intervals': 'temporal_resolution_gap_intervals',
                   'analysis_blocks': 'analysis_blocks', **{k:k for k in _censor_counts(es, ds)}}
        if any(sum(c[k] for c in cats.values()) != role_stats[role][v] for k, v in mapping.items()):
            raise ValueError('B1 category/role count conservation failed')
    total = lambda key: sum(s[key] for s in role_stats.values())
    if (total('annotations')-len(segments) != total('intervals')
            or total('intervals') != total('valid_intervals')+total('temporal_resolution_gap_intervals')
            or len(trace) != total('intervals')):
        raise ValueError('B1 interval/trace conservation failed')
    summary = {'status': 'b1a_rule_candidates_complete', 'input_tracks': len(tracks),
               'input_annotations': total('annotations'), 'input_segments': len(segments),
               'input_intervals': total('intervals'), 'valid_intervals': total('valid_intervals'),
               'temporal_resolution_gap_intervals': total('temporal_resolution_gap_intervals'),
               'analysis_blocks': len(all_blocks), 'event_counts': _event_counts(events),
               'diagnostic_counts': dict(sorted(Counter(d['reason'] for d in diagnostics).items())),
               'censor_counts': _censor_counts(events, diagnostics),
               'role_stats': role_stats, 'coverage': coverage_summary(events, links, categories),
               'window_event_links': len(links), 'cache_windows': len(windows),
               'pilot_windows': sum(w['phase3a_pilot_member'] for w in windows),
               'manual_completed': 0, 'rule_candidates_are_human_ground_truth': False,
               'neural_training': False, 'holdout_evaluated': False, 'new_prediction_scores_computed': False}
    documents = {'blocks.json': all_blocks, 'events.json': events, 'diagnostics.json': diagnostics,
                 'state_trace.json': trace, 'window_event_links.json': links, 'summary.json': summary,
                 'protocol.json': {'schema': 'stop_restart_realclock_v1', 'config': config,
                                   'config_sha256': digest(config), 'offline_gt_only': True,
                                   'speed_estimator': 'raw_displacement_over_actual_dt',
                                   'interpolation': False, 'forecast_stitching': False}}
    return arrays, documents


def _binding(info, config_path, config):
    return {'schema': 'scenario_events_b1a_v1', 'scope': 'original_fit_select_cache_target_instances',
            'parent': info['parent'], 'parent_owner': info['owner'], 'parent_identity': info['identity'],
            'dataset_release': info['dataset_release'], 'config_path': str(config_path),
            'config_sha256': digest(config), 'references': dict(info['references']), 'sources': sources(PROJECT),
            'runtime': {'python': sys.version, 'numpy': str(np.__version__), 'yaml': str(yaml.__version__),
                        'platform': platform.platform()}, 'neural_training': False, 'holdout_evaluated': False,
            'new_prediction_scores_computed': False}


def _check(binding):
    if sources(PROJECT) != binding['sources']:
        raise ValueError('B1 source changed')
    check_references(binding['references'])


def _files(output, hashes, marker=False):
    actual = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    if actual != set(hashes) | ({'completed.json'} if marker else set()):
        raise ValueError('B1 file coverage mismatch')
    for name, expected in hashes.items():
        path = (output/name).resolve()
        if not path.is_relative_to(output) or sha256_file(path) != expected:
            raise ValueError('B1 artifact checksum mismatch')


def prepare_events(parent, parent_owner, config_path, output):
    output, config_path = Path(output).resolve(), Path(config_path).resolve()
    if output.exists():
        raise FileExistsError('use a new event output directory')
    source_hashes, config_hash = sources(PROJECT), sha256_file(config_path)
    config = load_config(config_path)
    info = parent_snapshot(parent, parent_owner)
    add_reference(info['references'], config_path, config_hash)
    binding = _binding(info, config_path, config)
    if binding['sources'] != source_hashes:
        raise ValueError('B1 source changed during parent verification')
    output.mkdir(parents=True, exist_ok=False)
    _write(output/'progress.json', {'state': 'analysing'})
    try:
        arrays, documents = _analyse(info, config)
        documents.update({'run.json': {'identity': digest(binding), 'binding': binding},
                          'parent_proof.json': info['proof']})
        hashes = {}
        for name, data in arrays.items():
            _save_npz(output/name, data); _compare_npz(output/name, data)
            hashes[name] = sha256_file(output/name)
        for name, data in documents.items():
            _write(output/name, data); hashes[name] = sha256_file(output/name)
        _check(binding)
        _write(output/'progress.json', {'state': 'complete'})
        hashes['progress.json'] = sha256_file(output/'progress.json')
        _files(output, hashes)
        _write(output/'completed.json', {'identity': digest(binding), 'sha256': hashes})
    except BaseException:
        _write(output/'progress.json', {'state': 'failed'})
        raise
    return documents['summary.json']


def verify_events(output):
    output = Path(output).resolve()
    marker_path = output/'completed.json'
    if not marker_path.is_file():
        raise ValueError('incomplete event output')
    marker_hash, marker = sha256_file(marker_path), read_json(marker_path)
    _files(output, marker['sha256'], marker=True)
    run, proof = read_json(output/'run.json'), read_json(output/'parent_proof.json')
    binding = run['binding']
    if digest(binding) != run['identity'] or marker['identity'] != run['identity']:
        raise ValueError('B1 run identity mismatch')
    _check(binding)
    info = parent_snapshot(binding['parent'], binding['parent_owner'], proof)
    config_path = Path(binding['config_path']).resolve()
    config = load_config(config_path)
    add_reference(info['references'], config_path)
    if digest(binding) != digest(_binding(info, config_path, config)):
        raise ValueError('B1 policy/runtime/source replay mismatch')
    arrays, documents = _analyse(info, config)
    if set(marker['sha256']) != set(arrays) | set(documents) | {'run.json', 'parent_proof.json', 'progress.json'}:
        raise ValueError('B1 expected file coverage mismatch')
    for name, value in arrays.items():
        _compare_npz(output/name, value)
    for name, value in documents.items():
        if digest(read_json(output/name)) != digest(value):
            raise ValueError('B1 semantic replay mismatch: '+name)
    if read_json(output/'progress.json') != {'state': 'complete'}:
        raise ValueError('B1 progress replay mismatch')
    _check(binding); _files(output, marker['sha256'], marker=True)
    if sha256_file(marker_path) != marker_hash:
        raise ValueError('B1 marker changed during replay')
    return {**documents['summary.json'], 'events_semantically_replayed': True, 'identity': run['identity']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path); parser.add_argument('--parent-owner', type=Path)
    parser.add_argument('--config', type=Path); parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only and any(v is None for v in (args.parent, args.parent_owner, args.config)):
        parser.error('--parent/--parent-owner/--config required')
    result = (verify_events(args.output_root) if args.verify_only else
              prepare_events(args.parent, args.parent_owner, args.config, args.output_root))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)

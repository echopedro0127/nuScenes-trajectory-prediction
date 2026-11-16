"""B0a immutable real-clock tracks and existing-window mappings; no new scores."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
from pyquaternion import Quaternion

from trajectory_prediction.analysis.track_timeline import build_tracks, digest
from trajectory_prediction.data.geometry import global_to_agent
from trajectory_prediction.data.metadata_stream import iter_json_objects
from trajectory_prediction.evaluation.track_sources import (
    add_reference,
    check_references,
    parent_snapshot,
    read,
    sources,
)
from trajectory_prediction.utils.provenance import sha256_file

PROJECT = Path(__file__).resolve().parents[3]
ROLES = ('independent_fit', 'independent_select')
TABLES = ('sample', 'sample_annotation', 'instance', 'scene', 'log', 'category')


def _write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def _index(path, wanted=None):
    result = {}
    for row in iter_json_objects(path):
        token = row.get('token')
        if not isinstance(token, str) or not token:
            raise ValueError('missing metadata table token')
        if wanted is not None and token not in wanted:
            continue
        if token in result:
            raise ValueError('duplicate metadata table token')
        result[token] = row
    return result


def _window_keys(parent):
    unique = {}
    for role in ROLES:
        for row in read(parent / 'predictions' / (role + '-keys.json')):
            if row['role'] != role:
                raise ValueError('window role mismatch')
            key = role, row['instance'], row['sample']
            selected = {name: row[name] for name in ('role', 'instance', 'sample', 'scene', 'log', 'row')}
            if key in unique and unique[key] != selected:
                raise ValueError('conflicting repeated window identity')
            unique[key] = selected
    return [unique[k] for k in sorted(unique)]


def _load(parent_info, metadata_root=None):
    refs = dict(parent_info['parent_references'])
    bound = {name: [(p, h) for p, h in refs.items() if Path(p).name == name + '.json']
             for name in ('sample', 'scene', 'log')}
    if any(not values or len({h for p, h in values}) != 1 for values in bound.values()):
        raise ValueError('unambiguous bound metadata source required')
    metadata = (Path(metadata_root).resolve() if metadata_root is not None else
                Path(bound['sample'][0][0]).parent)
    for name in TABLES:
        path = metadata / (name + '.json')
        add_reference(refs, path, bound[name][0][1] if name in bound else None)
    cache = Path(parent_info['cache_root'])
    for role in ROLES:
        manifest_path = cache / role / 'manifest.json'
        if str(manifest_path.resolve()) not in refs:
            raise ValueError('unbound source cache manifest')
        manifest = read(manifest_path)
        archive = (manifest_path.parent / manifest['npz_file']).resolve()
        if not archive.is_relative_to(manifest_path.parent.resolve()):
            raise ValueError('cache archive escapes manifest directory')
        add_reference(refs, archive, manifest['npz_sha256'])
    scenes = _index(metadata / 'scene.json')
    scene_by_name = {s['name']: s for s in scenes.values()}
    if len(scene_by_name) != len(scenes):
        raise ValueError('duplicate source scene name')
    logs = _index(metadata / 'log.json')
    windows = _window_keys(Path(parent_info['parent']))
    original_windows = read(Path(parent_info['parent']) / 'events/windows.json')
    parent_window_ids = {(w['role'], w['instance'], w['sample']): w['window_id'] for w in original_windows}
    if (len(parent_window_ids) != len(original_windows)
            or set(parent_window_ids) != {(w['role'], w['instance'], w['sample']) for w in windows}):
        raise ValueError('parent window evidence coverage mismatch')
    targets = {role: {w['instance'] for w in windows if w['role'] == role} for role in ROLES}
    if any(not value for value in targets.values()):
        raise ValueError('both nonempty development roles required')
    role_logs, cache_data, releases = {}, {}, set()
    for role in ROLES:
        manifest = read(cache / role / 'manifest.json')
        if manifest['split'] != role:
            raise ValueError('source cache role mismatch')
        releases.add(manifest['dataset_version'])
        with np.load(cache / role / manifest['npz_file'], allow_pickle=False) as z:
            names = ('history_xy', 'future_xy', 'history_mask', 'anchor_xy', 'instance_token', 'sample_token', 'scene_name')
            data = {n: z[n] for n in names}
        n = len(data['instance_token'])
        shapes = {'history_xy': (n, 5, 2), 'future_xy': (n, 12, 2), 'anchor_xy': (n, 2)}
        if any(data[k].shape != shape or data[k].dtype != np.float32 or not np.isfinite(data[k]).all()
               for k, shape in shapes.items()):
            raise ValueError('finite float32 source GT shapes required')
        if data['history_mask'].shape != (n, 5) or data['history_mask'].dtype != np.bool_:
            raise ValueError('source GT history mask required')
        for name in ('instance_token', 'sample_token', 'scene_name'):
            if data[name].shape != (n,) or data[name].dtype.kind != 'U':
                raise ValueError('aligned source GT Unicode keys required')
        if any(str(name) not in scene_by_name for name in data['scene_name']):
            raise ValueError('source scene missing from metadata')
        role_logs[role] = {scene_by_name[str(name)]['log_token'] for name in data['scene_name']}
        if not role_logs[role] <= set(logs):
            raise ValueError('source log missing from metadata')
        cache_data[role] = data
    if len(releases) != 1:
        raise ValueError('source dataset releases differ')
    samples = _index(metadata / 'sample.json')
    instances = _index(metadata / 'instance.json', set().union(*targets.values()))
    categories = {t: r['name'] for t, r in _index(metadata / 'category.json').items()}
    annotations, scanned = [], 0
    target_set = set().union(*targets.values())
    for row in iter_json_objects(metadata / 'sample_annotation.json'):
        scanned += 1
        if row.get('instance_token') in target_set:
            annotations.append(row)
    release = next(iter(releases))
    timeline = build_tracks(annotations, samples, instances, scenes, categories, targets, role_logs, release)
    return {'metadata_root': str(metadata), 'dataset_release': release, 'references': refs,
            'cache_data': cache_data, 'windows': windows, 'parent_window_ids': parent_window_ids,
            'timeline': timeline, 'annotations_scanned': scanned}


def _derive(data):
    timeline, windows, cache_data = data['timeline'], data['windows'], data['cache_data']
    arrays, metadata, track_index = {}, [], []
    positions = {}
    for role in ROLES:
        tracks = [t for t in timeline['tracks'] if t['role'] == role]
        rows = []
        for track in tracks:
            start = len(rows)
            rows.extend([{**r, 'instance_token': track['instance']} for r in track['annotations']])
            track_index.append({k: v for k, v in track.items() if k != 'annotations'} |
                               {'role_row_start': start, 'role_row_stop': len(rows)})
            positions[role, track['instance']] = {r['token']: r for r in track['annotations']}
        arrays[role + '-tracks.npz'] = {
            'timestamp_us': np.array([r['timestamp_us'] for r in rows], np.int64),
            'world_xy': np.array([r['world_xy'] for r in rows], np.float64),
            'rotation': np.array([r['rotation'] for r in rows], np.float64),
            **{name: np.array([r[field] for r in rows], dtype=dtype) for name, field, dtype in (
                ('annotation_token', 'token', 'U32'), ('sample_token', 'sample_token', 'U32'),
                ('instance_token', 'instance_token', 'U32'), ('segment_id', 'segment_id', 'U64'),
                ('actual_dt_us', 'actual_dt_us', np.int64), ('delta_valid', 'delta_valid', bool))}}
        selected = [w for w in windows if w['role'] == role]
        window_rows = []
        source = cache_data[role]
        for window in selected:
            i = window['row']
            if (type(i) is not int or not 0 <= i < len(source['sample_token'])
                    or str(source['instance_token'][i]) != window['instance']
                    or str(source['sample_token'][i]) != window['sample']
                    or str(source['scene_name'][i]) != window['scene']
                    or not source['history_mask'][i].all()):
                raise ValueError('cached window identity/history mismatch')
            track = next(t for t in tracks if t['instance'] == window['instance'])
            if track['log'] != window['log'] or track['scene'] != window['scene']:
                raise ValueError('cached window scene/log mismatch')
            lookup = positions[role, window['instance']]
            anchors = [r for r in lookup.values() if r['sample_token'] == window['sample']]
            if len(anchors) != 1:
                raise ValueError('missing/duplicate cached anchor annotation')
            anchor, past, future = anchors[0], [], []
            current = anchor
            for _ in range(4):
                if current['prev'] not in lookup:
                    raise ValueError('cached window missing history annotation')
                current = lookup[current['prev']]
                past.append(current)
            current = anchor
            for _ in range(12):
                if current['next'] not in lookup:
                    raise ValueError('cached window missing future annotation')
                current = lookup[current['next']]
                future.append(current)
            sequence = [*reversed(past), anchor, *future]
            xy = np.asarray(anchor['world_xy'], np.float32)
            yaw = Quaternion(anchor['rotation']).yaw_pitch_roll[0]
            h = global_to_agent(np.array([r['world_xy'] for r in sequence[:5]], np.float32), xy, yaw)
            f = global_to_agent(np.array([r['world_xy'] for r in future], np.float64), xy, yaw)
            if (not np.array_equal(xy, source['anchor_xy'][i]) or not np.array_equal(h, source['history_xy'][i])
                    or not np.array_equal(f, source['future_xy'][i])):
                raise ValueError('raw annotation/cache geometry mismatch')
            window_rows.append(sequence)
            contiguous = len({r['segment_id'] for r in sequence}) == 1
            window_id = hashlib.sha256(f"{role}:{window['instance']}:{window['sample']}".encode()).hexdigest()
            if data['parent_window_ids'][role, window['instance'], window['sample']] != window_id:
                raise ValueError('parent window ID mismatch')
            metadata.append({**window, 'window_id': window_id, 'role_window_row': len(window_rows) - 1,
                             'anchor_annotation_token': anchor['token'], 'contiguous_annotation_segment': contiguous,
                             'status': 'ok' if contiguous else 'not_evaluable',
                             'reason': None if contiguous else 'source_annotation_gap'})
        times = np.array([[r['timestamp_us'] for r in seq] for seq in window_rows], np.int64)
        arrays[role + '-windows.npz'] = {
            'annotation_token': np.array([[r['token'] for r in seq] for seq in window_rows], 'U32'),
            'sample_token': np.array([[r['sample_token'] for r in seq] for seq in window_rows], 'U32'),
            'actual_timestamp_us': times, 'actual_offset_us': times - times[:, 4:5],
            'nominal_step': np.arange(-4, 13, dtype=np.int16)}
    summary = {'status': 'tracks_and_clock_complete', 'scope': 'phase3a_pilot_target_instances',
               'target_instances': len(timeline['tracks']), 'annotation_rows': sum(len(t['annotations']) for t in timeline['tracks']),
               'window_count': len(windows), 'source_frame_gap_pairs': len(timeline['gaps']),
               'annotation_contiguous_segments': len(timeline['segments']),
               'annotations_scanned': data['annotations_scanned'], 'nominal_time_is_exact': False,
               'canonical_episodes_extracted': False, 'new_prediction_scores_computed': False,
               'neural_training': False, 'holdout_evaluated': False}
    audit = {'cached_windows_exactly_rebuilt': len(metadata), 'annotation_chain_validated': True,
             'actual_timestamp_dtype': 'int64_us', 'world_coordinate_dtype': 'float64_m',
             'interpolated_or_filled_gt_points': 0, 'continuous_segments_are_not_episodes': True}
    return arrays, {'tracks.json': track_index, 'segments.json': timeline['segments'], 'gaps.json': timeline['gaps'],
                    'windows.json': metadata, 'summary.json': summary, 'audit.json': audit}


def _save_npz(path, arrays):
    np.savez_compressed(path, **arrays)


def _compare_npz(path, expected):
    with np.load(path, allow_pickle=False) as z:
        if set(z.files) != set(expected) or any(z[n].dtype != v.dtype or not np.array_equal(z[n], v)
                                              for n, v in expected.items()):
            raise ValueError('track/window array replay mismatch')


def _check(binding):
    if sources(PROJECT) != binding['sources']:
        raise ValueError('B0 source changed')
    check_references(binding['references'])


def _binding(info, data, source_hashes):
    return {'schema': 'scenario_tracks_b0a_v1', 'scope': 'phase3a_pilot_target_instances',
            'parent': info['parent'], 'parent_owner': info['owner'], 'parent_identity': info['identity'],
            'metadata_root': data['metadata_root'], 'dataset_release': data['dataset_release'],
            'sources': source_hashes, 'references': data['references'],
            'runtime': {'python': sys.version, 'numpy': str(np.__version__), 'platform': platform.platform()},
            'neural_training': False, 'holdout_evaluated': False}


def _protocol():
    return {'scope': 'phase3a_pilot_target_instances', 'nominal_dt_s': .5, 'nominal_time_is_exact': False,
            'timestamp_unit': 'us', 'world_coordinate_frame': 'global_xy_m',
            'source_frame_gap_policy': 'split_annotation_segment_no_GT_fill',
            'canonical_episodes_extracted': False}


def prepare_tracks(parent, parent_owner, output, metadata_root=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('use a new B0a output directory')
    source_hashes = sources(PROJECT)
    info = parent_snapshot(parent, parent_owner)
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'progress.json', {'state': 'assembling'})
    try:
        data = _load(info, metadata_root)
        arrays, documents = _derive(data)
        binding = _binding(info, data, source_hashes)
        documents.update({'run.json': {'identity': digest(binding), 'binding': binding}, 'parent_proof.json': info['proof'],
                          'protocol.json': _protocol()})
        hashes = {}
        for name, values in arrays.items():
            _save_npz(output / name, values)
            _compare_npz(output / name, values)
            hashes[name] = sha256_file(output / name)
        for name, value in documents.items():
            _write(output / name, value)
            hashes[name] = sha256_file(output / name)
        _check(binding)
        _write(output / 'progress.json', {'state': 'complete'})
        hashes['progress.json'] = sha256_file(output / 'progress.json')
        actual = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
        if actual != set(hashes) or any(sha256_file(output / n) != h for n, h in hashes.items()):
            raise ValueError('B0 artifact coverage/contents changed')
        _write(output / 'completed.json', {'identity': digest(binding), 'sha256': hashes})
    except BaseException:
        _write(output / 'progress.json', {'state': 'failed'})
        raise
    return documents['summary.json']


def verify_tracks(output):
    output = Path(output).resolve()
    marker_path = output / 'completed.json'
    if not marker_path.is_file():
        raise ValueError('incomplete B0a output')
    marker_hash, marker = sha256_file(marker_path), read(marker_path)
    actual = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    if actual != set(marker['sha256']) | {'completed.json'}:
        raise ValueError('B0 file coverage mismatch')
    for name, expected in marker['sha256'].items():
        path = (output / name).resolve()
        if not path.is_relative_to(output) or sha256_file(path) != expected:
            raise ValueError('B0 artifact checksum mismatch')
    run, proof = read(output / 'run.json'), read(output / 'parent_proof.json')
    binding = run['binding']
    if digest(binding) != run['identity'] or marker['identity'] != run['identity']:
        raise ValueError('B0 run identity mismatch')
    _check(binding)
    info = parent_snapshot(binding['parent'], binding['parent_owner'], proof)
    if info['identity'] != binding['parent_identity']:
        raise ValueError('B0 parent identity changed')
    data = _load(info, binding['metadata_root'])
    if data['references'] != binding['references'] or data['dataset_release'] != binding['dataset_release']:
        raise ValueError('B0 source snapshot mismatch')
    if binding != _binding(info, data, sources(PROJECT)):
        raise ValueError('B0 run policy/runtime replay mismatch')
    arrays, documents = _derive(data)
    documents['protocol.json'] = _protocol()
    if set(marker['sha256']) != set(arrays) | set(documents) | {'run.json', 'parent_proof.json', 'progress.json'}:
        raise ValueError('B0 expected file coverage mismatch')
    for name, value in arrays.items():
        _compare_npz(output / name, value)
    for name, expected in documents.items():
        if read(output / name) != expected:
            raise ValueError('B0 semantic replay mismatch: ' + name)
    if read(output / 'progress.json') != {'state': 'complete'}:
        raise ValueError('B0 progress replay mismatch')
    _check(binding)
    for name, expected in marker['sha256'].items():
        if sha256_file(output / name) != expected:
            raise ValueError('B0 artifact changed during replay')
    if sha256_file(marker_path) != marker_hash:
        raise ValueError('B0 marker changed during replay')
    return {**documents['summary.json'], 'track_and_window_replayed': True, 'identity': run['identity']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--parent-owner', type=Path)
    parser.add_argument('--metadata-root', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if not args.verify_only and (args.parent is None or args.parent_owner is None):
        parser.error('--parent and --parent-owner required')
    result = (verify_tracks(args.output_root) if args.verify_only else
              prepare_tracks(args.parent, args.parent_owner, args.output_root, args.metadata_root))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)

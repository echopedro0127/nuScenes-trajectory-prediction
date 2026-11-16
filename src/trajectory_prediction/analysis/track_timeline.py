"""Pure real-clock annotation chains; continuous segments are not episodes."""
import hashlib
import json
import math


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _token(value, *, empty=False):
    if not isinstance(value, str) or len(value) > 32 or '\0' in value or (not value and not empty):
        raise ValueError('nonempty source token of at most 32 characters required')
    return value


def _vector(value, length):
    if (not isinstance(value, (list, tuple)) or len(value) != length
            or any(type(v) not in (int, float) or not math.isfinite(v) for v in value)):
        raise ValueError('finite source coordinate/rotation vector required')
    return [float(v) for v in value]


def _timestamp(value):
    if type(value) is not int or not 0 < value < 2**63:
        raise ValueError('positive int64 source timestamp required')
    return value


def build_tracks(annotations, samples, instances, scenes, categories, targets_by_role,
                 logs_by_role, dataset_release):
    if (not isinstance(dataset_release, str) or not dataset_release
            or not targets_by_role or set(targets_by_role) != set(logs_by_role)
            or not set(targets_by_role) <= {'independent_fit', 'independent_select'}):
        raise ValueError('explicit development role scope and dataset release required')
    owners, log_owners = {}, {}
    for role in sorted(targets_by_role):
        for instance in targets_by_role[role]:
            _token(instance)
            if instance in owners:
                raise ValueError('instance crosses role boundary')
            owners[instance] = role
        for log in logs_by_role[role]:
            _token(log)
            if log in log_owners:
                raise ValueError('log crosses role boundary')
            log_owners[log] = role
    grouped, all_tokens = {}, set()
    for row in annotations:
        token = _token(row.get('token'))
        instance = _token(row.get('instance_token'))
        if token in all_tokens or instance not in owners:
            raise ValueError('duplicate annotation token or out-of-scope target')
        all_tokens.add(token)
        grouped.setdefault(instance, {})[token] = row
    if set(grouped) != set(owners):
        raise ValueError('missing target annotations')

    checked_samples = {}
    def sample(token):
        _token(token)
        if token in checked_samples:
            return checked_samples[token]
        row = samples.get(token)
        if not isinstance(row, dict) or row.get('token') != token:
            raise ValueError('missing/inconsistent source sample')
        _timestamp(row.get('timestamp'))
        scene_token = _token(row.get('scene_token'))
        if scene_token not in scenes:
            raise ValueError('missing source scene')
        for field, reverse, direction in [('prev', 'next', -1), ('next', 'prev', 1)]:
            other = _token(row.get(field), empty=True)
            if other:
                neighbor = samples.get(other)
                if (not isinstance(neighbor, dict) or neighbor.get('token') != other
                        or neighbor.get(reverse) != token or neighbor.get('scene_token') != scene_token
                        or direction * (_timestamp(neighbor.get('timestamp')) - row['timestamp']) <= 0):
                    raise ValueError('sample chain is not reciprocal/chronological within scene')
        checked_samples[token] = row
        return row

    tracks, segments, gaps = [], [], []
    for instance in sorted(owners):
        role, available = owners[instance], grouped[instance]
        info = instances.get(instance)
        if not isinstance(info, dict) or info.get('token') != instance:
            raise ValueError('missing/inconsistent instance metadata')
        count = info.get('nbr_annotations')
        if type(count) is not int or count < 1 or count != len(available):
            raise ValueError('instance annotation count mismatch')
        category = categories.get(_token(info.get('category_token')))
        if not isinstance(category, str) or not category:
            raise ValueError('missing category metadata')
        token, previous = _token(info.get('first_annotation_token')), ''
        normalized, visited, seen_samples, breaks, track_gaps = [], set(), set(), [0], []
        scene_token, log, scene_name = None, None, None
        while token:
            if token in visited or token not in available:
                raise ValueError('annotation cycle/broken chain')
            visited.add(token)
            raw = available[token]
            if raw.get('prev') != previous or raw.get('instance_token') != instance:
                raise ValueError('annotation prev/instance mismatch')
            current = sample(raw.get('sample_token'))
            if current['token'] in seen_samples:
                raise ValueError('duplicate instance/sample')
            seen_samples.add(current['token'])
            scene = scenes[current['scene_token']]
            if scene_token is None:
                scene_token, log, scene_name = current['scene_token'], _token(scene.get('log_token')), scene.get('name')
                if log_owners.get(log) != role or not isinstance(scene_name, str) or not scene_name:
                    raise ValueError('scene/log role mismatch')
            if current['scene_token'] != scene_token:
                raise ValueError('instance crosses scene/log boundary')
            missing = []
            if normalized:
                last = sample(normalized[-1]['sample_token'])
                cursor, seen = last['next'], {last['token']}
                while cursor != current['token']:
                    if not cursor or cursor in seen:
                        raise ValueError('annotation sample is unreachable in scene chain')
                    seen.add(cursor)
                    middle = sample(cursor)
                    if (middle['scene_token'] != scene_token
                            or not last['timestamp'] < middle['timestamp'] < current['timestamp']):
                        raise ValueError('annotation timestamps are not increasing')
                    missing.append(cursor)
                    cursor = middle['next']
                if missing:
                    breaks.append(len(normalized))
                    track_gaps.append({'role': role, 'instance': instance, 'before_annotation': previous,
                                       'after_annotation': token, 'before_sample': last['token'],
                                       'after_sample': current['token'], 'missing_sample_tokens': missing,
                                       'elapsed_us': current['timestamp'] - last['timestamp']})
            xyz, rotation = _vector(raw.get('translation'), 3), _vector(raw.get('rotation'), 4)
            try:
                norm = math.fsum(v * v for v in rotation)
            except OverflowError as error:
                raise ValueError('invalid rotation norm') from error
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError('invalid rotation norm')
            normalized.append({'token': token, 'sample_token': current['token'],
                               'prev': _token(raw.get('prev'), empty=True),
                               'next': _token(raw.get('next'), empty=True),
                               'sample_prev': current['prev'], 'sample_next': current['next'],
                               'timestamp_us': current['timestamp'], 'world_xy': xyz[:2], 'rotation': rotation})
            previous, token = token, normalized[-1]['next']
        if len(visited) != count or previous != info.get('last_annotation_token'):
            raise ValueError('annotation endpoint/orphan/count mismatch')
        semantic = digest({'schema': 'target_track_v1', 'dataset_release': dataset_release,
                           'role': role, 'instance': instance, 'scene': scene_token, 'log': log,
                           'category': category, 'annotations': normalized})
        for start, stop in zip(breaks, [*breaks[1:], len(normalized)], strict=True):
            first, last = normalized[start], normalized[stop - 1]
            segment_id = digest({'schema': 'track_segment_v1', 'track_semantic_sha256': semantic,
                                 'role': role, 'log': log, 'scene': scene_token, 'instance': instance,
                                 'first_annotation': first['token'], 'last_annotation': last['token']})
            segments.append({'segment_id': segment_id, 'role': role, 'instance': instance,
                             'scene': scene_name, 'log': log, 'start_index': start, 'stop_index': stop,
                             'first_annotation': first['token'], 'last_annotation': last['token'],
                             'start_timestamp_us': first['timestamp_us'], 'end_timestamp_us': last['timestamp_us']})
            for i in range(start, stop):
                normalized[i].update(segment_id=segment_id, delta_valid=i > start,
                                     actual_dt_us=normalized[i]['timestamp_us'] - normalized[i - 1]['timestamp_us']
                                     if i > start else 0)
        gaps.extend(track_gaps)
        tracks.append({'role': role, 'instance': instance, 'scene': scene_name, 'scene_token': scene_token,
                       'log': log, 'category': category, 'semantic_sha256': semantic, 'annotations': normalized})
    tracks.sort(key=lambda r: (r['role'], r['log'], r['scene'], r['instance']))
    segments.sort(key=lambda r: (r['role'], r['log'], r['scene'], r['instance'], r['start_index']))
    gaps.sort(key=lambda r: (r['role'], r['instance'], r['before_annotation']))
    return {'tracks': tracks, 'segments': segments, 'gaps': gaps}

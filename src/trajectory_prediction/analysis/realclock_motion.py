"""Offline GT interval speeds and analysis blocks; no interpolation."""
import itertools
import math

import numpy as np

from trajectory_prediction.analysis.track_timeline import digest

FIELDS = {'rule_version', 'stop_speed_mps', 'restart_speed_mps', 'prior_history_us',
          'stop_dwell_us', 'restart_dwell_us', 'minimum_dwell_intervals', 'max_interval_us'}


def validate_config(config):
    if not isinstance(config, dict) or set(config) != FIELDS:
        raise ValueError('exact B1a config fields required')
    if config['rule_version'] != 'stop_restart_realclock_v1':
        raise ValueError('unsupported event rule version')
    for key in FIELDS - {'rule_version', 'stop_speed_mps', 'restart_speed_mps'}:
        if type(config[key]) is not int or not 0 < config[key] < 2**63:
            raise ValueError('positive int64 config duration/count required')
    for key in ('stop_speed_mps', 'restart_speed_mps'):
        if type(config[key]) not in (int, float) or not math.isfinite(config[key]):
            raise ValueError('finite speed config required')
    if (not 0 <= config['stop_speed_mps'] < config['restart_speed_mps']
            or config['minimum_dwell_intervals'] < 2):
        raise ValueError('invalid stop/restart thresholds or minimum interval count')
    return dict(config)


def build_motion(track, segment, arrays, config, release):
    config = validate_config(config)
    a, b = segment['start_index'], segment['stop_index']
    start, stop = track['role_row_start'], track['role_row_stop']
    if (any(type(v) is not int for v in (a, b, start, stop))
            or not 0 <= start < stop <= len(arrays['timestamp_us']) or not 0 <= a < b <= stop-start):
        raise ValueError('invalid track/segment row bounds')
    sl = slice(start+a, start+b)
    times, xy = arrays['timestamp_us'][sl], arrays['world_xy'][sl]
    tokens, samples = arrays['annotation_token'][sl], arrays['sample_token'][sl]
    n = b-a
    if (times.dtype != np.int64 or times.shape != (n,) or xy.dtype != np.float64
            or xy.shape != (n, 2) or not np.isfinite(xy).all() or np.any(times <= 0)
            or np.any(np.diff(times) <= 0) or tokens.shape != (n,) or samples.shape != (n,)
            or tokens.dtype.kind != 'U' or samples.dtype.kind != 'U'
            or any(not str(t) or len(str(t)) > 32 for t in [*tokens, *samples])
            or len(set(tokens)) != n or str(tokens[0]) != segment['first_annotation']
            or str(tokens[-1]) != segment['last_annotation']):
        raise ValueError('invalid finite real-clock points/tokens')
    times, tokens, samples = times.tolist(), tokens.tolist(), samples.tolist()
    common = {k: track[k] for k in ('role', 'instance', 'scene', 'log', 'category')}
    common.update(dataset_release=release, track_semantic_sha256=track['semantic_sha256'],
                  segment_id=segment['segment_id'], config_sha256=digest(config))
    rows, blocks, diagnostics = [], [], []
    for i in range(n-1):
        dt = times[i+1]-times[i]
        with np.errstate(over='ignore', invalid='ignore'):
            delta = xy[i+1]-xy[i]
        speed = math.hypot(float(delta[0]), float(delta[1]))/(dt/1e6)
        if not math.isfinite(speed):
            raise ValueError('non-finite interval speed')
        valid = dt <= config['max_interval_us']
        row = {'interval_id': digest([common, tokens[i], tokens[i+1]]),
                   'segment_id': segment['segment_id'], 'role': track['role'], 'instance': track['instance'],
                   'left_token': tokens[i], 'right_token': tokens[i+1], 'left_sample': samples[i],
                   'right_sample': samples[i+1], 'start_us': times[i], 'end_us': times[i+1], 'dt_us': dt,
                   'speed_mps': speed, 'valid': valid, 'reason': None if valid else 'temporal_resolution_gap',
                   'block_id': None}
        rows.append(row)
    cuts = [0, *[i+1 for i, r in enumerate(rows) if not r['valid']], n]
    for left, right in itertools.pairwise(cuts):
        block_id = digest({'schema': 'analysis_block_v1', **common,
                           'first_annotation': tokens[left], 'last_annotation': tokens[right-1]})
        block = {**common, 'block_id': block_id, 'first_annotation': tokens[left],
                 'last_annotation': tokens[right-1], 'start_us': times[left], 'end_us': times[right-1],
                 'interval_count': right-left-1}
        blocks.append(block)
        for row in rows[left:right-1]:
            row['block_id'] = block_id
        if right-left == 1:
            diagnostics.append({**block, 'reason': 'no_velocity_interval', 'status': 'not_evaluable'})
    for row in rows:
        if not row['valid']:
            diagnostics.append({**common, **row, 'status': 'not_evaluable'})
    return {'blocks': blocks, 'intervals': rows, 'diagnostics': diagnostics}

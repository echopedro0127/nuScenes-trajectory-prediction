"""Offline GT window candidates; never consumed by prediction or model selection."""
import copy
import hashlib

import numpy as np

RULE_VERSION = 'scenario_window_events_v1'
TAGS = ('hard_brake_candidate', 'stop_event', 'stop_go_candidate',
        's_turn_candidate', 'u_turn_candidate')
EVENT_RULES = {
    'speed_median_window': 3, 'braking_acceleration_mps2': -2.,
    'braking_duration_steps': 2, 'braking_prior_speed_mps': 2.,
    'stop_speed_mps': .3, 'stop_duration_steps': 2, 'stop_prior_speed_mps': 1.,
    'prior_window_steps': 4, 'restart_speed_mps': 1., 'restart_duration_steps': 2,
    'turn_min_speed_mps': .5, 's_turn_absolute_yaw_deg': 45.,
    's_turn_max_net_yaw_deg': 15., 's_turn_each_direction_deg': 20.,
    'u_turn_net_yaw_deg': 135., 'manual_review_cap_per_log_tag': 20,
}


def _runs(mask):
    start = None
    for i, value in enumerate([*mask, False]):
        if value and start is None:
            start = i
        elif not value and start is not None:
            yield start, i - 1
            start = None


def _turn_segments(points, mask):
    with np.errstate(over='ignore', invalid='ignore'):
        delta = np.diff(points.astype(np.float64), axis=0)
        speed = np.hypot(delta[:, 0], delta[:, 1]) / .5
    interval_valid = mask[1:] & mask[:-1]
    if not np.isfinite(speed[interval_valid]).all():
        raise ValueError('GT speed exceeds finite range')
    reliable = interval_valid & (speed >= .5)
    angles = np.arctan2(delta[:, 1], delta[:, 0])
    segments = []
    for start, end in _runs(reliable):
        if end <= start:
            continue
        changes = (np.diff(angles[start:end + 1]) + np.pi) % (2 * np.pi) - np.pi
        degrees = np.degrees(changes)
        positive = float(degrees[degrees > 0].sum())
        negative = float(-degrees[degrees < 0].sum())
        net = float(degrees.sum())
        labels = []
        if positive + negative >= 45 and abs(net) <= 15 and positive >= 20 and negative >= 20:
            labels.append('s_turn_candidate')
        if abs(net) >= 135:
            labels.append('u_turn_candidate')
        segments.append({'start_interval': start, 'end_interval': end,
                         'onset_s': start * .5, 'end_s': (end + 1) * .5,
                         'absolute_yaw_deg': positive + negative, 'net_yaw_deg': net,
                         'positive_yaw_deg': positive, 'negative_yaw_deg': negative,
                         'labels': labels})
    return segments


def turn_shape_tags(future, dt=.5):
    points = np.asarray(future)
    if (type(dt) not in (int, float) or dt != .5 or points.ndim != 2
            or points.shape[1] != 2 or len(points) < 2 or points.dtype.kind != 'f'
            or not np.isfinite(points).all()):
        raise ValueError('finite floating positions[T>=2,2] at dt=.5 required')
    return {tag for segment in _turn_segments(points, np.ones(len(points), bool))
            for tag in segment['labels']}


def _event(tag, start, end, status, reason=None):
    return {'tag': tag, 'start_interval': start, 'end_interval': end,
            'onset_s': start * .5 - 2., 'end_s': (end + 1) * .5 - 2.,
            'confirmation_s': start * .5 - 1. if status == 'candidate' else None,
            'status': status, 'reason': reason}


def _duration_candidates(predicate, valid, smooth, smooth_valid, tag, prior_speed):
    events = []
    for start, end in _runs(predicate & valid):
        if end < 4:
            continue
        if start < 4:
            if tag == 'hard_brake_candidate':
                events.append(_event(tag, start, end, 'not_evaluable', 'onset_before_window'))
            continue
        short = end - start + 1 < 2
        if short and end < 15 and valid[end + 1]:
            continue
        if not smooth_valid[start - 4:start].all():
            events.append(_event(tag, start, end, 'not_evaluable', 'insufficient_prior_history'))
            continue
        peak = float(smooth[start - 4:start].max())
        if peak >= prior_speed:
            if short:
                events.append(_event(tag, start, end, 'not_evaluable', 'insufficient_followup'))
                continue
            event = _event(tag, start, end, 'candidate')
            event['prior_peak_speed_mps'] = peak
            events.append(event)
    return events


def annotate_window(history, future, history_mask):
    h, f, mask = np.asarray(history), np.asarray(future), np.asarray(history_mask)
    if (h.shape != (5, 2) or f.shape != (12, 2) or mask.shape != (5,)
            or mask.dtype != np.bool_ or h.dtype.kind != 'f' or f.dtype.kind != 'f'
            or not np.isfinite(h).all() or not np.isfinite(f).all()):
        raise ValueError('finite floating history[5,2]/future[12,2] and bool mask required')
    points = np.concatenate((h, f)).astype(np.float64)
    point_valid = np.r_[mask, np.ones(12, bool)]
    raw_valid = point_valid[:-1] & point_valid[1:]
    with np.errstate(over='ignore', invalid='ignore'):
        delta = np.diff(points, axis=0)
        raw = np.hypot(delta[:, 0], delta[:, 1]) / .5
    if not np.isfinite(raw[raw_valid]).all():
        raise ValueError('GT speed exceeds finite range')
    raw[~raw_valid] = np.nan
    smooth, smooth_valid = np.full(16, np.nan), np.zeros(16, bool)
    for i in range(16):
        neighbors = np.clip([i - 1, i, i + 1], 0, 15)
        if raw_valid[neighbors].all():
            smooth[i], smooth_valid[i] = np.median(raw[neighbors]), True
    acceleration, acceleration_valid = np.full(16, np.nan), np.zeros(16, bool)
    with np.errstate(over='ignore', invalid='ignore'):
        for i in range(1, 16):
            if smooth_valid[i - 1:i + 1].all():
                acceleration[i] = (smooth[i] - smooth[i - 1]) / .5
                acceleration_valid[i] = True
    if not np.isfinite(acceleration[acceleration_valid]).all():
        raise ValueError('GT acceleration exceeds finite range')
    events = _duration_candidates(acceleration <= -2, acceleration_valid, smooth, smooth_valid,
                                  'hard_brake_candidate', 2.)
    stops = _duration_candidates(smooth <= .3, smooth_valid, smooth, smooth_valid, 'stop_event', 1.)
    events.extend(stops)
    initial = ('initially_stationary' if smooth[3] <= .3 else 'moving') if smooth_valid[3] else 'unknown'
    confirmed_stops = {e['start_interval'] + 1: e for e in stops if e['status'] == 'candidate'}
    high_runs = list(_runs((smooth >= 1.) & smooth_valid))
    confirmed_restarts = {a + 1: (a, b) for a, b in high_runs if a >= 4 and b - a + 1 >= 2}
    active, latched = None, initial == 'initially_stationary'
    states = []
    for i in range(4, 16):
        # A real low-speed dwell may fail the prior-speed rule. A later restart
        # still lacks a qualifying stop; retain that censoring instead of a negative.
        if i >= 5 and smooth_valid[i - 1:i + 1].all() and (smooth[i - 1:i + 1] <= .3).all():
            latched = True
        if i in confirmed_stops:
            active, latched = confirmed_stops[i], True
        if i in confirmed_restarts:
            a, b = confirmed_restarts[i]
            if active is not None and active['end_interval'] < a:
                event = _event('stop_go_candidate', active['start_interval'], b, 'candidate')
                event['restart_onset_s'] = a * .5 - 2.
                event['confirmation_s'] = (a + 2) * .5 - 2.
                events.append(event)
                active = None
            elif latched or (initial == 'unknown' and a > 4):
                events.append(_event('stop_go_candidate', a, b, 'not_evaluable', 'no_confirmed_stop_in_window'))
            latched = False
        state = 'unknown' if not smooth_valid[i] else (
            'stopped' if smooth[i] <= .3 else 'moving' if smooth[i] >= 1. else 'creeping' if latched else 'low_speed')
        states.append(state)
    restart_tail_recorded = False
    for a, b in high_runs:
        if a >= 4 and b == 15 and a == b and (active is not None or latched):
            event = _event('stop_go_candidate', active['start_interval'] if active is not None else a,
                           b, 'not_evaluable', 'insufficient_followup')
            event['restart_onset_s'] = a * .5 - 2.
            events.append(event)
            restart_tail_recorded = True
    if active is not None and active['confirmation_s'] > 5. and not restart_tail_recorded:
        events.append(_event('stop_go_candidate', active['start_interval'], 15,
                             'not_evaluable', 'insufficient_followup'))
    segments = _turn_segments(points[4:], point_valid[4:])
    for segment in segments:
        for tag in segment['labels']:
            events.append({'tag': tag, 'onset_s': segment['onset_s'], 'end_s': segment['end_s'],
                           'confirmation_s': segment['end_s'], 'status': 'candidate', 'reason': None,
                           'start_interval': segment['start_interval'] + 4,
                           'end_interval': segment['end_interval'] + 4})
    status = {}
    for tag in TAGS:
        selected = [e for e in events if e['tag'] == tag]
        positive = any(e['status'] == 'candidate' for e in selected)
        incomplete = [e for e in selected if e['status'] == 'not_evaluable']
        unavailable = tag in ('s_turn_candidate', 'u_turn_candidate') and not segments
        status[tag] = {'status': 'candidate' if positive else 'not_evaluable' if incomplete or unavailable else 'absent',
                       'reason': incomplete[0]['reason'] if incomplete and not positive else
                                 'no_reliable_direction_segment' if unavailable else None,
                       'incomplete_candidate_count': len(incomplete)}
    def serial(values):
        return [float(v) if np.isfinite(v) else None for v in values]
    return {'rule_version': RULE_VERSION, 'scope': 'window_only',
            'labels': sorted(tag for tag, value in status.items() if value['status'] == 'candidate'),
            'tag_status': status, 'initial_state': initial, 'speed_states': states,
            'events': sorted(events, key=lambda e: (e['onset_s'], e['tag'])),
            'direction_segments': segments,
            'trace': {'interval_start_s': (np.arange(16) * .5 - 2).tolist(),
                      'raw_speed_mps': serial(raw), 'raw_speed_valid': raw_valid.tolist(),
                      'smoothed_speed_mps': serial(smooth), 'smoothed_speed_valid': smooth_valid.tolist(),
                      'acceleration_mps2': serial(acceleration), 'acceleration_valid': acceleration_valid.tolist(),
                      'positions_xy': points.tolist(), 'point_valid': point_valid.tolist()}}


def build_review_manifest(windows):
    groups, seen = {}, set()
    for window in windows:
        key = (window['role'], window['instance'], window['sample'])
        if key in seen:
            raise ValueError('duplicate review window identity')
        seen.add(key)
        for tag in sorted(set(window['labels'])):
            if tag not in TAGS:
                raise ValueError('unknown rule label')
            row = {k: window[k] for k in ('window_id', 'role', 'instance', 'sample', 'log', 'rule_version')}
            text = ':'.join(('s4d-review-v1', window['role'], window['log'], tag, window['instance'], window['sample']))
            row.update(rule_label=tag, review_status='pending', reviewer='', review_reason='',
                       manual_label='', review_history=[], selection_hash=hashlib.sha256(text.encode()).hexdigest())
            groups.setdefault((window['role'], window['log'], tag), []).append(row)
    return [row for key in sorted(groups) for row in sorted(groups[key], key=lambda r: (r['selection_hash'], r['window_id']))[:20]]


def update_review(row, *, status, reviewer, reason='', manual_label=''):
    if (status not in ('confirmed', 'rejected', 'corrected') or not isinstance(reviewer, str)
            or not reviewer.strip() or not isinstance(reason, str)
            or (status in ('rejected', 'corrected') and not reason.strip())
            or not isinstance(manual_label, str)
            or (status == 'corrected' and (manual_label not in (*TAGS, 'none')
                                           or manual_label == row['rule_label']))):
        raise ValueError('reviewer and appropriate revision status/reason/label required')
    result = copy.deepcopy(row)
    result.setdefault('review_history', []).append({k: row.get(k, '') for k in
                                                   ('review_status', 'reviewer', 'review_reason', 'manual_label')})
    result.update(review_status=status, reviewer=reviewer.strip(), review_reason=reason,
                  manual_label=row['rule_label'] if status == 'confirmed' else 'none' if status == 'rejected' else manual_label)
    return result

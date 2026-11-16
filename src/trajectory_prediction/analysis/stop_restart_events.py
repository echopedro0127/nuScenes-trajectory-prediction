"""Deterministic offline rule candidates, with explicit censoring and state trace."""
import math
from itertools import groupby

from trajectory_prediction.analysis.realclock_motion import validate_config
from trajectory_prediction.analysis.track_timeline import digest


def _checked(block, intervals, config):
    if block['config_sha256'] != digest(config):
        raise ValueError('block config identity mismatch')
    rows = sorted(intervals, key=lambda r: r['start_us'])
    if len(rows) != block['interval_count'] or len({r['interval_id'] for r in rows}) != len(rows):
        raise ValueError('interval coverage/identity mismatch')
    time, token = block['start_us'], block['first_annotation']
    for r in rows:
        if (r['block_id'] != block['block_id'] or r['valid'] is not True
                or any(type(r[k]) is not int for k in ('start_us', 'end_us', 'dt_us'))
                or r['start_us'] != time or r['left_token'] != token
                or r['dt_us'] != r['end_us']-r['start_us'] or not 0 < r['dt_us'] <= config['max_interval_us']
                or type(r['speed_mps']) not in (int, float) or not math.isfinite(r['speed_mps'])
                or r['speed_mps'] < 0):
            raise ValueError('invalid linked real-clock interval')
        time, token = r['end_us'], r['right_token']
    if time != block['end_us'] or token != block['last_annotation']:
        raise ValueError('block interval endpoint mismatch')
    return rows


def extract_events(block, intervals, config):
    cfg = validate_config(config)
    rows = _checked(block, intervals, cfg)
    common = {k: block[k] for k in ('role', 'instance', 'scene', 'log', 'category', 'dataset_release',
              'track_semantic_sha256', 'segment_id', 'block_id', 'config_sha256')}
    common.update(block_start_us=block['start_us'], block_end_us=block['end_us'])
    events, diagnostics, trace = [], [], []
    context = None

    def event(kind, start, confirm, end, key, **extra):
        return {**common, '_key': key, 'event_type': kind+'_candidate', 'status': 'confirmed_by_rule',
                'onset_token': start['left_token'], 'onset_us': start['start_us'],
                'confirmation_token': confirm['right_token'], 'confirmation_us': confirm['end_us'],
                'end_token': end[0], 'end_us': end[1], 'left_censored': False, 'right_censored': False,
                'low_dwell_end_token': None, 'low_dwell_end_us': None, 'low_end_censored': None,
                'episode_end_token': None, 'episode_end_us': None, **extra}

    def diag(reason, run, key, **extra):
        row = {**common, '_key': key, 'reason': reason,
               'status': 'not_evaluable' if reason.startswith(('insufficient', 'restart_without')) else 'rule_diagnostic',
               'onset_token': run[0]['left_token'], 'start_us': run[0]['start_us'],
               'end_token': run[-1]['right_token'], 'end_us': run[-1]['end_us'],
               'observed_duration_us': sum(r['dt_us'] for r in run), 'interval_count': len(run),
               'initially_low_speed': run[0]['start_us'] == block['start_us']
               and run[0]['speed_mps'] <= cfg['stop_speed_mps'], **extra}
        diagnostics.append(row)
        return row

    def category(r):
        return 'low' if r['speed_mps'] <= cfg['stop_speed_mps'] else (
            'high' if r['speed_mps'] >= cfg['restart_speed_mps'] else 'mid')

    for kind, grouped in groupby(enumerate(rows), key=lambda pair: category(pair[1])):
        indexed = list(grouped)
        indices, run = [p[0] for p in indexed], [p[1] for p in indexed]
        first, last = run[0], run[-1]
        tail = indices[-1] == len(rows)-1
        before = context
        emitted = {}
        elapsed, confirmation = 0, None
        duration = cfg['stop_dwell_us'] if kind == 'low' else cfg['restart_dwell_us']
        for j, row in enumerate(run):
            elapsed += row['dt_us']
            if confirmation is None and elapsed >= duration and j+1 >= cfg['minimum_dwell_intervals']:
                confirmation = j
        key = f'{kind}:{indices[0]}'
        if kind == 'low':
            if before is not None:
                reason = 'low_run_with_existing_'+('stop_context' if before['kind'] == 'stop' else 'low_context')
                d = diag(reason, run, key, dwell_satisfied=confirmation is not None)
                emitted[len(run)-1] = [d['_key']]
            elif confirmation is None:
                d = diag('insufficient_followup' if tail else 'short_low_speed_run', run, key)
                emitted[len(run)-1] = [d['_key']]
            else:
                onset = first['start_us']
                prior = [r for r in rows[:indices[0]] if onset-cfg['prior_history_us'] <= r['start_us']
                         and r['end_us'] <= onset]
                enough = block['start_us'] <= onset-cfg['prior_history_us']
                moving = any(r['speed_mps'] >= cfg['restart_speed_mps'] for r in prior)
                if enough and moving:
                    stop = event('stop', first, run[confirmation],
                                 (block['last_annotation'], block['end_us']), key,
                                 right_censored=True, low_dwell_end_token=last['right_token'],
                                 low_dwell_end_us=last['end_us'], low_end_censored=tail,
                                 episode_end_token=block['last_annotation'], episode_end_us=block['end_us'],
                                 prior_interval_ids=[r['interval_id'] for r in prior])
                    events.append(stop)
                    context = {'kind': 'stop', 'event': stop}
                    emitted[confirmation] = [key]
                else:
                    reason = 'insufficient_prior_history' if not enough else 'no_recent_motion_evidence'
                    origin = ('initially_low_speed' if onset == block['start_us'] else reason)
                    d = diag(reason, run, key, initially_low_speed=onset == block['start_us'], context_origin=origin,
                             prior_interval_ids=[r['interval_id'] for r in prior])
                    context = {'kind': 'unpaired', 'origin': origin}
                    emitted[confirmation] = [d['_key']]
        elif kind == 'high' and before is not None:
            if confirmation is None:
                reason = 'insufficient_followup' if tail else 'short_restart_run'
                if before['kind'] == 'unpaired':
                    reason = ('insufficient_followup_restart_without_confirmed_stop' if tail else
                              'short_restart_run_without_confirmed_stop')
                d = diag(reason, run, key, context_origin=before.get('origin'))
                emitted[len(run)-1] = [d['_key']]
            else:
                confirm = run[confirmation]
                if before['kind'] == 'stop':
                    stop = before['event']
                    stop.update(end_token=first['left_token'], end_us=first['start_us'], right_censored=False,
                                episode_end_token=first['left_token'], episode_end_us=first['start_us'])
                    restart = event('restart', first, confirm, (confirm['right_token'], confirm['end_us']), key,
                                    _stop_key=stop['_key'])
                    pair = event('stop_go', {'left_token': stop['onset_token'], 'start_us': stop['onset_us']},
                                 confirm, (confirm['right_token'], confirm['end_us']), key+':pair',
                                 _stop_key=stop['_key'], _restart_key=key)
                    events.extend([restart, pair])
                    emitted[confirmation] = [key, key+':pair']
                else:
                    d = diag('restart_without_confirmed_stop', run[:confirmation+1], key,
                             context_origin=before['origin'])
                    emitted[confirmation] = [d['_key']]
                context = None
        elapsed = 0
        for j, row in enumerate(run):
            elapsed += row['dt_us']
            active = before
            if kind == 'low' and before is None and confirmation is not None and j >= confirmation:
                active = context
            if kind == 'high' and before is not None and confirmation is not None and j >= confirmation:
                active = None
            pending = (kind == 'low' and before is None and (confirmation is None or j < confirmation)) or (
                kind == 'high' and before is not None and (confirmation is None or j < confirmation))
            state = ('LOW_PENDING' if kind == 'low' else 'RESTART_PENDING') if pending else (
                'STOPPED' if active and active['kind'] == 'stop' else
                'UNPAIRED_LOW' if active else 'MOVING')
            if kind == 'mid' and active:
                state += '_CREEPING'
            trace.append({'block_id': block['block_id'], 'interval_id': row['interval_id'],
                          'state_before': trace[-1]['state_after'] if trace else 'UNKNOWN',
                          'state_after': state, 'pending_us': elapsed if pending else 0,
                          'pending_intervals': j+1 if pending else 0,
                          'context_stop_id': active['event']['_key'] if active and active['kind'] == 'stop' else None,
                          'unpaired_origin': active.get('origin') if active else None,
                          'emitted_keys': emitted.get(j, [])})
    ids = {}
    for kind in ('stop_candidate', 'restart_candidate', 'stop_go_candidate'):
        for row in events:
            if row['event_type'] != kind:
                continue
            key = row.pop('_key')
            for internal, external in (('_stop_key', 'stop_id'), ('_restart_key', 'restart_id')):
                if internal in row:
                    row[external] = ids[row.pop(internal)]
            row['event_id'] = digest({'schema': 'stop_restart_event_v1', **row})
            ids[key] = row['event_id']
    for row in diagnostics:
        key = row.pop('_key')
        row['diagnostic_id'] = digest({'schema': 'stop_restart_diagnostic_v1', **row})
        ids[key] = row['diagnostic_id']
    for row in trace:
        row['context_stop_id'] = ids.get(row['context_stop_id'])
        row['emitted_keys'] = [ids[k] for k in row['emitted_keys']]
    events.sort(key=lambda e: (e['onset_us'], e['event_type'], e['event_id']))
    diagnostics.sort(key=lambda d: (d['start_us'], d['reason'], d['diagnostic_id']))
    return {'events': events, 'diagnostics': diagnostics, 'state_trace': trace}

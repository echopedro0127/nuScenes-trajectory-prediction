"""Descriptive event coverage; never stitch forecasts or multiply event counts."""
from collections import defaultdict

import numpy as np

TYPES = ('stop_candidate', 'restart_candidate', 'stop_go_candidate')
METRICS = {'observed_interval_overlap': 'context_overlap', 'onset_in_future': 'future_onset',
           'confirmation_in_future': 'future_confirmation', 'entire_event_observed': 'entire_event',
           'episode_end_covered': 'event_end'}


def link_windows(events, windows, window_arrays):
    if len({e['event_id'] for e in events}) != len(events):
        raise ValueError('duplicate event identity')
    if len({w['window_id'] for w in windows}) != len(windows):
        raise ValueError('duplicate window identity')
    grouped = defaultdict(list)
    for e in events:
        if (e['event_type'] not in TYPES or type(e['right_censored']) is not bool
                or any(type(e[k]) is not int for k in ('onset_us', 'confirmation_us', 'end_us'))
                or not e['onset_us'] < e['confirmation_us'] <= e['end_us']):
            raise ValueError('invalid event boundaries/type')
        grouped[e['role'], e['instance']].append(e)
    links = []
    for w in windows:
        times, i = window_arrays[w['role']]['actual_timestamp_us'], w['role_window_row']
        if (times.dtype != np.int64 or times.ndim != 2 or times.shape[1] != 17
                or type(i) is not int or not 0 <= i < len(times)
                or type(w['phase3a_pilot_member']) is not bool):
            raise ValueError('invalid window clock/row/member')
        clock = times[i]
        if np.any(clock <= 0) or np.any(np.diff(clock) <= 0):
            raise ValueError('nonpositive/nonmonotone window clock')
        first, anchor, last = int(clock[0]), int(clock[4]), int(clock[-1])
        for e in grouped[w['role'], w['instance']]:
            onset, confirmation, end = e['onset_us'], e['confirmation_us'], e['end_us']
            future = anchor <= onset < last
            flags = {'observed_interval_overlap': max(first, onset) < min(last, end),
                     'onset_in_future': future, 'confirmation_in_future': future and confirmation <= last,
                     'episode_end_covered': not e['right_censored'] and first <= end <= last,
                     'entire_event_observed': not e['right_censored'] and first <= onset and end <= last}
            if any(flags.values()):
                links.append({'event_id': e['event_id'], 'window_id': w['window_id'],
                              'role': w['role'], 'phase3a_pilot_member': w['phase3a_pilot_member'], **flags})
    return sorted(links, key=lambda r: (r['role'], r['event_id'], r['window_id']))


def coverage_summary(events, links, categories_by_role=None):
    results = []
    for universe in ('full_cache', 'pilot'):
        selected = [r for r in links if universe == 'full_cache' or r['phase3a_pilot_member']]
        for role in ('independent_fit', 'independent_select'):
            cats = {e['category'] for e in events if e['role'] == role}
            if categories_by_role:
                cats.update(categories_by_role[role])
            for category in ['__all__', *sorted(cats)]:
                for kind in TYPES:
                    ids = {e['event_id'] for e in events if e['role'] == role and e['event_type'] == kind
                           and (category == '__all__' or e['category'] == category)}
                    row = {'universe': universe, 'role': role, 'category': category, 'event_type': kind,
                           'events_total': len(ids), 'status': 'descriptive_only' if ids else 'not_evaluable',
                           'reason': None if ids else 'no_rule_candidates'}
                    for flag, metric in METRICS.items():
                        covered = {r['event_id'] for r in selected if r['event_id'] in ids and r[flag]}
                        name = 'entire_event_observed' if metric == 'entire_event' else metric
                        row[name+'_events'] = len(covered)
                        row[metric+'_uncovered_events'] = len(ids)-len(covered)
                    results.append(row)
    return results

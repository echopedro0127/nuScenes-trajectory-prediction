"""Assemble source clocks and offline rule-candidate coverage, never model inputs."""
import hashlib
from collections import Counter, defaultdict

import numpy as np

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation.query_clock import LEADS, STATUS, TOLERANCE, window_queries

ROLES = ('independent_fit', 'independent_select')


def event_coverage(events, windows):
    grouped = defaultdict(list)
    for w in windows:
        grouped[w['block_id']].append(w)
    summaries, links = [], []
    for e in sorted(events,key=lambda e:e['event_id']):
        first_link = len(links)
        matched, eligible, before = set(), set(), set()
        for w in grouped[e['block_id']]:
            times = w['time_us']
            if times[0] > e['end_us'] or times[-1] < e['onset_us']:
                continue
            eligible.add(w['window_id'])
            for k,q in enumerate(times[5:16],1):
                if q >= e['onset_us']:
                    continue
                before.add(w['window_id'])
                for h in (500000,1000000):
                    if abs((e['onset_us']-q)-h) <= TOLERANCE:
                        matched.add(h)
                        links.append({'event_id':e['event_id'],'window_id':w['window_id'],'query_step':k,
                                      'query_us':q,'pre_lead_us':h,'actual_pre_lead_us':e['onset_us']-q,
                                      'confirmation_covered':e['confirmation_token'] in w['tokens'][5+k:],
                                      'query_has_two_seconds_history':times[0] <= q-2000000,
                                      'label_source':'rule_candidate'})
        own_links = links[first_link:]
        per_lead = {}
        for h in (500000,1000000):
            selected = [r for r in own_links if r['pre_lead_us'] == h]
            per_lead[str(h)] = {'query_links':len(selected),
                                'unique_source_queries':len({r['query_us'] for r in selected}),
                                'windows':len({r['window_id'] for r in selected}),
                                'confirmation_covered_links':sum(r['confirmation_covered'] for r in selected),
                                'insufficient_history_links':sum(not r['query_has_two_seconds_history'] for r in selected),
                                'insufficient_prediction_horizon_links':sum(not r['confirmation_covered'] for r in selected)}
        reasons = ['human_review_not_supplied']
        if not eligible:reasons.append('no_existing_anchor')
        if not matched:reasons.append('no_pre_lead_query')
        if any(not r['query_has_two_seconds_history'] for r in own_links):reasons.append('insufficient_query_history')
        if any(not r['confirmation_covered'] for r in own_links):reasons.append('insufficient_prediction_horizon')
        summaries.append({**{k:e[k] for k in ('event_id','event_type','role','instance','log','category','block_id',
                                             'right_censored','low_end_censored')},
                          'label_source':'rule_candidate','linked_windows':len(eligible),
                          'windows_with_pre_onset_query':len(before),'pre_lead_covered_us':sorted(matched),
                          'pre_lead_match_counts':per_lead,'coverage_reasons':reasons,
                          'status':'query_coverage_available' if matched else 'no_pre_lead_query',
                          'human_review_status':'not_supplied'})
    return summaries, links


def _points(data):
    lookup, used_rows, logs = {}, defaultdict(set), defaultdict(set)
    tracks = {}
    for t in data['tracks']:
        role = t['role']
        if role not in ROLES or role not in data['track_arrays'] or (role,t['instance']) in tracks:
            raise ValueError('invalid or duplicate source track role/identity')
        tracks[role,t['instance']] = t
        logs[role].add(t['log'])
        a = data['track_arrays'][role]
        n = len(a['timestamp_us'])
        if (a['timestamp_us'].dtype != np.dtype('int64') or a['world_xy'].dtype != np.dtype('float64')
                or a['world_xy'].shape != (n,2) or not np.isfinite(a['world_xy']).all()
                or any(a[k].shape != (n,) or a[k].dtype.kind != 'U' for k in
                       ('annotation_token','sample_token','instance_token','segment_id'))):
            raise ValueError('invalid source GT point shape/dtype/coordinates')
        lo,hi = t['role_row_start'],t['role_row_stop']
        if type(lo) is not int or type(hi) is not int or not 0 <= lo < hi <= n:
            raise ValueError('invalid source track bounds')
        times = a['timestamp_us'][lo:hi]
        if (times <= 0).any() or (np.diff(times) <= 0).any():
            raise ValueError('invalid track clock')
        for j in range(lo,hi):
            token = str(a['annotation_token'][j])
            if not token or (role,token) in lookup or j in used_rows[role] or str(a['instance_token'][j]) != t['instance']:
                raise ValueError('duplicate or foreign source point')
            used_rows[role].add(j)
            lookup[role,token] = {**t,'time_us':int(a['timestamp_us'][j]),
                                  'sample':str(a['sample_token'][j]),'segment_id':str(a['segment_id'][j]),'token':token}
    if logs[ROLES[0]] & logs[ROLES[1]]:
        raise ValueError('development role log intersection')
    if set(data['track_arrays']) != set(used_rows) or any(len(used_rows[r]) != len(data['track_arrays'][r]['timestamp_us']) for r in used_rows):
        raise ValueError('incomplete source track coverage')
    return lookup, tracks


def derive_audit(data):
    if not isinstance(data['dataset_release'],str) or not data['dataset_release']:
        raise ValueError('explicit dataset release required')
    points, tracks = _points(data)
    blocks, token_blocks = {}, {}
    grouped_points = defaultdict(list)
    for p in points.values():
        grouped_points[p['role'],p['instance']].append(p)
    for b in data['blocks']:
        if (b['block_id'] in blocks or (b['role'],b['instance']) not in tracks
                or any(type(b[k]) is not int for k in ('start_us','end_us','interval_count'))
                or b['interval_count'] < 0 or not 0 < b['start_us'] <= b['end_us']):
            raise ValueError('invalid source block identity')
        if any(b[k] != tracks[b['role'],b['instance']][k] for k in ('scene','log','category')):
            raise ValueError('block track foreign key mismatch')
        selected = [p for p in grouped_points[b['role'],b['instance']] if b['start_us'] <= p['time_us'] <= b['end_us']]
        if (len(selected) != b['interval_count']+1 or selected[0]['token'] != b['first_annotation']
                or selected[-1]['token'] != b['last_annotation']
                or selected[0]['time_us'] != b['start_us'] or selected[-1]['time_us'] != b['end_us']
                or any(p['segment_id'] != b['segment_id'] for p in selected)):
            raise ValueError('block clock/segment coverage mismatch')
        for p in selected:
            key = b['role'],p['token']
            if key in token_blocks:
                raise ValueError('overlapping source blocks')
            token_blocks[key] = b['block_id']
        blocks[b['block_id']] = b
    if set(token_blocks) != set(points):
        raise ValueError('incomplete block point coverage')
    event_ids = set()
    for e in data['events']:
        b = blocks.get(e['block_id'])
        if (e['event_id'] in event_ids or b is None
                or e['event_type'] not in ('stop_candidate','restart_candidate','stop_go_candidate')
                or any(e[k] != b[k] for k in ('role','instance','scene','log','category'))
                or type(e['right_censored']) is not bool
                or any(type(e[k]) is not int for k in ('onset_us','confirmation_us','end_us'))
                or not e['onset_us'] < e['confirmation_us'] <= e['end_us']):
            raise ValueError('invalid rule event/block identity or clock')
        for field in ('onset','confirmation','end'):
            key = e['role'],e[field+'_token']
            if key not in points or token_blocks[key] != e['block_id'] or points[key]['time_us'] != e[field+'_us']:
                raise ValueError('event token/time foreign key conflict')
        event_ids.add(e['event_id'])

    windows, times, tokens, samples, packets, seen, cache_rows, array_rows = [], [], [], [], {}, set(), set(), defaultdict(set)
    for w in sorted(data['windows'],key=lambda w:(w['role'],w['window_id'])):
        role = w['role']
        expected_id = hashlib.sha256(f"{role}:{w['instance']}:{w['sample']}".encode()).hexdigest()
        i = w['role_window_row']
        if (role not in ROLES or role not in data['window_arrays'] or type(i) is not int or i < 0
                or type(w['row']) is not int or w['row'] < 0 or (role,w['row']) in cache_rows
                or w['window_id'] != expected_id or w['window_id'] in seen
                or i in array_rows[role] or w['contiguous_annotation_segment'] is not True
                or w['status'] != 'ok' or w['reason'] is not None):
            raise ValueError('invalid cached window identity/row/status')
        a = data['window_arrays'][role]
        n = len(a['actual_timestamp_us'])
        if (i >= n or a['actual_timestamp_us'].shape != (n,17) or a['actual_timestamp_us'].dtype != np.dtype('int64')
                or any(a[k].shape != (n,17) or a[k].dtype.kind != 'U' for k in ('annotation_token','sample_token'))
                or a['actual_offset_us'].shape != (n,17) or a['actual_offset_us'].dtype != np.dtype('int64')
                or a['nominal_step'].dtype != np.dtype('int16') or not np.array_equal(a['nominal_step'],np.arange(-4,13))):
            raise ValueError('invalid cached window arrays')
        t = a['actual_timestamp_us'][i]; ts = a['annotation_token'][i]; ss = a['sample_token'][i]
        if (t <= 0).any() or (np.diff(t) <= 0).any() or not np.array_equal(a['actual_offset_us'][i],t-t[4]):
            raise ValueError('invalid cached offset clock')
        block_ids = set()
        for j in range(17):
            key = role,str(ts[j]); p = points.get(key)
            if (p is None or p['time_us'] != int(t[j]) or p['sample'] != str(ss[j])
                    or any(w[k] != p[k] for k in ('instance','scene','log'))):
                raise ValueError('window source token/clock/identity mismatch')
            block_ids.add(token_blocks[key])
            namespace = {'schema':'realclock_source_packet_v1','dataset_release':data['dataset_release'],
                         **{k:p[k] for k in ('role','instance','scene','log')},'annotation_token':str(ts[j])}
            packet = {**namespace,'key_digest':digest(namespace),'source_us':p['time_us']}
            packets[packet['key_digest']] = packet
        if len(block_ids) != 1 or str(ts[4]) != w['anchor_annotation_token'] or str(ss[4]) != w['sample']:
            raise ValueError('window gap or anchor identity mismatch')
        seen.add(w['window_id']);cache_rows.add((role,w['row']));array_rows[role].add(i)
        windows.append({**w,'block_id':next(iter(block_ids)),'time_us':t.tolist(),'tokens':ts.tolist()})
        times.append(t);tokens.append(ts);samples.append(ss)
    if (set(data['window_arrays']) != set(array_rows)
            or any(len(array_rows[r]) != len(data['window_arrays'][r]['actual_timestamp_us']) for r in array_rows)):
        raise ValueError('incomplete cached window coverage')
    clocks = np.stack(times)
    mapping = window_queries(clocks)
    coverage, links = event_coverage(data['events'],windows)
    groups = []
    all_queries,all_targets = set(),set()
    for role in ROLES:
        role_logs = sorted({w['log'] for w in windows if w['role'] == role})
        for log in ['__all__',*role_logs]:
            indices = [i for i,w in enumerate(windows) if w['role'] == role and (log == '__all__' or w['log'] == log)]
            for lead,h in enumerate(LEADS):
                codes = mapping['status_code'][indices,:,lead]
                status_counts = {s:int((codes == j).sum()) for j,s in enumerate(STATUS)}
                query_tokens = {token for i in indices for token in windows[i]['tokens'][5:16]}
                target_tokens = {windows[i]['tokens'][int(mapping['target_index'][i,k,lead])]
                                 for i in indices for k in range(11) if mapping['valid'][i,k,lead]}
                if log == '__all__':
                    all_queries.update((role,t) for t in query_tokens)
                    all_targets.update((role,t) for t in target_tokens)
                eligible_events = [e for e in coverage if e['role'] == role and (log == '__all__' or e['log'] == log)]
                groups.append({'role':role,'log':log,'lead_us':h,'windows':len(indices),
                               'queries':len(indices)*11,'matching_attempts':int(codes.size),
                               'unique_query_source_packets':len(query_tokens),'unique_target_annotations':len(target_tokens),
                               'instances':len({windows[i]['instance'] for i in indices}),
                               'events':len(eligible_events),'status_counts':status_counts,
                               'matched_fraction':status_counts['matched']/codes.size if codes.size else None,
                               'fraction_reason':None if codes.size else 'empty_group'})
    counts = Counter(mapping['status_code'].reshape(-1).tolist())
    summary = {'status':'realclock_query_audit_complete','windows':len(windows),'queries':len(windows)*11,
               'matching_attempts':int(mapping['status_code'].size),'status_counts':{s:counts[j] for j,s in enumerate(STATUS)},
               'events':len(coverage),'events_with_pre_lead_query':sum(bool(e['pre_lead_covered_us']) for e in coverage),
               'source_packets':len(packets),'groups':groups,'label_source':'rule_candidate','human_labels_imported':False,
               'unique_query_source_packets':len(all_queries),'unique_target_annotations':len(all_targets),
               'neural_training':False,'holdout_evaluated':False,'new_prediction_scores_computed':False}
    arrays = {'window_clocks.npz':{'actual_timestamp_us':clocks,'annotation_token':np.stack(tokens),
                                  'sample_token':np.stack(samples)},'query_matches.npz':mapping}
    docs = {'windows.json':windows,'source_packets.json':[packets[k] for k in sorted(packets)],
            'event_coverage.json':coverage,'event_query_links.json':links,'summary.json':summary,
            'compatibility.json':{'old_phase3a':'nominal_dt_v1_only','realclock_learning':'not_adapted',
                                  'realclock_physics':'not_executed','inference_performed':False},
            'noise_namespace.json':{'schema':'realclock_source_packet_v1','noise_seeds':[2027,2028,2029],
                                    'key_fields':['dataset_release','role','scene','log','instance','annotation_token'],
                                    'excluded':['window_id','query_step','strategy','model_seed','event_type'],
                                    'rng_executed':False,'source_packet_manifest':'source_packets.json'}}
    return {'arrays':arrays,'documents':docs}

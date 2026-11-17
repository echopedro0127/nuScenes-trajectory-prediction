"""Token-bound human boundary declarations; immutable rule events remain evidence."""
import copy
import csv
import io
import itertools
from collections import Counter, defaultdict

from trajectory_prediction.evaluation.candidate_verdicts import (
    make_template as verdict_template,
)
from trajectory_prediction.evaluation.candidate_verdicts import (
    source_candidates,
    validate_records,
)
from trajectory_prediction.evaluation.event_review_data import (
    MAX_CELL_CODEPOINTS,
    REVIEW_WHITESPACE,
)

LOCKED = ('source_overlay_identity', 'source_review_identity', 'source_event_id',
          'source_event_type', 'role', 'instance', 'block_id')
TOKENS = ('onset_token', 'confirmation_token', 'end_token', 'low_dwell_end_token')
EDITABLE = ('action', *TOKENS, 'reviewer', 'reason')
COLUMNS = (*LOCKED, *EDITABLE)
PAIR_STATUSES = ('derived_declared_components', 'rejected_components', 'unresolved_components')


def make_template(overlay_identity, review_identity, data, verdicts):
    if not isinstance(overlay_identity, str) or not overlay_identity:
        raise ValueError('explicit overlay identity required')
    checked = validate_records(verdicts, verdict_template(review_identity, data))
    return [dict(source_overlay_identity=overlay_identity,
                 **{k:r[k] for k in LOCKED if k != 'source_overlay_identity'},
                 action='pending', **{k:'' for k in EDITABLE if k != 'action'})
            for r in checked if r['decision'] == 'needs_correction']


def csv_text(rows):
    out = io.StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=COLUMNS, lineterminator='\r\n')
    writer.writeheader()
    writer.writerows(rows)
    return out.getvalue()


def validate_corrections(rows, baseline):
    expected = {r['source_event_id']:r for r in baseline}
    seen = set()
    for r in rows:
        if (not isinstance(r, dict) or set(r) != set(COLUMNS)
                or any(not isinstance(v, str) or len(v) > MAX_CELL_CODEPOINTS for v in r.values())):
            raise ValueError('exact bounded string correction columns required')
        key = r['source_event_id']
        if key not in expected or key in seen or any(r[k] != expected[key][k] for k in LOCKED):
            raise ValueError('correction locked identity/coverage mismatch')
        if r['action'] == 'pending':
            if any(r[k] != '' for k in (*TOKENS, 'reviewer', 'reason')):
                raise ValueError('pending correction fields must be empty')
        elif r['action'] == 'propose':
            if any(not r[k].strip(REVIEW_WHITESPACE) for k in ('reviewer', 'reason')):
                raise ValueError('proposal requires reviewer and reason')
            required = TOKENS if r['source_event_type'] == 'stop_candidate' else TOKENS[:3]
            if any(not r[k] for k in required):
                raise ValueError('proposal requires complete applicable tokens')
            if r['source_event_type'] == 'restart_candidate' and r['low_dwell_end_token']:
                raise ValueError('restart cannot declare low dwell token')
        else:
            raise ValueError('unknown correction action')
        seen.add(key)
    if seen != set(expected):
        raise ValueError('missing eligible correction rows')
    return sorted(rows, key=lambda r:r['source_event_id'])


def parse_corrections(text, baseline):
    reader = csv.DictReader(io.StringIO(text.lstrip('\ufeff'), newline=''), strict=True)
    try:
        rows = list(reader)
    except csv.Error as error:
        raise ValueError('malformed correction CSV') from error
    if reader.fieldnames != list(COLUMNS):
        raise ValueError('exact correction CSV header required')
    return validate_corrections(rows, baseline)


def _clocks(data, atomic):
    clocks = {}
    for b in data['blocks']:
        tokens, times = b['annotation_tokens'], b['time_us']
        if (not isinstance(tokens, list) or not isinstance(times, list) or len(tokens) != len(times)
                or len(tokens) < 1 or any(not isinstance(t, str) or not t for t in tokens)
                or len(set(tokens)) != len(tokens)
                or any(type(t) is not int or t <= 0 for t in times)
                or any(a >= z for a,z in itertools.pairwise(times))
                or times[0] != b['start_us'] or times[-1] != b['end_us']):
            raise ValueError('invalid correction block token clock')
        clocks[b['block_id']] = dict(zip(tokens, times, strict=True))
    for e in atomic.values():
        clock = clocks[e['block_id']]
        names = TOKENS if e['event_type'] == 'stop_candidate' else TOKENS[:3]
        for name in names:
            if clock.get(e[name]) != e[name.removesuffix('_token')+'_us']:
                raise ValueError('original event token/time mismatch')
    return clocks


def _correct(event, row, clock, block):
    result = copy.deepcopy(event)
    result['status'] = 'corrected_declared'
    names = TOKENS if event['event_type'] == 'stop_candidate' else TOKENS[:3]
    for name in names:
        token = row[name]
        if token not in clock:
            raise ValueError('foreign or cross-block correction token')
        result[name] = token
        result[name.removesuffix('_token')+'_us'] = clock[token]
    if not result['onset_us'] < result['confirmation_us'] <= result['end_us']:
        raise ValueError('correction boundary order conflict')
    if event['right_censored'] and result['end_us'] != block['end_us']:
        raise ValueError('cannot truncate censored observation boundary')
    if event['event_type'] == 'stop_candidate':
        if not result['confirmation_us'] <= result['low_dwell_end_us'] <= result['end_us']:
            raise ValueError('low dwell boundary order conflict')
        if event['low_end_censored'] and result['low_dwell_end_us'] != block['end_us']:
            raise ValueError('cannot truncate low censored observation boundary')
        result['episode_end_us'], result['episode_end_token'] = result['end_us'], result['end_token']
    else:
        if result['confirmation_us'] != result['end_us']:
            raise ValueError('restart confirmation must equal end')
    # The event_id identifies the immutable source, not this changed boundary view.
    return result


def derive_corrections(overlay_identity, review_identity, data, verdicts, rows):
    baseline = make_template(overlay_identity, review_identity, data, verdicts)
    rows = validate_corrections(rows, baseline)
    atomic, pairs, _ = source_candidates(data)
    clocks = _clocks(data, atomic)
    blocks = {b['block_id']:b for b in data['blocks']}
    proposals = {r['source_event_id']:r for r in rows if r['action'] == 'propose'}
    decisions = {r['source_event_id']:r['decision'] for r in verdicts}
    effective, events = {}, []
    for key, source in sorted(atomic.items()):
        e = (_correct(source, proposals[key], clocks[source['block_id']], blocks[source['block_id']])
             if key in proposals else copy.deepcopy(source))
        effective[key] = e
        events.append({'source_event': copy.deepcopy(source), 'effective_event': e,
                           'status': 'corrected_declared' if key in proposals else decisions[key],
                           'correction': copy.deepcopy(proposals.get(key))})
    derived = []
    for p in pairs:
        s, r = effective[p['stop_id']], effective[p['restart_id']]
        if (s['end_us'] != r['onset_us'] or r['confirmation_us'] != r['end_us']
                or s['end_token'] != r['onset_token']):
            raise ValueError('correction composite relationship conflict')
        view = copy.deepcopy(p)
        for field, child in (('onset', s), ('confirmation', r), ('end', r)):
            view[field+'_us'], view[field+'_token'] = child[field+'_us'], child[field+'_token']
        statuses = [('corrected_declared' if k in proposals else decisions[k])
                    for k in (p['stop_id'], p['restart_id'])]
        usable = all(v in ('corrected_declared', 'confirmed') for v in statuses)
        status = ('rejected_components' if 'rejected' in statuses else
                  'derived_declared_components' if usable else 'unresolved_components')
        view['status'] = status
        derived.append({'source_pair': copy.deepcopy(p), 'effective_pair': view,
                            'component_statuses': statuses,
                            'status': status})
    by_block = defaultdict(list)
    for key,e in effective.items():
        # A rejected candidate remains evidence, but is not an effective event.
        if decisions[key] != 'rejected':
            by_block[e['block_id']].append(e)
    for values in by_block.values():
        ordered = sorted(values, key=lambda e:(e['onset_us'], e['end_us'], e['event_id']))
        if any(a['end_us'] > b['onset_us'] for a,b in itertools.pairwise(ordered)):
            raise ValueError('effective event overlap conflict')

    def counts(selected):
        proposed = sum(r['action'] == 'propose' for r in selected)
        censored = sum(atomic[r['source_event_id']]['right_censored'] for r in selected)
        return {'eligible': len(selected), 'pending': len(selected)-proposed, 'proposed': proposed,
                    'right_censored': censored, 'proposed_fraction': proposed/len(selected) if selected else None,
                    'low_end_censored': sum(atomic[r['source_event_id']].get('low_end_censored') is True for r in selected),
                    'fraction_status': 'defined' if selected else 'not_evaluable',
                    'fraction_reason': None if selected else 'no_eligible_corrections'}

    groups, pair_groups = [], []
    for role in ('independent_fit', 'independent_select'):
        for category in ['__all__', *sorted({b['category'] for b in data['blocks'] if b['role'] == role})]:
            for kind in ('stop_candidate', 'restart_candidate'):
                selected = [r for r in rows if r['role'] == role and r['source_event_type'] == kind
                            and (category == '__all__' or atomic[r['source_event_id']]['category'] == category)]
                groups.append(dict(role=role, category=category, event_type=kind, **counts(selected)))
            selected_pairs = [p for p in derived if p['source_pair']['role'] == role
                              and (category == '__all__' or p['source_pair']['category'] == category)]
            pair_counts = Counter(p['status'] for p in selected_pairs)
            pair_groups.append({'role':role, 'category':category, 'pair_total':len(selected_pairs),
                                'status_counts':{k:pair_counts[k] for k in PAIR_STATUSES}})
    n = len(proposals)
    pair_counts = Counter(p['status'] for p in derived)
    summary = dict(counts(rows), status='no_eligible_corrections' if not rows else
                   'corrections_declared' if n else 'no_boundary_declarations',
                   source_atomic_total=len(atomic), source_pair_total=len(pairs),
                   pair_status_counts={k:pair_counts[k] for k in PAIR_STATUSES},
                   low_end_censored=sum(atomic[r['source_event_id']].get('low_end_censored') is True for r in rows),
                   human_boundary_declarations_imported=bool(n), official_event_ground_truth=False,
                   exhaustive_event_annotation=False, neural_training=False, holdout_evaluated=False,
                   new_prediction_scores_computed=False, groups=groups, pair_groups=pair_groups)
    return {'corrections': copy.deepcopy(rows), 'effective_events': events,
                'derived_pairs': derived, 'summary': summary}

"""Synthetic declarations only; never human labels or scientific results."""
import copy

import pytest


@pytest.fixture
def correction_data():
    common = {'role': 'independent_fit', 'instance': 'i', 'block_id': 'b', 'scene': 'scene',
                  'log': 'log', 'category': 'vehicle.car'}
    stop = dict(common, event_id='s', event_type='stop_candidate', onset_us=1000000,
                confirmation_us=2000000, end_us=4000000, right_censored=False,
                low_end_censored=False, low_dwell_end_us=3000000)
    restart = dict(common, event_id='r', event_type='restart_candidate', onset_us=4000000,
                   confirmation_us=5000000, end_us=5000000, right_censored=False, stop_id='s')
    pair = dict(common, event_id='p', event_type='stop_go_candidate', onset_us=1000000,
                confirmation_us=5000000, end_us=5000000, right_censored=False,
                stop_id='s', restart_id='r')
    other = dict(stop, event_id='z', block_id='c', instance='j', role='independent_select',
                 right_censored=True, low_end_censored=True, low_dwell_end_us=6000000,
                 end_us=6000000)
    events = [stop, restart, pair, other]
    for e in events:
        for field in ('onset', 'confirmation', 'end'):
            e[field+'_token'] = e['block_id']+str(e[field+'_us']//500000)
        if e['event_type'] == 'stop_candidate':
            e['low_dwell_end_token'] = e['block_id']+str(e['low_dwell_end_us']//500000)
            e['episode_end_us'], e['episode_end_token'] = e['end_us'], e['end_token']
    blocks = [dict(common, start_us=500000, end_us=6000000,
                   annotation_tokens=['b'+str(i) for i in range(1, 13)],
                   time_us=[i*500000 for i in range(1, 13)]),
              dict(common, role='independent_select', instance='j', block_id='c',
                   start_us=500000, end_us=6000000,
                   annotation_tokens=['c'+str(i) for i in range(1, 13)],
                   time_us=[i*500000 for i in range(1, 13)])]
    queue = [dict(task_id='task-'+e['event_id'], task_kind='event_check',
                  **{k:e[k] for k in ('role', 'instance', 'block_id', 'event_id')}) for e in events]
    return {'events': events, 'blocks': blocks, 'queue': queue}


def verdicts(data, decisions=None):
    from trajectory_prediction.evaluation.candidate_verdicts import make_template
    rows = make_template('review', data)
    for r in rows:
        d = (decisions or {}).get(r['source_event_id'], 'needs_correction')
        r.update(decision=d, reviewer='' if d == 'pending' else 'fixture-reader',
                 reason='' if d == 'pending' else 'Synthetic declaration')
    return rows


def propose(rows, key, **fields):
    r = next(r for r in rows if r['source_event_id'] == key)
    block = r['block_id']
    default = {'onset_token': block+'2', 'confirmation_token': block+'4', 'end_token': block+'8',
                   'low_dwell_end_token': block+'6'}
    if key == 'r':
        default.update(onset_token='b8', confirmation_token='b10', end_token='b10',
                       low_dwell_end_token='')
    if key == 'z':
        default.update(end_token='c12', low_dwell_end_token='c12')
    r.update(action='propose', reviewer='fixture-reader', reason='Synthetic boundary correction',
             **(default | fields))


def api(data, decisions=None):
    from trajectory_prediction.evaluation import boundary_corrections as mod
    v = verdicts(data, decisions)
    base = mod.make_template('overlay', 'review', data, v)
    return mod, v, base


def test_only_needs_correction_and_empty_denominator(correction_data):
    mod, v, rows = api(correction_data, {'s':'pending', 'r':'confirmed', 'z':'needs_correction'})
    assert [r['source_event_id'] for r in rows] == ['z']
    v = verdicts(correction_data, {'s':'pending', 'r':'pending', 'z':'pending'})
    assert mod.make_template('overlay', 'review', correction_data, v) == []
    result = mod.derive_corrections('overlay', 'review', correction_data, v, [])
    assert result['summary']['status'] == 'no_eligible_corrections'
    assert result['summary']['eligible'] == result['summary']['proposed'] == 0
    assert result['summary']['proposed_fraction'] is None


def test_proposal_uses_real_tokens_preserves_source_and_censor(correction_data):
    mod, v, rows = api(correction_data)
    before = copy.deepcopy(correction_data)
    propose(rows, 'z', onset_token='c3', confirmation_token='c5')
    result = mod.derive_corrections('overlay', 'review', correction_data, v, rows)
    event = next(e for e in result['effective_events'] if e['source_event']['event_id'] == 'z')
    assert event['status'] == 'corrected_declared'
    assert event['effective_event']['onset_us'] == 1500000
    assert event['effective_event']['right_censored'] is True
    assert event['effective_event']['low_end_censored'] is True
    assert event['source_event']['onset_us'] == 1000000
    assert correction_data == before
    assert result['summary']['eligible'] == 3 and result['summary']['proposed'] == 1
    assert result['summary']['pending'] == 2


def test_corrected_view_does_not_claim_rule_confirmed_boundaries(correction_data):
    correction_data['events'][3]['status'] = 'confirmed_by_rule'
    mod, v, rows = api(correction_data)
    propose(rows, 'z', onset_token='c3', confirmation_token='c5')
    result = mod.derive_corrections('overlay', 'review', correction_data, v, rows)
    event = next(e for e in result['effective_events'] if e['source_event']['event_id'] == 'z')
    assert event['source_event']['status'] == 'confirmed_by_rule'
    assert event['effective_event']['status'] == 'corrected_declared'


def test_group_denominators_preserve_all_eligible_rows(correction_data):
    mod, v, rows = api(correction_data)
    propose(rows, 'z')
    summary = mod.derive_corrections('overlay', 'review', correction_data, v, rows)['summary']
    groups = [g for g in summary['groups'] if g['category'] == '__all__']
    for field in ('eligible', 'pending', 'proposed', 'right_censored', 'low_end_censored'):
        assert sum(g[field] for g in groups) == summary[field]


@pytest.mark.parametrize('stop,restart,expected', [
    ('rejected', 'confirmed', 'rejected_components'),
    ('pending', 'rejected', 'rejected_components'),
    ('confirmed', 'confirmed', 'derived_declared_components'),
    ('uncertain', 'confirmed', 'unresolved_components'),
])
def test_relation_outcomes_and_group_denominators(correction_data, stop, restart, expected):
    mod, v, rows = api(correction_data, {'s':stop, 'r':restart, 'z':'pending'})
    result = mod.derive_corrections('overlay', 'review', correction_data, v, rows)
    summary = result['summary']
    assert result['derived_pairs'][0]['status'] == expected
    assert result['derived_pairs'][0]['effective_pair']['status'] == expected
    assert summary['pair_status_counts'][expected] == 1
    groups = [g for g in summary['pair_groups'] if g['category'] == '__all__']
    assert sum(g['pair_total'] for g in groups) == summary['source_pair_total']
    for status, count in summary['pair_status_counts'].items():
        assert sum(g['status_counts'][status] for g in groups) == count


def test_paired_edits_and_unresolved_components(correction_data):
    mod, v, rows = api(correction_data)
    propose(rows, 's', end_token='b9')
    propose(rows, 'r', onset_token='b9', confirmation_token='b11', end_token='b11')
    result = mod.derive_corrections('overlay', 'review', correction_data, v, rows)
    pair = result['derived_pairs'][0]
    assert pair['status'] == 'derived_declared_components'
    assert pair['effective_pair']['end_us'] == 5500000
    assert pair['source_pair']['end_us'] == 5000000
    rows[0]  # row order is deliberately unrelated to pair order
    pending = mod.make_template('overlay', 'review', correction_data, v)
    assert mod.derive_corrections('overlay', 'review', correction_data, v, pending)['derived_pairs'][0]['status'] == 'unresolved_components'


@pytest.mark.parametrize('kind', ['foreign_token', 'wrong_block', 'equal', 'reverse', 'low_after_end',
                                 'right_censor', 'low_censor', 'restart_low', 'pair_conflict',
                                 'locked', 'missing', 'duplicate', 'extra', 'pending_note',
                                 'blank_reviewer', 'unknown_action', 'long', 'ineligible'])
def test_invalid_corrections_rejected(correction_data, kind):
    mod, v, rows = api(correction_data)
    propose(rows, 's')
    s = next(r for r in rows if r['source_event_id'] == 's')
    if kind == 'foreign_token': s['onset_token'] = 'missing'
    elif kind == 'wrong_block': s['onset_token'] = 'c2'
    elif kind == 'equal': s['confirmation_token'] = s['onset_token']
    elif kind == 'reverse': s['onset_token'] = 'b9'
    elif kind == 'low_after_end': s['low_dwell_end_token'] = 'b10'
    elif kind == 'right_censor': propose(rows, 'z', end_token='c11')
    elif kind == 'low_censor': propose(rows, 'z', low_dwell_end_token='c11')
    elif kind == 'restart_low': propose(rows, 'r', low_dwell_end_token='b9')
    elif kind == 'pair_conflict': s['end_token'] = 'b9'
    elif kind == 'locked': s['role'] = 'independent_select'
    elif kind == 'missing': rows.pop()
    elif kind == 'duplicate': rows.append(dict(rows[0]))
    elif kind == 'extra': s['unexpected'] = ''
    elif kind == 'pending_note': rows = mod.make_template('overlay', 'review', correction_data, v); rows[0]['reason'] = ' '
    elif kind == 'blank_reviewer': s['reviewer'] = '\ufeff'
    elif kind == 'unknown_action': s['action'] = 'confirmed'
    elif kind == 'long': s['reason'] = 'a'*8193
    elif kind == 'ineligible': v = verdicts(correction_data, {'s':'confirmed'})
    with pytest.raises(ValueError):
        mod.derive_corrections('overlay', 'review', correction_data, v, rows)


def test_csv_order_bom_multiline_and_exact_schema(correction_data):
    mod, _, rows = api(correction_data)
    propose(rows, 'z'); rows[-1]['reason'] = 'Synthetic, "quoted"\nreason'
    assert mod.parse_corrections('\ufeff'+mod.csv_text(rows[::-1]), mod.make_template('overlay', 'review', correction_data, verdicts(correction_data))) == rows
    with pytest.raises(ValueError): mod.parse_corrections(mod.csv_text(rows)+'"unclosed', rows)
    with pytest.raises(ValueError): mod.parse_corrections('bad\n', rows)


def test_singleton_block_is_valid_evidence_but_cannot_supply_event_tokens(correction_data):
    # Real B2a includes blocks with one point and no speed interval.
    correction_data['blocks'].append(dict(correction_data['blocks'][0], block_id='singleton',
                                        annotation_tokens=['only'], time_us=[500000],
                                        start_us=500000, end_us=500000))
    mod, v, rows = api(correction_data)
    assert mod.derive_corrections('overlay', 'review', correction_data, v, rows)['summary']['pending'] == 3


@pytest.mark.parametrize('kind', ['duplicate_token', 'backwards_clock', 'wrong_endpoint', 'original_token_time'])
def test_invalid_source_clock_rejected(correction_data, kind):
    mod, v, rows = api(correction_data)
    propose(rows, 's')
    b = correction_data['blocks'][0]
    if kind == 'duplicate_token': b['annotation_tokens'][1] = b['annotation_tokens'][0]
    elif kind == 'backwards_clock': b['time_us'][1] = b['time_us'][0]
    elif kind == 'wrong_endpoint': b['end_us'] += 1
    else: correction_data['events'][0]['onset_token'] = 'b3'
    with pytest.raises(ValueError): mod.derive_corrections('overlay', 'review', correction_data, v, rows)


@pytest.mark.parametrize('touch', [True, False])
def test_unrelated_overlap_is_rejected_but_touch_allowed(correction_data, touch):
    # Add an independent, complete synthetic stop/restart pair to the first block.
    stop, restart, pair = copy.deepcopy(correction_data['events'][:3])
    for e, key in zip((stop, restart, pair), ('s3', 'r3', 'p3'), strict=True): e['event_id'] = key
    restart['stop_id'] = pair['stop_id'] = 's3'; pair['restart_id'] = 'r3'
    for e in (stop, restart, pair):
        for f in ('onset', 'confirmation', 'end'):
            e[f+'_us'] += 500000; e[f+'_token'] = 'b'+str(e[f+'_us']//500000)
    stop['low_dwell_end_us'] += 500000; stop['low_dwell_end_token'] = 'b7'
    correction_data['events'] += [stop, restart, pair]
    correction_data['queue'] += [dict(task_id='task-'+e['event_id'], task_kind='event_check',
                                    **{k:e[k] for k in ('role', 'instance', 'block_id', 'event_id')}) for e in (stop,restart,pair)]
    mod, v, rows = api(correction_data)
    propose(rows, 's', onset_token='b1', confirmation_token='b2', low_dwell_end_token='b3', end_token='b4')
    propose(rows, 'r', onset_token='b4', confirmation_token='b5', end_token='b5')
    propose(rows, 's3', onset_token='b5' if touch else 'b4', confirmation_token='b6',
            low_dwell_end_token='b7', end_token='b9')
    propose(rows, 'r3', onset_token='b9', confirmation_token='b11', end_token='b11', low_dwell_end_token='')
    if touch:
        assert mod.derive_corrections('overlay', 'review', correction_data, v, rows)['summary']['proposed'] == 4
    else:
        with pytest.raises(ValueError, match='overlap'):
            mod.derive_corrections('overlay', 'review', correction_data, v, rows)

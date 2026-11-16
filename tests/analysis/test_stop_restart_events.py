import pytest
from test_realclock_motion import config

from trajectory_prediction.analysis.track_timeline import digest

ORIGIN = 1000000


def case(speeds, dts=None, **changes):
    cfg = config(**changes)
    dts = dts if dts is not None else [500000]*len(speeds)
    rows, now = [], ORIGIN
    for i, (v, dt) in enumerate(zip(speeds, dts, strict=True)):
        rows.append({'interval_id': f'i{i}', 'block_id': 'block', 'valid': True, 'reason': None,
                     'left_token': f'a{i}', 'right_token': f'a{i+1}', 'start_us': now,
                     'end_us': now+dt, 'dt_us': dt, 'speed_mps': v})
        now += dt
    block = {'block_id': 'block', 'segment_id': 'segment', 'role': 'independent_fit',
             'instance': 'instance', 'scene': 'scene', 'log': 'log', 'category': 'vehicle.car',
             'dataset_release': 'v1.0-trainval', 'track_semantic_sha256': 'track',
             'config_sha256': digest(cfg), 'first_annotation': 'a0', 'last_annotation': f'a{len(speeds)}',
             'start_us': ORIGIN, 'end_us': now, 'interval_count': len(rows)}
    return block, rows, cfg


def run(speeds, dts=None, **changes):
    from trajectory_prediction.analysis.stop_restart_events import extract_events
    return extract_events(*case(speeds, dts, **changes))


def by_type(result, kind):
    return [e for e in result['events'] if e['event_type'] == kind+'_candidate']


def test_stop_creep_cancel_restart_exact_boundaries():
    result = run([2.]*4+[0.]*2+[.5, 1.2, .5, 1.2, 1.2])
    stop, restart, pair = [by_type(result, k)[0] for k in ('stop', 'restart', 'stop_go')]
    assert (stop['onset_us'], stop['confirmation_us'], stop['low_dwell_end_us'], stop['end_us']) == (
        ORIGIN+2000000, ORIGIN+3000000, ORIGIN+3000000, ORIGIN+4500000)
    assert restart['onset_us'] == ORIGIN+4500000
    assert restart['confirmation_us'] == restart['end_us'] == ORIGIN+5500000
    assert pair['onset_us'] == stop['onset_us']
    assert pair['end_us'] == pair['confirmation_us'] == restart['end_us']
    assert pair['stop_id'] == stop['event_id'] and pair['restart_id'] == restart['event_id']
    assert [d['reason'] for d in result['diagnostics']] == ['short_restart_run']
    assert len(result['state_trace']) == 11


def test_initial_low_restart_is_diagnostic_only_and_consumed_once():
    result = run([0., 0.]+[1.2]*5)
    assert result['events'] == []
    assert [d['reason'] for d in result['diagnostics']] == [
        'insufficient_prior_history', 'restart_without_confirmed_stop']
    assert result['diagnostics'][-1]['context_origin'] == 'initially_low_speed'


def test_stop_right_censoring():
    stop = by_type(run([2.]*4+[0.]*3), 'stop')[0]
    assert stop['end_us'] == ORIGIN+3500000
    assert stop['right_censored'] is True and stop['low_end_censored'] is True


def test_two_intervals_shorter_than_real_second_not_confirmed():
    result = run([2.]*4+[0.]*2, [500000]*4+[499883]*2)
    assert result['events'] == []
    assert result['diagnostics'][0]['reason'] == 'insufficient_followup'


def test_inclusive_speed_boundaries_and_no_duplicate_stop():
    result = run([1.]*4+[.3]*2+[.5]+[.3]*2+[1.]*2)
    assert len(by_type(result, 'stop')) == len(by_type(result, 'restart')) == 1
    assert 'low_run_with_existing_stop_context' in [d['reason'] for d in result['diagnostics']]


def test_multiple_stop_restart_cycles():
    result = run([2.]*4+[0.]*2+[2.]*2+[0.]*2+[2.]*2)
    assert [len(by_type(result, k)) for k in ('stop', 'restart', 'stop_go')] == [2, 2, 2]


def test_no_recent_motion_and_unpaired_tail():
    result = run([.5]*4+[0.]*2+[1.2])
    assert result['events'] == []
    assert [d['reason'] for d in result['diagnostics']] == [
        'no_recent_motion_evidence', 'insufficient_followup_restart_without_confirmed_stop']


def test_short_low_and_no_low_context():
    assert run([2.]*8)['diagnostics'] == []
    result = run([2.]*4+[0., 2., 2.])
    assert result['events'] == []
    assert result['diagnostics'][0]['reason'] == 'short_low_speed_run'


def test_semantic_order_and_config_identity():
    from trajectory_prediction.analysis.stop_restart_events import extract_events
    b, rows, cfg = case([2.]*4+[0.]*2+[2.]*2)
    a = extract_events(b, rows, cfg)
    assert extract_events(b, list(reversed(rows)), cfg) == a
    changed = run([2.]*4+[0.]*2+[2.]*2, stop_speed_mps=.2)
    assert a['events'][0]['event_id'] != changed['events'][0]['event_id']


def test_partial_prior_interval_excluded_and_diagnostic_keeps_evidence():
    result = run([2., .5, .5, .5, 0., 0.], [500000, 500000, 500000, 700000, 500000, 500000])
    assert result['events'] == []
    assert result['diagnostics'][0]['reason'] == 'no_recent_motion_evidence'
    assert result['diagnostics'][0]['prior_interval_ids'] == ['i1', 'i2', 'i3']


def test_initial_short_low_run_retains_boundary_flag():
    assert run([0.])['diagnostics'][0]['initially_low_speed'] is True


@pytest.mark.parametrize('field,value', [('speed_mps', float('nan')), ('speed_mps', -1.),
                                      ('dt_us', 0), ('left_token', 'broken')])
def test_invalid_interval_rejected(field, value):
    from trajectory_prediction.analysis.stop_restart_events import extract_events
    b, rows, cfg = case([2., 0.])
    rows[-1][field] = value
    with pytest.raises(ValueError):
        extract_events(b, rows, cfg)

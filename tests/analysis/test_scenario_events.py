"""Window-only offline GT rules; no predictor inputs or canonical episodes."""
import numpy as np
import pytest

from trajectory_prediction.analysis.scenario_events import (
    annotate_window,
    build_review_manifest,
    turn_shape_tags,
    update_review,
)


def trajectory(speeds):
    speeds = np.asarray(speeds, float)
    assert speeds.shape == (16,)
    x = np.r_[0., np.cumsum(speeds * .5)] - speeds[:4].sum() * .5
    xy = np.column_stack((x, np.zeros(17)))
    return xy[:5], xy[5:], np.ones(5, bool)


def test_s_turn_not_lost_by_net_yaw_cancellation():
    angle = np.radians([0, 10, 20, 30, 20, 10, 0, 0, 0, 0, 0, 0])
    future = np.cumsum(np.column_stack((np.cos(angle), np.sin(angle))), axis=0)
    assert 's_turn_candidate' in turn_shape_tags(future)


def test_stop_breaks_turn_segments():
    angle = np.radians([0, 15, 30, 0, 0, 30, 15, 0, 0, 0, 0, 0])
    delta = np.column_stack((np.cos(angle), np.sin(angle)))
    delta[3:5] = 0
    points = np.vstack((np.zeros((1, 2)), np.cumsum(delta, axis=0)))
    assert 's_turn_candidate' not in turn_shape_tags(points)


@pytest.mark.parametrize('angles,expected', [
    ([0, 30, 60, 90, 120, 150], True),
    ([170, 175, -179, -175, -170, -165], False),
])
def test_u_turn_and_angle_wrap(angles, expected):
    theta = np.radians(angles)
    xy = np.vstack((np.zeros((1, 2)), np.cumsum(np.column_stack((np.cos(theta), np.sin(theta))), axis=0)))
    assert ('u_turn_candidate' in turn_shape_tags(xy)) is expected


def test_stationary_start_is_not_a_new_stop():
    r = annotate_window(*trajectory([0.] * 16))
    assert r['initial_state'] == 'initially_stationary'
    assert 'stop_event' not in r['labels']
    assert r['tag_status']['s_turn_candidate']['status'] == 'not_evaluable'


def test_brake_stop_creep_and_restart_have_interval_evidence():
    h, f, mask = trajectory([3.] * 4 + [3, 2, 1, 0, 0, .5, .5, 1.5, 1.5, 1.5, 1.5, 1.5])
    r = annotate_window(h, f, mask)
    assert {'hard_brake_candidate', 'stop_event', 'stop_go_candidate'} <= set(r['labels'])
    stop = next(e for e in r['events'] if e['tag'] == 'stop_event' and e['status'] == 'candidate')
    assert stop['onset_s'] == 1.5 and stop['confirmation_s'] == 2.5
    assert 'creeping' in r['speed_states']
    assert all(e['end_s'] <= 6 for e in r['events'])


def test_single_short_stop_does_not_pass_duration_rule():
    speeds = [3.] * 16
    speeds[8] = 0
    r = annotate_window(*trajectory(speeds))
    assert 'stop_event' not in r['labels']


def test_terminal_stop_is_censored_not_confirmed_by_boundary_padding():
    r = annotate_window(*trajectory([3.] * 15 + [0.]))
    assert 'stop_event' not in r['labels']
    assert r['tag_status']['stop_event']['status'] == 'not_evaluable'
    assert any(e['reason'] == 'insufficient_followup' for e in r['events'])


def test_missing_prior_history_is_not_filled_with_padding():
    h, f, mask = trajectory([3.] * 4 + [0.] * 4 + [3.] * 8)
    mask[:4] = False
    r = annotate_window(h, f, mask)
    assert 'stop_event' not in r['labels']
    assert any(e['tag'] == 'stop_event' and e['reason'] == 'insufficient_prior_history'
               for e in r['events'])


def test_restart_after_initial_stationary_is_left_censored():
    r = annotate_window(*trajectory([0.] * 8 + [2.] * 8))
    assert 'stop_go_candidate' not in r['labels']
    assert r['tag_status']['stop_go_candidate']['status'] == 'not_evaluable'
    assert any(e['reason'] == 'no_confirmed_stop_in_window' for e in r['events'])


def test_creep_below_restart_threshold_is_not_restart():
    r = annotate_window(*trajectory([3.] * 4 + [3, 2, 1, 0, 0, .5, .5, .5, .5, .5, .5, .5]))
    assert 'stop_event' in r['labels'] and 'stop_go_candidate' not in r['labels']
    assert 'creeping' in r['speed_states']


def test_thresholds_and_timestamps_match_speed_intervals():
    r = annotate_window(*trajectory([2.] * 4 + [2, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]))
    assert 'hard_brake_candidate' in r['labels']
    assert r['trace']['interval_start_s'] == [-2 + i * .5 for i in range(16)]
    assert len(r['trace']['raw_speed_mps']) == 16
    assert r['trace']['acceleration_mps2'][0] is None


@pytest.mark.parametrize('kind', ['shape', 'nan', 'mask', 'dtype'])
def test_invalid_gt_is_rejected(kind):
    h, f, mask = trajectory([2.] * 16)
    if kind == 'shape':
        f = f[:11]
    elif kind == 'nan':
        f[0, 0] = np.nan
    elif kind == 'mask':
        mask = mask.astype(int)
    else:
        h = h.astype(int)
    with pytest.raises(ValueError):
        annotate_window(h, f, mask)


def test_review_selection_is_deterministic_capped_and_unreviewed():
    windows = [{'window_id': str(i), 'role': 'independent_select', 'instance': 'i' + str(i),
                'sample': 's' + str(i), 'log': 'log', 'labels': ['stop_event'],
                'rule_version': 'scenario_window_events_v1'} for i in range(25)]
    a = build_review_manifest(windows)
    assert a == build_review_manifest(windows[::-1])
    assert len(a) == 20 and all(r['review_status'] == 'pending' for r in a)
    assert all(r['rule_label'] == 'stop_event' and r['manual_label'] == '' for r in a)
    with pytest.raises(ValueError, match='duplicate'):
        build_review_manifest(windows + windows[:1])


def test_review_revision_keeps_original_rule_and_reviewer_history():
    row = build_review_manifest([{'window_id': 'w', 'role': 'independent_fit', 'instance': 'i',
                                  'sample': 's', 'log': 'l', 'labels': ['stop_event'],
                                  'rule_version': 'scenario_window_events_v1'}])[0]
    changed = update_review(row, status='corrected', reviewer='reviewer', reason='short stop', manual_label='none')
    assert changed['rule_label'] == 'stop_event' and changed['manual_label'] == 'none'
    assert row['review_status'] == 'pending' and len(changed['review_history']) == 1
    with pytest.raises(ValueError):
        update_review(row, status='confirmed', reviewer='', reason='')


def test_failed_prior_speed_precondition_is_not_a_terminal_unknown_stop():
    r = annotate_window(*trajectory([.5] * 15 + [0.]))
    assert r['tag_status']['stop_event']['status'] == 'absent'


def test_manual_label_must_be_a_known_text_label():
    row = build_review_manifest([{'window_id': 'w', 'role': 'independent_fit', 'instance': 'i',
                                  'sample': 's', 'log': 'l', 'labels': ['stop_event'],
                                  'rule_version': 'scenario_window_events_v1'}])[0]
    with pytest.raises(ValueError):
        update_review(row, status='corrected', reviewer='r', reason='x', manual_label=True)


def test_exact_u_turn_threshold_uses_wrapped_displacement_directions():
    points = np.vstack((np.zeros((1, 2)), np.cumsum([[1., 0.], [0., 1.], [-1., 1.]], axis=0)))
    assert 'u_turn_candidate' in turn_shape_tags(points)


def test_restart_after_unqualified_stop_is_not_a_negative_stop_go():
    r = annotate_window(*trajectory([.5] * 4 + [0.] * 4 + [2.] * 8))
    assert 'stop_event' not in r['labels']
    assert r['tag_status']['stop_go_candidate']['status'] == 'not_evaluable'
    assert any(e['reason'] == 'no_confirmed_stop_in_window' for e in r['events'])


def test_single_terminal_restart_has_one_censored_event():
    r = annotate_window(*trajectory([3.] * 13 + [0., 0., 2.]))
    events = [e for e in r['events'] if e['tag'] == 'stop_go_candidate']
    assert len(events) == 1 and events[0]['reason'] == 'insufficient_followup'
    assert r['tag_status']['stop_go_candidate']['incomplete_candidate_count'] == 1

import numpy as np
import pytest


def event(**extra):
    return {'event_id': 'e', 'event_type': 'stop_candidate', 'role': 'independent_fit', 'instance': 'i',
                'category': 'vehicle.car', 'onset_us': 12000000, 'confirmation_us': 13000000,
                'end_us': 13500000, 'right_censored': True} | extra


def window(first=10000000, anchor=12000000, last=13000000, pilot=True, name='w'):
    times = np.concatenate([np.linspace(first, anchor, 5, dtype=np.int64),
                            np.linspace(anchor, last, 13, dtype=np.int64)[1:]])
    return [{'window_id': name, 'role': 'independent_fit', 'instance': 'i',
             'role_window_row': 0, 'phase3a_pilot_member': pilot}], {
                 'independent_fit': {'actual_timestamp_us': times[None]}}


def test_right_censored_confirmation_covered_but_not_entire_event():
    from trajectory_prediction.evaluation.event_windows import coverage_summary, link_windows
    links = link_windows([event()], *window())
    assert len(links) == 1
    assert links[0]['onset_in_future'] and links[0]['confirmation_in_future']
    assert not links[0]['entire_event_observed']
    summary = coverage_summary([event()], links)
    row = next(r for r in summary if r['universe'] == 'pilot' and r['role'] == 'independent_fit'
               and r['category'] == '__all__' and r['event_type'] == 'stop_candidate')
    assert row['events_total'] == row['future_confirmation_events'] == 1
    assert row['entire_event_uncovered_events'] == 1


def test_onset_at_future_right_endpoint_no_overlap():
    from trajectory_prediction.evaluation.event_windows import link_windows
    assert link_windows([event()], *window(anchor=10000000, first=8000000, last=12000000)) == []


def test_history_context_is_not_future_onset():
    from trajectory_prediction.evaluation.event_windows import link_windows
    links = link_windows([event()], *window(anchor=12500000, last=14000000))
    assert links[0]['observed_interval_overlap']
    assert not links[0]['onset_in_future'] and not links[0]['confirmation_in_future']


def test_entire_event_and_multiple_windows_not_double_counted():
    from trajectory_prediction.evaluation.event_windows import coverage_summary, link_windows
    e = event(right_censored=False)
    w, arrays = window(last=14000000, pilot=False)
    w += [dict(w[0], window_id='w2', phase3a_pilot_member=True)]
    links = link_windows([e], w, arrays)
    assert len(links) == 2 and all(r['entire_event_observed'] for r in links)
    summary = coverage_summary([e], links)
    assert all(r['events_total'] <= 1 for r in summary)


def test_zero_events_and_no_matching_instance():
    from trajectory_prediction.evaluation.event_windows import coverage_summary, link_windows
    assert link_windows([event(instance='other')], *window()) == []
    assert all(r['events_total'] == 0 for r in coverage_summary([], []))


@pytest.mark.parametrize('kind', ['duplicate_event', 'duplicate_window', 'bad_times'])
def test_invalid_mapping_rejected(kind):
    from trajectory_prediction.evaluation.event_windows import link_windows
    events, (windows, arrays) = [event()], window()
    if kind == 'duplicate_event': events *= 2
    elif kind == 'duplicate_window': windows *= 2
    else: arrays['independent_fit']['actual_timestamp_us'][0, 5] = 1
    with pytest.raises(ValueError):
        link_windows(events, windows, arrays)

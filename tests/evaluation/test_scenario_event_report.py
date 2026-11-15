import json

import numpy as np
import pytest

from trajectory_prediction.evaluation import scenario_event_report as events
from trajectory_prediction.evaluation.scenario_preflight import run_preflight
from trajectory_prediction.evaluation.scenario_publication import publish_predictions
from trajectory_prediction.evaluation.scenario_replay import PROTOCOL_V1
from trajectory_prediction.evaluation.scenario_report import build_report
from trajectory_prediction.utils.provenance import sha256_file


def inputs():
    keys = [{'role': 'independent_fit', 'instance': 'i', 'sample': 's', 'scene': 'scene',
             'log': 'l', 'row': 0, 'noise_seed': seed, 'condition': condition}
            for condition in ('noise_05', 'noise_outlier') for seed in (2027, 2028, 2029)]
    p = np.zeros((6, 11, 12, 2), np.float32)
    for i, curve in enumerate(([2, .2, .2, .2], [0, 0, 0, 0], [2, .2, .2, 2]), 3):
        for k, value in zip(range(4, 8), curve, strict=True):
            p[i, k - 1, k + 1, 0] = value
    return p, np.zeros((6, 12, 2), np.float32), keys


def test_pairing_states_and_absolute_errors():
    p, y, keys = inputs()
    arrays, paired = events.compute_events(p, y, keys, 'model-identity')
    assert arrays['status'].tolist() == ['recovered', 'unaffected', 'not_recovered']
    assert arrays['confirm_delay_s'][0] == 1
    assert np.isnan(arrays['confirm_delay_s'][1:]).all()
    summary = events.summarize_events(arrays, paired)
    assert (summary['eligible'], summary['affected'], summary['unaffected']) == (3, 2, 1)
    assert summary['recovered_fraction_of_affected'] == .5
    assert summary['window_count'] == 1
    assert summary['mean_reference_error_m'] == 0
    assert summary['mean_outlier_error_m'] > 0
    json.dumps(summary, allow_nan=False)


def test_missing_pair_and_wrong_gt_rejected():
    p, y, keys = inputs()
    with pytest.raises(ValueError, match='pair|cohort'):
        events.compute_events(p[:-1], y[:-1], keys[:-1], 'model')
    y[3, 0, 0] = 1
    with pytest.raises(ValueError, match='GT'):
        events.compute_events(p, y, keys, 'model')


def test_bad_recovery_reducer_rejected(monkeypatch):
    p, y, keys = inputs()
    arrays, paired = events.compute_events(p, y, keys, 'model')
    original = events.summarize_recovery
    def wrong(*args):
        result = original(*args)
        result['recovered'] = 99
        return result
    monkeypatch.setattr(events, 'summarize_recovery', wrong)
    with pytest.raises(ValueError, match='scalar'):
        events.summarize_events(arrays, paired)


def test_cross_seed_pairing_is_independently_rejected(monkeypatch):
    p, y, keys = inputs()
    p[:] = 0
    original = events.pair_first_outlier
    def wrong(*args, **kwargs):
        result = original(*args, **kwargs)
        result['outlier_indices'] = result['outlier_indices'][1:] + result['outlier_indices'][:1]
        return result
    monkeypatch.setattr(events, 'pair_first_outlier', wrong)
    with pytest.raises(ValueError, match='scalar.*pair'):
        events.compute_events(p, y, keys, 'model')


def test_empty_group_keeps_reason():
    p, y, keys = inputs()
    arrays, paired = events.compute_events(p, y, keys, 'model')
    summary = events.summarize_events({k: v[:0] for k, v in arrays.items()}, paired[:0])
    assert summary['status'] == 'not_evaluable'
    assert summary['reason'] == 'empty_group'
    assert summary['recovered_fraction_of_affected'] is None


def test_event_log_macro_differs_from_window_mean():
    p, y, keys = inputs()
    arrays, paired = events.compute_events(p, y, keys, 'model')
    curves = np.array([[0.] * 4] * 6 + [[2.] * 4] * 3)
    arrays = {k: np.concatenate([v] * 3) for k, v in arrays.items()}
    arrays['excess'] = curves
    rows = [{**paired[i % 3], 'instance': f'w{i // 3}', 'log': 'a' if i < 6 else 'b'} for i in range(9)]
    pooled = events.summarize_event_logs(arrays, rows)[0]
    assert pooled['mean_excess_auc_m_s'] == 1
    assert pooled['log_macro_mean_excess_auc_m_s'] == 1.5
    assert pooled['valid_log_count'] == 2
    assert pooled['affected_log_count'] == 1
    assert pooled['log_macro_recovered_fraction_of_affected'] == 0


def test_end_to_end_labels_and_tamper(scenario_parent, tmp_path, monkeypatch):
    ready, published, queries, output = (tmp_path / n for n in ('ready', 'p', 'q', 'e'))
    run_preflight(scenario_parent['parent'], scenario_parent['owner'], PROTOCOL_V1, ready)
    publish_predictions(ready, published)
    build_report(published, queries)
    result = events.build_event_report(queries, output)
    assert result['event_shards'] == 32
    assert result['window_count'] == 2
    assert result['manual_completed'] == 0
    assert result['full_s6_complete'] is False
    assert events.verify_event_report(output)['scalar_reference_verified']
    with pytest.raises(FileExistsError):
        events.build_event_report(queries, output)
    groups = json.loads((output / 'event_groups.json').read_text())
    groups[0]['eligible'] += 1
    (output / 'event_groups.json').write_text(json.dumps(groups))
    marker = json.loads((output / 'completed.json').read_text())
    marker['sha256']['event_groups.json'] = sha256_file(output / 'event_groups.json')
    (output / 'completed.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='semantic|replay'):
        events.verify_event_report(output)
    original = events._calculate
    def interrupt(binding):
        for item in original(binding):
            yield item
            raise RuntimeError('simulated interruption')
    with monkeypatch.context() as m:
        m.setattr(events, '_calculate', interrupt)
        incomplete = tmp_path / 'interrupted'
        with pytest.raises(RuntimeError, match='interruption'):
            events.build_event_report(queries, incomplete)
        assert not (incomplete / 'completed.json').exists()
        with pytest.raises(ValueError, match='incomplete'):
            events.verify_event_report(incomplete)
    binding = events._prepare(queries)
    (scenario_parent['parent'] / 'records.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum|reference'):
        events._recheck(binding)


def test_positive_labels_remain_pending(tmp_path):
    role = 'independent_fit'
    (tmp_path / 'gt').mkdir()
    history = np.stack((np.arange(-2., .1, .5), np.zeros(5)), axis=1).astype(np.float32)
    future = np.tile(np.array([.5, 0], np.float32), (12, 1))
    np.savez_compressed(tmp_path / 'gt' / f'{role}.npz', history_xy=history[None],
                        future_xy=future[None], history_mask=np.ones((1, 5), bool))
    (tmp_path / f'{role}-keys.json').write_text(json.dumps([{
        'gt_index': 0, 'instance': 'i', 'sample': 's', 'scene': 'scene', 'log': 'l', 'row': 0}]))
    labels = events._labels(tmp_path, role, {'low_speed_mps': 2., 'turn_yaw_deg': 15.,
                                           'high_curvature_yaw_deg': 45., 'acceleration_mps2': 1.})
    assert labels[0]['tag_status']['stop_event']['status'] == 'candidate'
    review = events.build_review_manifest(labels)
    assert review
    assert all(r['review_status'] == 'pending' and r['manual_label'] == '' for r in review)

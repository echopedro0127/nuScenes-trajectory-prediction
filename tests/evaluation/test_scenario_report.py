import json

import numpy as np
import pytest

from trajectory_prediction.evaluation import scenario_report as report
from trajectory_prediction.evaluation.scenario_preflight import run_preflight
from trajectory_prediction.evaluation.scenario_publication import publish_predictions
from trajectory_prediction.evaluation.scenario_replay import PROTOCOL_V1
from trajectory_prediction.utils.provenance import sha256_file


def test_same_target_and_denominators():
    y = np.zeros((3, 12, 2), np.float32)
    p = np.zeros((3, 11, 12, 2), np.float32)
    p[:, 0, :, 0] = 3
    p[:, 0, 3, 0] = 7
    p[:, 1, :, 0] = 1
    history = np.zeros((3, 5, 2), np.float32)
    keys = [{'instance': 'a', 'sample': 'b', 'noise_seed': s,
             'condition': 'clean', 'log': 'l'} for s in (2027, 2028, 2029)]
    arrays, audit = report.compute_queries(p, y, history, np.ones((3, 5), bool))
    assert arrays['lead2_gain_m'][0, 0] == 6
    assert arrays['lead2_target_column'].tolist() == list(range(2, 12))
    assert not arrays['direction_valid'].any()
    assert np.isnan(arrays['direction_longitudinal_m']).all()
    assert audit['scalar_checked_values'] > 0
    groups = report.summarize_queries(arrays, keys)
    group = next(g for g in groups if g['log'] is None)
    assert group['m1_m5']['2']['window_count'] == 1
    assert group['m1_m5']['2']['query_count'] == 30
    assert group['m2']['2']['pair_count'] == 27
    assert group['m4']['reason'] == 'no_valid_heading'
    assert group['m4']['mean_lateral_abs_m'] is None
    assert group['remaining_miss']['query_denominators'] == [3] * 11
    assert group['remaining_miss']['miss_update_counts'][:2] == [3, 0]
    json.dumps(groups, allow_nan=False)


def test_missing_noise_and_duplicate_rejected():
    y = np.zeros((3, 12, 2), np.float32)
    p = np.zeros((3, 11, 12, 2), np.float32)
    a, _ = report.compute_queries(p, y, np.zeros((3, 5, 2), np.float32), np.ones((3, 5), bool))
    keys = [{'instance': 'a', 'sample': 'b', 'noise_seed': 2027,
             'condition': 'clean', 'log': 'l'} for _ in range(3)]
    with pytest.raises(ValueError, match='duplicate'):
        report.summarize_queries(a, keys)
    keys = [{**k, 'noise_seed': 2027 + i} for i, k in enumerate(keys)]
    with pytest.raises(ValueError, match='noise cohort'):
        report.summarize_queries({k: v[:2] if v.ndim > 1 else v for k, v in a.items()}, keys[:2])


@pytest.fixture
def published(scenario_parent, tmp_path):
    ready = tmp_path / 'ready'
    run_preflight(scenario_parent['parent'], scenario_parent['owner'], PROTOCOL_V1, ready)
    path = tmp_path / 'predictions'
    publish_predictions(ready, path)
    return path


def test_report_roundtrip_and_tampering(published, tmp_path):
    path = tmp_path / 'report'
    summary = report.build_report(published, path)
    assert summary['metric_shards'] == 32
    assert summary['full_s6_complete'] is False
    assert report.verify_report(path)['scalar_reference_verified']
    assert len(list((path / 'queries').glob('*.npz'))) == 32
    with pytest.raises(FileExistsError):
        report.build_report(published, path)
    # Change a reported statistic AND update its checksum: semantic replay must reject.
    groups = json.loads((path / 'groups.json').read_text())
    groups[0]['m1_m5']['1']['mean_error_m'] += 1
    (path / 'groups.json').write_text(json.dumps(groups))
    marker = json.loads((path / 'completed.json').read_text())
    marker['sha256']['groups.json'] = sha256_file(path / 'groups.json')
    (path / 'completed.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='summary|group'):
        report.verify_report(path)


def test_scalar_audit_detects_wrong_vector_math(monkeypatch):
    original = report.fixed_lead_errors
    def wrong(*args, **kwargs):
        value = original(*args, **kwargs)
        value['values'] += 1
        return value
    monkeypatch.setattr(report, 'fixed_lead_errors', wrong)
    with pytest.raises(ValueError, match='scalar'):
        report.compute_queries(np.zeros((1, 11, 12, 2), np.float32),
                               np.zeros((1, 12, 2), np.float32),
                               np.zeros((1, 5, 2), np.float32), np.ones((1, 5), bool))


def test_window_mean_and_log_macro_are_distinct():
    y = np.zeros((9, 12, 2), np.float32)
    p = np.zeros((9, 11, 12, 2), np.float32)
    p[6:, ..., 0] = 9
    arrays, _ = report.compute_queries(p, y, np.zeros((9, 5, 2), np.float32), np.ones((9, 5), bool))
    keys = [{'instance': f'w{i // 3}', 'sample': 's', 'noise_seed': 2027 + i % 3,
             'condition': 'clean', 'log': 'l1' if i < 6 else 'l2'} for i in range(9)]
    pooled = next(g for g in report.summarize_queries(arrays, keys) if g['log'] is None)
    assert pooled['m1_m5']['2']['mean_error_m'] == 3
    assert pooled['log_macro_mean_error_m']['2'] == 4.5
    assert pooled['m1_m5']['2']['window_count'] == 3
    assert pooled['valid_log_count'] == 2


@pytest.mark.parametrize('field', ['harmful_fraction', 'window_macro_harmful_fraction', 'median_pair_gain_m'])
def test_scalar_audit_rejects_wrong_reducer(monkeypatch, field):
    p = np.zeros((3, 11, 12, 2), np.float32)
    a, _ = report.compute_queries(p, np.zeros((3, 12, 2), np.float32),
                                  np.zeros((3, 5, 2), np.float32), np.ones((3, 5), bool))
    original = report.summarize_revisions
    def wrong(*args):
        value = original(*args)
        value[field] = 1.
        return value
    monkeypatch.setattr(report, 'summarize_revisions', wrong)
    keys = [{'instance': 'a', 'sample': 'b', 'noise_seed': s,
             'condition': 'clean', 'log': 'l'} for s in (2027, 2028, 2029)]
    with pytest.raises(ValueError, match='scalar'):
        report.summarize_queries(a, keys)


@pytest.mark.parametrize('field,value', [('valid_query_count', 30), ('heading_coverage', 1.), ('status', 'ok')])
def test_scalar_audit_rejects_wrong_direction_reducer(monkeypatch, field, value):
    a, _ = report.compute_queries(np.zeros((3, 11, 12, 2), np.float32),
                                  np.zeros((3, 12, 2), np.float32),
                                  np.zeros((3, 5, 2), np.float32), np.ones((3, 5), bool))
    original = report.summarize_directional
    def wrong(*args):
        result = original(*args)
        result[field] = value
        return result
    monkeypatch.setattr(report, 'summarize_directional', wrong)
    keys = [{'instance': 'a', 'sample': 'b', 'noise_seed': s,
             'condition': 'clean', 'log': 'l'} for s in (2027, 2028, 2029)]
    with pytest.raises(ValueError, match='scalar'):
        report.summarize_queries(a, keys)

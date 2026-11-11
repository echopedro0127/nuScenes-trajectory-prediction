"""Protocol and frozen inference contracts; no nuScenes dependency."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from trajectory_prediction.evaluation.scenario_replay import time_grid, validate_protocol


def test_checked_in_yaml_matches_approved_protocol(protocol):
    path = Path(__file__).parents[2] / 'configs/analysis/scenario_evaluation.yaml'
    assert validate_protocol(yaml.safe_load(path.read_text(encoding='utf-8'))) == protocol


@pytest.fixture
def protocol():
    # The approved plan is the RED-stage source before the YAML is created.
    text = (Path(__file__).parents[2] / 'docs/superpowers/plans/'
            '2026-11-10-scenario-evaluation-phase3a.md').read_text(encoding='utf-8')
    return yaml.safe_load(text.split('```yaml\n')[1].split('```')[0])


def test_protocol_preserves_frozen_values_and_time_index(protocol):
    result = validate_protocol(protocol)
    assert result == protocol and result is not protocol
    result['inference']['threads'] = 9
    assert protocol['inference']['threads'] == 2
    grid = time_grid(protocol)
    assert [len(grid[m]['query_steps']) for m in (1, 2, 4)] == [11, 10, 8]
    assert grid[2]['query_rows'] == list(range(10))
    assert grid[2]['target_columns'] == list(range(2, 12))
    assert grid[2]['target_times_s'][0] == 1.5
    assert len(grid[2]['revision_query_steps']) == 9


@pytest.mark.parametrize('key,value', [
    ('dt', .1), ('dt', float('nan')), ('dt', True),
    ('lead_steps', [0, 2, 4]), ('lead_steps', [1, 2, 2]),
    ('query_steps', list(range(11))), ('schema', 'unknown'),
    ('roles', ['independent_holdout']), ('conditions', ['missing']),
    ('model_seeds', [2027, 2027, 2029]), ('noise_seeds', [2027]),
    ('quantile_method', 'nearest'), ('missing_expected_pair', 'skip'),
])
def test_protocol_rejects_unsupported_or_unsafe_contract(protocol, key, value):
    protocol[key] = value
    with pytest.raises(ValueError, match=key):
        validate_protocol(protocol)


def test_protocol_requires_all_fields_and_rejects_unknown_fields(protocol):
    missing = copy.deepcopy(protocol)
    del missing['schema']
    with pytest.raises(ValueError, match='schema'):
        validate_protocol(missing)
    protocol['unversioned_override'] = True
    with pytest.raises(ValueError, match='unversioned_override'):
        validate_protocol(protocol)


@pytest.mark.parametrize('path,value', [
    (('inference', 'flat_batch_size'), 1),
    (('inference', 'compatibility'), 'tolerant'),
    (('recovery', 'last_query_step'), 8),
    (('recovery', 'require_no_relapse'), False),
    (('event_labels', 'stop_duration_steps'), 1),
])
def test_nested_contract_cannot_be_changed_without_version(protocol, path, value):
    protocol[path[0]][path[1]] = value
    with pytest.raises(ValueError, match=path[1]):
        validate_protocol(protocol)


def test_frozen_replay_matches_old_metrics_and_has_no_training(scenario_parent, monkeypatch):
    from trajectory_prediction.evaluation.scenario_inputs import assemble_inputs
    from trajectory_prediction.evaluation.scenario_replay import replay_streams

    def forbidden(*args, **kwargs):
        raise AssertionError('training is forbidden during frozen replay')

    monkeypatch.setattr(torch.optim.Adam, '__init__', forbidden)
    monkeypatch.setattr(torch.Tensor, 'backward', forbidden)
    f = scenario_parent
    inputs = assemble_inputs(f['parent'], f['owner'])
    role = 'independent_select'
    for strategy in ('s2027_baseline7', 's2028_plus_ca', 's2029_guarded_ca'):
        result = replay_streams(inputs, role, strategy, [0, 8])
        with np.load(f['parent'] / 'per_update.npz') as z:
            assert np.array_equal(result['metrics'], z[role + '__' + strategy][[0, 8]])
        assert result['forecast'].shape == (2, 11, 12, 2)
        assert result['forecast'].dtype == np.float32
        assert result['exact_parent_metrics']
    physical = replay_streams(inputs, role, 'kf_nis', [0])
    assert np.array_equal(physical['forecast'], inputs.datasets[role]['candidates'][[0], :, 3])


def test_future_changes_do_not_change_published_prefix(scenario_parent):
    from trajectory_prediction.evaluation.scenario_inputs import assemble_inputs
    from trajectory_prediction.evaluation.scenario_replay import replay_streams
    f = scenario_parent
    inputs = assemble_inputs(f['parent'], f['owner'])
    role = 'independent_select'
    before = replay_streams(inputs, role, 's2027_guarded_ca', [0], compare_parent=False)
    # Alter only later query inputs; target timestamps in an earlier forecast may be later.
    inputs.datasets[role]['features'][0, 4:] += 30
    inputs.datasets[role]['candidates'][0, 4:] += 90
    inputs.datasets[role]['ca_allowed'][0, 4:] = False
    inputs.gt_windows[role][0]['history_xy'][:] += 500
    after = replay_streams(inputs, role, 's2027_guarded_ca', [0], compare_parent=False)
    assert np.array_equal(before['forecast'][:, :4], after['forecast'][:, :4])


def test_changed_checkpoint_is_rejected_after_assembly(scenario_parent):
    from trajectory_prediction.evaluation.scenario_inputs import assemble_inputs
    from trajectory_prediction.evaluation.scenario_replay import replay_streams
    f = scenario_parent
    inputs = assemble_inputs(f['parent'], f['owner'])
    path = f['parent'] / 's2027_plus_ca.pt'
    payload = torch.load(path, weights_only=True)
    payload['model']['extra.bias'][0] += 1
    torch.save(payload, path)
    with pytest.raises(ValueError, match='checkpoint|checksum'):
        replay_streams(inputs, 'independent_select', 's2027_plus_ca', [0])


def test_parent_verification_executes_in_owner_and_binds_proof(scenario_parent):
    from trajectory_prediction.evaluation.scenario_inputs import assemble_inputs
    from trajectory_prediction.evaluation.scenario_replay import verify_parent
    f = scenario_parent
    inputs = assemble_inputs(f['parent'], f['owner'])
    proof = verify_parent(inputs)
    assert proof['identity'] == inputs.parent_identity
    assert proof['all_seed_metrics_groups_weights_replayed']
    assert proof['owner_project_root'] == str(f['owner'].resolve())
    assert not proof['holdout_evaluated']


def test_preflight_receipt_is_complete_replayable_and_not_full_evaluation(
        scenario_parent, protocol, tmp_path):
    from trajectory_prediction.evaluation.scenario_preflight import (
        run_preflight,
        verify_preflight,
    )
    f = scenario_parent
    output = tmp_path / 'preflight'
    result = run_preflight(f['parent'], f['owner'], protocol, output)
    assert result['status'] == 's0_s2_verified'
    assert result['models_checked'] == 9
    assert result['physical_strategies_checked'] == 7
    assert result['new_metrics_computed'] is False
    assert verify_preflight(output)['saved_forecasts_and_legacy_metrics_replayed']
    with pytest.raises(FileExistsError):
        run_preflight(f['parent'], f['owner'], protocol, output)
    (output / 'report.json').write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        verify_preflight(output)


def test_receipt_recomputes_report_not_only_file_checksums(scenario_parent, protocol, tmp_path):
    from trajectory_prediction.evaluation.scenario_preflight import run_preflight, verify_preflight
    from trajectory_prediction.utils.provenance import sha256_file
    f = scenario_parent
    output = tmp_path / 'preflight'
    run_preflight(f['parent'], f['owner'], protocol, output)
    path = output / 'report.json'
    report = json.loads(path.read_text())
    report['checks'][0]['checked_streams'] += 1
    path.write_text(json.dumps(report))
    marker_path = output / 'preflight_completed.json'
    marker = json.loads(marker_path.read_text())
    marker['sha256']['report.json'] = sha256_file(path)
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='report|summary'):
        verify_preflight(output)


@pytest.mark.parametrize('field,value', [
    ('fit_only_normalization_verified', False),
    ('seed2027_two_controls_match_parent', False),
    ('owner_project_root', 'wrong-owner'), ('command', ['wrong-command']),
    ('stdout', '{}\n'),
])
def test_saved_parent_proof_revalidates_all_semantics(
        scenario_parent, protocol, tmp_path, field, value):
    from trajectory_prediction.evaluation.scenario_preflight import run_preflight, verify_preflight
    from trajectory_prediction.utils.provenance import sha256_file
    f = scenario_parent
    output = tmp_path / 'preflight'
    run_preflight(f['parent'], f['owner'], protocol, output)
    path = output / 'parent_proof.json'
    proof = json.loads(path.read_text())
    proof[field] = value
    path.write_text(json.dumps(proof))
    marker_path = output / 'preflight_completed.json'
    marker = json.loads(marker_path.read_text())
    marker['sha256']['parent_proof.json'] = sha256_file(path)
    marker_path.write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='proof'):
        verify_preflight(output)

"""A file-backed integration audit stays separate from full Phase 3A results."""
import json
import runpy
from pathlib import Path

import pytest


def test_fixed_lead_audit_checks_saved_predictions(scenario_parent, tmp_path):
    from trajectory_prediction.evaluation.scenario_preflight import run_preflight
    from trajectory_prediction.evaluation.scenario_replay import PROTOCOL_V1
    f = scenario_parent
    preflight = tmp_path / 'preflight'
    run_preflight(f['parent'], f['owner'], PROTOCOL_V1, preflight)
    module = runpy.run_path(str(Path(__file__).parents[2] / 'scripts/check_scenario_fixed_lead.py'))
    output = tmp_path / 's3'
    result = module['check_fixed_leads'](preflight, output)
    assert result['lead_queries_per_stream'] == {'1': 11, '2': 10, '4': 8}
    assert result['prediction_shards'] == 32
    assert result['checked_error_values'] == 18 * 16 * 29
    assert result['full_phase3a_evaluation'] is False
    assert result['all_scalar_comparisons_within_tolerance']
    assert json.loads((output / 'completed.json').read_text())['identity'] == result['identity']
    with pytest.raises(FileExistsError):
        module['check_fixed_leads'](preflight, output)


@pytest.mark.parametrize('kind', ['proof_identity', 'marker_switched'])
def test_s2_proof_and_snapshot_cannot_be_switched_after_verification(
        scenario_parent, tmp_path, monkeypatch, kind):
    from trajectory_prediction.evaluation.scenario_preflight import run_preflight
    from trajectory_prediction.evaluation.scenario_replay import PROTOCOL_V1
    f = scenario_parent
    preflight = tmp_path / 'preflight'
    run_preflight(f['parent'], f['owner'], PROTOCOL_V1, preflight)
    module = runpy.run_path(str(Path(__file__).parents[2] / 'scripts/check_scenario_fixed_lead.py'))
    check = module['check_fixed_leads']
    original = check.__globals__['verify_preflight']

    def switched(path):
        proof = original(path)
        if kind == 'proof_identity':
            proof['identity'] = 'another-preflight'
        else:
            marker_path = preflight / 'preflight_completed.json'
            marker = json.loads(marker_path.read_text())
            marker['identity'] = 'switched-after-verification'
            marker_path.write_text(json.dumps(marker))
        return proof

    monkeypatch.setitem(check.__globals__, 'verify_preflight', switched)
    with pytest.raises(ValueError, match='identity|snapshot|changed'):
        check(preflight, tmp_path / 's3')

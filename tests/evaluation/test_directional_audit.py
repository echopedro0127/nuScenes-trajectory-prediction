import json
import runpy
from pathlib import Path

import pytest


def test_directional_subset_audit_preserves_common_gt_frames(scenario_parent, tmp_path):
    from trajectory_prediction.evaluation.scenario_preflight import run_preflight
    from trajectory_prediction.evaluation.scenario_replay import PROTOCOL_V1
    f = scenario_parent
    parent = tmp_path / 'preflight'
    run_preflight(f['parent'], f['owner'], PROTOCOL_V1, parent)
    module = runpy.run_path(str(Path(__file__).parents[2] / 'scripts/check_scenario_directions.py'))
    result = module['check_directions'](parent, tmp_path / 's4b')
    assert result['query_pairs_checked'] == 18 * 16 * 10
    assert result['prediction_shards'] == 32
    assert result['full_phase3a_evaluation'] is False
    assert result['common_gt_masks_verified']
    rows = json.loads((tmp_path / 's4b/groups.json').read_text())
    assert len(rows) == 2 * 16 * 3
    assert all(r['directional']['valid_query_count'] == 30 for r in rows)
    with pytest.raises(FileExistsError):
        module['check_directions'](parent, tmp_path / 's4b')


def test_direction_audit_captures_source_before_parent_verification(scenario_parent, tmp_path, monkeypatch):
    from trajectory_prediction.evaluation.scenario_preflight import run_preflight
    from trajectory_prediction.evaluation.scenario_replay import PROTOCOL_V1
    f = scenario_parent
    parent = tmp_path / 'preflight'
    run_preflight(f['parent'], f['owner'], PROTOCOL_V1, parent)
    module = runpy.run_path(str(Path(__file__).parents[2] / 'scripts/check_scenario_directions.py'))
    check = module['check_directions']
    namespace = check.__globals__
    original_verify, original_hash = namespace['verify_preflight'], namespace['sha256_file']
    changed = False

    def verify(path):
        nonlocal changed
        result = original_verify(path)
        changed = True
        return result

    def digest(path):
        return 'f' * 64 if changed and Path(path).name == 'directional_metrics.py' else original_hash(path)

    monkeypatch.setitem(namespace, 'verify_preflight', verify)
    monkeypatch.setitem(namespace, 'sha256_file', digest)
    with pytest.raises(ValueError, match='source'):
        check(parent, tmp_path / 's4b')

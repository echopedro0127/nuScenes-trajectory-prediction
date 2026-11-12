import json
import runpy
from pathlib import Path

import pytest


def test_revision_subset_audit_is_recomputable_and_scoped(scenario_parent, tmp_path):
    from trajectory_prediction.evaluation.scenario_preflight import run_preflight
    from trajectory_prediction.evaluation.scenario_replay import PROTOCOL_V1
    f = scenario_parent
    parent = tmp_path / 'preflight'
    run_preflight(f['parent'], f['owner'], PROTOCOL_V1, parent)
    module = runpy.run_path(str(Path(__file__).parents[2] / 'scripts/check_scenario_revisions.py'))
    result = module['check_revisions'](parent, tmp_path / 's4a')
    assert result['gain_values_checked'] == 18 * 16 * 26
    assert result['error_values_checked'] == 18 * 16 * 29
    assert result['remaining_flags_checked'] == 18 * 16 * 11
    assert result['full_phase3a_evaluation'] is False
    rows = json.loads((tmp_path / 's4a/groups.json').read_text())
    assert len(rows) == 2 * 16 * 3 * 3
    for row in rows:
        r = row['revision']
        assert r['beneficial_count'] + r['harmful_count'] + r['neutral_count'] == r['pair_count']
        assert row['train_seed'] is None or row['train_seed'] in [2027, 2028, 2029]
    with pytest.raises(FileExistsError):
        module['check_revisions'](parent, tmp_path / 's4a')


def test_source_snapshot_precedes_slow_parent_verification(scenario_parent, tmp_path, monkeypatch):
    from trajectory_prediction.evaluation.scenario_preflight import run_preflight
    from trajectory_prediction.evaluation.scenario_replay import PROTOCOL_V1
    f = scenario_parent
    parent = tmp_path / 'preflight'
    run_preflight(f['parent'], f['owner'], PROTOCOL_V1, parent)
    module = runpy.run_path(str(Path(__file__).parents[2] / 'scripts/check_scenario_revisions.py'))
    check = module['check_revisions']
    namespace = check.__globals__
    original_verify, original_hash = namespace['verify_preflight'], namespace['sha256_file']
    changed = False

    def verify(path):
        nonlocal changed
        result = original_verify(path)
        changed = True
        return result

    def digest(path):
        if changed and Path(path).name == 'revision_metrics.py':
            return 'f' * 64
        return original_hash(path)

    monkeypatch.setitem(namespace, 'verify_preflight', verify)
    monkeypatch.setitem(namespace, 'sha256_file', digest)
    with pytest.raises(ValueError, match='source|源码'):
        check(parent, tmp_path / 's4a')
    assert not (tmp_path / 's4a/completed.json').exists()

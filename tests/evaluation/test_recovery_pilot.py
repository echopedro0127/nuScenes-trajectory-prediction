import json
import runpy
from pathlib import Path

import pytest


def test_complete_pair_pilot_and_independent_replay(scenario_parent, tmp_path):
    from trajectory_prediction.evaluation.scenario_preflight import run_preflight
    from trajectory_prediction.evaluation.scenario_replay import PROTOCOL_V1
    f = scenario_parent
    parent = tmp_path / 'preflight'
    run_preflight(f['parent'], f['owner'], PROTOCOL_V1, parent)
    module = runpy.run_path(str(Path(__file__).parents[2] / 'scripts/check_scenario_recovery.py'))
    output = tmp_path / 's4c'
    r = module['check_recovery'](parent, output)
    assert r['strategy_event_records'] == 2 * 3 * 16
    assert r['unique_window_noise_pairs'] == 6
    assert r['group_rows'] == 32 and r['full_phase3a_evaluation'] is False
    assert module['check_recovery'](parent, output, verify_only=True)['independent_replay_verified']
    with pytest.raises(FileExistsError):
        module['check_recovery'](parent, output)
    groups = json.loads((output / 'groups.json').read_text())
    for g in groups:
        assert g['eligible'] == g['affected'] + g['unaffected']
        assert g['affected'] == g['recovered'] + g['not_recovered']
        assert g['eligible'] == 3
    (output / 'events.json').write_text('[]')
    with pytest.raises(ValueError, match='checksum'):
        module['check_recovery'](parent, output, verify_only=True)


def test_source_change_during_verification_is_rejected(scenario_parent, tmp_path, monkeypatch):
    from trajectory_prediction.evaluation.scenario_preflight import run_preflight
    from trajectory_prediction.evaluation.scenario_replay import PROTOCOL_V1
    f = scenario_parent
    parent = tmp_path / 'preflight'
    run_preflight(f['parent'], f['owner'], PROTOCOL_V1, parent)
    module = runpy.run_path(str(Path(__file__).parents[2] / 'scripts/check_scenario_recovery.py'))
    check = module['check_recovery']
    namespace = check.__globals__
    old_verify, old_hash = namespace['verify_preflight'], namespace['sha256_file']
    changed = False

    def verify(path):
        nonlocal changed
        result = old_verify(path)
        changed = True
        return result

    def digest(path):
        return 'f' * 64 if changed and Path(path).name == 'event_metrics.py' else old_hash(path)

    monkeypatch.setitem(namespace, 'verify_preflight', verify)
    monkeypatch.setitem(namespace, 'sha256_file', digest)
    with pytest.raises(ValueError, match='source'):
        check(parent, tmp_path / 's4c')


def test_verify_only_detects_own_marker_changed_mid_replay(scenario_parent, tmp_path, monkeypatch):
    from trajectory_prediction.evaluation.scenario_preflight import run_preflight
    from trajectory_prediction.evaluation.scenario_replay import PROTOCOL_V1
    f = scenario_parent
    parent, output = tmp_path / 'preflight', tmp_path / 's4c'
    run_preflight(f['parent'], f['owner'], PROTOCOL_V1, parent)
    module = runpy.run_path(str(Path(__file__).parents[2] / 'scripts/check_scenario_recovery.py'))
    check = module['check_recovery']
    check(parent, output)
    original = check.__globals__['verify_preflight']

    def changed(path):
        proof = original(path)
        marker_path = output / 'completed.json'
        marker = json.loads(marker_path.read_text())
        marker['identity'] = 'changed-mid-replay'
        marker_path.write_text(json.dumps(marker))
        return proof

    monkeypatch.setitem(check.__globals__, 'verify_preflight', changed)
    with pytest.raises(ValueError, match='output|snapshot'):
        check(parent, output, verify_only=True)

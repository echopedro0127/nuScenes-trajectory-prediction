import json
import runpy
from pathlib import Path

import numpy as np
import pytest


def test_label_pipeline_deduplicates_noise_and_preserves_pending_review(scenario_parent, tmp_path):
    from trajectory_prediction.evaluation.scenario_preflight import run_preflight
    from trajectory_prediction.evaluation.scenario_replay import PROTOCOL_V1
    f = scenario_parent
    parent = tmp_path / 'preflight'
    run_preflight(f['parent'], f['owner'], PROTOCOL_V1, parent)
    module = runpy.run_path(str(Path(__file__).parents[2] / 'scripts/check_scenario_labels.py'))
    output = tmp_path / 'labels'
    result = module['check_labels'](parent, output)
    assert result['unique_windows'] == 2  # 18 noisy streams are the same 2 GT windows.
    assert result['manual_review_completed'] == 0 and result['scope'] == 'window_only'
    assert module['check_labels'](parent, output, verify_only=True)['labels_and_manifest_recomputed']
    rows = json.loads((output / 'windows.json').read_text())
    assert len(rows) == 2 and all(len(r['trace']['raw_speed_mps']) == 16 for r in rows)
    assert (output / 'review.html').exists()
    with pytest.raises(FileExistsError):
        module['check_labels'](parent, output)
    (output / 'review.json').write_text('[] changed')
    with pytest.raises(ValueError, match='checksum'):
        module['check_labels'](parent, output, verify_only=True)


def test_case_plot_is_a_standalone_gt_image(tmp_path):
    from trajectory_prediction.analysis.scenario_events import annotate_window
    module = runpy.run_path(str(Path(__file__).parents[2] / 'scripts/check_scenario_labels.py'))
    points = np.column_stack((np.arange(-4, 13) * .5, np.zeros(17)))
    window = {**annotate_window(points[:5], points[5:], np.ones(5, bool)),
              'window_id': 'fixture', 'role': 'independent_select'}
    path = tmp_path / 'case.png'
    module['plot_gt_window'](window, path)
    assert path.read_bytes().startswith(b'\x89PNG\r\n\x1a\n')


def test_label_source_frozen_before_parent_verification(scenario_parent, tmp_path, monkeypatch):
    from trajectory_prediction.evaluation.scenario_preflight import run_preflight
    from trajectory_prediction.evaluation.scenario_replay import PROTOCOL_V1
    f = scenario_parent
    parent = tmp_path / 'preflight'
    run_preflight(f['parent'], f['owner'], PROTOCOL_V1, parent)
    module = runpy.run_path(str(Path(__file__).parents[2] / 'scripts/check_scenario_labels.py'))
    check = module['check_labels']
    namespace = check.__globals__
    old_verify, old_hash = namespace['verify_preflight'], namespace['sha256_file']
    changed = False

    def verify(path):
        nonlocal changed
        result = old_verify(path)
        changed = True
        return result

    def digest(path):
        return 'f' * 64 if changed and Path(path).name == 'scenario_events.py' else old_hash(path)

    monkeypatch.setitem(namespace, 'verify_preflight', verify)
    monkeypatch.setitem(namespace, 'sha256_file', digest)
    with pytest.raises(ValueError, match='source'):
        check(parent, tmp_path / 'labels')


def test_bound_inputs_can_be_rechecked_after_rendering(tmp_path):
    from trajectory_prediction.utils.provenance import sha256_file
    module = runpy.run_path(str(Path(__file__).parents[2] / 'scripts/check_scenario_labels.py'))
    (tmp_path / 'run.json').write_text('{}')
    (tmp_path / 'preflight_completed.json').write_text('{}')
    gt = tmp_path / 'gt.bin'
    gt.write_bytes(b'GT')
    marker = {'sha256': {'run.json': sha256_file(tmp_path / 'run.json')}}
    marker_hash = sha256_file(tmp_path / 'preflight_completed.json')
    refs = {'gt': {'path': str(gt), 'sha256': sha256_file(gt)}}
    module['_check_bound_inputs'](tmp_path, marker, marker_hash, refs, {})
    gt.write_bytes(b'changed during plotting')
    with pytest.raises(ValueError, match='GT|parent'):
        module['_check_bound_inputs'](tmp_path, marker, marker_hash, refs, {})

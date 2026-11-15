import json
import shutil
import sys

import pytest

from trajectory_prediction.evaluation import scenario_pipeline as pipeline
from trajectory_prediction.evaluation.scenario_preflight import run_preflight
from trajectory_prediction.evaluation.scenario_replay import PROTOCOL_V1
from trajectory_prediction.utils.provenance import sha256_file


@pytest.fixture
def frozen(scenario_parent, tmp_path):
    ready, freeze = tmp_path / 'ready', tmp_path / 'freeze'
    run_preflight(scenario_parent['parent'], scenario_parent['owner'], PROTOCOL_V1, ready)
    pipeline.freeze_protocol(ready, freeze, purpose='fixture')
    return freeze


def test_freeze_before_inference_and_source_change(scenario_parent, tmp_path, monkeypatch):
    ready, freeze = tmp_path / 'ready', tmp_path / 'freeze'
    run_preflight(scenario_parent['parent'], scenario_parent['owner'], PROTOCOL_V1, ready)
    def forbidden(*args, **kwargs):
        raise AssertionError('freeze cannot infer')
    monkeypatch.setattr(pipeline, 'publish_predictions', forbidden)
    result = pipeline.freeze_protocol(ready, freeze, purpose='fixture')
    assert result['resource_plan']['shards'] == 32
    assert pipeline.verify_freeze(freeze)['identity'] == result['identity']
    with pytest.raises(FileExistsError):
        pipeline.freeze_protocol(ready, freeze, purpose='fixture')
    original = pipeline._sources
    monkeypatch.setattr(pipeline, '_sources', lambda: {**original(), 'changed.py': 'bad'})
    with pytest.raises(ValueError, match='source'):
        pipeline.verify_freeze(freeze)


def test_pipeline_completion_and_rehashed_summary_tamper(frozen, tmp_path):
    output = tmp_path / 'run'
    summary = pipeline.run_pipeline(frozen, output)
    assert summary['formal_full_run'] is False
    assert summary['unique_windows'] == {'independent_fit': 1, 'independent_select': 1}
    assert pipeline.verify_pipeline(output)['top_level_semantic_replay_verified']
    for name in ('protocol.json', 'keys.json', 'event_review.csv', 'audit.json', 'completed.json'):
        assert (output / name).is_file()
    with pytest.raises(FileExistsError):
        pipeline.run_pipeline(frozen, output)
    summary['unique_windows']['independent_fit'] = 99
    (output / 'summary.json').write_text(json.dumps(summary))
    marker = json.loads((output / 'completed.json').read_text())
    marker['sha256']['summary.json'] = sha256_file(output / 'summary.json')
    (output / 'completed.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='semantic'):
        pipeline.verify_pipeline(output)


def test_stage_failure_never_publishes_top_completion(frozen, tmp_path, monkeypatch):
    def fail(*args):
        raise RuntimeError('stage failure')
    monkeypatch.setattr(pipeline, 'build_report', fail)
    output = tmp_path / 'partial'
    with pytest.raises(RuntimeError, match='stage failure'):
        pipeline.run_pipeline(frozen, output)
    assert (output / 'predictions' / 'completed.json').is_file()
    assert not (output / 'completed.json').exists()
    progress = json.loads((output / 'progress.json').read_text())
    assert progress['state'] == 'failed'
    assert progress['completed_stages'] == ['predictions']
    with pytest.raises(ValueError, match='incomplete'):
        pipeline.verify_pipeline(output)


def test_cli_rejects_preflight_not_bound_to_freeze(frozen, tmp_path, monkeypatch):
    old = pipeline._read(frozen / 'freeze.json')['binding']['preflight']
    other = tmp_path / 'other-ready'
    shutil.copytree(old, other)
    def forbidden(*args):
        raise AssertionError('mismatched CLI must not start execution')
    monkeypatch.setattr(pipeline, 'run_pipeline', forbidden)
    monkeypatch.setattr(sys, 'argv', ['run_scenario_evaluation.py', '--preflight', str(other),
                                    '--freeze-root', str(frozen), '--output-root', str(tmp_path / 'run')])
    with pytest.raises(SystemExit) as error:
        pipeline.main()
    assert error.value.code == 2


@pytest.mark.parametrize('change', ['dirty', 'head'])
def test_formal_freeze_rechecks_git_before_publication(scenario_parent, tmp_path, monkeypatch, change):
    ready, output = tmp_path / 'ready', tmp_path / 'freeze'
    run_preflight(scenario_parent['parent'], scenario_parent['owner'], PROTOCOL_V1, ready)
    counts = {'status': 0, 'rev-parse': 0}
    def git(*args):
        key = args[0]
        counts[key] += 1
        if key == 'status':
            return ' M src/changed.py' if change == 'dirty' and counts[key] > 1 else ''
        return ('b' if change == 'head' and counts[key] > 1 else 'a') * 40
    monkeypatch.setattr(pipeline, '_git', git)
    with pytest.raises(ValueError, match='source|HEAD|commit|clean'):
        pipeline.freeze_protocol(ready, output)
    assert not output.exists()

import json

import numpy as np
import pytest

from trajectory_prediction.evaluation import scenario_publication as publication
from trajectory_prediction.evaluation.scenario_preflight import run_preflight
from trajectory_prediction.evaluation.scenario_replay import PROTOCOL_V1
from trajectory_prediction.utils.provenance import sha256_file


@pytest.fixture
def readiness(scenario_parent, tmp_path):
    path = tmp_path / 'ready'
    run_preflight(scenario_parent['parent'], scenario_parent['owner'], PROTOCOL_V1, path)
    return path


def test_plan_has_no_inference(readiness, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('plan must not infer')
    monkeypatch.setattr(publication, 'replay_streams', forbidden)
    plan = publication.plan_publication(readiness)
    assert plan['shards'] == 32
    assert plan['forecast_bytes'] == 304128
    assert plan['unique_windows'] == {'independent_fit': 1, 'independent_select': 1}


def test_publish_verify_and_refuse_overwrite(readiness, tmp_path):
    path = tmp_path / 'published'
    result = publication.publish_predictions(readiness, path)
    assert result['shards'] == 32
    assert publication.verify_publication(path)['frozen_replay_verified']
    run = json.loads((path / 'run.json').read_text())
    registry = run['binding']['strategies']
    assert sum(s['train_seed'] is None for s in registry) == 7
    with pytest.raises(FileExistsError):
        publication.publish_predictions(readiness, path)


def test_interruption_never_completes(readiness, tmp_path, monkeypatch):
    original = publication.replay_streams
    calls = 0
    def interrupt(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('simulated interruption')
        return original(*args, **kwargs)
    monkeypatch.setattr(publication, 'replay_streams', interrupt)
    path = tmp_path / 'partial'
    with pytest.raises(RuntimeError, match='interruption'):
        publication.publish_predictions(readiness, path)
    assert not (path / 'completed.json').exists()
    assert json.loads((path / 'progress.json').read_text()) == {
        'state': 'failed', 'completed_shards': 1, 'expected_shards': 32}
    with pytest.raises(ValueError, match='incomplete'):
        publication.verify_publication(path)


def test_stray_shard_rejected(readiness, tmp_path):
    path = tmp_path / 'published'
    publication.publish_predictions(readiness, path)
    (path / 'predictions' / 'stray.npz').write_bytes(b'bad')
    with pytest.raises(ValueError, match='coverage'):
        publication.verify_publication(path)


def test_rehashed_prediction_tamper_rejected(readiness, tmp_path):
    path = tmp_path / 'published'
    publication.publish_predictions(readiness, path)
    shard = path / 'predictions' / 'independent_fit__raw_cv.npz'
    with np.load(shard, allow_pickle=False) as saved:
        arrays = {k: saved[k] for k in saved.files}
    # GT y=0: reflection preserves distance error, but is a different prediction.
    arrays['forecast'][..., 1] *= -1
    np.savez_compressed(shard, **arrays)
    marker = json.loads((path / 'completed.json').read_text())
    marker['sha256'][shard.relative_to(path).as_posix()] = sha256_file(shard)
    (path / 'completed.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError, match='frozen replay'):
        publication.verify_publication(path)


def test_source_change_during_write_rejected(readiness, tmp_path, monkeypatch):
    original = publication.replay_streams
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        (readiness / 'report.json').write_text('{}')
        return result
    monkeypatch.setattr(publication, 'replay_streams', mutate)
    path = tmp_path / 'changed'
    with pytest.raises(ValueError, match='changed'):
        publication.publish_predictions(readiness, path)
    assert not (path / 'completed.json').exists()


def test_previous_shard_change_blocks_completion(readiness, tmp_path, monkeypatch):
    original = publication.replay_streams
    path = tmp_path / 'changed'
    calls = 0
    def mutate(*args, **kwargs):
        nonlocal calls
        calls += 1
        result = original(*args, **kwargs)
        if calls == 2:
            first = next((path / 'predictions').glob('*.npz'))
            first.write_bytes(b'changed after initial write')
        return result
    monkeypatch.setattr(publication, 'replay_streams', mutate)
    with pytest.raises(ValueError, match='artifact changed'):
        publication.publish_predictions(readiness, path)
    assert not (path / 'completed.json').exists()

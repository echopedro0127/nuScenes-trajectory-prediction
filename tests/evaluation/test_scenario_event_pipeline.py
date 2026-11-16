import json
from pathlib import Path

import numpy as np
import pytest
from test_scenario_tracks import expand_target_cache
from test_scenario_tracks import track_parent as track_parent  # noqa: PLC0414

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation import scenario_tracks
from trajectory_prediction.utils.provenance import sha256_file

CONFIG = Path(__file__).resolve().parents[2]/'configs/analysis/scenario_events_b1a.yaml'


def write(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def event_parent(track_parent, tmp_path, monkeypatch):
    parent, owner, metadata = expand_target_cache(track_parent)
    b_owner = tmp_path/'b0-owner'
    script = b_owner/'scripts/prepare_scenario_tracks.py'
    script.parent.mkdir(parents=True)
    script.write_text("import argparse,json\nfrom pathlib import Path\np=argparse.ArgumentParser()\n"
                      "p.add_argument('--output-root');p.add_argument('--verify-only',action='store_true')\n"
                      "a=p.parse_args();r=Path(a.output_root);s=json.loads((r/'summary.json').read_text())\n"
                      "i=json.loads((r/'run.json').read_text())['identity']\n"
                      "print(json.dumps({**s,'track_and_window_replayed':True,'identity':i}))\n")
    output = tmp_path/'b0b'
    with monkeypatch.context() as m:
        m.setattr(scenario_tracks, 'PROJECT', b_owner)
        scenario_tracks.prepare_tracks(parent, owner, output, metadata_root=metadata, scope='full_cache')
    return output, b_owner, metadata


def test_events_fixture_independent_rebuild(event_parent, tmp_path):
    from trajectory_prediction.evaluation.scenario_events import prepare_events, verify_events
    parent, owner, _ = event_parent
    output = tmp_path/'events'
    summary = prepare_events(parent, owner, CONFIG, output)
    assert summary['input_tracks'] == 4 and summary['input_annotations'] == 68
    assert summary['input_intervals'] == summary['valid_intervals'] == 64
    assert all(n == 0 for n in summary['event_counts'].values())
    for role in ('independent_fit', 'independent_select'):
        cat = summary['role_stats'][role]['category_counts']['vehicle.car']
        assert cat['input_intervals'] == cat['valid_intervals'] == 32
        assert cat['input_segments'] == cat['analysis_blocks'] == 2
        assert cat['right_censored_stops'] == cat['left_censored_stop_runs'] == 0
    assert verify_events(output)['events_semantically_replayed'] is True
    with pytest.raises(FileExistsError):
        prepare_events(parent, owner, CONFIG, output)


@pytest.mark.parametrize('kind', ['summary', 'protocol', 'event', 'trace', 'extra', 'schema', 'proof'])
def test_events_rehashed_tampering_rejected(event_parent, tmp_path, kind):
    from trajectory_prediction.evaluation.scenario_events import prepare_events, verify_events
    parent, owner, _ = event_parent
    output = tmp_path/'events'
    prepare_events(parent, owner, CONFIG, output)
    marker = json.loads((output/'completed.json').read_text())
    if kind == 'summary':
        path = output/'summary.json'; value = json.loads(path.read_text()); value['input_tracks'] = 999
    elif kind == 'protocol':
        path = output/'protocol.json'; value = json.loads(path.read_text()); value['offline_gt_only'] = False
    elif kind == 'event': path, value = output/'events.json', [{'event_id': 'invented'}]
    elif kind == 'trace': path, value = output/'state_trace.json', []
    elif kind == 'extra': path, value = output/'extra.json', {'extra': True}
    elif kind == 'schema':
        path = output/'run.json'; value = json.loads(path.read_text())
        value['binding']['schema'] = 'wrong'; value['identity'] = digest(value['binding'])
        marker['identity'] = value['identity']
    else:
        path = output/'parent_proof.json'; value = json.loads(path.read_text())
        payload = json.loads(value['stdout']); payload['track_and_window_replayed'] = 1
        value['stdout'] = json.dumps(payload)
    write(path, value); marker['sha256'][path.name] = sha256_file(path); write(output/'completed.json', marker)
    with pytest.raises(ValueError):
        verify_events(output)


def test_events_source_change_and_wrong_owner(event_parent, tmp_path):
    from trajectory_prediction.evaluation.scenario_events import prepare_events, verify_events
    parent, owner, metadata = event_parent
    with pytest.raises(ValueError, match='owner'):
        prepare_events(parent, tmp_path/'wrong', CONFIG, tmp_path/'wrong-output')
    output = tmp_path/'events'; prepare_events(parent, owner, CONFIG, output)
    (metadata/'sample_annotation.json').write_text('[]')
    with pytest.raises(ValueError, match='source|checksum'):
        verify_events(output)


def test_events_interruption_no_receipt(event_parent, tmp_path, monkeypatch):
    from trajectory_prediction.evaluation import scenario_events as mod
    parent, owner, _ = event_parent
    def fail(*args): raise RuntimeError('interrupted')
    monkeypatch.setattr(mod, '_save_npz', fail)
    output = tmp_path/'partial'
    with pytest.raises(RuntimeError): mod.prepare_events(parent, owner, CONFIG, output)
    assert not (output/'completed.json').exists()
    with pytest.raises(ValueError, match='incomplete'): mod.verify_events(output)


@pytest.mark.parametrize('kind', ['protocol_bool', 'run_bool'])
def test_integer_not_accepted_as_frozen_boolean(event_parent, tmp_path, kind):
    from trajectory_prediction.evaluation.scenario_events import prepare_events, verify_events
    parent, owner, _ = event_parent
    output = tmp_path/'events'; prepare_events(parent, owner, CONFIG, output)
    marker = json.loads((output/'completed.json').read_text())
    path = output/('protocol.json' if kind == 'protocol_bool' else 'run.json')
    value = json.loads(path.read_text())
    if kind == 'protocol_bool': value['offline_gt_only'] = 1
    else:
        value['binding']['neural_training'] = 0
        value['identity'] = digest(value['binding']); marker['identity'] = value['identity']
    write(path, value); marker['sha256'][path.name] = sha256_file(path); write(output/'completed.json', marker)
    with pytest.raises(ValueError, match='replay'):
        verify_events(output)


@pytest.mark.parametrize('text', ['rule_version: a\nrule_version: b', 'unknown: 1',
                                CONFIG.read_text().replace('stop_speed_mps: 0.3', 'stop_speed_mps: true')])
def test_invalid_yaml_rejected_before_parent(tmp_path, monkeypatch, text):
    from trajectory_prediction.evaluation import scenario_events as mod
    path = tmp_path/'config.yaml'; path.write_text(text)
    def forbidden(*args): pytest.fail('invalid config must fail before native parent')
    monkeypatch.setattr(mod, 'parent_snapshot', forbidden)
    with pytest.raises(ValueError):
        mod.prepare_events(tmp_path/'no-parent', tmp_path/'no-owner', path, tmp_path/'out')


def test_kinematics_dtype_tamper_rejected(event_parent, tmp_path):
    from trajectory_prediction.evaluation.scenario_events import prepare_events, verify_events
    parent, owner, _ = event_parent
    output = tmp_path/'events'; prepare_events(parent, owner, CONFIG, output)
    path = output/'independent_fit-kinematics.npz'
    with np.load(path, allow_pickle=False) as z: arrays = {k:z[k] for k in z.files}
    arrays['start_us'] = arrays['start_us'].astype(np.int32); np.savez_compressed(path, **arrays)
    marker = json.loads((output/'completed.json').read_text())
    marker['sha256'][path.name] = sha256_file(path); write(output/'completed.json', marker)
    with pytest.raises(ValueError, match='replay'):
        verify_events(output)

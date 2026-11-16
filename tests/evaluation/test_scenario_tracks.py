import hashlib
import json

import numpy as np
import pytest

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.data.geometry import global_to_agent
from trajectory_prediction.evaluation import scenario_tracks as tracks
from trajectory_prediction.utils.provenance import sha256_file


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def seal(root, identity, marker='completed.json'):
    names = [p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and p.name != marker]
    write(root / marker, {'identity': identity, 'sha256': {n: sha256_file(root / n) for n in names}})


@pytest.fixture
def track_parent(tmp_path):
    owner, parent = tmp_path / 'owner', tmp_path / 'parent'
    metadata, cache = tmp_path / 'v1.0-trainval', tmp_path / 'cache'
    preflight, freeze, phase2 = tmp_path / 'preflight', tmp_path / 'freeze', tmp_path / 'phase2'
    script = owner / 'scripts/run_scenario_evaluation.py'
    script.parent.mkdir(parents=True)
    script.write_text("import argparse,json\nfrom pathlib import Path\np=argparse.ArgumentParser()\n"
                      "p.add_argument('--output-root');p.add_argument('--verify-only',action='store_true')\n"
                      "a=p.parse_args();s=json.loads((Path(a.output_root)/'summary.json').read_text())\n"
                      "print(json.dumps({**s,'top_level_semantic_replay_verified':True}))\n")
    samples, annotations, instances, scenes, logs, parent_windows = [], [], [], [], [], []
    for role in ('independent_fit', 'independent_select'):
        scene, log, instance = 'scene-' + role, 'log-' + role, 'i-' + role
        for i in range(17):
            samples.append({'token': f's-{role}-{i}', 'timestamp': 1000000 + (i // 2) * 1000000 + (400000 if i % 2 else 0),
                            'prev': f's-{role}-{i-1}' if i else '',
                            'next': f's-{role}-{i+1}' if i < 16 else '', 'scene_token': scene})
            annotations.append({'token': f'a-{role}-{i}', 'instance_token': instance,
                                'sample_token': f's-{role}-{i}', 'prev': f'a-{role}-{i-1}' if i else '',
                                'next': f'a-{role}-{i+1}' if i < 16 else '',
                                'translation': [i * .5, 0., 0.], 'rotation': [1., 0., 0., 0.]})
        instances.append({'token': instance, 'category_token': 'cat', 'nbr_annotations': 17,
                          'first_annotation_token': f'a-{role}-0', 'last_annotation_token': f'a-{role}-16'})
        scenes.append({'token': scene, 'name': scene, 'log_token': log})
        logs.append({'token': log})
        parent_windows.append({'role': role, 'instance': instance, 'sample': f's-{role}-4',
                               'window_id': hashlib.sha256(f'{role}:{instance}:s-{role}-4'.encode()).hexdigest()})
        anchor = np.array([2., 0.], np.float32)
        history = global_to_agent(np.array([[i * .5, 0.] for i in range(5)], np.float32), anchor, 0.)
        future = global_to_agent(np.array([[i * .5, 0.] for i in range(5, 17)], np.float64), anchor, 0.)
        archive = cache / role / 'gt.npz'
        archive.parent.mkdir(parents=True)
        np.savez_compressed(archive, history_xy=history[None], future_xy=future[None],
                            history_mask=np.ones((1, 5), bool), anchor_xy=anchor[None],
                            instance_token=np.array([instance]), sample_token=np.array([f's-{role}-4']),
                            scene_name=np.array([scene]))
        write(archive.parent / 'manifest.json', {'npz_file': 'gt.npz', 'npz_sha256': sha256_file(archive),
                                                'dataset_version': 'v1.0-trainval', 'split': role})
        write(parent / 'predictions' / f'{role}-keys.json', [
            {'role': role, 'instance': instance, 'sample': f's-{role}-4', 'scene': scene, 'log': log,
             'row': 0, 'gt_index': 0, 'condition': c, 'noise_seed': s, 'stream_index': j}
            for j, (c, s) in enumerate((c, s) for c in ('clean', 'noise_05', 'noise_outlier') for s in (2027, 2028, 2029))])
    for name, rows in [('sample', samples), ('sample_annotation', annotations), ('instance', instances),
                       ('scene', scenes), ('log', logs), ('category', [{'token': 'cat', 'name': 'vehicle.car'}])]:
        write(metadata / (name + '.json'), rows)
    write(phase2 / 'run.json', {'binding': {'source_cache_root': str(cache)}})
    references = {str(p): {'path': str(p), 'sha256': sha256_file(p)} for p in
                  [phase2 / 'run.json', *metadata.glob('*.json'), *cache.rglob('*.json'), *cache.rglob('*.npz')]}
    pb = {'parent_path': str(phase2), 'references': references}
    write(preflight / 'run.json', {'identity': digest(pb), 'binding': pb})
    seal(preflight, digest(pb), 'preflight_completed.json')
    fb = {'preflight': str(preflight), 'sources': {'scripts/run_scenario_evaluation.py': sha256_file(script)},
          'purpose': 'fixture'}
    write(freeze / 'freeze.json', {'identity': digest(fb), 'binding': fb})
    seal(freeze, digest(fb))
    binding = {'schema': 'scenario_phase3a_pipeline_v1', 'freeze_root': str(freeze),
               'freeze_identity': digest(fb), 'freeze_marker_sha256': sha256_file(freeze / 'completed.json')}
    write(parent / 'run.json', {'identity': digest(binding), 'binding': binding})
    write(parent / 'summary.json', {'status': 'phase3a_verified', 'formal_full_run': False,
                                   'purpose': 'fixture', 'neural_training': False, 'holdout_evaluated': False})
    write(parent / 'events/windows.json', parent_windows)
    seal(parent, digest(binding))
    return parent, owner, metadata


def test_real_clock_window_roundtrip_and_rehashed_tamper(track_parent, tmp_path):
    parent, owner, metadata = track_parent
    output = tmp_path / 'tracks'
    result = tracks.prepare_tracks(parent, owner, output, metadata_root=metadata)
    assert result['target_instances'] == 2
    assert result['annotation_rows'] == 34
    assert result['window_count'] == 2
    assert result['canonical_episodes_extracted'] is False
    assert tracks.verify_tracks(output)['track_and_window_replayed']
    original_ids = {(w['role'], w['instance'], w['sample']): w['window_id']
                    for w in json.loads((parent / 'events/windows.json').read_text())}
    for w in json.loads((output / 'windows.json').read_text()):
        assert w['window_id'] == original_ids[w['role'], w['instance'], w['sample']]
    with np.load(output / 'independent_fit-windows.npz', allow_pickle=False) as z:
        assert z['actual_offset_us'][0, 5] == 400000
        assert z['nominal_step'].tolist() == list(range(-4, 13))
    proof = json.loads((output / 'parent_proof.json').read_text())
    assert proof['owner_root'] == str(owner.resolve())
    assert proof['pythonpath'] == str((owner / 'src').resolve())
    with pytest.raises(FileExistsError):
        tracks.prepare_tracks(parent, owner, output, metadata_root=metadata)
    summary = json.loads((output / 'summary.json').read_text())
    summary['window_count'] = 99
    write(output / 'summary.json', summary)
    marker = json.loads((output / 'completed.json').read_text())
    marker['sha256']['summary.json'] = sha256_file(output / 'summary.json')
    write(output / 'completed.json', marker)
    with pytest.raises(ValueError, match='replay'):
        tracks.verify_tracks(output)


def test_source_change_and_wrong_owner_refused(track_parent, tmp_path):
    parent, owner, metadata = track_parent
    with pytest.raises(ValueError, match='owner'):
        tracks.prepare_tracks(parent, tmp_path / 'wrong-owner', tmp_path / 'bad', metadata_root=metadata)
    output = tmp_path / 'ok'
    tracks.prepare_tracks(parent, owner, output, metadata_root=metadata)
    (metadata / 'sample_annotation.json').write_text('[]')
    with pytest.raises(ValueError, match='checksum|source'):
        tracks.verify_tracks(output)


def test_interruption_does_not_complete(track_parent, tmp_path, monkeypatch):
    parent, owner, metadata = track_parent
    def fail(*args, **kwargs):
        raise RuntimeError('interrupted')
    monkeypatch.setattr(tracks, '_save_npz', fail)
    output = tmp_path / 'partial'
    with pytest.raises(RuntimeError, match='interrupted'):
        tracks.prepare_tracks(parent, owner, output, metadata_root=metadata)
    assert not (output / 'completed.json').exists()
    with pytest.raises(ValueError, match='incomplete'):
        tracks.verify_tracks(output)


@pytest.mark.parametrize('kind', ['protocol', 'proof', 'extra_file', 'run_flags', 'run_schema', 'run_scope', 'run_runtime'])
def test_rehashed_policy_changes_rejected(track_parent, tmp_path, kind):
    parent, owner, metadata = track_parent
    output = tmp_path / 'run'
    tracks.prepare_tracks(parent, owner, output, metadata_root=metadata)
    marker = json.loads((output / 'completed.json').read_text())
    if kind == 'protocol':
        path = output / 'protocol.json'; value = json.loads(path.read_text()); value['nominal_dt_s'] = .25
        write(path, value)
    elif kind == 'proof':
        path = output / 'parent_proof.json'; value = json.loads(path.read_text())
        payload = json.loads(value['stdout']); payload['top_level_semantic_replay_verified'] = 1
        value['stdout'] = json.dumps(payload); write(path, value)
    elif kind == 'extra_file':
        path = output / 'extra.txt'; path.write_text('stray file')
    else:
        path = output / 'run.json'; value = json.loads(path.read_text())
        if kind == 'run_flags': value['binding']['neural_training'] = True
        elif kind == 'run_schema': value['binding']['schema'] = 'unknown'
        elif kind == 'run_scope': value['binding']['scope'] = 'all_objects'
        else: value['binding']['runtime']['numpy'] = 'different-runtime'
        value['identity'] = digest(value['binding']); marker['identity'] = value['identity']; write(path, value)
    marker['sha256'][path.name] = sha256_file(path)
    write(output / 'completed.json', marker)
    with pytest.raises(ValueError, match='replay|proof|policy|protocol|coverage|identity'):
        tracks.verify_tracks(output)


def test_stray_nested_file_blocks_completion(track_parent, tmp_path, monkeypatch):
    parent, owner, metadata = track_parent
    original = tracks._save_npz
    def inject(path, arrays):
        stray = path.parent / 'stray' / 'note.txt'
        stray.parent.mkdir(exist_ok=True)
        stray.write_text('unexpected')
        original(path, arrays)
    monkeypatch.setattr(tracks, '_save_npz', inject)
    output = tmp_path / 'changed'
    with pytest.raises(ValueError, match='coverage'):
        tracks.prepare_tracks(parent, owner, output, metadata_root=metadata)
    assert not (output / 'completed.json').exists()


def expand_target_cache(track_parent, defect=None):
    parent, owner, metadata = track_parent
    cache = metadata.parent / 'cache'
    annotations = json.loads((metadata / 'sample_annotation.json').read_text())
    instances = json.loads((metadata / 'instance.json').read_text())
    for role in tracks.ROLES:
        original = [a for a in annotations if a['instance_token'] == 'i-' + role]
        for prefix in ('new', 'decoy'):
            instance = prefix + '-' + role
            copied = []
            for a in original:
                copied.append({**a, 'token': prefix + a['token'], 'instance_token': instance,
                               'prev': prefix + a['prev'] if a['prev'] else '',
                               'next': prefix + a['next'] if a['next'] else '',
                               'translation': [a['translation'][0] + 10., 0., 0.]})
            annotations.extend(copied)
            instances.append({'token': instance, 'category_token': 'cat', 'nbr_annotations': 17,
                              'first_annotation_token': copied[0]['token'], 'last_annotation_token': copied[-1]['token']})
        archive = cache / role / 'gt.npz'
        with np.load(archive, allow_pickle=False) as z:
            arrays = {n: np.concatenate([z[n], z[n]], axis=0) for n in z.files}
        arrays['instance_token'] = np.array(['i-' + role, 'new-' + role])
        arrays['anchor_xy'][1] += np.array([10., 0.], np.float32)
        if defect == 'duplicate_key' and role == 'independent_fit':
            arrays['instance_token'][1] = arrays['instance_token'][0]
        if defect == 'role_overlap' and role == 'independent_select':
            arrays['instance_token'][1] = 'new-independent_fit'
        np.savez_compressed(archive, **arrays)
        manifest_path = archive.parent / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        manifest['npz_sha256'] = sha256_file(archive)
        write(manifest_path, manifest)
    write(metadata / 'sample_annotation.json', annotations)
    write(metadata / 'instance.json', instances)
    preflight = metadata.parent / 'preflight'
    ready = json.loads((preflight / 'run.json').read_text())
    for ref in ready['binding']['references'].values():
        ref['sha256'] = sha256_file(ref['path'])
    ready['identity'] = digest(ready['binding'])
    write(preflight / 'run.json', ready)
    seal(preflight, ready['identity'], 'preflight_completed.json')
    return parent, owner, metadata


def test_full_cache_scope_covers_new_targets_without_decoy_objects(track_parent, tmp_path):
    parent, owner, metadata = expand_target_cache(track_parent)
    pilot, full = tmp_path / 'pilot', tmp_path / 'full'
    assert tracks.prepare_tracks(parent, owner, pilot, metadata_root=metadata)['target_instances'] == 2
    result = tracks.prepare_tracks(parent, owner, full, metadata_root=metadata, scope='full_cache')
    assert result['scope'] == 'original_fit_select_cache_target_instances'
    assert result['target_instances'] == 4
    assert result['annotation_rows'] == 68
    assert result['window_count'] == 4
    assert tracks.verify_tracks(full)['track_and_window_replayed'] is True
    old_tracks = json.loads((pilot / 'tracks.json').read_text())
    new_tracks = json.loads((full / 'tracks.json').read_text())
    for old in old_tracks:
        new = next(t for t in new_tracks if (t['role'], t['instance']) == (old['role'], old['instance']))
        assert new['semantic_sha256'] == old['semantic_sha256']
    windows = json.loads((full / 'windows.json').read_text())
    assert sum(w['phase3a_pilot_member'] for w in windows) == 2
    assert {w['row'] for w in windows} == {0, 1}
    assert not any('decoy' in w['instance'] for w in windows)


@pytest.mark.parametrize('defect', ['duplicate_key', 'role_overlap'])
def test_full_cache_invalid_scope_membership_rejected(track_parent, tmp_path, defect):
    parent, owner, metadata = expand_target_cache(track_parent, defect)
    output = tmp_path / 'invalid'
    with pytest.raises(ValueError, match='duplicate|role|overlap'):
        tracks.prepare_tracks(parent, owner, output, metadata_root=metadata, scope='full_cache')
    assert not (output / 'completed.json').exists()


def test_unknown_scope_is_rejected_before_parent_replay(track_parent, tmp_path, monkeypatch):
    parent, owner, metadata = track_parent
    def unexpected(*args, **kwargs):
        pytest.fail('unknown scope must be rejected before parent replay')
    monkeypatch.setattr(tracks, 'parent_snapshot', unexpected)
    with pytest.raises(ValueError, match='scope'):
        tracks.prepare_tracks(parent, owner, tmp_path / 'unknown', metadata_root=metadata, scope='all_objects')


def test_full_cache_scope_tamper_rejected_after_rehash(track_parent, tmp_path):
    parent, owner, metadata = expand_target_cache(track_parent)
    output = tmp_path / 'full'
    tracks.prepare_tracks(parent, owner, output, metadata_root=metadata, scope='full_cache')
    run = json.loads((output / 'run.json').read_text())
    run['binding']['scope'] = 'phase3a_pilot_target_instances'
    run['identity'] = digest(run['binding'])
    write(output / 'run.json', run)
    marker = json.loads((output / 'completed.json').read_text())
    marker['identity'] = run['identity']
    marker['sha256']['run.json'] = sha256_file(output / 'run.json')
    write(output / 'completed.json', marker)
    with pytest.raises(ValueError, match='policy|scope|replay'):
        tracks.verify_tracks(output)

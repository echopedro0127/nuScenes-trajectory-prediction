"""Identity, GT provenance and resource integrity for scenario evaluation."""
import json

import numpy as np
import pytest

from trajectory_prediction.data.cache import load_cache, save_cache
from trajectory_prediction.evaluation.scenario_inputs import assemble_inputs
from trajectory_prediction.utils.provenance import sha256_file


def test_assembly_separates_gt_from_model_inputs(scenario_parent):
    f = scenario_parent
    result = assemble_inputs(f['parent'], f['owner'])
    role = 'independent_select'
    gt = result.gt_windows[role][0]
    assert gt['history_xy'].shape == (5, 2)
    assert gt['future_xy'].shape == (12, 2)
    assert gt['history_mask'].all()
    assert gt['coordinate_frame'] == 'anchor_agent_y_forward'
    assert gt['units'] == 'm'
    assert gt['query_times_s'].tolist() == [.5 * k for k in range(1, 12)]
    assert gt['history_times_s'].tolist() == [-2., -1.5, -1., -.5, 0.]
    assert gt['window_key'] == (role, role + '-instance', role + '-sample')
    assert 'history_xy' not in result.datasets[role]
    assert result.parent_identity and result.references
    assert result.scene_reference['path'].endswith('scene.json')


@pytest.mark.parametrize('field,value', [
    ('instance', 'wrong'), ('sample', 'wrong'), ('scene', 'wrong'),
    ('log', 'wrong'), ('role', 'independent_holdout'), ('seed', 2030),
    ('row', 1), ('condition', 'missing'),
])
def test_record_identity_and_declared_cohorts_are_strict(scenario_parent, field, value):
    f = scenario_parent
    f['records']['independent_select'][0][field] = value
    (f['parent'] / 'records.json').write_text(json.dumps(f['records']))
    f['seal']()
    with pytest.raises(ValueError):
        assemble_inputs(f['parent'], f['owner'])


def test_duplicate_stream_and_missing_noise_pair_are_errors(scenario_parent):
    f = scenario_parent
    rows = f['records']['independent_select']
    rows[1] = dict(rows[0])
    (f['parent'] / 'records.json').write_text(json.dumps(f['records']))
    f['seal']()
    with pytest.raises(ValueError, match='duplicate|cohort'):
        assemble_inputs(f['parent'], f['owner'])


@pytest.mark.parametrize('kind', ['future', 'mask', 'frequency', 'partition'])
def test_gt_cache_contract_is_checked(scenario_parent, kind):
    f = scenario_parent
    role = 'independent_select'
    arrays, manifest = load_cache(f['cache'] / role, role)
    if kind == 'future':
        arrays['future_xy'][0, 0, 0] += 1
    elif kind == 'mask':
        arrays['history_mask'][0, 0] = False
    elif kind == 'frequency':
        manifest['frequency_hz'] = 10
    else:
        manifest['partition_identity'] = 'wrong-partition'
    save_cache(f['cache'] / role, arrays, manifest)
    with pytest.raises(ValueError):
        assemble_inputs(f['parent'], f['owner'])


@pytest.mark.parametrize('kind', ['nan', 'shape', 'dtype'])
def test_parent_arrays_reject_invalid_model_inputs(scenario_parent, kind):
    f = scenario_parent
    data = f['datasets']['independent_select']
    if kind == 'nan':
        data['candidates'][0, 0, 0, 0, 0] = np.nan
    elif kind == 'shape':
        data['features'] = data['features'][:, :10]
    else:
        data['features'] = data['features'].astype('float64')
    np.savez_compressed(f['parent'] / 'independent_select.npz', **data)
    f['seal']()
    with pytest.raises(ValueError):
        assemble_inputs(f['parent'], f['owner'])


def test_missing_checkpoint_and_tampering_are_not_skipped(scenario_parent):
    f = scenario_parent
    checkpoint = f['parent'] / 's2027_baseline7.pt'
    checkpoint.unlink()
    with pytest.raises(FileNotFoundError, match='s2027_baseline7'):
        assemble_inputs(f['parent'], f['owner'])


def test_wrong_owner_and_changed_parent_source_are_rejected(scenario_parent):
    f = scenario_parent
    with pytest.raises(ValueError, match='owner|project'):
        assemble_inputs(f['parent'], f['parent'])
    (f['owner'] / 'scripts/run_noise_aware_stability.py').write_text('changed')
    with pytest.raises(ValueError, match='source|checksum'):
        assemble_inputs(f['parent'], f['owner'])


def test_roles_cannot_share_a_log_even_when_file_hashes_are_updated(scenario_parent):
    f = scenario_parent
    path = f['owner'] / 'scene.json'
    scenes = json.loads(path.read_text())
    scenes[1]['log_token'] = scenes[0]['log_token']
    path.write_text(json.dumps(scenes))
    f['binding']['references']['scene_metadata']['sha256'] = sha256_file(path)
    for row in f['records']['independent_select']:
        row['log'] = scenes[0]['log_token']
    (f['parent'] / 'records.json').write_text(json.dumps(f['records']))
    f['seal']()
    with pytest.raises(ValueError, match='overlap'):
        assemble_inputs(f['parent'], f['owner'])


@pytest.mark.parametrize('kind', ['missing_stats', 'select_normalization', 'wrong_seed'])
def test_checkpoint_metadata_and_fit_only_stats_are_required(scenario_parent, kind):
    import torch
    f = scenario_parent
    path = f['parent'] / 's2027_baseline7.pt'
    payload = torch.load(path, weights_only=True)
    if kind == 'missing_stats':
        del payload['stats']
    elif kind == 'select_normalization':
        payload['stats']['mean'][0] += .2
    else:
        payload['train_seed'] = 2028
    torch.save(payload, path)
    f['seal']()
    with pytest.raises(ValueError, match='identity|stats|fit only'):
        assemble_inputs(f['parent'], f['owner'])

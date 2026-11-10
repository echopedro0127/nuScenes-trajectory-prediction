"""Bound local frozen models, static maps and development-only noisy streams."""
import json
from pathlib import Path

import numpy as np
import torch

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.maps import MapConfig, MapProvider
from trajectory_prediction.evaluation.runner import _mapping_sha256
from trajectory_prediction.models.factory import build_model
from trajectory_prediction.online.independent_map_training import load_pair
from trajectory_prediction.robustness.benchmark import ROLES, load_development, verify
from trajectory_prediction.robustness.noisy_candidates import build_candidates
from trajectory_prediction.robustness.packets import PROFILES, make_packets
from trajectory_prediction.utils.provenance import sha256_file

CONDITIONS = ('clean', 'noise_05', 'noise_outlier')


def reference(path):
    path = Path(path).resolve()
    return {'path': str(path), 'sha256': sha256_file(path)}


def load_models(root, model_config):
    root = Path(root)
    model_config = Path(model_config)
    payload = torch.load(root/'map-training/seed-2027/best.pt', map_location='cpu', weights_only=True)
    pair, _ = load_pair(root/'map-training/seed-2027/best.pt', payload['input_metadata'])
    checkpoint = root/'gru-seed-2027/best.pt'
    expected = payload['input_metadata']['base_model']
    if (sha256_file(checkpoint) != expected['checkpoint_sha256']
            or _mapping_sha256(load_yaml(model_config)) != expected['model_config_sha256']):
        raise ValueError('frozen GRU/map parent identity mismatch')
    base = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if base['metadata'] != payload['input_metadata']['upstream_training']:
        raise ValueError('frozen GRU training roles mismatch')
    gru = build_model(load_yaml(model_config)['model'])
    gru.load_state_dict(base['model'], strict=True)
    for model in (gru, pair):
        model.requires_grad_(False)
        model.eval()
    return gru, pair, payload


def prepare_inputs(root, phase1, model_config, dataroot):
    root, phase1, dataroot = Path(root).resolve(), Path(phase1).resolve(), Path(dataroot).resolve()
    parent = verify(phase1)
    run = json.loads((phase1/'run.json').read_text())['binding']
    loaded, manifests = load_development(root/'cache', dataroot/'v1.0-trainval/scene.json')
    gru, pair, payload = load_models(root, model_config)
    partition = payload['input_metadata']['partition_identity']
    for role in ROLES:
        if (manifests[role]['npz_sha256'] != run['parents'][role]['archive_sha256']
                or manifests[role]['partition_identity'] != partition):
            raise ValueError('Phase 1/model/development partition mismatch')
    tables = {name: json.loads((dataroot/'v1.0-trainval'/f'{name}.json').read_text(encoding='utf-8'))
              for name in ('sample', 'scene', 'log')}
    samples = {r['token']: r for r in tables['sample']}
    scenes = {r['name']: r for r in tables['scene']}
    logs = {r['token']: r for r in tables['log']}
    references = {str(path): reference(path) for path in
                  [phase1/'completed.json', phase1/'run.json', phase1/'samples.json', model_config,
                   root/'gru-seed-2027/best.pt', root/'map-training/seed-2027/best.pt',
                   *[dataroot/'v1.0-trainval'/f'{name}.json' for name in tables],
                   *sorted((dataroot/'maps/expansion').glob('*.json')),
                   *[root/'cache'/role/'manifest.json' for role in ROLES]]}
    for role in ROLES:
        name = json.loads((root/'cache'/role/'manifest.json').read_text())['npz_file']
        references[str(root/'cache'/role/name)] = reference(root/'cache'/role/name)
    binding = {'phase1_identity': parent['identity'], 'references': references,
               'model_root': str(root), 'phase1_root': str(phase1), 'model_config': str(Path(model_config).resolve()),
               'dataroot': str(dataroot), 'conditions': list(CONDITIONS), 'noise_seeds': run['noise_seeds'],
               'partition_identity': partition, 'candidate_model_policy': 'frozen_clean_trained_recomputed_noisy_inputs',
               'coordinate_assumption': 'fixed_metadata_heading_and_map_reference', 'holdout_evaluated': False}
    provider = MapProvider(dataroot, MapConfig())
    context = (loaded, run, gru, pair, samples, scenes, logs, provider)
    return binding, context


def rebuild_one(context, sample, condition, seed):
    loaded, _, gru, pair, samples, scenes, logs, provider = context
    arrays, _ = loaded[sample['role']]
    i = sample['row']
    location = logs[scenes[str(arrays['scene_name'][i])]['log_token']]['location']
    timestamp = samples[str(arrays['sample_token'][i])]['timestamp']
    key = str(arrays['instance_token'][i])+':'+str(arrays['sample_token'][i])
    packets = make_packets(arrays['history_xy'][i], arrays['future_xy'][i, :11], key, PROFILES[condition], seed)
    return build_candidates(packets, gru, pair['original'], pair['map'], provider.get(location),
                            arrays['anchor_xy'][i], float(arrays['anchor_yaw'][i]), timestamp)


def generate_development(context):
    loaded, run, *_ = context
    datasets, records = {}, {}
    for role in ROLES:
        arrays, _ = loaded[role]
        records[role] = []
        values = {name: [] for name in ('features', 'candidates', 'target')}
        selected = [s for s in run['samples'] if s['role'] == role]
        for condition in CONDITIONS:
            for seed in run['noise_seeds']:
                for sample in selected:
                    result = rebuild_one(context, sample, condition, seed)
                    for name in ('features', 'candidates'):
                        values[name].append(result[name])
                    values['target'].append(arrays['future_xy'][sample['row']])
                    records[role].append({**sample, 'condition': condition, 'seed': seed})
                print(json.dumps({'prepared_role': role, 'condition': condition, 'seed': seed,
                                  'streams_so_far': len(records[role])}), flush=True)
        datasets[role] = {name: np.array(value, np.float32) for name, value in values.items()}
    return datasets, records

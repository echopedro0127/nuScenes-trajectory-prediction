"""Assemble evaluation GT separately from immutable Phase 2D model inputs."""
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.data.cache import load_cache
from trajectory_prediction.evaluation.scenario_replay import PROTOCOL_V1
from trajectory_prediction.utils.provenance import sha256_file


@dataclass(frozen=True)
class ScenarioInputs:
    parent: Path
    parent_project_root: Path
    parent_identity: str
    binding: dict
    datasets: dict
    records: dict
    gt_windows: dict
    references: dict
    scene_reference: dict
    owner_roots: dict
    legacy_updates: dict
    legacy_means: dict


def _json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _reference(path, references, expected=None):
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f'required parent resource missing: {path}; see development §4.3')
    name = str(path)
    digest = references[name]['sha256'] if name in references else sha256_file(path)
    if expected is not None and digest != expected:
        raise ValueError(f'parent checksum mismatch: {path}')
    references[name] = {'path': name, 'sha256': digest}
    return references[name]


def _inside(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root) or Path(name).is_absolute():
        raise ValueError('parent manifest path escapes owner project/output')
    return path


def _lineage(parent, owner, references, owners, seen):
    """Record actual Git owners rather than guessing from output path depth."""
    if str(parent) in seen:
        raise ValueError('parent lineage cycle')
    seen.add(str(parent))
    if not parent.is_relative_to(owner):
        raise ValueError('parent output must belong to explicit owner project root')
    _reference(parent / 'completed.json', references)
    marker = _json(parent / 'completed.json')
    names = marker.get('sha256', {})
    if 'run.json' not in names:
        raise ValueError('parent completion missing run.json')
    for name, digest in names.items():
        _reference(_inside(parent, name), references, digest)
    run = _json(parent / 'run.json')
    binding = run.get('binding', run)
    if run.get('identity') != marker['identity'] or digest_json(binding) != marker['identity']:
        raise ValueError('parent run identity mismatch')
    owners[marker['identity']] = str(owner)
    for name, digest in binding.get('sources', {}).items():
        _reference(_inside(owner, name), references, digest)
    for ref in binding.get('references', {}).values():
        _reference(ref['path'], references, ref['sha256'])
    scenes = [ref for ref in binding.get('references', {}).values()
              if Path(ref['path']).name == 'scene.json']
    if 'scene_metadata' in binding:
        ref = binding['scene_metadata']
        _reference(ref['path'], references, ref['sha256'])
        scenes.append(ref)
    upstream = binding.get('parent_path')
    if upstream:
        upstream = Path(upstream).resolve()
        # Git supplies the actual worktree owner; the resolved map is bound to the run.
        process = subprocess.run(
            ['git', '-c', 'safe.directory=*', '-C', str(upstream),
             'rev-parse', '--show-toplevel'], check=True, capture_output=True, text=True)
        upstream_owner = Path(process.stdout.strip()).resolve()
        upstream_binding, upstream_scene = _lineage(
            upstream, upstream_owner, references, owners, seen)
        del upstream_binding
        if binding.get('parent_identity') != _json(upstream / 'completed.json')['identity']:
            raise ValueError('upstream parent identity mismatch')
        if upstream_scene:
            scenes.append(upstream_scene)
    unique = {(str(Path(r['path']).resolve()), r['sha256']) for r in scenes}
    if len(unique) > 1:
        raise ValueError('inconsistent scene metadata references')
    return binding, scenes[0] if scenes else None


def _validate_arrays(data, n):
    shapes = {'features': (n, 11, 18), 'candidates': (n, 11, 10, 12, 2),
              'target': (n, 12, 2), 'ca_allowed': (n, 11),
              'guard_yaw': (n, 11), 'guard_min_speed': (n, 11)}
    if n < 1 or set(data) != set(shapes):
        raise ValueError('nonempty complete Phase 2D model input fields required')
    for name, shape in shapes.items():
        value = data[name]
        dtype = np.dtype(bool if name == 'ca_allowed' else np.float32)
        if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
            raise ValueError(f'invalid parent {name} shape/dtype/nonfinite values')


def assemble_inputs(parent, parent_project_root):
    """Preflight hashes, identity and GT sources; never trains or reads holdout."""
    parent, owner = Path(parent).resolve(), Path(parent_project_root).resolve()
    if not (owner / 'scripts/run_noise_aware_stability.py').is_file():
        raise ValueError('explicit parent owner project must provide stability verifier')
    references, owners = {}, {}
    binding, scene_ref = _lineage(parent, owner, references, owners, set())
    if scene_ref is None:
        raise ValueError('bound scene metadata source is required')
    for key, expected in [('model_seeds', PROTOCOL_V1['model_seeds']),
                          ('noise_seeds', PROTOCOL_V1['noise_seeds']),
                          ('variants', PROTOCOL_V1['gate_variants'])]:
        if binding.get(key) != expected:
            raise ValueError(f'parent {key} differs from frozen protocol')
    scenes = _json(scene_ref['path'])
    scene_logs = {s['name']: s['log_token'] for s in scenes}
    if len(scene_logs) != len(scenes) or not all(scene_logs.values()):
        raise ValueError('unique scene/log metadata required')
    roles = PROTOCOL_V1['roles']
    records = _json(parent / 'records.json')
    if set(records) != set(roles):
        raise ValueError('only development fit/select roles allowed')
    cache_root = Path(binding['source_cache_root']).resolve()
    datasets, windows, manifests = {}, {}, {}
    seen_logs, seen_scenes, seen_keys = set(), set(), set()
    for role in roles:
        for name in (role + '.npz', 'records.json', 'per_update.npz'):
            if str(parent / name) not in references:
                raise ValueError(f'parent completion does not bind {name}')
        with np.load(parent / (role + '.npz'), allow_pickle=False) as z:
            data = {k: z[k] for k in z.files}
        rows = records[role]
        _validate_arrays(data, len(rows))
        original, manifest = load_cache(cache_root / role, role)
        manifests[role] = manifest
        if (manifest.get('frequency_hz') != 2 or manifest.get('history_seconds') != 2
                or manifest.get('future_seconds') != 6):
            raise ValueError('GT time grid must be regular 2Hz, 2s history and 6s future')
        for name in ('manifest.json', manifest['npz_file']):
            _reference(cache_root / role / name, references)
        keys = list(zip(original['instance_token'], original['sample_token'], strict=True))
        if len(set(keys)) != len(keys):
            raise ValueError('duplicate GT instance/sample key')
        lookup = {key: i for i, key in enumerate(keys)}
        streams, cohorts, selected = set(), {}, []
        role_logs, role_scenes, role_keys = set(), set(), set()
        for i, row in enumerate(rows):
            if (row.get('role') != role or row.get('condition') not in PROTOCOL_V1['conditions']
                    or type(row.get('seed')) is not int
                    or row['seed'] not in PROTOCOL_V1['noise_seeds']):
                raise ValueError('record role/condition/noise seed mismatch')
            key = (row.get('instance'), row.get('sample'))
            index = lookup.get(key)
            if (index is None or type(row.get('row')) is not int or row['row'] != index
                    or row.get('scene') != original['scene_name'][index]
                    or scene_logs.get(row['scene']) != row.get('log')):
                raise ValueError('window identity/row/scene/log mismatch')
            stream = (*key, row['condition'], row['seed'])
            if stream in streams:
                raise ValueError('duplicate stream key')
            streams.add(stream)
            cohorts.setdefault(key, set()).add((row['condition'], row['seed']))
            if not np.array_equal(data['target'][i], original['future_xy'][index]):
                raise ValueError('GT future differs from parent target')
            if not original['history_mask'][index].all():
                raise ValueError('v1 requires complete model GT history')
            role_logs.add(row['log'])
            role_scenes.add(row['scene'])
            role_keys.add(key)
            selected.append({
                'window_key': (role, *key), 'source_row': index,
                'scene': row['scene'], 'log': row['log'],
                'history_xy': original['history_xy'][index].copy(),
                'history_mask': original['history_mask'][index].copy(),
                'future_xy': original['future_xy'][index].copy(),
                'anchor_xy': original['anchor_xy'][index].copy(),
                'anchor_yaw': float(original['anchor_yaw'][index]),
                'history_times_s': np.arange(-4, 1) * .5,
                'query_times_s': np.arange(1, 12) * .5,
                'future_times_s': np.arange(1, 13) * .5,
                'coordinate_frame': 'anchor_agent_y_forward', 'units': 'm',
                'source_manifest': str(cache_root / role / 'manifest.json'),
                'source_archive': str(cache_root / role / manifest['npz_file']),
            })
        expected = {(c, s) for c in PROTOCOL_V1['conditions'] for s in PROTOCOL_V1['noise_seeds']}
        if any(group != expected for group in cohorts.values()):
            raise ValueError('incomplete condition/noise cohort or missing expected pair')
        # Check entire role cache, not just the sampled pilot windows.
        all_scenes = set(original['scene_name'].tolist())
        if not all_scenes <= set(scene_logs):
            raise ValueError('GT scene missing from bound metadata')
        all_logs = {scene_logs[s] for s in all_scenes}
        if seen_logs & all_logs or seen_scenes & all_scenes or seen_keys & set(keys):
            raise ValueError('development role scene/log/key overlap')
        seen_logs.update(all_logs)
        seen_scenes.update(all_scenes)
        seen_keys.update(keys)
        datasets[role], windows[role] = data, selected
    for name in ('partition_identity', 'preprocessing_config_sha256'):
        values = [manifests[r].get(name) for r in roles]
        if not values[0] or values[0] != values[1]:
            raise ValueError(f'GT role {name} mismatch')
    fit = datasets[roles[0]]['features'].reshape(-1, 18)
    stats = {'mean': fit.mean(0, dtype=np.float64),
             'std': np.maximum(fit.std(0, dtype=np.float64), 1e-4)}
    for seed in PROTOCOL_V1['model_seeds']:
        for variant in PROTOCOL_V1['gate_variants']:
            path = parent / f's{seed}_{variant}.pt'
            if str(path) not in references:
                raise ValueError(f'parent completion missing checkpoint: {path.name}')
            payload = torch.load(path, map_location='cpu', weights_only=True)
            if (payload.get('train_seed') != seed or payload.get('variant') != variant
                    or 'model' not in payload or 'stats' not in payload):
                raise ValueError('checkpoint model identity/stats missing')
            if (payload['stats'].get('fit_streams') != len(datasets[roles[0]]['features'])
                    or any(not np.array_equal(payload['stats'].get(k), v)
                           for k, v in stats.items())):
                raise ValueError('checkpoint normalization must use fit only')
    identity = _json(parent / 'completed.json')['identity']
    with np.load(parent / 'per_update.npz', allow_pickle=False) as z:
        legacy_updates = {k: z[k] for k in z.files}
    if str(parent / 'per_trajectory.csv') not in references:
        raise ValueError('parent completion missing per_trajectory.csv')
    frame = pd.read_csv(parent / 'per_trajectory.csv', float_precision='round_trip',
                        dtype={'variant': 'string'}, low_memory=False)
    legacy_means = {}
    for row in frame.to_dict('records'):
        if row['strategy'] not in PROTOCOL_V1['physical_candidates']:
            continue
        key = (row['role'], row['instance'], row['sample'], row['condition'],
               int(row['seed']), row['strategy'])
        if key in legacy_means:
            raise ValueError('duplicate legacy physical metric key')
        value = np.array([row[k] for k in ('ade', 'fde', 'jitter')], np.float64)
        if not np.isfinite(value).all():
            raise ValueError('nonfinite legacy metric')
        legacy_means[key] = value
    return ScenarioInputs(parent, owner, identity, binding, datasets, records,
                          windows, references, scene_ref, owners, legacy_updates, legacy_means)

"""Tiny file-backed Phase 2D contract; no dataset download or training."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.data.cache import save_cache
from trajectory_prediction.online.batched_evaluation import forecast_metrics
from trajectory_prediction.robustness.motion_learning import MotionGate, predict_motion
from trajectory_prediction.utils.provenance import sha256_file


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


@pytest.fixture
def scenario_parent(tmp_path):
    owner = tmp_path / 'owner'
    parent = owner / 'outputs/noise-aware/phase2d'
    parent.mkdir(parents=True)
    cache = owner / 'cache'
    script = owner / 'scripts/run_noise_aware_stability.py'
    script.parent.mkdir()
    # A real subprocess contract fixture; production runs the real parent verifier.
    script.write_text(
        "import json,sys\nfrom pathlib import Path\n"
        "p=Path(sys.argv[sys.argv.index('--output-root')+1])\n"
        "i=json.loads((p/'completed.json').read_text())['identity']\n"
        "print(json.dumps({'identity':i,'all_seed_metrics_groups_weights_replayed':True,"
        "'fit_only_normalization_verified':True,'seed2027_two_controls_match_parent':True,"
        "'holdout_evaluated':False}))\n", encoding='utf-8')
    scenes = owner / 'scene.json'
    write_json(scenes, [{'name': role + '-scene', 'log_token': role + '-log'}
                        for role in ('independent_fit', 'independent_select')])
    rng = np.random.default_rng(71)
    datasets, records = {}, {}
    for role in ('independent_fit', 'independent_select'):
        xy = np.column_stack((np.arange(-4, 13) * .5, np.zeros(17))).astype('float32')
        arrays = {
            'history_xy': xy[:5][None], 'history_features': np.zeros((1, 5, 6), 'float32'),
            'history_mask': np.ones((1, 5), bool), 'future_xy': xy[5:][None],
            'anchor_xy': np.array([[100, 200]], 'float32'),
            'anchor_yaw': np.array([.3], 'float32'),
            'current_speed': np.array([1], 'float32'),
            'instance_token': np.array([role + '-instance']),
            'sample_token': np.array([role + '-sample']),
            'scene_name': np.array([role + '-scene']),
        }
        save_cache(cache / role, arrays, {
            'schema_version': 1, 'split': role, 'frequency_hz': 2,
            'history_seconds': 2., 'future_seconds': 6.,
            'partition_identity': 'fixture-partition',
            'preprocessing_config_sha256': 'fixture-config',
        })
        records[role] = [{'role': role, 'row': 0, 'log': role + '-log', 'scene': role + '-scene',
                          'instance': role + '-instance', 'sample': role + '-sample',
                          'condition': c, 'seed': s}
                         for c in ('clean', 'noise_05', 'noise_outlier')
                         for s in (2027, 2028, 2029)]
        datasets[role] = {
            'features': rng.normal(size=(9, 11, 18)).astype('float32'),
            'candidates': rng.normal(size=(9, 11, 10, 12, 2)).astype('float32'),
            'target': np.repeat(arrays['future_xy'], 9, axis=0),
            'ca_allowed': np.tile(np.arange(11) % 2 == 0, (9, 1)),
            'guard_yaw': np.ones((9, 11), 'float32'),
            'guard_min_speed': np.ones((9, 11), 'float32'),
        }
        np.savez_compressed(parent / (role + '.npz'), **datasets[role])
    fit = datasets['independent_fit']['features'].reshape(-1, 18)
    stats = {'mean': fit.mean(0, dtype=np.float64).tolist(),
             'std': np.maximum(fit.std(0, dtype=np.float64), 1e-4).tolist(),
             'fit_streams': 9}
    updates = {}
    for seed in (2027, 2028, 2029):
        for variant in ('baseline7', 'plus_ca', 'guarded_ca'):
            model = MotionGate(seed=seed).eval()
            name = f's{seed}_{variant}'
            torch.save({'train_seed': seed, 'variant': variant,
                        'model': model.state_dict(), 'stats': stats}, parent / (name + '.pt'))
            for role, data in datasets.items():
                pred, _ = predict_motion(model, stats, data['features'], data['candidates'],
                                         variant, ca_allowed=data['ca_allowed'])
                updates[role + '__' + name] = forecast_metrics(pred, data['target'])
    np.savez_compressed(parent / 'per_update.npz', **updates)
    means = []
    for role, data in datasets.items():
        for name, column in [('raw_cv', 0), ('alpha_beta', 1), ('kf_cv', 2), ('kf_nis', 3),
                             ('bounded_ca', 7), ('brake_stop', 8), ('hold_position', 9)]:
            values = forecast_metrics(data['candidates'][:, :, column], data['target']).mean(1)
            for row, value in zip(records[role], values, strict=True):
                means.append({**row, 'strategy': name,
                              **dict(zip(('ade', 'fde', 'jitter'), value, strict=True))})
    pd.DataFrame(means).to_csv(parent / 'per_trajectory.csv', index=False)
    write_json(parent / 'records.json', records)
    binding = {
        'source_cache_root': str(cache), 'model_seeds': [2027, 2028, 2029],
        'noise_seeds': [2027, 2028, 2029],
        'variants': ['baseline7', 'plus_ca', 'guarded_ca'],
        'sources': {'scripts/run_noise_aware_stability.py': sha256_file(script)},
        'references': {'scene_metadata': {'path': str(scenes), 'sha256': sha256_file(scenes)}},
    }

    def seal():
        write_json(parent / 'run.json', {'identity': digest_json(binding), 'binding': binding})
        names = [p.name for p in parent.iterdir() if p.is_file() and p.name != 'completed.json']
        write_json(parent / 'completed.json', {'identity': digest_json(binding),
                   'sha256': {n: sha256_file(parent / n) for n in names}})

    seal()
    return {'owner': owner, 'parent': parent, 'cache': cache, 'binding': binding,
            'datasets': datasets, 'records': records, 'seal': seal}

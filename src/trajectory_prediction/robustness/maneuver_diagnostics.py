"""Ground-truth maneuver tags for offline evaluation only; never gate inputs."""
import numpy as np
import pandas as pd

from trajectory_prediction.analysis.maneuvers import describe_motion, tag_maneuvers
from trajectory_prediction.robustness.filters import SETTINGS


def build_tags(datasets, records, thresholds):
    tags, targets = {}, {}
    for role, rows in records.items():
        for i, row in enumerate(rows):
            key = role, int(row['row'])
            target = datasets[role]['target'][i]
            if key in targets:
                if not np.array_equal(targets[key], target):
                    raise ValueError('repeated trajectory has inconsistent evaluation target')
                continue
            stats = describe_motion(target, .5)
            tags[key] = {'role': role, 'row': key[1], 'tags': tag_maneuvers(stats, thresholds), 'motion': stats}
            targets[key] = target
    return [tags[key] for key in sorted(tags)]


def grouped_metrics(frame, tags):
    lookup = pd.DataFrame([{k: row[k] for k in ('role', 'row', 'tags')} for row in tags])
    joined = frame.merge(lookup, on=['role', 'row'], how='left', validate='many_to_one')
    if joined['tags'].isna().any():
        raise ValueError('missing offline maneuver tags')
    joined = joined.explode('tags').rename(columns={'tags': 'tag'})
    columns = ['role', 'condition', 'tag', 'strategy']
    result = []
    for key, rows in joined.groupby(columns, sort=True):
        result.append({**dict(zip(columns, key, strict=True)),
                       **{name: float(rows[name].mean()) for name in ('ade', 'fde', 'jitter')},
                       'unique_trajectories': int(rows['row'].nunique()), 'logs': int(rows['log'].nunique()),
                       'trajectory_seed_records': len(rows)})
    return result


def event_diagnostics(datasets, records, tags, full_weights):
    lookup = {(t['role'], t['row']): t['tags'] for t in tags}
    values = []
    for role, rows in records.items():
        for i, row in enumerate(rows):
            for k in range(11):
                outlier = row['condition'] == 'noise_outlier' and k+1 in (4, 8)
                for tag in ['all', *lookup[role, row['row']]]:
                    values.append({'role': role, 'condition': row['condition'], 'tag': tag,
                                   'event': 'injected_outlier' if outlier else 'other_update', 'row': row['row'],
                                   'nis_alarm': bool(datasets[role]['features'][i, k, 12] > np.log1p(SETTINGS['nis_threshold'])),
                                   'kf_nis_weight': float(full_weights[role][i, k, 3]),
                                   'raw_cv_weight': float(full_weights[role][i, k, 0])})
    frame = pd.DataFrame(values)
    columns = ['role', 'condition', 'tag', 'event']
    result = []
    for key, rows in frame.groupby(columns, sort=True):
        result.append({**dict(zip(columns, key, strict=True)), 'updates': len(rows),
                       'unique_trajectories': int(rows['row'].nunique()),
                       **{name: float(rows[name].mean()) for name in ('nis_alarm', 'kf_nis_weight', 'raw_cv_weight')}})
    return result

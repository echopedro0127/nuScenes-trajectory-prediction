"""Paired gate-training-seed statistics; noise repeats are averaged within a model."""
import numpy as np


def seed_statistics(frame, extra=()):
    groups = ['role', 'condition', *extra]
    selected = frame[frame.train_seed.notna()].copy()
    selected['train_seed'] = selected['train_seed'].astype(int)
    models = selected.groupby([*groups, 'variant', 'train_seed'], sort=True)[['ade', 'fde', 'jitter']].mean().reset_index()
    across, paired, paired_summary = [], [], []
    for key, rows in models.groupby([*groups, 'variant'], sort=True):
        item = dict(zip([*groups, 'variant'], key, strict=True))
        item['training_seeds'] = sorted(rows.train_seed.astype(int).tolist())
        for metric in ('ade', 'fde', 'jitter'):
            item[metric+'_mean'] = float(rows[metric].mean())
            item[metric+'_std'] = float(rows[metric].std(ddof=1)) if len(rows) > 1 else None
        across.append(item)
    for key, rows in models.groupby(groups, sort=True):
        context = dict(zip(groups, key, strict=True))
        lookup = {(int(r.train_seed), r.variant): r for r in rows.itertuples()}
        for a, b in (('plus_ca', 'baseline7'), ('guarded_ca', 'plus_ca')):
            differences = []
            for seed in sorted(rows.train_seed.unique()):
                if (int(seed), a) not in lookup or (int(seed), b) not in lookup:
                    raise ValueError('paired controls must have the same training seeds')
                first, second = lookup[int(seed), a], lookup[int(seed), b]
                item = {**context, 'contrast': a+'-minus-'+b, 'train_seed': int(seed)}
                for metric in ('ade', 'fde', 'jitter'):
                    delta = float(getattr(first, metric)-getattr(second, metric))
                    item[metric+'_delta'] = delta
                    item[metric+'_percent'] = 100*delta/float(getattr(second, metric)) if getattr(second, metric) else None
                paired.append(item)
                differences.append(item['ade_delta'])
            paired_summary.append({**context, 'contrast': a+'-minus-'+b, 'training_seed_count': len(differences),
                                   'improved_seeds': int(np.sum(np.array(differences) < 0)),
                                   'ade_delta_mean': float(np.mean(differences)),
                                   'ade_delta_std': float(np.std(differences, ddof=1)) if len(differences) > 1 else None})
    return {'per_model': models.to_dict('records'), 'across_seeds': across,
            'paired': paired, 'paired_summary': paired_summary}

"""Explicit noise-seed then window reductions; no pooled-seed quantile claims."""
import numpy as np

from trajectory_prediction.evaluation.physics_clean_metrics import _groups
from trajectory_prediction.robustness.source_gaussian import SEEDS


def _ordered(records):
    if (not isinstance(records,list) or len(records) != 3
            or any(type(r.get('seed')) is not int for r in records)
            or {r['seed'] for r in records} != set(SEEDS)):
        raise ValueError('all three distinct frozen noise seeds required')
    return sorted(records,key=lambda r:r['seed'])


def _arrays(values,shared,windows,signed=False):
    counts = shared['valid'].sum(axis=1,dtype=np.int64);valid = counts > 0
    for a in values:
        if (a.shape != (len(windows),3) or a.dtype != np.dtype('float64') or not np.isfinite(a).all()
                or np.any(a[~valid] != 0) or (not signed and (a < 0).any())):
            raise ValueError('invalid per-window seed values')
    stack = np.stack(values)
    return {'seed_window_values':stack,'mean_over_noise_seeds':stack.mean(axis=0),
            'valid_windows':valid,'query_counts_per_seed':counts}


def _log_macro(groups,field,output_field):
    for g in groups:
        values = [x[field] for x in groups if x['role'] == g['role'] and x['lead_us'] == g['lead_us']
                  and x['log'] != '__all__' and x[field] is not None]
        g[output_field] = (float(np.mean(values)) if values else None) if g['log'] == '__all__' else g[field]


def aggregate_scores(records,shared,windows):
    records = _ordered(records)
    arrays = _arrays([r['window_metrics']['window_mean_error_m'] for r in records],shared,windows)
    maps = []
    expected = {(role,log,h) for role,log,indices,col,h in _groups(windows)}
    for r in records:
        wm = r['window_metrics']
        if (wm['valid_windows'].dtype != np.dtype('bool') or not np.array_equal(wm['valid_windows'],arrays['valid_windows'])
                or wm['query_counts'].dtype != np.dtype('int64')
                or not np.array_equal(wm['query_counts'],arrays['query_counts_per_seed'])):
            raise ValueError('noise seeds must share valid windows and query counts')
        mapping = {(g['role'],g['log'],g['lead_us']):g for g in r['groups']}
        if len(mapping) != len(r['groups']) or set(mapping) != expected:raise ValueError('missing/duplicate seed groups')
        maps.append(mapping)
    groups = []
    for role,log,indices,col,h in _groups(windows):
        valid = arrays['valid_windows'][indices,col];key = role,log,h
        values = arrays['mean_over_noise_seeds'][indices,col][valid]
        seed_means = arrays['seed_window_values'][:,indices,col][:,valid].mean(axis=1) if len(values) else None
        source = [m[key] for m in maps];first = source[0]
        queries = int(shared['valid'][indices,:,col].sum())
        if any(g['valid_queries'] != queries or g['valid_windows'] != int(valid.sum()) for g in source):
            raise ValueError('seed group denominator mismatch')
        row = {k:first[k] for k in ('role','log','lead_us','windows','valid_windows','not_evaluable_windows',
                    'unique_source_queries','unique_gt_targets','valid_logs',
                    'actual_lead_us_min','actual_lead_us_median','actual_lead_us_max')}
        row.update(noise_seeds=list(SEEDS),repeats=3,valid_queries_per_seed=queries,repeated_query_evaluations=queries*3,
                   mean_window_error_m=float(values.mean()) if len(values) else None,
                   noise_seed_std_window_macro_m=float(seed_means.std(ddof=1)) if seed_means is not None else None,
                   query_error_gt2m_count_over_repeats=sum(g['query_error_gt2m_count'] for g in source),
                   status='defined' if len(values) else 'not_evaluable',reason=None if len(values) else 'no_valid_queries')
        for field in ('query_p50_error_m','query_p95_error_m','query_error_gt2m_fraction'):
            samples = [g[field] for g in source]
            if len(values) and any(type(v) is not float or not np.isfinite(v) for v in samples):
                raise ValueError('invalid per-seed tail/proportion')
            row[field+'_seed_mean'] = float(np.mean(samples)) if len(values) else None
            row[field+'_seed_std_ddof1'] = float(np.std(samples,ddof=1)) if len(values) else None
        groups.append(row)
    _log_macro(groups,'mean_window_error_m','log_macro_mean_error_m')
    arrays['mean_error_over_noise_seeds'] = arrays.pop('mean_over_noise_seeds')
    return {'groups':groups,'arrays':arrays}


def aggregate_differences(records,shared,windows,metric):
    if metric not in ('mean_window_gain_m','mean_window_extra_error_m'):raise ValueError('explicit signed difference metric required')
    records = _ordered(records)
    arrays = _arrays([r['values'] for r in records],shared,windows,signed=True)
    groups = []
    for role,log,indices,col,h in _groups(windows):
        valid = arrays['valid_windows'][indices,col];values = arrays['mean_over_noise_seeds'][indices,col][valid]
        seed_means = arrays['seed_window_values'][:,indices,col][:,valid].mean(axis=1) if len(values) else None
        count = int(shared['valid'][indices,:,col].sum())
        groups.append({'role':role,'log':log,'lead_us':h,'noise_seeds':list(SEEDS),'valid_windows':len(values),
                       'valid_queries_per_seed':count,'repeated_query_evaluations':count*3,
                       metric:float(values.mean()) if len(values) else None,
                       'noise_seed_std_window_macro_m':float(seed_means.std(ddof=1)) if seed_means is not None else None,
                       'reason':None if len(values) else 'no_valid_queries'})
    _log_macro(groups,metric,'log_macro_'+metric)
    return {'groups':groups,'arrays':arrays}

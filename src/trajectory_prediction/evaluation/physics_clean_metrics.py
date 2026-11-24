"""Masked window/log reductions and matched-target paired differences."""
import numpy as np

from trajectory_prediction.evaluation.query_clock import LEADS

ROLES = ('independent_fit','independent_select')


def _validate(errors,shared,windows):
    e,v = np.asarray(errors),shared['valid']
    if (e.shape != (len(windows),11,3) or e.dtype != np.dtype('float64')
            or v.shape != e.shape or v.dtype != np.dtype('bool') or not np.isfinite(e).all()
            or (e < 0).any() or np.any(e[~v] != 0)
            or any(w['role'] not in ROLES or not isinstance(w['log'],str) or not w['log'] for w in windows)):
        raise ValueError('finite nonnegative float64 errors with shared valid mask required')
    return e,v


def window_reduce(values,valid):
    counts = valid.sum(axis=1,dtype=np.int64)
    means = np.divide(np.where(valid,values,0.).sum(axis=1),counts,
                     out=np.zeros(counts.shape,np.float64),where=counts > 0)
    return {'window_mean_error_m':means,'valid_windows':counts > 0,'query_counts':counts}


def _groups(windows):
    for role in ROLES:
        for log in ['__all__',*sorted({w['log'] for w in windows if w['role'] == role})]:
            indices = np.array([i for i,w in enumerate(windows) if w['role'] == role and (log == '__all__' or w['log'] == log)],np.int64)
            for col,h in enumerate(LEADS):yield role,log,indices,col,h


def summarize_errors(errors,shared,windows):
    e,v = _validate(errors,shared,windows)
    reduced = window_reduce(e,v);groups = []
    for role,log,indices,col,h in _groups(windows):
        mask = v[indices,:,col];values = e[indices,:,col][mask]
        present = reduced['valid_windows'][indices,col]
        means = reduced['window_mean_error_m'][indices,col][present]
        query_tokens = {windows[i]['tokens'][k+5] for i in indices for k in range(11) if v[i,k,col]}
        target_tokens = {windows[i]['tokens'][int(shared['target_index'][i,k,col])] for i in indices for k in range(11) if v[i,k,col]}
        actual_leads = shared['actual_lead_us'][indices,:,col][mask]
        groups.append({'role':role,'log':log,'lead_us':h,'windows':len(indices),'valid_windows':int(present.sum()),
                       'not_evaluable_windows':int((~present).sum()),'valid_queries':len(values),
                       'unique_source_queries':len(query_tokens),'unique_gt_targets':len(target_tokens),
                       'valid_logs':len({windows[int(i)]['log'] for i in indices[present]}),
                       'mean_window_error_m':float(means.mean()) if len(means) else None,
                       'query_p50_error_m':float(np.quantile(values,.5,method='linear')) if len(values) else None,
                       'query_p95_error_m':float(np.quantile(values,.95,method='linear')) if len(values) else None,
                       'query_error_gt2m_count':int((values > 2).sum()),
                       'query_error_gt2m_fraction':float((values > 2).mean()) if len(values) else None,
                       'actual_lead_us_min':int(actual_leads.min()) if len(values) else None,
                       'actual_lead_us_median':float(np.median(actual_leads)) if len(values) else None,
                       'actual_lead_us_max':int(actual_leads.max()) if len(values) else None,
                       'status':'defined' if len(values) else 'not_evaluable',
                       'reason':None if len(values) else 'no_valid_queries'})
    for g in groups:
        logs = [r['mean_window_error_m'] for r in groups if r['role'] == g['role'] and r['lead_us'] == g['lead_us'] and r['log'] != '__all__' and r['mean_window_error_m'] is not None]
        g['log_macro_mean_error_m'] = (float(np.mean(logs)) if logs else None) if g['log'] == '__all__' else g['mean_window_error_m']
    return {'groups':groups,'window_metrics':reduced}


def paired_gains(results,shared,windows,comparisons):
    groups,arrays = [],{}
    for name,reference,compared in comparisons:
        for strategy in (reference,compared):
            _validate(results[strategy]['errors'],shared,windows)
            if not np.array_equal(results[strategy]['valid'],shared['valid']):
                raise ValueError('paired strategies must share identical valid targets')
        gains = results[reference]['errors']-results[compared]['errors']
        reduced = window_reduce(gains,shared['valid'])
        arrays[name] = reduced['window_mean_error_m']
        arrays['valid_windows'],arrays['query_counts'] = reduced['valid_windows'],reduced['query_counts']
        for role,log,indices,col,h in _groups(windows):
            present = reduced['valid_windows'][indices,col]
            values = arrays[name][indices,col][present]
            groups.append({'comparison':name,'reference_strategy':reference,'compared_strategy':compared,
                           'role':role,'log':log,'lead_us':h,'valid_windows':len(values),
                           'valid_queries':int(shared['valid'][indices,:,col].sum()),
                           'mean_window_gain_m':float(values.mean()) if len(values) else None,
                           'reason':None if len(values) else 'no_valid_queries'})
    return {'groups':groups,'arrays':arrays}

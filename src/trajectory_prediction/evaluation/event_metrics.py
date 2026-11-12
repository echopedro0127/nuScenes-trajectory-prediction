"""First-outlier excess position error; recovery never implies state recovery."""
import numpy as np

from trajectory_prediction.evaluation.streaming_metrics import fixed_lead_errors


def recovery_from_excess(excess, dt=.5, threshold_m=.5, consecutive_updates=2):
    if (type(dt) not in (int, float) or dt != .5
            or type(threshold_m) not in (int, float) or threshold_m != .5
            or type(consecutive_updates) is not int or consecutive_updates != 2):
        raise ValueError('v1 requires dt=.5, threshold=.5 and two confirmation updates')
    a = np.asarray(excess)
    if (a.ndim != 2 or a.shape[1] != 4 or a.dtype.kind != 'f'
            or not np.isfinite(a).all() or (a < 0).any()):
        raise ValueError('complete nonnegative finite floating [events,4] excess required')
    a = a.astype(np.float64)
    affected = (a > threshold_m).any(axis=1)
    recovered = np.zeros(len(a), bool)
    delay = np.full(len(a), np.nan)
    first, confirm = np.full(len(a), -1, np.int64), np.full(len(a), -1, np.int64)
    for i in np.flatnonzero(affected):
        damaged = int(np.flatnonzero(a[i] > threshold_m)[0])
        first[i] = damaged + 4
        for j in range(damaged + 2, 4):
            if np.all(a[i, j - 1:] <= threshold_m):
                recovered[i], delay[i], confirm[i] = True, j * dt, j + 4
                break
    with np.errstate(over='ignore', invalid='ignore'):
        auc = (a[:, :-1] * (dt / 2) + a[:, 1:] * (dt / 2)).sum(axis=1)
    if not np.isfinite(auc).all():
        raise ValueError('excess AUC exceeds finite float64 range')
    return {'affected': affected, 'recovered': recovered, 'confirm_delay_s': delay,
            'first_affected_step': first, 'confirm_step': confirm, 'excess_auc': auc,
            'status': np.where(~affected, 'unaffected', np.where(recovered, 'recovered', 'not_recovered'))}


def pair_first_outlier(predictions, target, records, *, strategy_identity):
    errors = fixed_lead_errors(predictions, target, lead_steps=2)['values']
    if (not isinstance(strategy_identity, str) or not strategy_identity
            or not isinstance(records, list) or len(records) != len(errors)):
        raise ValueError('aligned records and explicit model/strategy identity required')
    pairs, roles = {}, set()
    for i, row in enumerate(records):
        if (row.get('role') not in ('independent_fit', 'independent_select')
                or row.get('condition') not in ('noise_05', 'noise_outlier')
                or type(row.get('seed')) is not int or row['seed'] not in (2027, 2028, 2029)
                or type(row.get('row')) is not int or row['row'] < 0
                or any(not isinstance(row.get(k), str) or not row[k]
                       for k in ('instance', 'sample', 'scene', 'log'))
                or row.get('strategy_identity', strategy_identity) != strategy_identity):
            raise ValueError('invalid role/condition/seed/window/model pairing metadata')
        roles.add(row['role'])
        key = row['role'], row['instance'], row['sample'], row['seed'], strategy_identity
        item = pairs.setdefault(key, {})
        if row['condition'] in item:
            raise ValueError('duplicate pairing key/condition')
        item[row['condition']] = i
    if len(roles) != 1:
        raise ValueError('one development role per paired evaluation required')
    reference, outlier, keys = [], [], []
    for key, pair in sorted(pairs.items()):
        if set(pair) != {'noise_05', 'noise_outlier'}:
            raise ValueError('missing expected reference/outlier pair')
        a, b = pair['noise_05'], pair['noise_outlier']
        if (any(records[a][field] != records[b][field] for field in ('scene', 'log', 'row'))
                or not np.array_equal(np.asarray(target)[a], np.asarray(target)[b])):
            raise ValueError('paired GT or metadata mismatch')
        reference.append(a)
        outlier.append(b)
        keys.append(key)
    e_ref, e_out = errors[reference, 3:7], errors[outlier, 3:7]
    steps, columns = np.arange(4, 8), np.arange(5, 9)
    p = np.asarray(predictions, np.float64)
    with np.errstate(over='ignore', invalid='ignore'):
        delta = p[outlier][:, steps - 1, columns] - p[reference][:, steps - 1, columns]
        distance = np.hypot(delta[..., 0], delta[..., 1])
    if not np.isfinite(distance).all():
        raise ValueError('paired prediction distance exceeds finite float64 range')
    return {'excess': np.maximum(0., e_out - e_ref),
            'reference_errors': e_ref, 'outlier_errors': e_out, 'prediction_distance': distance,
            'query_steps': steps, 'target_columns': columns, 'relative_times_s': np.arange(4) * .5,
            'pair_keys': keys, 'reference_indices': reference, 'outlier_indices': outlier}


def summarize_recovery(excess, window_keys, noise_seeds):
    result = recovery_from_excess(excess)
    if len(window_keys) != len(excess) or len(noise_seeds) != len(excess):
        raise ValueError('aligned event keys/noise seeds required')
    groups, seen = {}, set()
    for i, (key, seed) in enumerate(zip(window_keys, noise_seeds, strict=True)):
        if not isinstance(key, (str, tuple)) or not key or type(seed) is not int or seed < 0:
            raise ValueError('nonempty window keys and integer noise seeds required')
        try:
            if (key, seed) in seen:
                raise ValueError('duplicate window/noise event key')
            seen.add((key, seed))
            groups.setdefault(key, []).append(i)
        except TypeError as exc:
            raise ValueError('hashable window keys required') from exc
    eligible, affected, recovered = len(excess), int(result['affected'].sum()), int(result['recovered'].sum())
    with np.errstate(over='ignore', invalid='ignore'):
        mean_auc = float(np.mean([result['excess_auc'][indices].mean()
                                  for indices in groups.values()])) if groups else None
    if mean_auc is not None and not np.isfinite(mean_auc):
        raise ValueError('recovery summary exceeds finite float64 range')
    return {'eligible': eligible, 'affected': affected, 'unaffected': eligible - affected,
            'recovered': recovered, 'not_recovered': affected - recovered,
            'window_count': len(groups), 'noise_seed_count': len(set(noise_seeds)),
            'mean_excess_auc_m_s': mean_auc,
            'recovered_fraction_of_affected': recovered / affected if affected else None,
            'not_recovered_fraction_of_affected': (affected - recovered) / affected if affected else None,
            'unaffected_fraction_of_eligible': (eligible - affected) / eligible if eligible else None,
            'mean_confirm_delay_s_recovered_events':
                float(result['confirm_delay_s'][result['recovered']].mean()) if recovered else None,
            'status': 'ok' if eligible else 'not_evaluable',
            'reason': None if eligible else 'empty_group'}

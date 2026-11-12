"""Same-target updates and explicitly named reduction units; S3 stays frozen."""
import numpy as np

from trajectory_prediction.evaluation.streaming_metrics import fixed_lead_errors


def revision_gains(predictions, target, *, lead_steps, tolerance_m=.05):
    if type(tolerance_m) not in (int, float) or tolerance_m != .05:
        raise ValueError('v1 revision tolerance_m must equal 0.05')
    current = fixed_lead_errors(predictions, target, lead_steps=lead_steps)
    steps, columns = current['query_steps'][1:], current['target_columns'][1:]
    p, y = np.asarray(predictions), np.asarray(target)
    with np.errstate(over='ignore', invalid='ignore'):
        delta = p[:, steps - 2, columns].astype(np.float64) - y[:, columns].astype(np.float64)
        previous = np.hypot(delta[..., 0], delta[..., 1])
        values = previous - current['values'][:, 1:]
    if not np.isfinite(previous).all() or not np.isfinite(values).all():
        raise ValueError('revision error exceeds finite float64 range')
    return {'values': values, 'previous_error_m': previous,
            'current_error_m': current['values'][:, 1:],
            'query_steps': steps, 'query_rows': steps - 1, 'previous_query_rows': steps - 2,
            'target_columns': columns, 'query_times_s': steps * .5,
            'target_times_s': (columns + 1) * .5,
            'beneficial': values > tolerance_m, 'harmful': values < -tolerance_m,
            'neutral': (values >= -tolerance_m) & (values <= tolerance_m),
            'valid_mask': np.ones(values.shape, dtype=bool)}


def _cohort(values, window_keys, noise_seeds):
    x = np.asarray(values)
    if (x.ndim != 2 or x.shape[1] < 1 or x.dtype.kind != 'f'
            or not np.isfinite(x).all() or len(window_keys) != len(x)
            or len(noise_seeds) != len(x)):
        raise ValueError('finite floating [streams,queries] and aligned keys/seeds required')
    groups, seen = {}, set()
    for i, (key, seed) in enumerate(zip(window_keys, noise_seeds, strict=True)):
        if not isinstance(key, (str, tuple)) or not key or type(seed) is not int or seed < 0:
            raise ValueError('nonempty window keys and integer noise seeds required')
        try:
            pair = key, seed
            if pair in seen:
                raise ValueError('duplicate window/noise seed')
            seen.add(pair)
            groups.setdefault(key, []).append(i)
        except TypeError as exc:
            raise ValueError('window keys must be hashable') from exc
    counts = {'stream_count': len(x), 'window_count': len(groups),
              'noise_seed_count': len(set(noise_seeds)), 'noise_repeat_count': len(x)}
    return x.astype(np.float64), groups, counts


def _window_means(x, groups):
    # queries → repeats within the same window → equally weighted windows
    with np.errstate(over='ignore', invalid='ignore'):
        result = np.array([x[indices].mean(axis=1).mean() for indices in groups.values()])
    if not np.isfinite(result).all():
        raise ValueError('window mean exceeds finite float64 range')
    return result


def _checked(result):
    if any(isinstance(v, float) and not np.isfinite(v) for v in result.values()):
        raise ValueError('summary exceeds finite float64 range')
    return result


def summarize_revisions(values, window_keys, noise_seeds):
    x, groups, counts = _cohort(values, window_keys, noise_seeds)
    pair_count = x.size
    result = {**counts, 'pair_count': pair_count,
              'status': 'ok' if pair_count else 'not_evaluable',
              'reason': None if pair_count else 'empty_group',
              'mean_gain_m': float(_window_means(x, groups).mean()) if pair_count else None,
              'median_pair_gain_m': float(np.median(x)) if pair_count else None}
    for name, mask in [('beneficial', x > .05), ('harmful', x < -.05),
                       ('neutral', (x >= -.05) & (x <= .05))]:
        result[name + '_count'] = int(mask.sum())
        result[name + '_fraction'] = float(mask.mean()) if pair_count else None
        result['window_macro_' + name + '_fraction'] = (
            float(_window_means(mask, groups).mean()) if pair_count else None)
    return _checked(result)


def summarize_tail(values, window_keys, noise_seeds):
    x, groups, counts = _cohort(values, window_keys, noise_seeds)
    if (x < 0).any():
        raise ValueError('position errors must be nonnegative')
    count = x.size
    windows = _window_means(x, groups)
    result = {**counts, 'query_count': count, 'status': 'ok' if count else 'not_evaluable',
              'reason': None if count else 'empty_group',
              'mean_error_m': float(windows.mean()) if count else None,
              'miss_count': int((x > 2).sum()),
              'miss_fraction': float((x > 2).mean()) if count else None,
              'p95_window_mean_error': float(np.quantile(windows, .95, method='linear'))
                                       if count else None}
    for q, name in [(.5, 'p50_query_error'), (.95, 'p95_query_error'), (.99, 'p99_query_error')]:
        result[name] = float(np.quantile(x, q, method='linear')) if count else None
    return _checked(result)


def remaining_miss_flags(predictions, target):
    # Reuse S3 input validation but preserve original per-query remaining length.
    fixed_lead_errors(predictions, target, lead_steps=1)
    p, y = np.asarray(predictions), np.asarray(target)
    with np.errstate(over='ignore', invalid='ignore'):
        delta = p.astype(np.float64) - y[:, None].astype(np.float64)
        errors = np.hypot(delta[..., 0], delta[..., 1])
    if not np.isfinite(errors).all():
        raise ValueError('remaining error exceeds finite float64 range')
    miss = np.stack([(errors[:, k - 1, k:] > 2).any(axis=1) for k in range(1, 12)], axis=1)
    return {'miss': miss, 'query_steps': np.arange(1, 12),
            'remaining_points': np.arange(11, 0, -1)}

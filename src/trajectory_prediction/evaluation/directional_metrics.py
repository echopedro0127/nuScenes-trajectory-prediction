"""Common past-GT reference frames; invalidity affects only direction metrics."""
import numpy as np

from trajectory_prediction.evaluation.streaming_metrics import fixed_lead_errors


def gt_reference_directions(history, future, history_mask):
    h, y, mask = np.asarray(history), np.asarray(future), np.asarray(history_mask)
    if (h.ndim != 3 or h.shape[1:] != (5, 2) or len(h) < 1
            or y.shape != (len(h), 12, 2) or mask.shape != (len(h), 5)
            or mask.dtype != np.bool_ or h.dtype.kind != 'f' or y.dtype.kind != 'f'
            or not np.isfinite(h).all() or not np.isfinite(y).all()):
        raise ValueError('finite GT history[N,5,2], future[N,12,2] and bool history mask required')
    points = np.concatenate((h, y), axis=1).astype(np.float64)
    available = np.concatenate((mask, np.ones((len(h), 12), bool)), axis=1)
    directions = np.zeros((len(h), 11, 2), np.float64)
    valid, fresh = np.zeros((len(h), 11), bool), np.zeros((len(h), 11), bool)
    sources = np.full((len(h), 11), -999, np.int64)
    last = np.zeros((len(h), 2), np.float64)
    last_step = np.full(len(h), -999, np.int64)
    known = np.zeros(len(h), bool)
    for end in range(1, 16):  # GT segments end at steps -3..0, then 1..11.
        with np.errstate(over='ignore', invalid='ignore'):
            delta = points[:, end] - points[:, end - 1]
            distance = np.hypot(delta[:, 0], delta[:, 1])
        if not np.isfinite(distance).all():
            raise ValueError('GT displacement exceeds finite float64 range')
        reliable = available[:, end] & available[:, end - 1] & (distance >= .25)
        last[reliable] = delta[reliable] / distance[reliable, None]
        last_step[reliable] = end - 4
        known |= reliable
        if end >= 5:
            row = end - 5
            directions[:, row] = last
            valid[:, row], fresh[:, row], sources[:, row] = known, reliable, last_step
    return {'directions': directions, 'valid_mask': valid, 'fresh_mask': fresh,
            'source_steps': sources}


def directional_errors(predictions, target, history, history_mask):
    base = fixed_lead_errors(predictions, target, lead_steps=2)
    frame = gt_reference_directions(history, target, history_mask)
    steps, cols = base['query_steps'], base['target_columns']
    directions, valid = frame['directions'][:, :10], frame['valid_mask'][:, :10]
    p, y = np.asarray(predictions), np.asarray(target)
    delta = p[:, steps - 1, cols].astype(np.float64) - y[:, cols].astype(np.float64)
    normal = np.stack((-directions[..., 1], directions[..., 0]), axis=-1)
    with np.errstate(over='ignore', invalid='ignore'):
        longitudinal = np.sum(delta * directions, axis=-1)
        lateral = np.sum(delta * normal, axis=-1)
    if (not np.isfinite(longitudinal[valid]).all()
            or not np.isfinite(lateral[valid]).all()):
        raise ValueError('directional projection exceeds finite float64 range')
    longitudinal = np.where(valid, longitudinal, np.nan)
    lateral = np.where(valid, lateral, np.nan)
    return {'longitudinal_signed_m': longitudinal, 'lateral_signed_m': lateral,
            'longitudinal_abs_m': np.abs(longitudinal), 'lateral_abs_m': np.abs(lateral),
            'valid_mask': valid, 'reference_directions': directions,
            'fresh_heading_mask': frame['fresh_mask'][:, :10],
            'heading_source_steps': frame['source_steps'][:, :10],
            'reason': np.where(valid, 'ok', 'no_past_reliable_heading'),
            'query_steps': steps, 'query_rows': steps - 1, 'target_columns': cols,
            'query_times_s': base['query_times_s'], 'target_times_s': base['target_times_s']}


def summarize_directional(longitudinal, lateral, valid_mask, window_keys, noise_seeds):
    x, y, mask = np.asarray(longitudinal), np.asarray(lateral), np.asarray(valid_mask)
    if (x.ndim != 2 or x.shape[1] < 1 or x.shape != y.shape or x.shape != mask.shape
            or mask.dtype != np.bool_ or x.dtype.kind != 'f' or y.dtype.kind != 'f'
            or len(window_keys) != len(x) or len(noise_seeds) != len(x)
            or not np.isfinite(x[mask]).all() or not np.isfinite(y[mask]).all()
            or np.isinf(x).any() or np.isinf(y).any()):
        raise ValueError('aligned projections/mask required; valid entries must be finite')
    groups, seen = {}, set()
    for i, (key, seed) in enumerate(zip(window_keys, noise_seeds, strict=True)):
        if not isinstance(key, (str, tuple)) or not key or type(seed) is not int or seed < 0:
            raise ValueError('nonempty window keys and integer noise seeds required')
        try:
            if (key, seed) in seen:
                raise ValueError('duplicate window/noise key')
            seen.add((key, seed))
            groups.setdefault(key, []).append(i)
        except TypeError as exc:
            raise ValueError('hashable window keys required') from exc
    valid_streams = mask.any(axis=1)
    evaluable_windows = sum(any(valid_streams[i] for i in indices) for indices in groups.values())
    count = int(mask.sum())
    result = {
        'status': 'ok' if count else 'not_evaluable',
        'reason': None if count else ('empty_group' if not len(x) else 'no_valid_heading'),
        'query_count': mask.size, 'valid_query_count': count,
        'invalid_query_count': mask.size - count, 'stream_count': len(x),
        'evaluable_stream_count': int(valid_streams.sum()), 'window_count': len(groups),
        'evaluable_window_count': evaluable_windows, 'noise_seed_count': len(set(noise_seeds)),
        'heading_coverage': count / mask.size if mask.size else None,
    }
    for name, values in [('mean_longitudinal_abs_m', np.abs(x)),
                          ('mean_longitudinal_signed_m', x), ('mean_lateral_abs_m', np.abs(y))]:
        window_means = []
        with np.errstate(over='ignore', invalid='ignore'):
            for indices in groups.values():
                repeated = [values[i, mask[i]].mean(dtype=np.float64)
                            for i in indices if valid_streams[i]]
                if repeated:
                    window_means.append(np.mean(repeated))
            value = float(np.mean(window_means)) if window_means else None
        if value is not None and not np.isfinite(value):
            raise ValueError('directional summary exceeds finite float64 range')
        result[name] = value
    return result

"""Pure fixed-lead metrics on the original anchor-relative prediction grid."""
import numpy as np


def fixed_lead_errors(predictions, target, *, lead_steps):
    """Return float64 distances; k is one-based and array rows are zero-based.

    Both float32 and float64 inputs are accepted. Invalid tail queries are
    excluded rather than padded. All supplied coordinates must remain finite,
    including slots that this particular lead does not score.
    """
    if type(lead_steps) is not int or lead_steps not in (1, 2, 4):
        raise ValueError('lead_steps must be a v1 integer in [1, 2, 4]')
    p, y = np.asarray(predictions), np.asarray(target)
    if (y.ndim != 3 or y.shape[1:] != (12, 2) or len(y) < 1
            or p.shape != (len(y), 11, 12, 2)):
        raise ValueError('nonempty predictions [N,11,12,2] and target [N,12,2] required')
    if (p.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or y.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(p).all() or not np.isfinite(y).all()):
        raise ValueError('finite float32/float64 predictions and GT required')
    steps = np.arange(1, 13 - lead_steps, dtype=np.int64)
    rows = steps - 1
    columns = steps + lead_steps - 1
    with np.errstate(over='ignore', invalid='ignore'):
        delta = p[:, rows, columns, :].astype(np.float64) - y[:, columns, :].astype(np.float64)
        values = np.hypot(delta[..., 0], delta[..., 1])
    if not np.isfinite(values).all():
        raise ValueError('fixed-lead error exceeds finite float64 range')
    return {
        'values': values, 'query_steps': steps, 'query_rows': rows,
        'target_columns': columns, 'query_times_s': steps * .5,
        'target_times_s': (columns + 1) * .5,
        'valid_mask': np.ones(values.shape, dtype=bool),
        'excluded_query_steps': np.arange(13 - lead_steps, 12, dtype=np.int64),
    }

"""Paired log-cluster bootstrap of pooled trajectory error differences."""
import numpy as np


def paired_log_bootstrap(values, logs, *, replicates=5000, seed=20271109):
    values, logs = np.asarray(values, np.float64), np.asarray(logs)
    if (values.ndim != 1 or not len(values) or logs.shape != values.shape or logs.dtype.kind not in "US"
            or not np.isfinite(values).all() or np.any(logs == "") or type(replicates) is not int or replicates < 1):
        raise ValueError("finite paired differences and nonempty log identifiers required")
    groups = sorted(set(logs.tolist()))
    if len(groups) < 2:
        raise ValueError("at least two log clusters required")
    sums = np.array([values[logs == group].sum() for group in groups])
    counts = np.array([(logs == group).sum() for group in groups])
    chosen = np.random.Generator(np.random.PCG64(seed)).integers(0, len(groups), size=(replicates, len(groups)))
    draws = sums[chosen].sum(1) / counts[chosen].sum(1)
    interval = np.quantile(draws, [.025, .975], method="linear").tolist()
    summary = {"difference": float(values.mean()), "interval": interval, "superior": bool(interval[1] < 0),
               "log_count": len(groups), "replicates": replicates, "seed": seed,
               "logs": {group: {"trajectories": int(counts[i]), "difference": float(sums[i] / counts[i])}
                        for i, group in enumerate(groups)}}
    return summary, draws

"""Frozen P6.5 mean-coefficient matching; weights remain in [0, 1]."""

import math

ARMS = ("fixed_005", "adaptive_matched_005", "fixed_010", "adaptive_matched_010")


def strength_settings(arm, train_mean_weight):
    if arm not in ARMS:
        raise ValueError("unknown strength arm")
    if not math.isfinite(train_mean_weight) or not 0 < train_mean_weight <= 1:
        raise ValueError("invalid trusted training mean")
    alpha = .05 if arm.endswith("005") else .10
    adaptive = arm.startswith("adaptive")
    return {"target_mean_weight": alpha, "revision_kind": "revision_adaptive" if adaptive else "revision_fixed",
        "revision_weight": alpha/train_mean_weight if adaptive else alpha}

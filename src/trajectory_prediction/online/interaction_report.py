"""Matched-row ablation reports, coverage and qualified cache-path timing."""
import json
from pathlib import Path

import numpy as np

from trajectory_prediction.online.evaluator import aggregate_metrics
from trajectory_prediction.online.interaction_provenance import SUITE_NAMES

STRATEGIES = ("frozen", "sliding", *SUITE_NAMES)


def validate_rows(frame):
    keys = ["instance_token", "sample_token", "update_step"]
    if frame.empty or set(frame.strategy) != set(STRATEGIES) or frame.duplicated(keys + ["strategy"]).any():
        raise ValueError("all nine strategies must have unique matched update rows")
    if not (frame.groupby(keys).strategy.nunique() == len(STRATEGIES)).all():
        raise ValueError("ablation update key mismatch")


def write_interaction_outputs(frame, output_dir, provenance):
    validate_rows(frame)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    frame.to_csv(root / "per_update.csv", index=False, lineterminator="\n")
    summaries, fallbacks = {}, {}
    for name, group in frame.groupby("strategy"):
        row = aggregate_metrics(group.to_dict(orient="records"))
        row["update_count"] = row.pop("sample_count")
        row["sample_count"] = int(group.sample_index.nunique())
        for column in ("feature_ms", "encoder_head_ms", "cached_replay_prepare_ms"):
            row[column + "_p50"] = float(np.percentile(group[column], 50))
            row[column + "_p95"] = float(np.percentile(group[column], 95))
        summaries[name] = row
        fallbacks[name] = int(group.fallback.sum())
    fused = frame[frame.strategy == "fusion"]
    selected = int(fused.neighbor_count.sum())
    history = int(fused.neighbor_history_points.sum())
    summary = {"provenance": provenance, "strategies": summaries, "fallback_updates": fallbacks,
        "sample_count": int(fused.sample_index.nunique()), "update_count": len(fused),
        "coverage": {"map": float(fused.map_covered.mean()), "neighbors": float(fused.neighbor_covered.mean()),
                     "either": float((fused.map_covered | fused.neighbor_covered).mean())},
        "neighbor_quality": {"selected_neighbors": selected, "valid_history_points": history,
            "valid_velocity_points": int(fused.neighbor_velocity_points.sum()),
            "history_valid_fraction": history / (selected * 5) if selected else 0.,
            "velocity_valid_fraction": float(fused.neighbor_velocity_points.sum()) / history if history else 0.,
            "frame_age_p50_us": float(fused.loc[fused.neighbor_covered, "neighbor_frame_age_us"].median()) if selected else None,
            "frame_age_p95_us": float(fused.loc[fused.neighbor_covered, "neighbor_frame_age_us"].quantile(.95)) if selected else None},
        "failure_count": 0, "oracle": "target_position_and_neighbor_annotations",
        "timing_scope": "cached_replay_including_label_snapshot_plus_feature_transfer_and_prediction; excludes_metadata_disk_reads_and_index_build",
        "encoder_head_scope": "encoder_head_plus_return_transfer; synchronized; no_warmup; feasibility_only"}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for field, filename in (("update_step", "by_update.csv"), ("remaining_steps", "by_horizon.csv"), ("maneuver", "by_maneuver.csv")):
        grouped = frame.copy()
        if field == "maneuver":
            grouped[field] = grouped[field].str.split("|")
            grouped = grouped.explode(field)
        grouped.groupby(["strategy", field], as_index=False).agg(update_count=("sample_index", "size"),
            ade=("ade", "mean"), fde=("fde", "mean"), jitter=("jitter", "mean"),
            map_coverage=("map_covered", "mean"), neighbor_coverage=("neighbor_covered", "mean"),
            fallback_count=("fallback", "sum"), latency_ms=("latency_ms", "mean")
        ).to_csv(root / filename, index=False, lineterminator="\n")
    return summary

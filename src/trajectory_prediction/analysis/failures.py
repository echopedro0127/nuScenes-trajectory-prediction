from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.analysis.maneuvers import describe_motion, tag_maneuvers
from trajectory_prediction.config import data_config_sha256, load_yaml, validate_data_config
from trajectory_prediction.data.cache import load_cache
from trajectory_prediction.evaluation.metrics import displacement_metrics

_IDENTIFIERS = ("instance_token", "sample_token", "scene_name")
_STAT_KEYS = (
    "mean_speed_mps",
    "max_speed_mps",
    "mean_acceleration_mps2",
    "yaw_change_deg",
    "displacement_m",
)


def _validate_top_k(top_k: object) -> int:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer")
    return top_k


def select_worst_cases(frame: pd.DataFrame, top_k: int = 15) -> pd.DataFrame:
    """Select the union of the top ADE and FDE rows with deterministic ordering."""
    top_k = _validate_top_k(top_k)
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("frame must be a pandas DataFrame")  # noqa: TRY004
    required = set(_IDENTIFIERS[:2]) | {"ade", "fde"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"failure frame missing columns: {sorted(missing)}")
    if frame.empty:
        return frame.copy()
    if not np.isfinite(frame[["ade", "fde"]].to_numpy(dtype=float)).all():
        raise ValueError("ade and fde must contain only finite values")
    identity = list(_IDENTIFIERS[:2])
    ade = frame.sort_values(
        ["ade", *identity], ascending=[False, True, True], kind="mergesort"
    ).head(top_k)
    fde = frame.sort_values(
        ["fde", *identity], ascending=[False, True, True], kind="mergesort"
    ).head(top_k)
    return (
        pd.concat([ade.assign(rank_source="ade"), fde.assign(rank_source="fde")])
        .drop_duplicates(identity, keep="first")
        .reset_index(drop=True)
    )


def _validate_arrays(arrays: Mapping[str, object], prediction: object) -> tuple[dict[str, list[str]], np.ndarray, np.ndarray]:
    if not isinstance(arrays, Mapping):
        raise ValueError("arrays must be a mapping")  # noqa: TRY004
    missing = set(_IDENTIFIERS) | {"future_xy"}
    absent = missing.difference(arrays)
    if absent:
        raise ValueError(f"analysis arrays missing: {sorted(absent)}")
    target = np.asarray(arrays["future_xy"])
    predicted = np.asarray(prediction)
    for name, value in (("future_xy", target), ("prediction", predicted)):
        if value.ndim != 3 or value.shape[1:] != (12, 2):
            raise ValueError(f"{name} must have shape [N, 12, 2]")
        if not np.issubdtype(value.dtype, np.floating) or not np.isfinite(value).all():
            raise ValueError(f"{name} must be a finite floating-point array")
    if target.shape != predicted.shape or target.shape[0] == 0:
        raise ValueError("prediction and future_xy must share a non-empty shape [N, 12, 2]")
    identifiers: dict[str, list[str]] = {}
    for name in _IDENTIFIERS:
        values = np.asarray(arrays[name])
        if values.ndim != 1 or values.shape[0] != target.shape[0]:
            raise ValueError(f"identifier {name} must have length {target.shape[0]}")
        if values.dtype.kind not in "US":
            raise ValueError(f"identifier {name} must contain strings")
        converted = [str(value) for value in values.tolist()]
        if any(not value for value in converted):
            raise ValueError(f"identifier {name} must contain nonempty strings")
        identifiers[name] = converted
    keys = list(zip(identifiers["instance_token"], identifiers["sample_token"], strict=True))
    if len(keys) != len(set(keys)):
        raise ValueError("identifier instance/sample pairs must be unique")
    return identifiers, predicted.astype(np.float32, copy=False), target.astype(np.float32, copy=False)


def analyze_arrays(
    arrays: Mapping[str, object],
    prediction: np.ndarray,
    thresholds: Mapping[str, object],
    top_k: int = 15,
    dt: float = 0.5,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build exploded maneuver rows, aggregate metrics, and worst-case records."""
    identifiers, predicted, target = _validate_arrays(arrays, prediction)
    if isinstance(dt, bool) or not isinstance(dt, (int, float)) or not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError("dt must be a finite positive number")
    per_sample_metrics = displacement_metrics(
        torch.from_numpy(predicted).double(), torch.from_numpy(target).double()
    )
    frame = pd.DataFrame(identifiers)
    frame["ade"] = per_sample_metrics["ade"].numpy()
    frame["fde"] = per_sample_metrics["fde"].numpy()
    frame["miss"] = per_sample_metrics["miss"].numpy()
    stats = [describe_motion(future, float(dt)) for future in target]
    frame["maneuver_tags"] = [tag_maneuvers(item, thresholds) for item in stats]
    for key in _STAT_KEYS:
        frame[key] = [item[key] for item in stats]

    exploded = frame.explode("maneuver_tags", ignore_index=True).rename(
        columns={"maneuver_tags": "maneuver"}
    )
    maneuver = (
        exploded.groupby("maneuver", as_index=False, sort=True)
        .agg(
            sample_count=("sample_token", "nunique"),
            ade=("ade", "mean"),
            fde=("fde", "mean"),
            miss=("miss", "mean"),
        )
    )
    worst = select_worst_cases(frame.drop(columns=["maneuver_tags"]), top_k=top_k)
    return exploded, maneuver, worst


def _json_safe_records(frame: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for row in frame.to_dict(orient="records"):
        clean: dict[str, object] = {}
        for key, value in row.items():
            if isinstance(value, np.generic):
                value = value.item()
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError("analysis output contains non-finite values")
            clean[key] = value
        records.append(clean)
    return records


def write_analysis(
    output_dir: Path,
    per_sample: pd.DataFrame,
    maneuver: pd.DataFrame,
    worst: pd.DataFrame,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_sample.to_csv(output_dir / "per_sample_analysis.csv", index=False, lineterminator="\n")
    maneuver.to_csv(output_dir / "maneuver_metrics.csv", index=False, lineterminator="\n")
    (output_dir / "worst_cases.json").write_text(
        json.dumps(_json_safe_records(worst), indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name == "data" and resolved.parent.parent.name == "configs":
        return resolved.parents[2]
    return resolved.parent


def _reorder_prediction(prediction: np.ndarray, rows: pd.DataFrame, arrays: Mapping[str, object]) -> np.ndarray:
    expected = list(zip(np.asarray(arrays["instance_token"]).tolist(), np.asarray(arrays["sample_token"]).tolist(), strict=True))
    if not {"instance_token", "sample_token"}.issubset(rows.columns):
        raise ValueError("per_sample.csv must contain instance_token and sample_token")
    actual = list(zip(rows["instance_token"].astype(str), rows["sample_token"].astype(str), strict=True))
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise ValueError("evaluation identifiers do not match cache identifiers")
    positions = {key: index for index, key in enumerate(actual)}
    return prediction[np.asarray([positions[key] for key in expected], dtype=np.int64)]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export maneuver-conditioned trajectory failures")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/mini.yaml"))
    parser.add_argument("--split", required=True)
    parser.add_argument("--evaluation-dir", type=Path, required=True)
    parser.add_argument("--thresholds", type=Path, default=Path("configs/analysis/maneuvers.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/analysis"))
    parser.add_argument("--top-k", type=int, default=15)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_config_path = args.data_config.resolve()
    project_root = _project_root(data_config_path)
    config = validate_data_config(load_yaml(data_config_path))
    if args.split not in config["splits"]:
        raise ValueError(f"split {args.split!r} is not listed in data configuration")
    cache_root = Path(config["cache_root"])
    if not cache_root.is_absolute():
        cache_root = project_root / cache_root
    arrays, _ = load_cache(cache_root / args.split, args.split, expected_config_sha256=data_config_sha256(config))
    evaluation_dir = args.evaluation_dir if args.evaluation_dir.is_absolute() else project_root / args.evaluation_dir
    prediction_path = evaluation_dir / "predictions.npz"
    per_sample_path = evaluation_dir / "per_sample.csv"
    with np.load(prediction_path, allow_pickle=False) as payload:
        if "prediction" not in payload:
            raise ValueError("predictions.npz must contain prediction")
        prediction = np.asarray(payload["prediction"])
    rows = pd.read_csv(per_sample_path)
    prediction = _reorder_prediction(prediction, rows, arrays)
    threshold_path = args.thresholds if args.thresholds.is_absolute() else project_root / args.thresholds
    thresholds = load_yaml(threshold_path)
    per_sample, maneuver, worst = analyze_arrays(arrays, prediction, thresholds, top_k=args.top_k)
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    write_analysis(output_dir, per_sample, maneuver, worst)


if __name__ == "__main__":
    main()

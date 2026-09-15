from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.analysis.maneuvers import describe_motion, tag_maneuvers
from trajectory_prediction.config import data_config_sha256, load_yaml, validate_data_config
from trajectory_prediction.data.cache import load_cache
from trajectory_prediction.models.physics import finite_difference_cv
from trajectory_prediction.online.corrector import ResidualCorrector
from trajectory_prediction.online.dataset import CorrectionDataset
from trajectory_prediction.online.replay import replay_updates


def _points(name: str, value: np.ndarray) -> np.ndarray:
    points = np.asarray(value)
    if points.ndim != 3 or points.shape[1:] != (12, 2):
        raise ValueError(f"{name} must have shape [N, 12, 2]")
    if not np.issubdtype(points.dtype, np.floating) or not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite floating-point data")
    return points.astype(np.float32, copy=False)


def _trajectory(name: str, value: np.ndarray) -> np.ndarray:
    points = np.asarray(value)
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] == 0:
        raise ValueError(f"{name} must have shape [T >= 1, 2]")
    if not np.issubdtype(points.dtype, np.floating) or not np.isfinite(points).all():
        raise ValueError(f"{name} must be finite floating-point data")
    return points.astype(np.float32, copy=False)


def update_metrics(
    current: np.ndarray,
    target: np.ndarray,
    previous_overlap: np.ndarray,
    latency_ms: float,
) -> dict[str, float]:
    """Compute rolling ADE/FDE, forecast jitter, and inference latency."""
    current_array = _trajectory("current", current)
    target_array = _trajectory("target", target)
    previous_array = _trajectory("previous_overlap", previous_overlap)
    if current_array.shape != target_array.shape or current_array.shape != previous_array.shape:
        raise ValueError("current, target, and previous_overlap must share shape [T, 2]")
    if isinstance(latency_ms, bool) or not isinstance(latency_ms, (int, float)) or not math.isfinite(float(latency_ms)) or latency_ms < 0:
        raise ValueError("latency_ms must be finite and nonnegative")
    error = np.linalg.norm(current_array - target_array, axis=-1)
    jitter = np.linalg.norm(current_array - previous_array, axis=-1)
    return {
        "ade": float(error.mean()),
        "fde": float(error[-1]),
        "jitter": float(jitter.mean()),
        "latency_ms": float(latency_ms),
    }


def aggregate_metrics(rows: Iterable[Mapping[str, object]]) -> dict[str, float | int]:
    """Aggregate finite online metric rows and report p50/p95 latency."""
    records = list(rows)
    if not records:
        raise ValueError("at least one online metric row is required")
    values: dict[str, np.ndarray] = {}
    for name in ("ade", "fde", "jitter", "latency_ms"):
        try:
            array = np.asarray([float(row[name]) for row in records], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"online rows must contain numeric {name}") from exc
        if not np.isfinite(array).all():
            raise ValueError("online rows must contain only finite values")
        values[name] = array
    return {
        "sample_count": len(records),
        "ade": float(values["ade"].mean()),
        "fde": float(values["fde"].mean()),
        "jitter": float(values["jitter"].mean()),
        "latency_p50_ms": float(np.percentile(values["latency_ms"], 50)),
        "latency_p95_ms": float(np.percentile(values["latency_ms"], 95)),
    }


def validate_corrector_metadata(
    metadata: Mapping[str, object],
    data_config_sha256: str,
    evaluation_split: str,
) -> None:
    """Reject checkpoints trained or selected on the split being evaluated."""
    if not isinstance(metadata, Mapping):
        raise TypeError("corrector checkpoint metadata must be a mapping")
    if metadata.get("data_config_sha256") != data_config_sha256:
        raise ValueError("corrector checkpoint data configuration mismatch")
    train_split = metadata.get("train_split")
    val_split = metadata.get("val_split")
    same_split_validation = metadata.get("same_split_validation", False)
    if not isinstance(train_split, str) or not isinstance(val_split, str):
        raise TypeError("corrector checkpoint must record train_split and val_split")
    if not train_split or not val_split:
        raise ValueError("corrector checkpoint train and validation splits must be different")
    if train_split == val_split and same_split_validation is not True:
        raise ValueError("corrector checkpoint train and validation splits must be different")
    if evaluation_split in {train_split, val_split}:
        raise ValueError("corrector checkpoint split must differ from evaluation split")


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(callable_object, device: torch.device) -> tuple[np.ndarray, float]:
    _synchronize(device)
    start = time.perf_counter_ns()
    result = callable_object()
    _synchronize(device)
    return result, (time.perf_counter_ns() - start) / 1_000_000.0


def evaluate_arrays(
    base_prediction: np.ndarray,
    future_xy: np.ndarray,
    initial_history: np.ndarray,
    corrector: ResidualCorrector | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Replay three causal strategies for every sample and update."""
    base = _points("base_prediction", base_prediction)
    future = _points("future_xy", future_xy)
    if base.shape != future.shape:
        raise ValueError("base_prediction and future_xy must share shape [N, 12, 2]")
    history = np.asarray(initial_history)
    if history.ndim != 3 or history.shape[0] != base.shape[0] or history.shape[1] < 1 or history.shape[2] != 2:
        raise ValueError("initial_history must have shape [N, H >= 1, 2]")
    if not np.issubdtype(history.dtype, np.floating) or not np.isfinite(history).all():
        raise ValueError("initial_history must be finite floating-point data")
    selected_device = device or torch.device("cpu")
    if corrector is not None:
        corrector = corrector.to(selected_device).eval()
    rows: list[dict[str, object]] = []
    for sample_index in range(base.shape[0]):
        previous: dict[str, np.ndarray | None] = {name: None for name in ("frozen", "sliding", "corrected")}
        updates = replay_updates(history[sample_index], future[sample_index], history_steps=history.shape[1])
        for update in updates:
            remaining = update.remaining_target
            step = update.update_step
            frozen, frozen_latency = _timed(
                lambda sample_index=sample_index, step=step: base[sample_index, step:].copy(),
                selected_device,
            )
            sliding, sliding_latency = _timed(
                lambda update=update, remaining=remaining: finite_difference_cv(
                    torch.from_numpy(update.observed_history[None].copy()).to(selected_device),
                    dt=0.5,
                    future_steps=remaining.shape[0],
                ).detach().cpu().numpy()[0],
                selected_device,
            )

            def corrected_forecast(sample_index=sample_index, step=step) -> np.ndarray:
                forecast = base[sample_index, step:].copy()
                if corrector is None:
                    return forecast
                dataset = CorrectionDataset(
                    base[sample_index : sample_index + 1],
                    future[sample_index : sample_index + 1],
                    history[sample_index : sample_index + 1],
                    update_step=step,
                )
                item = dataset[0]
                with torch.inference_mode():
                    offsets = corrector(item["features"][None].to(selected_device), item["mask"][None].to(selected_device))
                return forecast + offsets[0, step:].detach().cpu().numpy()

            corrected, corrected_latency = _timed(corrected_forecast, selected_device)
            forecasts = {"frozen": (frozen, frozen_latency), "sliding": (sliding, sliding_latency), "corrected": (corrected, corrected_latency)}
            for strategy, (forecast, latency) in forecasts.items():
                overlap = previous[strategy]
                previous_overlap = forecast if overlap is None else overlap[1:]
                values = update_metrics(forecast, remaining, previous_overlap, latency)
                rows.append({"sample_index": sample_index, "update_step": step, "strategy": strategy, **values})
                previous[strategy] = forecast
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("online replay produced no updates")
    expected = frame.groupby("sample_index", sort=False)["strategy"].nunique()
    if not (expected == 3).all():
        raise ValueError("all samples must produce all three online strategies")
    return frame


def write_online_outputs(frame: pd.DataFrame, output_dir: Path) -> None:
    required = {"sample_index", "update_step", "strategy", "ade", "fde", "jitter", "latency_ms"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"online frame missing columns: {sorted(missing)}")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "online_per_update.csv", index=False, lineterminator="\n")
    summaries = {
        strategy: aggregate_metrics(group.to_dict(orient="records"))
        for strategy, group in frame.groupby("strategy", sort=True)
    }
    (output_dir / "online_metrics.json").write_text(
        json.dumps(summaries, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    if "maneuver" in frame.columns:
        maneuver = (
            frame.groupby("maneuver", as_index=False, sort=True)
            .agg(sample_count=("sample_index", "nunique"), ade=("ade", "mean"), fde=("fde", "mean"), jitter=("jitter", "mean"), latency_ms=("latency_ms", "mean"))
        )
    else:
        maneuver = pd.DataFrame(columns=["maneuver", "sample_count", "ade", "fde", "jitter", "latency_ms"])
    maneuver.to_csv(output_dir / "online_maneuver_metrics.csv", index=False, lineterminator="\n")


def _project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name == "data" and resolved.parent.parent.name == "configs":
        return resolved.parents[2]
    return resolved.parent


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate causal online trajectory correction strategies")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/mini.yaml"))
    parser.add_argument("--split", required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--corrector-checkpoint", type=Path)
    parser.add_argument("--corrector-config", type=Path, default=Path("configs/models/corrector.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/online"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_config_path = args.data_config.resolve()
    project_root = _project_root(data_config_path)
    config = validate_data_config(load_yaml(data_config_path))
    if args.split not in config["splits"]:
        raise ValueError(f"split {args.split!r} is not listed in data configuration")
    config_hash = data_config_sha256(config)
    cache_root = Path(config["cache_root"])
    if not cache_root.is_absolute():
        cache_root = project_root / cache_root
    arrays, _ = load_cache(cache_root / args.split, args.split, expected_config_sha256=config_hash)
    prediction_path = args.base_predictions if args.base_predictions.is_absolute() else project_root / args.base_predictions
    with np.load(prediction_path, allow_pickle=False) as payload:
        if "prediction" not in payload:
            raise ValueError("base predictions must contain prediction")
        prediction = payload["prediction"]
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    corrector = None
    if args.corrector_checkpoint is not None:
        checkpoint_path = args.corrector_checkpoint if args.corrector_checkpoint.is_absolute() else project_root / args.corrector_checkpoint
        document = load_yaml(args.corrector_config if args.corrector_config.is_absolute() else project_root / args.corrector_config)
        model_config = document.get("model", {})
        corrector = ResidualCorrector(**model_config)
        payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if not isinstance(payload, Mapping):
            raise ValueError("corrector checkpoint payload must be a mapping")
        validate_corrector_metadata(payload.get("metadata"), config_hash, args.split)
        corrector.load_state_dict(payload["model"] if isinstance(payload, Mapping) and "model" in payload else payload)
    frame = evaluate_arrays(prediction, arrays["future_xy"], arrays["history_xy"], corrector=corrector, device=device)
    thresholds_path = project_root / "configs/analysis/maneuvers.yaml"
    thresholds = load_yaml(thresholds_path)
    tags = [tag_maneuvers(describe_motion(future, 0.5), thresholds) for future in arrays["future_xy"]]
    frame["maneuver"] = [tags[int(index)][0] for index in frame["sample_index"]]
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    write_online_outputs(frame, output_dir)


if __name__ == "__main__":
    main()

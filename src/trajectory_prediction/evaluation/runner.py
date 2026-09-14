from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from trajectory_prediction.config import data_config_sha256, load_yaml, validate_data_config
from trajectory_prediction.data.cache import load_cache
from trajectory_prediction.data.geometry import agent_to_global
from trajectory_prediction.evaluation.metrics import displacement_metrics, horizon_metrics
from trajectory_prediction.models.factory import build_model
from trajectory_prediction.models.physics import constant_velocity_heading, finite_difference_cv
from trajectory_prediction.training.dataset import TrajectoryDataset

FUTURE_STEPS = 12
FREQUENCY_HZ = 2
ALLOWED_HISTORY_SECONDS = (0.5, 1.0, 2.0)
_IDENTIFIER_NAMES = ("instance_token", "sample_token", "scene_name")
_LEARNED_MODELS = frozenset({"gru", "transformer"})


def _trajectory_array(name: str, value: object) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[1:] != (FUTURE_STEPS, 2):
        raise ValueError(f"{name} must have shape [N, 12, 2]")
    if not np.issubdtype(array.dtype, np.floating):
        raise ValueError(f"{name} must be a floating-point array")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _identifier_values(identifiers: Mapping[str, object], sample_count: int) -> dict[str, list[str]]:
    missing = set(_IDENTIFIER_NAMES).difference(identifiers)
    if missing:
        raise ValueError(f"identifier arrays missing: {sorted(missing)}")
    values: dict[str, list[str]] = {}
    for name in _IDENTIFIER_NAMES:
        array = np.asarray(identifiers[name])
        if array.ndim != 1 or len(array) != sample_count:
            raise ValueError(f"identifier {name} must have length {sample_count}")
        converted: list[str] = []
        for value in array.tolist():
            if not isinstance(value, str) or not value:
                raise ValueError(f"identifier {name} must contain nonempty strings")
            converted.append(value)
        values[name] = converted
    return values


def _validate_prediction_inputs(
    prediction: object,
    target: object,
    identifiers: Mapping[str, object],
) -> tuple[np.ndarray, np.ndarray, dict[str, list[str]]]:
    prediction_array = _trajectory_array("prediction", prediction)
    target_array = _trajectory_array("target", target)
    if prediction_array.shape != target_array.shape:
        raise ValueError("prediction and target must have the same shape [N, 12, 2]")
    if prediction_array.shape[0] == 0:
        raise ValueError("prediction and target must contain at least one sample")
    identifier_values = _identifier_values(identifiers, prediction_array.shape[0])
    return prediction_array, target_array, identifier_values


def write_evaluation(
    output_dir: Path,
    prediction: np.ndarray,
    target: np.ndarray,
    identifiers: Mapping[str, object],
    provenance: Mapping[str, object] | None = None,
) -> None:
    """Write local-coordinate metrics, per-sample rows, and reproducible arrays."""
    prediction_array, target_array, identifier_values = _validate_prediction_inputs(
        prediction, target, identifiers
    )
    if provenance is not None and not isinstance(provenance, Mapping):
        raise ValueError("provenance must be a mapping")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_tensor = torch.from_numpy(prediction_array).double()
    target_tensor = torch.from_numpy(target_array).double()
    metrics = displacement_metrics(prediction_tensor, target_tensor)
    horizons = horizon_metrics(prediction_tensor, target_tensor, frequency_hz=FREQUENCY_HZ)
    summary: dict[str, object] = {
        "sample_count": int(prediction_array.shape[0]),
        "ade": float(metrics["ade"].mean().item()),
        "fde": float(metrics["fde"].mean().item()),
        "miss_rate_2": float(metrics["miss"].mean().item()),
        "horizons": {
            name: {
                "ade": float(values["ade"].mean().item()),
                "fde": float(values["fde"].mean().item()),
                "miss_rate_2": float(values["miss"].mean().item()),
            }
            for name, values in horizons.items()
        },
    }
    if provenance is not None:
        summary["provenance"] = dict(provenance)
    serialized = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)
    (output_dir / "metrics.json").write_text(serialized + "\n", encoding="utf-8")

    frame = pd.DataFrame(identifier_values, columns=list(_IDENTIFIER_NAMES))
    frame["ade"] = metrics["ade"].numpy()
    frame["fde"] = metrics["fde"].numpy()
    frame["miss"] = metrics["miss"].numpy()
    frame.to_csv(output_dir / "per_sample.csv", index=False, lineterminator="\n")
    np.savez_compressed(
        output_dir / "predictions.npz",
        prediction=prediction_array,
        target=target_array,
    )


def write_official_predictions(
    output_path: Path,
    prediction_global: np.ndarray,
    identifiers: Mapping[str, object],
) -> None:
    """Write the single-mode nuScenes prediction JSON in cache order."""
    prediction_array = _trajectory_array("prediction_global", prediction_global)
    identifier_values = _identifier_values(identifiers, prediction_array.shape[0])
    keys = [
        f"{instance}_{sample}"
        for instance, sample in zip(
            identifier_values["instance_token"],
            identifier_values["sample_token"],
            strict=True,
        )
    ]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate instance/sample prediction keys")
    payload = {
        key: {
            "prediction": prediction_array[index].tolist(),
            "probabilities": [1.0],
        }
        for index, key in enumerate(keys)
    }
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _validate_history_seconds(value: float) -> float:
    if isinstance(value, bool) or not math.isfinite(value) or value not in ALLOWED_HISTORY_SECONDS:
        raise ValueError("history-seconds must be one of 0.5, 1.0, 2.0")
    return float(value)


def history_steps_from_seconds(history_seconds: float) -> int:
    """Convert an allowed 2 Hz history duration to inclusive point count."""
    seconds = _validate_history_seconds(history_seconds)
    return int(seconds * FREQUENCY_HZ) + 1


def validate_checkpoint_requirement(model_name: str, checkpoint: Path | None) -> None:
    if model_name in _LEARNED_MODELS and checkpoint is None:
        raise ValueError(f"checkpoint is required for learned model {model_name}")


def _history_suffix(array: np.ndarray, history_steps: int, name: str) -> np.ndarray:
    if array.ndim < 2 or array.shape[1] < history_steps:
        raise ValueError(f"{name} does not contain {history_steps} history steps")
    return array[:, -history_steps:]


def _dispatch_physics(
    model_name: str,
    arrays: Mapping[str, np.ndarray],
    history_steps: int,
    device: torch.device | None = None,
) -> np.ndarray:
    """Run a deterministic baseline using local-coordinate cache arrays."""
    if model_name not in {"finite_difference_cv", "constant_velocity_heading"}:
        raise ValueError(f"unknown physics model: {model_name}")
    if isinstance(history_steps, bool) or not isinstance(history_steps, int) or history_steps < 2:
        raise ValueError("history_steps must be an integer >= 2 for physics baselines")
    selected_device = device or torch.device("cpu")
    if model_name == "finite_difference_cv":
        history = _history_suffix(np.asarray(arrays["history_xy"]), history_steps, "history_xy")
        output = finite_difference_cv(
            torch.from_numpy(history).to(selected_device),
            dt=1.0 / FREQUENCY_HZ,
            future_steps=FUTURE_STEPS,
        )
    else:
        speed = torch.from_numpy(np.asarray(arrays["current_speed"])).to(selected_device)
        output = constant_velocity_heading(
            speed,
            dt=1.0 / FREQUENCY_HZ,
            future_steps=FUTURE_STEPS,
        )
    return output.detach().cpu().numpy().astype(np.float32, copy=False)


def _load_model_checkpoint(model: torch.nn.Module, checkpoint: Path, device: torch.device) -> None:
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if isinstance(payload, Mapping) and "model" in payload:
        state = payload["model"]
    else:
        state = payload
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint must contain a model state mapping")  # noqa: TRY004
    model.load_state_dict(state)


def _predict_learned(
    model_config: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    history_steps: int,
    checkpoint: Path,
    device: torch.device,
) -> np.ndarray:
    model = build_model(dict(model_config))
    _load_model_checkpoint(model, checkpoint, device)
    model.to(device)
    model.eval()
    dataset = TrajectoryDataset(arrays, history_steps=history_steps)
    loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
    predictions: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            features = batch["history_features"].to(device)
            mask = batch["history_mask"].to(device)
            output = model(features, mask)
            predictions.append(output.detach().cpu().numpy())
    if not predictions:
        raise ValueError("evaluation cache contains no samples")
    return np.concatenate(predictions, axis=0).astype(np.float32, copy=False)


def _predict_local(
    model_config: Mapping[str, object],
    arrays: Mapping[str, np.ndarray],
    history_steps: int,
    checkpoint: Path | None,
    device: torch.device,
) -> np.ndarray:
    model_name = model_config.get("name")
    if not isinstance(model_name, str):
        raise ValueError("model.name must be a string")  # noqa: TRY004
    validate_checkpoint_requirement(model_name, checkpoint)
    if model_name in {"finite_difference_cv", "constant_velocity_heading"}:
        if checkpoint is not None:
            raise ValueError(f"checkpoint is not used by physics model {model_name}")
        return _dispatch_physics(model_name, arrays, history_steps, device=device)
    if model_name in _LEARNED_MODELS:
        if checkpoint is None:  # defensive; requirement checked above
            raise ValueError(f"checkpoint is required for learned model {model_name}")
        return _predict_learned(model_config, arrays, history_steps, checkpoint, device)
    raise ValueError(f"unknown model: {model_name}")


def _inverse_transform_batch(prediction_local: np.ndarray, arrays: Mapping[str, np.ndarray]) -> np.ndarray:
    prediction_array = _trajectory_array("prediction_local", prediction_local)
    anchors = np.asarray(arrays["anchor_xy"])
    yaws = np.asarray(arrays["anchor_yaw"])
    if anchors.shape != (prediction_array.shape[0], 2):
        raise ValueError("anchor_xy must have shape [N, 2]")
    if yaws.shape != (prediction_array.shape[0],):
        raise ValueError("anchor_yaw must have shape [N]")
    if not np.isfinite(anchors).all() or not np.isfinite(yaws).all():
        raise ValueError("anchor transforms must contain finite values")
    return np.stack(
        [
            agent_to_global(prediction_array[index], anchors[index], float(yaws[index]))
            for index in range(prediction_array.shape[0])
        ],
        axis=0,
    )


def _mapping_sha256(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name == "data" and resolved.parent.parent.name == "configs":
        return resolved.parents[2]
    return resolved.parent


def _validate_model_config(config: object) -> dict:
    if not isinstance(config, dict) or not isinstance(config.get("model"), dict):
        raise ValueError("model configuration requires a model mapping")  # noqa: TRY004
    model = config["model"]
    if not isinstance(model.get("name"), str) or not model["name"].strip():
        raise ValueError("model.name must be a nonempty string")
    if model.get("future_steps") != FUTURE_STEPS:
        raise ValueError("model.future_steps must equal 12")
    return config


def build_evaluation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a static nuScenes trajectory model")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/mini.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/cv.yaml"))
    parser.add_argument("--split", required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--history-seconds", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/eval"))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_evaluation_parser().parse_args(argv)
    history_steps = history_steps_from_seconds(args.history_seconds)
    data_config_path = args.data_config.resolve()
    project_root = _project_root(data_config_path)
    data_config = validate_data_config(load_yaml(data_config_path))
    if args.split not in data_config["splits"]:
        raise ValueError(f"split {args.split!r} is not listed in data configuration")
    data_hash = data_config_sha256(data_config)
    cache_root = _resolve_path(data_config["cache_root"], project_root)
    arrays, _ = load_cache(
        cache_root / args.split,
        args.split,
        expected_config_sha256=data_hash,
    )
    model_config_path = _resolve_path(args.model_config, project_root)
    model_document = _validate_model_config(load_yaml(model_config_path))
    model_config = model_document["model"]
    checkpoint = _resolve_path(args.checkpoint, project_root) if args.checkpoint else None
    validate_checkpoint_requirement(model_config["name"], checkpoint)
    if checkpoint is not None and not checkpoint.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {checkpoint}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction_local = _predict_local(
        model_config,
        arrays,
        history_steps,
        checkpoint,
        device,
    )
    target = np.asarray(arrays["future_xy"], dtype=np.float32)
    identifiers = {name: arrays[name] for name in _IDENTIFIER_NAMES}
    provenance: dict[str, object] = {
        "data_config_sha256": data_hash,
        "model_config_sha256": _mapping_sha256(model_document),
        "split": args.split,
        "history_seconds": float(args.history_seconds),
        "history_steps": history_steps,
        "model_name": model_config["name"],
    }
    if checkpoint is not None:
        provenance["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output_dir = _resolve_path(args.output_dir, project_root)
    write_evaluation(output_dir, prediction_local, target, identifiers, provenance=provenance)
    prediction_global = _inverse_transform_batch(prediction_local, arrays)
    write_official_predictions(output_dir / "official_predictions.json", prediction_global, identifiers)


if __name__ == "__main__":
    main()

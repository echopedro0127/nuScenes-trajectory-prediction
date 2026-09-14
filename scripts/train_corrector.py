from __future__ import annotations

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import ConcatDataset, DataLoader

from trajectory_prediction.config import data_config_sha256, load_yaml, validate_data_config
from trajectory_prediction.data.cache import load_cache
from trajectory_prediction.online.corrector import ResidualCorrector, smoothness_loss
from trajectory_prediction.online.dataset import CorrectionDataset
from trajectory_prediction.training.checkpoint import save_checkpoint
from trajectory_prediction.utils.seed import seed_everything


def _project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name == "data" and resolved.parent.parent.name == "configs":
        return resolved.parents[2]
    return resolved.parent


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _load_predictions(path: Path, sample_count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "prediction" not in payload:
            raise ValueError("base predictions must contain a prediction array")
        prediction = np.asarray(payload["prediction"])
    if prediction.shape != (sample_count, 12, 2):
        raise ValueError(f"base prediction must have shape ({sample_count}, 12, 2)")
    if not np.issubdtype(prediction.dtype, np.floating) or not np.isfinite(prediction).all():
        raise ValueError("base prediction must be finite floating-point data")
    return prediction.astype(np.float32, copy=False)


def _train_epoch(
    model: ResidualCorrector,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    smoothness_weight: float,
) -> float:
    model.train()
    total = 0.0
    samples = 0
    for batch in loader:
        features = batch["features"].to(device)
        mask = batch["mask"].to(device)
        target = batch["target_offset"].to(device)
        offsets = model(features, mask)
        if not mask.any():
            continue
        loss = F.smooth_l1_loss(offsets[mask], target[mask])
        loss = loss + smoothness_weight * smoothness_loss(offsets, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        batch_size = int(features.shape[0])
        total += float(loss.detach().cpu()) * batch_size
        samples += batch_size
    if samples == 0:
        raise ValueError("correction dataset contains no valid target points")
    return total / samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a causal residual trajectory corrector")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/mini.yaml"))
    parser.add_argument("--split", required=True)
    parser.add_argument("--base-predictions", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/corrector.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/corrector"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    data_config_path = args.data_config.resolve()
    project_root = _project_root(data_config_path)
    data_config = validate_data_config(load_yaml(data_config_path))
    if args.split not in data_config["splits"]:
        raise ValueError(f"split {args.split!r} is not listed in data configuration")
    config_hash = data_config_sha256(data_config)
    cache_root = Path(data_config["cache_root"])
    if not cache_root.is_absolute():
        cache_root = project_root / cache_root
    arrays, _ = load_cache(cache_root / args.split, args.split, expected_config_sha256=config_hash)
    prediction_path = args.base_predictions if args.base_predictions.is_absolute() else project_root / args.base_predictions
    base_prediction = _load_predictions(prediction_path, len(arrays["future_xy"]))

    model_path = args.model_config if args.model_config.is_absolute() else project_root / args.model_config
    document = load_yaml(model_path)
    model_config = document.get("model")
    training_config = document.get("training")
    if not isinstance(model_config, dict) or not isinstance(training_config, dict):
        raise ValueError("model config must contain model and training mappings")  # noqa: TRY004
    feature_dim = _positive_int("feature_dim", model_config.get("feature_dim"))
    hidden_dim = _positive_int("hidden_dim", model_config.get("hidden_dim"))
    epochs = _positive_int("epochs", training_config.get("epochs"))
    batch_size = _positive_int("batch_size", training_config.get("batch_size"))
    seed = training_config.get("seed")
    seed = _positive_int("seed", seed) if seed != 0 else 0
    learning_rate = float(training_config.get("learning_rate", 0.001))
    weight_decay = float(training_config.get("weight_decay", 0.0))
    smoothness_weight = float(training_config.get("smoothness_weight", 0.05))
    if not all(np.isfinite(value) and value >= 0.0 for value in (learning_rate, weight_decay, smoothness_weight)) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive; weights must be finite and nonnegative")
    seed_everything(seed)
    datasets = [
        CorrectionDataset(base_prediction, arrays["future_xy"], arrays["history_xy"], update_step)
        for update_step in range(1, 12)
    ]
    loader = DataLoader(ConcatDataset(datasets), batch_size=batch_size, shuffle=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model = ResidualCorrector(feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    output_dir = args.output_dir if args.output_dir.is_absolute() else project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {"data_config_sha256": config_hash, "split": args.split, "base_prediction_sha256": hashlib.sha256(prediction_path.read_bytes()).hexdigest()}
    rows: list[dict[str, float | int]] = []
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        loss = _train_epoch(model, loader, optimizer, device, smoothness_weight)
        rows.append({"epoch": epoch, "loss": loss})
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, loss, metadata=metadata)
        if loss < best_loss:
            best_loss = loss
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, loss, metadata=metadata)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "loss"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

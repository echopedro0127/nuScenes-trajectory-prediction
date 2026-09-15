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


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_split_names(train_split: str, val_split: str, *, allow_same: bool = False) -> None:
    if not isinstance(train_split, str) or not train_split:
        raise ValueError("train split must be a non-empty string")
    if not isinstance(val_split, str) or not val_split:
        raise ValueError("validation split must be a non-empty string")
    if train_split == val_split and not allow_same:
        raise ValueError("train and validation splits must be different")


def _load_predictions(path: Path, sample_count: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "prediction" not in payload:
            raise ValueError("base prediction must contain a prediction array")
        prediction = np.asarray(payload["prediction"])
    if prediction.shape != (sample_count, 12, 2):
        raise ValueError(f"base prediction must have shape ({sample_count}, 12, 2)")
    if not np.issubdtype(prediction.dtype, np.floating) or not np.isfinite(prediction).all():
        raise ValueError("base prediction must be finite floating-point data")
    return prediction.astype(np.float32, copy=False)


def _build_datasets(
    base_prediction: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> list[CorrectionDataset]:
    return [
        CorrectionDataset(
            base_prediction,
            arrays["future_xy"],
            arrays["history_xy"],
            update_step,
        )
        for update_step in range(1, 12)
    ]


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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        batch_size = int(features.shape[0])
        total += float(loss.detach().cpu()) * batch_size
        samples += batch_size
    if samples == 0:
        raise ValueError("correction dataset contains no valid target points")
    return total / samples


def _evaluate_epoch(
    model: ResidualCorrector,
    datasets: list[CorrectionDataset],
    batch_size: int,
    device: torch.device,
) -> dict[str, float | int]:
    if not datasets:
        raise ValueError("at least one validation dataset is required")
    model.eval()
    update_losses: list[float] = []
    update_ades: list[float] = []
    update_fdes: list[float] = []
    with torch.inference_mode():
        for dataset in datasets:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
            loss_total = 0.0
            ade_total = 0.0
            fde_total = 0.0
            samples = 0
            for batch in loader:
                features = batch["features"].to(device)
                mask = batch["mask"].to(device)
                target = batch["target_offset"].to(device)
                offsets = model(features, mask)
                valid = mask[0]
                errors = torch.linalg.vector_norm(offsets - target, dim=-1)
                batch_ade = errors[:, valid].mean(dim=1)
                batch_fde = errors[:, -1]
                batch_size_actual = int(features.shape[0])
                loss_total += float(F.smooth_l1_loss(offsets[mask], target[mask])) * batch_size_actual
                ade_total += float(batch_ade.sum())
                fde_total += float(batch_fde.sum())
                samples += batch_size_actual
            if samples == 0:
                raise ValueError("validation dataset contains no samples")
            update_losses.append(loss_total / samples)
            update_ades.append(ade_total / samples)
            update_fdes.append(fde_total / samples)
    return {
        "loss": float(np.mean(update_losses)),
        "ade": float(np.mean(update_ades)),
        "fde": float(np.mean(update_fdes)),
        "samples": int(sum(len(dataset) for dataset in datasets)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a causal residual trajectory corrector")
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/mini.yaml"))
    parser.add_argument("--train-split", required=True)
    parser.add_argument("--val-split", required=True)
    parser.add_argument("--train-base-predictions", type=Path, required=True)
    parser.add_argument("--val-base-predictions", type=Path, required=True)
    parser.add_argument(
        "--allow-same-split-validation",
        action="store_true",
        help="allow training-time selection on the train split for mini smoke runs",
    )
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/corrector.yaml"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/corrector"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validate_split_names(args.train_split, args.val_split, allow_same=args.allow_same_split_validation)
    data_config_path = args.data_config.resolve()
    project_root = _project_root(data_config_path)
    data_config = validate_data_config(load_yaml(data_config_path))
    if args.train_split not in data_config["splits"] or args.val_split not in data_config["splits"]:
        raise ValueError("train and validation splits must be listed in data configuration")
    config_hash = data_config_sha256(data_config)
    cache_root = _resolve_path(data_config["cache_root"], project_root)
    train_arrays, _ = load_cache(
        cache_root / args.train_split,
        args.train_split,
        expected_config_sha256=config_hash,
    )
    val_arrays, _ = load_cache(
        cache_root / args.val_split,
        args.val_split,
        expected_config_sha256=config_hash,
    )
    train_scenes = set(train_arrays["scene_name"].tolist())
    val_scenes = set(val_arrays["scene_name"].tolist())
    overlap = train_scenes.intersection(val_scenes)
    if overlap and not args.allow_same_split_validation:
        raise ValueError(f"train/validation scene overlap: {sorted(overlap)}")

    train_prediction_path = _resolve_path(args.train_base_predictions, project_root)
    val_prediction_path = _resolve_path(args.val_base_predictions, project_root)
    train_prediction = _load_predictions(train_prediction_path, len(train_arrays["future_xy"]))
    val_prediction = _load_predictions(val_prediction_path, len(val_arrays["future_xy"]))

    model_path = _resolve_path(args.model_config, project_root)
    document = load_yaml(model_path)
    model_config = document.get("model")
    training_config = document.get("training")
    if not isinstance(model_config, dict) or not isinstance(training_config, dict):
        raise TypeError("model config must contain model and training mappings")
    feature_dim = _positive_int("feature_dim", model_config.get("feature_dim"))
    hidden_dim = _positive_int("hidden_dim", model_config.get("hidden_dim"))
    epochs = _positive_int("epochs", training_config.get("epochs"))
    batch_size = _positive_int("batch_size", training_config.get("batch_size"))
    patience = _positive_int("patience", training_config.get("patience"))
    seed = training_config.get("seed")
    seed = _positive_int("seed", seed) if seed != 0 else 0
    learning_rate = float(training_config.get("learning_rate", 0.001))
    weight_decay = float(training_config.get("weight_decay", 0.0))
    smoothness_weight = float(training_config.get("smoothness_weight", 0.05))
    if not all(np.isfinite(value) and value >= 0.0 for value in (learning_rate, weight_decay, smoothness_weight)) or learning_rate <= 0.0:
        raise ValueError("learning_rate must be positive; weights must be finite and nonnegative")
    seed_everything(seed)
    train_datasets = _build_datasets(train_prediction, train_arrays)
    val_datasets = _build_datasets(val_prediction, val_arrays)
    train_loader = DataLoader(ConcatDataset(train_datasets), batch_size=batch_size, shuffle=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    model = ResidualCorrector(feature_dim=feature_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    output_dir = _resolve_path(args.output_dir, project_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "data_config_sha256": config_hash,
        "train_split": args.train_split,
        "val_split": args.val_split,
        "same_split_validation": args.allow_same_split_validation,
        "train_base_prediction_sha256": hashlib.sha256(train_prediction_path.read_bytes()).hexdigest(),
        "val_base_prediction_sha256": hashlib.sha256(val_prediction_path.read_bytes()).hexdigest(),
    }
    rows: list[dict[str, float | int]] = []
    best_ade = float("inf")
    bad_epochs = 0
    for epoch in range(1, epochs + 1):
        train_loss = _train_epoch(model, train_loader, optimizer, device, smoothness_weight)
        validation = _evaluate_epoch(model, val_datasets, batch_size, device)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": float(validation["loss"]),
            "val_ade": float(validation["ade"]),
            "val_fde": float(validation["fde"]),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
        }
        rows.append(row)
        improved = row["val_ade"] < best_ade
        if improved:
            best_ade = float(row["val_ade"])
            bad_epochs = 0
        else:
            bad_epochs += 1
        save_checkpoint(output_dir / "last.pt", model, optimizer, epoch, best_ade, bad_epochs, metadata=metadata)
        if improved:
            save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best_ade, bad_epochs, metadata=metadata)
        if bad_epochs >= patience:
            break
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["epoch", "train_loss", "val_loss", "val_ade", "val_fde", "learning_rate"])
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()

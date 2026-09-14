from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from trajectory_prediction.config import (
    data_config_sha256,
    load_yaml,
    validate_data_config,
)
from trajectory_prediction.data.cache import load_cache
from trajectory_prediction.models.factory import build_model
from trajectory_prediction.training.dataset import TrajectoryDataset
from trajectory_prediction.training.engine import fit
from trajectory_prediction.utils.seed import seed_everything


def _resolve_path(value: str | Path, base: Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def _project_root(config_path: Path) -> Path:
    resolved = config_path.resolve()
    if resolved.parent.name == "data" and resolved.parent.parent.name == "configs":
        return resolved.parents[2]
    return resolved.parent


def _mapping_sha256(value: dict) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_model_config(config: object) -> dict:
    if not isinstance(config, dict):
        raise TypeError("model configuration must be a mapping")
    for key in ("model", "training"):
        if not isinstance(config.get(key), dict):
            raise TypeError(f"model configuration requires mapping: {key}")
    required_training = {
        "seed",
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "patience",
    }
    missing = required_training.difference(config["training"])
    if missing:
        raise ValueError(f"model training configuration missing keys: {sorted(missing)}")
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-config", type=Path, default=Path("configs/data/mini.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/gru.yaml"))
    parser.add_argument("--train-split", default="mini_train")
    parser.add_argument("--val-split", default="mini_val")
    parser.add_argument("--history-seconds", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/mini/gru"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if args.history_seconds not in {0.5, 1.0, 2.0}:
        raise ValueError("history-seconds must be one of 0.5, 1.0, 2.0")
    data_config_path = args.data_config.resolve()
    project_root = _project_root(data_config_path)
    data_config = validate_data_config(load_yaml(data_config_path))
    if args.train_split == args.val_split:
        raise ValueError("train and validation splits must be different")
    if args.train_split not in data_config["splits"] or args.val_split not in data_config["splits"]:
        raise ValueError("train and validation splits must be listed in data config")
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
    if overlap:
        raise ValueError(f"train/validation scene overlap: {sorted(overlap)}")
    history_steps = int(args.history_seconds * data_config["frequency_hz"]) + 1
    model_config_path = _resolve_path(args.model_config, project_root)
    model_config = _validate_model_config(load_yaml(model_config_path))
    seed_everything(int(model_config["training"]["seed"]))
    train_dataset = TrajectoryDataset(train_arrays, history_steps=history_steps)
    val_dataset = TrajectoryDataset(val_arrays, history_steps=history_steps)
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(model_config["training"]["batch_size"]),
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(model_config["training"]["batch_size"]),
        shuffle=False,
    )
    model = build_model(model_config["model"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    training = model_config["training"]
    run_metadata = {
        "data_config_sha256": config_hash,
        "model_config_sha256": _mapping_sha256(model_config),
        "train_split": args.train_split,
        "val_split": args.val_split,
        "history_steps": history_steps,
    }
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    fit(
        model,
        optimizer,
        train_loader,
        val_loader,
        device=device,
        output_dir=_resolve_path(args.output_dir, project_root),
        epochs=int(training["epochs"]),
        patience=int(training["patience"]),
        resume=args.resume,
        run_metadata=run_metadata,
    )


if __name__ == "__main__":
    main()

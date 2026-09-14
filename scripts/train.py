from __future__ import annotations

import argparse
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
    data_config = validate_data_config(load_yaml(args.data_config))
    if args.train_split not in data_config["splits"] or args.val_split not in data_config["splits"]:
        raise ValueError("train and validation splits must be listed in data config")
    config_hash = data_config_sha256(data_config)
    train_arrays, _ = load_cache(
        Path(data_config["cache_root"]) / args.train_split,
        args.train_split,
        expected_config_sha256=config_hash,
    )
    val_arrays, _ = load_cache(
        Path(data_config["cache_root"]) / args.val_split,
        args.val_split,
        expected_config_sha256=config_hash,
    )
    history_steps = int(args.history_seconds * data_config["frequency_hz"]) + 1
    model_config = load_yaml(args.model_config)
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
        output_dir=args.output_dir,
        epochs=int(training["epochs"]),
        patience=int(training["patience"]),
        resume=args.resume,
    )


if __name__ == "__main__":
    main()

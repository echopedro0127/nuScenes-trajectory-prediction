"""Scene-isolated control training followed by frozen-control map conditioning."""
import argparse
import json
import math
import re
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.context.maps import MapConfig
from trajectory_prediction.context.preparation import load_bundle, load_predictions
from trajectory_prediction.online.context_corrector import MapCorrector
from trajectory_prediction.online.context_dataset import ContextDataset
from trajectory_prediction.online.corrector import ResidualCorrector, smoothness_loss
from trajectory_prediction.training.checkpoint import save_checkpoint
from trajectory_prediction.utils.seed import seed_everything


def scene_partition(scenes):
    scenes = sorted(set(scenes))
    if len(scenes) < 2:
        raise ValueError("at least two scenes are required for isolated selection")
    count = max(1, math.ceil(len(scenes) / 4))
    return scenes[:-count], scenes[-count:]


def shared_contract(contract):
    keys = ("dataset_version", "config", "source", "preprocessing_sha256", "time_policy",
            "oracle", "coordinate_policy", "map_source_timestamp_us", "map_available_timestamp_us", "map_age")
    return {key: contract.get(key) for key in keys}


def validate_scene_metadata(metadata):
    partitions = []
    for field in ("train_scenes", "selection_scenes"):
        scenes = metadata.get(field)
        if not isinstance(scenes, list) or not scenes or any(not isinstance(s, str) or not s for s in scenes) or sorted(set(scenes)) != scenes:
            raise ValueError("checkpoint must record unique sorted scene partitions")
        if metadata.get(field + "_sha256") != digest_json(scenes):
            raise ValueError("checkpoint scene hash mismatch")
        partitions.append(set(scenes))
    if partitions[0] & partitions[1]:
        raise ValueError("train/selection scene overlap")
    def require_hash(value):
        if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
            raise ValueError("checkpoint missing valid source artifact hash")

    identity = metadata.get("base_model", {})
    if identity.get("model_name") != "gru" or identity.get("history_seconds") != 2.0 or identity.get("history_steps") != 5:
        raise ValueError("checkpoint requires fixed GRU base model identity")
    for field in ("checkpoint_sha256", "model_config_sha256", "data_config_sha256"):
        require_hash(identity.get(field))
    splits = []
    for field in ("train_provenance", "selection_provenance"):
        provenance = metadata.get(field, {})
        require_hash(provenance.get("context_sha256"))
        base = provenance.get("base", {})
        for name in ("npz_sha256", "keys_sha256", "metrics_sha256"):
            require_hash(base.get(name))
        if base.get("model_identity") != identity:
            raise ValueError("checkpoint base model provenance mismatch")
        contract = provenance.get("contract")
        if not isinstance(contract, dict) or contract.get("schema_version") != 1 or shared_contract(contract) != metadata.get("shared_contract"):
            raise ValueError("checkpoint stored source contract mismatch")
        for name in ("keys_sha256", "trajectory_sha256", "preprocessing_sha256"):
            require_hash(contract.get(name))
        if contract["preprocessing_sha256"] != identity["data_config_sha256"]:
            raise ValueError("checkpoint base preprocessing mismatch")
        source = contract.get("source")
        if not isinstance(source, dict) or not source:
            raise ValueError("checkpoint missing source files")
        for value in source.values():
            require_hash(value)
        MapConfig(**contract["config"])
        splits.append(contract.get("split"))
    if tuple(splits) not in (("mini_train", "mini_train"), ("train", "train_val")):
        raise ValueError("checkpoint train/selection split mismatch")


def load_map_checkpoint(path, contract, evaluation_scenes, expected_base_identity=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    if payload.get("checkpoint_schema") != 1 or metadata.get("shared_contract") != shared_contract(contract):
        raise ValueError("checkpoint context contract mismatch")
    validate_scene_metadata(metadata)
    if expected_base_identity is not None and metadata.get("base_model") != expected_base_identity:
        raise ValueError("checkpoint base model identity mismatch")
    if set(evaluation_scenes) & set(metadata["train_scenes"] + metadata["selection_scenes"]):
        raise ValueError("evaluation scene overlap with training or selection")
    dims = metadata["model_config"]
    model = MapCorrector(ResidualCorrector(hidden_dim=dims["fallback_hidden_dim"]), dims["hidden_dim"])
    model.load_state_dict(payload["model"], strict=True)
    return model.eval(), payload


def _forward(model, batch, device):
    args = [batch[name].to(device) for name in ("features", "mask")]
    if isinstance(model, MapCorrector):
        args.extend(batch[name].to(device) for name in ("map_features", "map_mask"))
    return model(*args)


def _epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    total_loss, total_ade, samples = 0., 0., 0
    for batch in loader:
        mask, target = batch["mask"].to(device), batch["target"].to(device)
        with torch.set_grad_enabled(optimizer is not None):
            offsets = _forward(model, batch, device)
            loss = F.smooth_l1_loss(offsets[mask], target[mask]) + .05 * smoothness_loss(offsets, mask)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
        n = len(mask)
        ade = (torch.linalg.vector_norm(offsets.detach() - target, dim=-1) * mask).sum(1) / mask.sum(1)
        total_loss += float(loss.detach().cpu()) * n
        total_ade += float(ade.sum().cpu())
        samples += n
    return {"loss": total_loss / samples, "ade": total_ade / samples}


def train_models(train_arrays, train_context, train_base, train_indices,
                 selection_arrays, selection_context, selection_base, selection_indices,
                 metadata, output_dir, *, epochs=2, batch_size=128, hidden_dim=64,
                 device=None, seed=2027, patience=10, learning_rate=.001):
    for name, value in (("epochs", epochs), ("batch_size", batch_size), ("hidden_dim", hidden_dim), ("patience", patience)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    device = device or torch.device("cpu")
    validate_scene_metadata(metadata)
    for arrays, indices, field in ((train_arrays, train_indices, "train_scenes"),
                                  (selection_arrays, selection_indices, "selection_scenes")):
        if sorted(set(arrays["scene_name"][indices].tolist())) != metadata[field]:
            raise ValueError("partition indices do not match scene metadata")
    seed_everything(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_dataset = ContextDataset(train_arrays, train_context, train_base, train_indices)
    selection_dataset = ContextDataset(selection_arrays, selection_context, selection_base, selection_indices)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    selection_loader = DataLoader(selection_dataset, batch_size=batch_size, shuffle=False)
    fallback = ResidualCorrector(hidden_dim=hidden_dim).to(device)
    logs = []
    metadata = {**metadata, "seed": seed, "epochs_requested": epochs,
                "model_config": {"fallback_hidden_dim": hidden_dim, "hidden_dim": hidden_dim},
                "learning_rate": learning_rate, "batch_size": batch_size, "smoothness_weight": .05,
                "control": "original_residual_architecture_retrained_on_same_train_scenes"}
    for stage in ("original", "map"):
        model = fallback if stage == "original" else MapCorrector(fallback, hidden_dim).to(device)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=.0001)
        best, best_state, bad_epochs = float("inf"), None, 0
        for epoch in range(1, epochs + 1):
            training = _epoch(model, train_loader, device, optimizer)
            validation = _epoch(model, selection_loader, device)
            logs.append({"stage": stage, "epoch": epoch, "train_loss": training["loss"],
                         "selection_loss": validation["loss"], "selection_ade": validation["ade"]})
            if validation["ade"] < best:
                best, best_state, bad_epochs = validation["ade"], deepcopy(model.state_dict()), 0
                if stage == "map":
                    metadata["map_selection_ade"] = best
                    save_checkpoint(output_dir / "best.pt", model, optimizer, epoch, best, metadata=metadata)
            else:
                bad_epochs += 1
            if bad_epochs >= patience:
                break
        model.load_state_dict(best_state)
        if stage == "original":
            metadata["original_selection_ade"] = best
    (output_dir / "training.json").write_text(json.dumps({"metadata": metadata, "epochs": logs,
        "parameters": {"original": sum(p.numel() for p in fallback.parameters()),
                       "map_total": sum(p.numel() for p in model.parameters())}}, indent=2, allow_nan=False) + "\n")
    return model.eval()


def choose_device(name):
    if name == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA unavailable")
    return torch.device("cuda" if name == "cuda" or (name == "auto" and torch.cuda.is_available()) else "cpu")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train Map-only with scene-isolated control and selection")
    for name in ("trajectory-cache", "context-cache", "base-predictions", "dataroot", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", required=True, choices=("mini_train", "train"))
    for name in ("selection-trajectory-cache", "selection-context-cache", "selection-base-predictions"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--selection-split", choices=("train_val",))
    parser.add_argument("--config", type=Path, default=Path("configs/context/map.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/context_corrector.yaml"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    config = MapConfig(**load_yaml(args.config))
    arrays, context, manifest = load_bundle(args.trajectory_cache, args.context_cache, args.split, args.dataroot, config)
    base, base_hashes = load_predictions(args.base_predictions, arrays, manifest["contract"])
    if args.split == "mini_train":
        if any((args.selection_trajectory_cache, args.selection_context_cache, args.selection_base_predictions, args.selection_split)):
            raise ValueError("mini selection must be an internal scene partition of mini_train")
        train_scenes, selection_scenes = scene_partition(arrays["scene_name"].tolist())
        val_arrays, val_context, val_manifest, val_base, val_hashes = arrays, context, manifest, base, base_hashes
        train_indices = np.flatnonzero(np.isin(arrays["scene_name"], train_scenes)).tolist()
        val_indices = np.flatnonzero(np.isin(arrays["scene_name"], selection_scenes)).tolist()
    else:
        if not all((args.selection_trajectory_cache, args.selection_context_cache, args.selection_base_predictions, args.selection_split)):
            raise ValueError("train requires separate train_val selection caches and predictions")
        val_arrays, val_context, val_manifest = load_bundle(args.selection_trajectory_cache,
            args.selection_context_cache, args.selection_split, args.dataroot, config)
        val_base, val_hashes = load_predictions(args.selection_base_predictions, val_arrays, val_manifest["contract"])
        if shared_contract(manifest["contract"]) != shared_contract(val_manifest["contract"]):
            raise ValueError("train/selection context contract mismatch")
        train_scenes = sorted(set(arrays["scene_name"].tolist()))
        selection_scenes = sorted(set(val_arrays["scene_name"].tolist()))
        train_indices, val_indices = list(range(len(base))), list(range(len(val_base)))
    if base_hashes["model_identity"] != val_hashes["model_identity"]:
        raise ValueError("train/selection base model identity mismatch")
    metadata = {"shared_contract": shared_contract(manifest["contract"]),
        "base_model": base_hashes["model_identity"],
        "train_scenes": train_scenes, "selection_scenes": selection_scenes,
        "train_scenes_sha256": digest_json(train_scenes), "selection_scenes_sha256": digest_json(selection_scenes),
        "train_provenance": {"context_sha256": manifest["npz_sha256"], "contract": manifest["contract"], "base": base_hashes},
        "selection_provenance": {"context_sha256": val_manifest["npz_sha256"], "contract": val_manifest["contract"], "base": val_hashes}}
    settings = load_yaml(args.model_config)
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    train_models(arrays, context, base, train_indices, val_arrays, val_context, val_base, val_indices,
                 metadata, args.output_dir, device=choose_device(args.device), **settings)
    print(json.dumps({"train_scenes": train_scenes, "selection_scenes": selection_scenes,
                      "train_updates": len(train_indices) * 11, "selection_updates": len(val_indices) * 11}))

"""Scene-isolated training of a common original control and independent ablations."""
import argparse
import json
import math
import subprocess
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.context.maps import MapConfig
from trajectory_prediction.context.neighbor_preparation import load_interaction_bundle
from trajectory_prediction.context.neighbors import NeighborConfig
from trajectory_prediction.context.preparation import load_predictions
from trajectory_prediction.online.context_training import (
    choose_device,
    scene_partition,
    shared_contract,
)
from trajectory_prediction.online.corrector import ResidualCorrector, smoothness_loss
from trajectory_prediction.online.interaction_corrector import (
    MODES,
    InteractionCorrector,
    matched_width,
)
from trajectory_prediction.online.interaction_dataset import InteractionDataset
from trajectory_prediction.online.interaction_provenance import (
    shared_neighbor_contract,
    validate_interaction_metadata,
)
from trajectory_prediction.training.checkpoint import save_checkpoint
from trajectory_prediction.utils.seed import seed_everything

MODEL_FIELDS = ("features", "mask", "map_features", "map_mask", "neighbor_features", "neighbor_mask")


def _epoch(model, loader, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum, ade_sum, count = 0., 0., 0
    for batch in loader:
        mask, target = batch["mask"].to(device), batch["target"].to(device)
        fields = MODEL_FIELDS if isinstance(model, InteractionCorrector) else MODEL_FIELDS[:2]
        with torch.set_grad_enabled(optimizer is not None):
            offsets = model(*(batch[name].to(device) for name in fields))
            loss = F.smooth_l1_loss(offsets[mask], target[mask]) + .05 * smoothness_loss(offsets, mask)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite interaction training loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
        n = len(mask)
        ade = (torch.linalg.vector_norm(offsets.detach() - target, dim=-1) * mask).sum(1) / mask.sum(1)
        loss_sum += float(loss.detach().cpu()) * n
        ade_sum += float(ade.sum().cpu())
        count += n
    return {"loss": loss_sum / count, "ade": ade_sum / count}


def train_suite(train_bundle, train_indices, selection_bundle, selection_indices, metadata, output_dir,
                *, epochs=2, batch_size=128, hidden_dim=64, seed=2027, patience=5, learning_rate=.001, device=None):
    for name, value in (("epochs", epochs), ("batch_size", batch_size), ("hidden_dim", hidden_dim), ("patience", patience)):
        if type(value) is not int or value < 1:
            raise ValueError(f"{name} must be positive integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    validate_interaction_metadata(metadata)
    for bundle, indices, key in ((train_bundle, train_indices, "train_scenes"),
                                 (selection_bundle, selection_indices, "selection_scenes")):
        if len(set(indices)) != len(indices) or sorted(set(bundle[0]["scene_name"][indices].tolist())) != metadata[key]:
            raise ValueError("training indices must match disjoint scene metadata")
    device = device or torch.device("cpu")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    train_data = InteractionDataset(*train_bundle[:3], train_bundle[3], train_indices)
    selection_data = InteractionDataset(*selection_bundle[:3], selection_bundle[3], selection_indices)
    selection_loader = DataLoader(selection_data, batch_size=batch_size, shuffle=False)
    models, logs, selections, parameters, progress = nn.ModuleDict(), [], {}, {}, {}
    names = ("original", *MODES, *("matched_" + mode for mode in MODES))
    for name in names:
        seed_everything(seed)
        if name == "original":
            model = ResidualCorrector(hidden_dim=hidden_dim)
        elif name in MODES:
            model = InteractionCorrector(deepcopy(models["original"]), name, hidden_dim)
        else:
            parameter_count = sum(p.numel() for p in models[name.removeprefix("matched_")].parameters())
            model = ResidualCorrector(hidden_dim=matched_width(parameter_count))
        model = model.to(device)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, generator=generator)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=.0001)
        best, state, bad, best_epoch = float("inf"), None, 0, 0
        for epoch in range(1, epochs + 1):
            training = _epoch(model, loader, device, optimizer)
            selection = _epoch(model, selection_loader, device)
            logs.append({"stage": name, "epoch": epoch, "train_loss": training["loss"],
                         "selection_loss": selection["loss"], "selection_ade": selection["ade"]})
            if selection["ade"] < best:
                best, state, bad, best_epoch = selection["ade"], deepcopy(model.state_dict()), 0, epoch
            else:
                bad += 1
            if bad >= patience:
                break
        model.load_state_dict(state)
        models[name] = model.eval()
        selections[name] = best
        progress[name] = {"best_epoch": best_epoch, "completed_epochs": epoch}
        parameters[name] = {"total": sum(p.numel() for p in model.parameters()),
                            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    for mode in MODES:
        reference = parameters[mode]["total"]
        parameters["matched_" + mode]["relative_parameter_difference"] = abs(parameters["matched_" + mode]["total"] - reference) / reference
    metadata = {**metadata, "architecture": "interaction_suite_v1", "hidden_dim": hidden_dim,
        "seed": seed, "epochs_requested": epochs, "batch_size": batch_size, "patience": patience,
        "learning_rate": learning_rate, "weight_decay": .0001, "smoothness_weight": .05,
        "device": str(device), "selection_ade": selections, "parameters": parameters,
        "stage_progress": progress, "checkpoint_epoch_policy": "maximum_completed_stage_epoch_see_stage_progress",
        "resume_supported": False, "parameter_control_policy": "match_total_parameters_context_fallback_frozen_control_all_trainable"}
    save_checkpoint(output / "best.pt", models, optimizer, max(item["completed_epochs"] for item in progress.values()), selections["fusion"], metadata=metadata)
    (output / "training.json").write_text(json.dumps({"metadata": metadata, "epochs": logs}, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return models.eval()


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train scene-isolated neighbor/map ablation suite")
    for name in ("trajectory-cache", "map-cache", "neighbor-cache", "base-predictions", "dataroot", "output-dir"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", required=True, choices=("mini_train", "train"))
    for name in ("selection-trajectory-cache", "selection-map-cache", "selection-neighbor-cache", "selection-base-predictions"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/context/neighbors.yaml"))
    parser.add_argument("--map-config", type=Path, default=Path("configs/context/map.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/interaction_corrector.yaml"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    map_config, config = MapConfig(**load_yaml(args.map_config)), NeighborConfig(**load_yaml(args.config))

    def load(trajectory, maps, neighbors, split, predictions):
        arrays, map_arrays, manifest, neighbor_arrays, neighbor_manifest = load_interaction_bundle(
            trajectory, maps, neighbors, split, args.dataroot, map_config, config)
        base, hashes = load_predictions(predictions, arrays, manifest["contract"])
        return (arrays, map_arrays, neighbor_arrays, base), manifest, neighbor_manifest, hashes

    train, map_manifest, neighbor_manifest, hashes = load(args.trajectory_cache, args.map_cache,
        args.neighbor_cache, args.split, args.base_predictions)
    selection_paths = (args.selection_trajectory_cache, args.selection_map_cache,
                        args.selection_neighbor_cache, args.selection_base_predictions)
    if args.split == "mini_train":
        if any(selection_paths):
            raise ValueError("mini selection must be internal scene partition")
        train_scenes, selection_scenes = scene_partition(train[0]["scene_name"].tolist())
        selection, val_map_manifest, val_neighbor_manifest, val_hashes = train, map_manifest, neighbor_manifest, hashes
        train_indices = np.flatnonzero(np.isin(train[0]["scene_name"], train_scenes)).tolist()
        selection_indices = np.flatnonzero(np.isin(train[0]["scene_name"], selection_scenes)).tolist()
    else:
        if not all(selection_paths):
            raise ValueError("train requires all separate train_val selection artifacts")
        selection, val_map_manifest, val_neighbor_manifest, val_hashes = load(*selection_paths[:3], "train_val", selection_paths[3])
        train_scenes = sorted(set(train[0]["scene_name"].tolist()))
        selection_scenes = sorted(set(selection[0]["scene_name"].tolist()))
        train_indices, selection_indices = list(range(len(train[3]))), list(range(len(selection[3])))
    metadata = {"shared_contract": shared_contract(map_manifest["contract"]),
        "neighbor_shared_contract": shared_neighbor_contract(neighbor_manifest["contract"]),
        "base_model": hashes["model_identity"], "train_scenes": train_scenes, "selection_scenes": selection_scenes,
        "train_scenes_sha256": digest_json(train_scenes), "selection_scenes_sha256": digest_json(selection_scenes),
        "train_provenance": {"context_sha256": map_manifest["npz_sha256"], "contract": map_manifest["contract"], "base": hashes},
        "selection_provenance": {"context_sha256": val_map_manifest["npz_sha256"], "contract": val_map_manifest["contract"], "base": val_hashes},
        "train_neighbor_provenance": {"context_sha256": neighbor_manifest["npz_sha256"], "contract": neighbor_manifest["contract"]},
        "selection_neighbor_provenance": {"context_sha256": val_neighbor_manifest["npz_sha256"], "contract": val_neighbor_manifest["contract"]},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()}
    settings = load_yaml(args.model_config)
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    models = train_suite(train, train_indices, selection, selection_indices, metadata, args.output_dir,
                         device=choose_device(args.device), **settings)
    print(json.dumps({"models": list(models), "train_updates": len(train_indices) * 11,
                      "selection_updates": len(selection_indices) * 11, "train_scenes": train_scenes,
                      "selection_scenes": selection_scenes}))

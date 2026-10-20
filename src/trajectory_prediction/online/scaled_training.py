"""Bounded batch training with identity-checked completed-stage reuse."""
import argparse
import json
import math
import os
import subprocess
import tempfile
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.context.maps import MapConfig
from trajectory_prediction.context.neighbors import NeighborConfig
from trajectory_prediction.context.scaled_preparation import load_scaled_bundle
from trajectory_prediction.online.batched_context import BatchedContext, forward_inputs
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
from trajectory_prediction.online.interaction_provenance import (
    shared_neighbor_contract,
    validate_interaction_metadata,
)
from trajectory_prediction.utils.provenance import sha256_file
from trajectory_prediction.utils.seed import seed_everything


def _save_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".stage-", suffix=".pt")
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _epoch(model, data, indices, batch_size, device, optimizer=None):
    model.train(optimizer is not None)
    totals = torch.zeros(3, device=device)
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        inputs = data.inputs(selected, device, context=model.mode if isinstance(model, InteractionCorrector) else False)
        target = data.labels(selected, device)
        mask = inputs["mask"]
        with torch.set_grad_enabled(optimizer is not None):
            offsets = forward_inputs(model, inputs)
            loss = F.smooth_l1_loss(offsets[mask], target[mask]) + .05 * smoothness_loss(offsets, mask)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite scaled training loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
        ade = (torch.linalg.vector_norm(offsets.detach() - target, dim=-1) * mask).sum(1) / mask.sum(1)
        n = len(selected)
        totals[0] += loss.detach() * n
        totals[1] += ade.sum()
        totals[2] += n
    result = totals.detach().cpu().numpy()
    return {"loss": float(result[0] / result[2]), "ade": float(result[1] / result[2])}


def train_scaled_suite(train, selection, metadata, output_dir, *, epochs=20, batch_size=4096,
                       hidden_dim=64, seed=2027, patience=5, learning_rate=.001, device=None):
    for value in (epochs, batch_size, hidden_dim, patience):
        if type(value) is not int or value < 1:
            raise ValueError("training dimensions must be positive integers")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning rate must be finite and positive")
    validate_interaction_metadata(metadata)
    for data, key in ((train, "train_scenes"), (selection, "selection_scenes")):
        if sorted(set(data.arrays["scene_name"].tolist())) != metadata[key]:
            raise ValueError("scaled data must exactly match scene partitions")
    device = device or torch.device("cpu")
    settings = {"epochs": epochs, "batch_size": batch_size, "hidden_dim": hidden_dim, "seed": seed,
                "patience": patience, "learning_rate": learning_rate, "weight_decay": .0001,
                "smoothness_weight": .05, "shuffle": "numpy_PCG64_per_stage"}
    files = [Path(__file__).with_name(name + ".py") for name in
             ("scaled_training", "batched_context", "interaction_corrector", "corrector",
              "context_training", "interaction_provenance")]
    files.append(Path(__file__).parents[1] / "utils" / "seed.py")
    source = {path.name: sha256_file(path) for path in files}
    identity = digest_json({"metadata": metadata, "settings": settings, "source": source,
                            "train_updates": train.size, "selection_updates": selection.size,
                            "train_content": train.fingerprint(), "selection_content": selection.fingerprint()})
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run_path = output / "run.json"
    if run_path.exists() and json.loads(run_path.read_text())["identity"] != identity:
        raise ValueError("completed run identity mismatch; use a new output directory")
    run_path.write_text(json.dumps({"identity": identity, "settings": settings, "source": source}, indent=2) + "\n")
    models, selections, progress, parameters, all_logs = nn.ModuleDict(), {}, {}, {}, []
    for name in ("original", *MODES, *("matched_" + mode for mode in MODES)):
        seed_everything(seed)
        if name == "original":
            model = ResidualCorrector(hidden_dim=hidden_dim)
        elif name in MODES:
            model = InteractionCorrector(deepcopy(models["original"]), name, hidden_dim)
        else:
            count = sum(p.numel() for p in models[name.removeprefix("matched_")].parameters())
            model = ResidualCorrector(hidden_dim=matched_width(count))
        model = model.to(device)
        stage_path = output / "stages" / (name + ".pt")
        if stage_path.exists():
            stage = torch.load(stage_path, map_location="cpu", weights_only=True)
            if stage.get("identity") != identity or stage.get("name") != name:
                raise ValueError("completed stage identity mismatch")
            model.load_state_dict(stage["model"], strict=True)
            print(json.dumps({"seed": seed, "stage": name, "reused_completed_stage": True}), flush=True)
        else:
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=.0001)
            generator = np.random.default_rng(seed)
            best, best_state, best_epoch, bad = float("inf"), None, 0, 0
            logs = []
            for epoch in range(1, epochs + 1):
                started = time.perf_counter()
                training = _epoch(model, train, generator.permutation(train.size), batch_size, device, optimizer)
                validation = _epoch(model, selection, np.arange(selection.size), batch_size, device)
                row = {"seed": seed, "stage": name, "epoch": epoch, "train_loss": training["loss"],
                       "selection_loss": validation["loss"], "selection_ade": validation["ade"],
                       "seconds": time.perf_counter() - started}
                logs.append(row)
                print(json.dumps(row), flush=True)
                (output / (name + "-progress.json")).write_text(json.dumps({"identity": identity, "epochs": logs}, indent=2) + "\n")
                if validation["ade"] < best:
                    best, best_state, best_epoch, bad = validation["ade"], {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, epoch, 0
                else:
                    bad += 1
                if bad >= patience:
                    break
            model.load_state_dict(best_state)
            stage = {"identity": identity, "name": name, "model": best_state, "selection_ade": best,
                     "best_epoch": best_epoch, "completed_epochs": epoch, "logs": logs}
            _save_torch(stage_path, stage)
        models[name] = model.eval()
        selections[name] = stage["selection_ade"]
        progress[name] = {"best_epoch": stage["best_epoch"], "completed_epochs": stage["completed_epochs"]}
        parameters[name] = {"total": sum(p.numel() for p in model.parameters()),
                            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        all_logs.extend(stage["logs"])
    for mode in MODES:
        total = parameters[mode]["total"]
        parameters["matched_" + mode]["relative_parameter_difference"] = abs(parameters["matched_" + mode]["total"] - total) / total
    final_metadata = {**metadata, **settings, "architecture": "interaction_suite_v1", "execution": "bounded_batch_v1",
        "training_config_sha256": digest_json(settings), "source_code_sha256": source, "run_identity": identity,
        "device": str(device), "selection_ade": selections, "stage_progress": progress, "parameters": parameters,
        "resume_supported": "completed_stages_only_restart_unfinished_stage", "torch_version": str(torch.__version__),
        "parameter_control_policy": "match_total_parameters_context_fallback_frozen_control_all_trainable"}
    _save_torch(output / "best.pt", {"checkpoint_schema": 1, "model": models.state_dict(), "metadata": final_metadata})
    (output / "training.json").write_text(json.dumps({"metadata": final_metadata, "epochs": all_logs}, indent=2) + "\n")
    return models.eval()


def _subset(bundle, indices):
    return tuple({name: value[indices] for name, value in part.items()} if isinstance(part, dict) else part[indices] for part in bundle)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Train bounded-memory whole-split ablations")
    for name in ("trajectory-root", "context-root", "prediction-root", "dataroot", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029])
    parser.add_argument("--mini", action="store_true")
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/trainval_interaction.yaml"))
    parser.add_argument("--map-config", type=Path, default=Path("configs/context/map.yaml"))
    parser.add_argument("--config", type=Path, default=Path("configs/context/neighbors.yaml"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    map_config, config = MapConfig(**load_yaml(args.map_config)), NeighborConfig(**load_yaml(args.config))
    splits = ("mini_train", "mini_train") if args.mini else ("train", "train_val")
    prediction_names = ("gru_train_eval", "gru_train_eval") if args.mini else ("gru_train_eval", "gru_train_val_eval")
    loaded = [load_scaled_bundle(args.trajectory_root / split, args.context_root / split, split, args.dataroot,
        map_config, config, args.prediction_root / name / "predictions.npz") for split, name in zip(splits, prediction_names, strict=True)]
    train, map_manifest, neighbor_manifest, hashes = loaded[0]
    selection, val_map, val_neighbors, val_hashes = loaded[1]
    if args.mini:
        train_scenes, selection_scenes = scene_partition(train[0]["scene_name"].tolist())
        all_data = train
        train = _subset(all_data, np.flatnonzero(np.isin(all_data[0]["scene_name"], train_scenes)))
        selection = _subset(all_data, np.flatnonzero(np.isin(all_data[0]["scene_name"], selection_scenes)))
    else:
        train_scenes = sorted(set(train[0]["scene_name"].tolist()))
        selection_scenes = sorted(set(selection[0]["scene_name"].tolist()))
    metadata = {"shared_contract": shared_contract(map_manifest["contract"]),
        "neighbor_shared_contract": shared_neighbor_contract(neighbor_manifest["contract"]), "base_model": hashes["model_identity"],
        "train_scenes": train_scenes, "selection_scenes": selection_scenes,
        "train_scenes_sha256": digest_json(train_scenes), "selection_scenes_sha256": digest_json(selection_scenes),
        "train_provenance": {"context_sha256": map_manifest["npz_sha256"], "contract": map_manifest["contract"], "base": hashes},
        "selection_provenance": {"context_sha256": val_map["npz_sha256"], "contract": val_map["contract"], "base": val_hashes},
        "train_neighbor_provenance": {"context_sha256": neighbor_manifest["npz_sha256"], "contract": neighbor_manifest["contract"]},
        "selection_neighbor_provenance": {"context_sha256": val_neighbors["npz_sha256"], "contract": val_neighbors["contract"]},
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()}
    settings = load_yaml(args.model_config)
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    device = choose_device(args.device)
    for seed in args.seeds:
        train_scaled_suite(BatchedContext(*train), BatchedContext(*selection), metadata,
                           args.output_root / ("seed-" + str(seed)), device=device, seed=seed, **settings)

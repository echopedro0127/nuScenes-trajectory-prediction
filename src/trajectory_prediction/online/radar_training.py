"""Scene-isolated five-condition mini Radar training with audited checkpoints."""
import argparse
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.context.radar import RadarConfig
from trajectory_prediction.context.radar_preparation import load_radar_bundle
from trajectory_prediction.online.context_training import choose_device, scene_partition
from trajectory_prediction.online.corrector import ResidualCorrector, smoothness_loss
from trajectory_prediction.online.interaction_corrector import matched_width
from trajectory_prediction.online.radar_corrector import RadarDataset, clone_fallback, forward_model
from trajectory_prediction.online.scaled_training import _save_torch
from trajectory_prediction.utils.provenance import sha256_file
from trajectory_prediction.utils.seed import seed_everything

NAMES = ("original", "radar", "matched_radar")
SOURCES = ("online/radar_corrector.py", "online/radar_training.py", "online/observation_noise.py",
           "online/batched_context.py", "online/corrector.py", "online/interaction_corrector.py",
           "context/radar.py", "context/radar_index.py", "context/radar_preparation.py", "utils/seed.py")


def source_hashes():
    root = Path(__file__).parents[1]
    return {name: sha256_file(root / name) for name in SOURCES}


def run_epoch(model, data, indices, batch_size, device, optimizer=None):
    model.train(optimizer is not None)
    loss_sum, ade_sum, count = 0., 0., 0
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        inputs, target = data.inputs(selected, device), data.labels(selected, device)
        with torch.set_grad_enabled(optimizer is not None):
            offsets = forward_model(model, inputs)
            mask = inputs["mask"]
            loss = F.smooth_l1_loss(offsets[mask], target[mask]) + .05 * smoothness_loss(offsets, mask)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite radar training loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
        ade = (torch.linalg.vector_norm(offsets.detach() - target, dim=-1) * mask).sum(1) / mask.sum(1)
        loss_sum += float(loss.detach()) * len(selected)
        ade_sum += float(ade.sum())
        count += len(selected)
    return {"loss": loss_sum / count, "ade": ade_sum / count}


def train_radar_suite(train, selection, metadata, output_dir, *, epochs=20, batch_size=1024,
                      hidden_dim=64, seed=2027, patience=5, learning_rate=.001, device=None):
    for value in (epochs, batch_size, hidden_dim, patience):
        if type(value) is not int or value < 1:
            raise ValueError("positive training settings required")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("positive learning rate required")
    if set(train.scenes) & set(selection.scenes):
        raise ValueError("radar train/selection scene overlap")
    if train.fingerprint() == selection.fingerprint():
        raise ValueError("training and selection data must differ")
    device = device or torch.device("cpu")
    settings = {"epochs": epochs, "batch_size": batch_size, "hidden_dim": hidden_dim, "seed": seed,
        "patience": patience, "learning_rate": learning_rate, "weight_decay": .0001, "smoothness_weight": .05}
    metadata = {**metadata, "train_scenes": train.scenes, "selection_scenes": selection.scenes,
        "train_updates": train.size, "selection_updates": selection.size, "source_code_sha256": source_hashes(),
        "train_fingerprint": train.fingerprint(), "selection_fingerprint": selection.fingerprint(), **settings}
    identity = digest_json(metadata)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    path = output / "run.json"
    if path.exists() and json.loads(path.read_text())["identity"] != identity:
        raise ValueError("radar run identity mismatch")
    path.write_text(json.dumps({"identity": identity, "settings": settings}, indent=2) + "\n")
    models, progress, logs, parameters = nn.ModuleDict(), {}, [], {}
    for name in NAMES:
        seed_everything(seed)
        if name == "original":
            model = ResidualCorrector(hidden_dim=hidden_dim)
        elif name == "radar":
            model = clone_fallback(models["original"], hidden_dim)
        else:
            model = ResidualCorrector(hidden_dim=matched_width(parameters["radar"]["total"]))
        model = model.to(device)
        stage_path = output / (name + ".pt")
        if stage_path.exists():
            stage = torch.load(stage_path, map_location="cpu", weights_only=True)
            if stage.get("identity") != identity or stage.get("name") != name:
                raise ValueError("radar stage identity mismatch")
            model.load_state_dict(stage["model"], strict=True)
        else:
            optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate, weight_decay=.0001)
            generator = np.random.default_rng(seed)
            best, best_state, best_epoch, bad, stage_logs = float("inf"), None, 0, 0, []
            for epoch in range(1, epochs + 1):
                started = time.perf_counter()
                training = run_epoch(model, train, generator.permutation(train.size), batch_size, device, optimizer)
                selected = run_epoch(model, selection, np.arange(selection.size), batch_size, device)
                row = {"seed": seed, "stage": name, "epoch": epoch, "train_loss": training["loss"],
                    "selection_loss": selected["loss"], "selection_ade": selected["ade"], "seconds": time.perf_counter() - started}
                stage_logs.append(row)
                print(json.dumps(row), flush=True)
                if selected["ade"] < best:
                    best, best_epoch, bad = selected["ade"], epoch, 0
                    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                else:
                    bad += 1
                if bad >= patience:
                    break
            model.load_state_dict(best_state)
            stage = {"identity": identity, "name": name, "model": best_state, "best_epoch": best_epoch,
                "completed_epochs": epoch, "selection_ade": best, "logs": stage_logs}
            _save_torch(stage_path, stage)
        models[name] = model.eval()
        progress[name] = {field: stage[field] for field in ("best_epoch", "completed_epochs", "selection_ade")}
        logs.extend(stage["logs"])
        parameters[name] = {"total": sum(p.numel() for p in model.parameters()), "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    final = {**metadata, "identity": identity, "architecture": "radar_suite_v1", "stage_progress": progress,
        "parameters": parameters, "device": str(device), "torch_version": str(torch.__version__),
        "parameter_control": "total_parameter_matched_frozen_pretrained_fallback_vs_all_trainable"}
    _save_torch(output / "best.pt", {"schema_version": 1, "model": models.state_dict(), "metadata": final})
    (output / "training.json").write_text(json.dumps({"metadata": final, "epochs": logs}, indent=2) + "\n")
    return models.eval()


def shared_contract(contract):
    return {name: contract[name] for name in ("config", "conditions", "perturbation_seed", "dataset_version", "filters", "nuscenes_version", "time_policy", "coordinate_policy", "motion_compensation", "oracle")}


def load_radar_suite(path, evaluation_contract, evaluation_scenes):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload["metadata"]
    if payload.get("schema_version") != 1 or metadata.get("architecture") != "radar_suite_v1":
        raise ValueError("unsupported radar checkpoint")
    contract = metadata["context_contract"]
    if contract["split"] != "mini_train" or evaluation_contract["split"] != "mini_val" or shared_contract(contract) != shared_contract(evaluation_contract) or contract["base"]["model_identity"] != evaluation_contract["base"]["model_identity"]:
        raise ValueError("radar checkpoint source/base/config mismatch")
    for name, digest in contract["source"].items():
        if name.startswith(contract["dataset_version"] + "/") and evaluation_contract["source"].get(name) != digest:
            raise ValueError("radar metadata source mismatch")
    if not metadata["train_scenes"] or not metadata["selection_scenes"] or set(metadata["train_scenes"]) & set(metadata["selection_scenes"]) or set(evaluation_scenes) & set(metadata["train_scenes"] + metadata["selection_scenes"]):
        raise ValueError("radar evaluation scene overlap")
    if metadata["source_code_sha256"] != source_hashes():
        raise ValueError("radar executable source identity mismatch")
    hidden = metadata["hidden_dim"]
    models = nn.ModuleDict({"original": ResidualCorrector(hidden_dim=hidden)})
    models["radar"] = clone_fallback(models["original"], hidden)
    count = sum(p.numel() for p in models["radar"].parameters())
    models["matched_radar"] = ResidualCorrector(hidden_dim=matched_width(count))
    models.load_state_dict(payload["model"], strict=True)
    if any(not torch.equal(value, models["radar"].fallback.state_dict()[key]) for key, value in models["original"].state_dict().items()):
        raise ValueError("radar fallback differs from original")
    return models.eval(), metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("trajectory-root", "prediction-root", "dataroot", "context-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/context/radar.yaml"))
    parser.add_argument("--model-config", type=Path, default=Path("configs/models/radar_corrector.yaml"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2027, 2028, 2029])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    arrays, cache, base, manifest = load_radar_bundle(args.trajectory_root / "mini_train", args.context_root / "mini_train",
        "mini_train", args.prediction_root / "gru_train_eval/predictions.npz", args.dataroot, RadarConfig(**load_yaml(args.config)))
    train_scenes, selection_scenes = scene_partition(arrays["scene_name"].tolist())
    train = RadarDataset(arrays, cache, base, np.flatnonzero(np.isin(arrays["scene_name"], train_scenes)))
    selection = RadarDataset(arrays, cache, base, np.flatnonzero(np.isin(arrays["scene_name"], selection_scenes)))
    metadata = {"context_contract": manifest["contract"], "cache_sha256": manifest["npz_sha256"],
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()}
    settings = load_yaml(args.model_config)
    if args.epochs is not None:
        settings["epochs"] = args.epochs
    for seed in args.seeds:
        train_radar_suite(train, selection, metadata, args.output_root / ("seed-" + str(seed)), seed=seed, device=choose_device(args.device), **settings)

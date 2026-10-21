"""P6.3 identity-bound eight-factor training; no validation-set access."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.context.gate_preparation import load_verified_gate_cache
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.corrector import smoothness_loss
from trajectory_prediction.online.horizon_features import HorizonDataset
from trajectory_prediction.online.horizon_gate import (
    VARIANTS,
    HorizonGate,
    blend_horizon,
    feature_mask,
    measure_horizon,
)
from trajectory_prediction.online.quality_gate import QualityGate, blend_forecasts
from trajectory_prediction.online.refinement_training import (
    completed_payload,
    load_initial,
    measure_model,
    state_fingerprint,
)
from trajectory_prediction.online.refinement_training import sources as parent_sources
from trajectory_prediction.online.scaled_training import _save_torch
from trajectory_prediction.utils.provenance import sha256_file
from trajectory_prediction.utils.seed import seed_everything

DEFAULT_SETTINGS = {"epochs": 10, "patience": 3, "batch_size": 4096, "learning_rate": .001,
    "weight_decay": .0001, "smoothness_weight": .05, "hidden_dim": 16, "input_dim": 19,
    "revision_weight": 0., "candidate": "map"}
CONFIG = Path(__file__).parents[3] / "configs/models/horizon_gate.yaml"


def configure_runtime():
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current not in (None, ":4096:8") or (current is None and torch.cuda.is_initialized()):
        raise ValueError("CuBLAS requires frozen :4096:8 configuration before CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def training_sources():
    package = Path(__file__).parents[1]
    return {"execution": {name: sha256_file(package/name) for name in (
        "online/horizon_features.py", "online/horizon_gate.py", "online/horizon_training.py")},
        "config": sha256_file(CONFIG), "parents": parent_sources()}


def validate_role(model, meta, variant, expected):
    if (variant not in VARIANTS or meta.get("variant") != variant or model.variant != variant
        or meta.get("parameters") != 337 or any(meta.get(k) != v for k, v in expected.items())
        or not 1 <= meta.get("best_epoch", 0) <= meta.get("completed_epochs", 0) <= expected["epochs"]
        or not torch.equal(model.feature_mask.cpu(), feature_mask(variant))):
        raise ValueError("horizon model role/settings mismatch")


def run_horizon_epoch(model, data, indices, batch_size, device, optimizer, revision_weight):
    if revision_weight != 0:
        raise ValueError("first-round horizon training excludes revision loss")
    model.train()
    totals = np.zeros(2, np.float64)
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start+batch_size]
        values = data.inputs(selected, device)
        target = data.labels(selected, device)
        weights = model(values["features"], values["valid"], values["mask"])
        forecast = blend_horizon(values["cv"], values["map"], weights)
        correction = weights[..., None]*(values["map"]-values["cv"])
        loss = F.smooth_l1_loss(forecast[values["mask"]], target[values["mask"]]) + .05*smoothness_loss(correction, values["mask"])
        if not torch.isfinite(loss):
            raise ValueError("nonfinite horizon training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
        totals += np.array([float(loss.detach()), 1])*len(selected)
    return {"loss": float(totals[0]/totals[1]), "supervised_loss": float(totals[0]/totals[1]), "revision_loss": 0.}


def train_horizon(
    train,
    selection,
    metadata,
    output_dir,
    *,
    initial,
    variant,
    epochs=10,
    batch_size=4096,
    patience=3,
    seed=2027,
    revision_weight=0.0,
    learning_rate=0.001,
    device=None,
):
    if set(train.scenes) & set(selection.scenes):
        raise ValueError("refinement scene overlap")
    if train.candidate != selection.candidate:
        raise ValueError("refinement candidate mismatch")
    if (
        any(type(v) is not int or v < 1 for v in (epochs, batch_size, patience))
        or any(not math.isfinite(v) for v in (revision_weight, learning_rate))
        or revision_weight != 0
        or learning_rate <= 0
    ):
        raise ValueError("invalid refinement settings")
    device = device or torch.device("cpu")
    if device.type == "cuda" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise ValueError("CuBLAS must be configured before horizon training")
    seed_everything(seed)
    model = HorizonGate(variant, initial).to(device)
    initial_identity = state_fingerprint(model.state_dict())
    settings = {
        "epochs": epochs,
        "batch_size": batch_size,
        "patience": patience,
        "seed": seed,
        "revision_weight": revision_weight,
        "learning_rate": learning_rate,
        "weight_decay": 0.0001,
        "smoothness_weight": 0.05,
        "hidden_dim": 16,
        "candidate": train.candidate,
        "input_dim": 19,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG", "not_set"),
        "variant": variant,
    }
    executable = training_sources()
    identity = digest_json(
        {
            "metadata": metadata,
            "settings": settings,
            "sources": executable,
            "initial": initial_identity,
            "train_content": train.fingerprint(),
            "selection_content": selection.fingerprint(),
        }
    )
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    run = root / "run.json"
    if run.exists() and json.loads(run.read_text())["identity"] != identity:
        raise ValueError("refinement run identity mismatch")
    run.write_text(
        json.dumps({"identity": identity, "settings": settings, "sources": executable}, indent=2)
        + "\n"
    )
    if (root / "completed.json").exists():
        payload = completed_payload(root)
        if (
            payload.get("checkpoint_schema") != 1
            or payload["metadata"].get("run_identity") != identity
        ):
            raise ValueError("refinement checkpoint identity mismatch")
        model.load_state_dict(payload["model"], strict=True)
        validate_role(model, payload["metadata"], variant, settings)
        return model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    generator = np.random.default_rng(seed)
    best, bad, best_state, logs = float("inf"), 0, None, []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        training = run_horizon_epoch(
            model,
            train,
            generator.permutation(train.size),
            batch_size,
            device,
            optimizer,
            revision_weight,
        )
        values, _ = measure_horizon(selection, model, device=device, batch_size=batch_size)
        ade = float(values[..., 0].mean())
        row = {
            "seed": seed,
            "candidate": train.candidate,
        "input_dim": 19,
        "variant": variant,
            "lambda": revision_weight,
            "epoch": epoch,
            **training,
            "selection_ade": ade,
            "selection_fde": float(values[..., 1].mean()),
            "selection_jitter": float(values[..., 2].mean()),
            "seconds": time.perf_counter() - started,
        }
        logs.append(row)
        print(json.dumps(row), flush=True)
        (root / "progress.json").write_text(json.dumps(logs, indent=2) + "\n")
        if ade < best:
            best, bad, best_epoch = ade, 0, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    if training_sources() != executable:
        raise ValueError("refinement executable changed during training")
    model.load_state_dict(best_state)
    final = {
        **metadata,
        **settings,
        "architecture": "horizon_gate_v1",
        "run_identity": identity,
        "source_code_sha256": executable,
        "initial_state_sha256": initial_identity,
        "train_content": train.fingerprint(),
        "selection_content": selection.fingerprint(),
        "initial_p6_state_sha256": state_fingerprint(initial),
        "train_scenes": train.scenes,
        "selection_scenes": selection.scenes,
        "best_epoch": best_epoch,
        "completed_epochs": epoch,
        "selection_ade": best,
        "parameters": sum(p.numel() for p in model.parameters()),
        "device": str(device),
        "torch_version": str(torch.__version__),
    }
    _save_torch(root / "best.pt", {"checkpoint_schema": 1, "model": best_state, "metadata": final})
    (root / "training.json").write_text(
        json.dumps({"metadata": final, "epochs": logs}, indent=2) + "\n"
    )
    marker = {
        "identity": identity,
        "checkpoint_sha256": sha256_file(root / "best.pt"),
        "training_sha256": sha256_file(root / "training.json"),
    }
    temporary = root / ".completed.tmp"
    temporary.write_text(json.dumps(marker, indent=2) + "\n")
    temporary.replace(root / "completed.json")
    return model.eval()


def load_horizon(path, contract, scenes, device, *, split, variant):
    path = Path(path)
    payload = completed_payload(path.parent)
    m = payload["metadata"]
    if (m.get("architecture") != "horizon_gate_v1" or m.get("source_code_sha256") != training_sources()
        or m.get("seed") != contract["seed"]
        or m.get("cublas_workspace_config") != ":4096:8"):
        raise ValueError("horizon checkpoint architecture/source/seed mismatch")
    for parent in m["candidate_contracts"].values():
        for key in ("seed", "gate_config", "conditions", "preparation_sources", "p4_training_identity"):
            if parent[key] != contract[key]:
                raise ValueError("horizon frozen contract mismatch")
        if (parent["references"]["checkpoint"] != contract["references"]["checkpoint"]
            or parent["base_hashes"]["model_identity"] != contract["base_hashes"]["model_identity"]):
            raise ValueError("horizon frozen candidate mismatch")
    for reference in m["candidate_manifests"].values():
        if sha256_file(Path(reference["path"])) != reference["sha256"]:
            raise ValueError("horizon training cache manifest changed")
    if split in ("train", "train_val"):
        if (m["candidate_contracts"][split] != contract
            or sorted(scenes) != m["train_scenes" if split == "train" else "selection_scenes"]):
            raise ValueError("horizon selection/train source mismatch")
    elif split != "val" or set(scenes) & set(m["train_scenes"]+m["selection_scenes"]):
        raise ValueError("horizon evaluation scene overlap/split")
    p6 = m["p6_checkpoint"]
    if sha256_file(Path(p6["path"])) != p6["sha256"]:
        raise ValueError("horizon initializer changed")
    initial = load_initial(p6["path"], m["candidate_contracts"])
    model = HorizonGate(variant, initial)
    if (state_fingerprint(initial) != m["initial_p6_state_sha256"]
        or state_fingerprint(model.state_dict()) != m["initial_state_sha256"]):
        raise ValueError("horizon initial state mismatch")
    model.load_state_dict(payload["model"], strict=True)
    validate_role(model, m, variant, DEFAULT_SETTINGS)
    return model.to(device).eval(), m


def check_initialization(data, initial, device):
    parent = QualityGate().to(device).eval()
    parent.load_state_dict(initial)
    baseline, scalar = measure_model(data, parent, device=device)
    mask = np.arange(12)[None] >= np.arange(1, 12)[:, None]
    expected = scalar[..., None]*mask[None, None]
    results = {}
    for variant in VARIANTS:
        model = HorizonGate(variant, initial).to(device).eval()
        values, weights = measure_horizon(data, model, device=device)
        np.testing.assert_allclose(values, baseline, atol=1e-6, rtol=1e-6)
        np.testing.assert_allclose(weights, expected, atol=1e-7, rtol=1e-6)
        results[variant] = {"update_count": data.size,
            "max_metric_difference": float(np.abs(values-baseline).max()),
            "max_weight_difference": float(np.abs(weights-expected).max())}
    print(json.dumps({"full_initial_parity": results}), flush=True)
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache-root", "p6-training-root", "output-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    configure_runtime()
    torch.set_num_threads(2)
    settings = load_yaml(CONFIG)
    if settings != {k: DEFAULT_SETTINGS[k] for k in settings}:
        raise ValueError("horizon config differs from frozen protocol")
    device = choose_device(args.device)
    for seed in (2027, 2028, 2029):
        print(json.dumps({"loading_horizon_training_seed": seed}), flush=True)
        loaded = {s: load_verified_gate_cache(args.cache_root/s/f"seed-{seed}", s, seed, device=device) for s in ("train", "train_val")}
        contracts = {s: value[1]["contract"] for s, value in loaded.items()}
        p6 = args.p6_training_root/f"seed-{seed}"/"best.pt"
        initial = load_initial(p6, contracts)
        metadata = {"candidate_contracts": contracts,
            "candidate_manifests": {s: {"path": str((args.cache_root/s/f"seed-{seed}"/"manifest.json").resolve()),
                "sha256": sha256_file(args.cache_root/s/f"seed-{seed}"/"manifest.json")} for s in loaded},
            "p6_checkpoint": {"path": str(p6.resolve()), "sha256": sha256_file(p6)}}
        train, selection = (HorizonDataset(loaded[s][0]) for s in ("train", "train_val"))
        full_parity = check_initialization(selection, initial, device)
        parent = QualityGate().to(device).eval()
        parent.load_state_dict(initial)
        keys = np.linspace(0, selection.size-1, 256, dtype=np.int64)
        values = selection.inputs(keys, device)
        for variant in VARIANTS:
            probe = HorizonGate(variant, initial).to(device).eval()
            with torch.inference_mode():
                old = blend_forecasts(values["cv"], values["map"], parent(values["quality"], values["valid"]))
                new = blend_horizon(values["cv"], values["map"], probe(values["features"], values["valid"], values["mask"]))
                torch.testing.assert_close(new[values["mask"]], old[values["mask"]], atol=1e-5, rtol=1e-6)
                initial_difference = float((new[values["mask"]]-old[values["mask"]]).abs().max())
            train_horizon(train, selection, {**metadata, "initial_probe_max_difference": initial_difference, "initial_full_metrics": full_parity[variant]},
                args.output_root/f"seed-{seed}"/variant, initial=initial, variant=variant, seed=seed, device=device, **settings)


if __name__ == "__main__":
    main()

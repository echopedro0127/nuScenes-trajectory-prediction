"""Identity-bound P6.1 calibration, candidate attribution, and paired gate training."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.context.gate_preparation import load_verified_gate_cache
from trajectory_prediction.online.batched_evaluation import forecast_metrics
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.corrector import smoothness_loss
from trajectory_prediction.online.gate_refinement import RefinementDataset, paired_revision
from trajectory_prediction.online.gate_training import training_sources as p6_sources
from trajectory_prediction.online.quality_gate import QualityGate, blend_forecasts
from trajectory_prediction.online.scaled_training import _save_torch
from trajectory_prediction.utils.provenance import sha256_file
from trajectory_prediction.utils.seed import seed_everything

MODEL_NAMES = ("original_gate", "finetune_0", "consistent_002", "consistent_010")


def sources():
    package = Path(__file__).parents[1]
    names = (
        "online/gate_refinement.py",
        "online/refinement_training.py",
        "online/batched_evaluation.py",
    )
    return {**{name: sha256_file(package / name) for name in names}, "p6": p6_sources()}


def state_fingerprint(state):
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        a = value.detach().cpu().contiguous().numpy()
        h.update(str((name, a.shape, a.dtype.str)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def fit_calibration(data, *, batch_size=4096, device=None):
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("positive calibration batch required")
    device = device or torch.device("cpu")
    grid = np.linspace(0, 1, 21)
    weights = torch.tensor(grid, dtype=torch.float32, device=device)[:, None, None, None]
    sums = np.zeros((55, 21), np.float64)
    counts = np.zeros(55, np.int64)
    for start in range(0, data.size, batch_size):
        indices = np.arange(start, min(start + batch_size, data.size))
        inputs = data.inputs(indices, device)
        target = data.labels(indices, device)
        _, _, k = data.indices(indices)
        valid = inputs["valid"].cpu().numpy()
        cells = k * 5 + np.clip(np.rint(inputs["age_s"].cpu().numpy() * 2).astype(np.int64), 0, 4)
        with torch.inference_mode():
            forecast = inputs["cv"][None] + weights * (inputs["map"] - inputs["cv"])[None]
            error = torch.linalg.vector_norm(forecast - target[None], dim=-1)
            ade = (error * inputs["mask"][None]).double().sum(-1) / inputs["mask"].sum(-1)[None]
            values = ade.cpu().numpy()
        for j in range(21):
            sums[:, j] += np.bincount(cells[valid], weights=values[j, valid], minlength=55)
        counts += np.bincount(cells[valid], minlength=55)
    chosen = grid[np.argmin(sums, axis=1)]
    chosen[counts == 0] = 0
    return {
        "constant": float(grid[np.argmin(sums.sum(0))]),
        "lookup": chosen.reshape(11, 5).tolist(),
        "grid": grid.tolist(),
        "cell_counts": counts.reshape(11, 5).tolist(),
        "cell_ade_sums": sums.tolist(),
        "valid_update_count": int(counts.sum()),
        "candidate": data.candidate,
    }


def measure_model(data, model, *, device=None, batch_size=4096):
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    metrics = np.empty((6, data.n, 11, 3), np.float64)
    weights = np.empty((6, data.n, 11), np.float32)
    for c in range(6):
        forecast = np.empty((data.n, 11, 12, 2), np.float32)
        for start in range(0, data.n * 11, batch_size):
            local = np.arange(start, min(start + batch_size, data.n * 11))
            indices = c * data.n * 11 + local
            inputs = data.inputs(indices, device)
            with torch.inference_mode():
                weight = model(inputs["quality"], inputs["valid"])
                value = blend_forecasts(inputs["cv"], inputs["map"], weight).cpu().numpy()
            i, k = local // 11, local % 11
            forecast[i, k] = value
            weights[c, i, k] = weight.cpu().numpy()
        metrics[c] = forecast_metrics(forecast, data.arrays["target"])
    return metrics, weights


def run_epoch(model, data, indices, batch_size, device, optimizer, revision_weight):
    model.train()
    totals = np.zeros(4, np.float64)
    for start in range(0, len(indices), batch_size):
        selected = indices[start : start + batch_size]
        inputs = data.inputs(selected, device)
        target = data.labels(selected, device)
        weight = model(inputs["quality"], inputs["valid"])
        forecast = blend_forecasts(inputs["cv"], inputs["map"], weight)
        correction = weight[:, None, None] * (inputs["map"] - inputs["cv"])
        supervised = F.smooth_l1_loss(
            forecast[inputs["mask"]], target[inputs["mask"]]
        ) + 0.05 * smoothness_loss(correction, inputs["mask"])
        revision = forecast.sum() * 0.0
        if revision_weight > 0:
            previous = data.inputs(data.previous_indices(selected), device)
            past_weight = model(previous["quality"], previous["valid"])
            past = blend_forecasts(previous["cv"], previous["map"], past_weight)
            _, _, k = data.indices(selected)
            revision = paired_revision(
                forecast, past, inputs["mask"], torch.from_numpy(k > 0).to(device)
            )
        loss = supervised + revision_weight * revision
        if not torch.isfinite(loss):
            raise ValueError("nonfinite paired training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        totals += np.array(
            [float(loss.detach()), float(supervised.detach()), float(revision.detach()), 1]
        ) * len(selected)
    return {
        k: float(totals[j] / totals[3])
        for j, k in enumerate(("loss", "supervised_loss", "revision_loss"))
    }


def completed_payload(root):
    root = Path(root)
    marker = json.loads((root / "completed.json").read_text())
    for name, field in (("best.pt", "checkpoint_sha256"), ("training.json", "training_sha256")):
        if not (root / name).is_file() or sha256_file(root / name) != marker[field]:
            raise ValueError("refinement completed artifact mismatch")
    payload = torch.load(root / "best.pt", map_location="cpu", weights_only=True)
    record = json.loads((root / "training.json").read_text())
    if (
        payload.get("checkpoint_schema") != 1
        or payload["metadata"].get("run_identity") != marker["identity"]
        or record["metadata"] != payload["metadata"]
        or len(record["epochs"]) != payload["metadata"]["completed_epochs"]
    ):
        raise ValueError("refinement completed log identity mismatch")
    return payload


def train_refinement(
    train,
    selection,
    metadata,
    output_dir,
    *,
    initial=None,
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
        or revision_weight < 0
        or learning_rate <= 0
    ):
        raise ValueError("invalid refinement settings")
    device = device or torch.device("cpu")
    seed_everything(seed)
    model = QualityGate().to(device)
    if initial is not None:
        model.load_state_dict(initial, strict=True)
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
    }
    executable = sources()
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
        return model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    generator = np.random.default_rng(seed)
    best, bad, best_state, logs = float("inf"), 0, None, []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        training = run_epoch(
            model,
            train,
            generator.permutation(train.size),
            batch_size,
            device,
            optimizer,
            revision_weight,
        )
        values, _ = measure_model(selection, model, device=device, batch_size=batch_size)
        ade = float(values[..., 0].mean())
        row = {
            "seed": seed,
            "candidate": train.candidate,
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
    if sources() != executable:
        raise ValueError("refinement executable changed during training")
    model.load_state_dict(best_state)
    final = {
        **metadata,
        **settings,
        "architecture": "gate_refinement_v1",
        "run_identity": identity,
        "source_code_sha256": executable,
        "initial_state_sha256": initial_identity,
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


def load_initial(path, contracts):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    m = payload["metadata"]
    if (
        payload.get("checkpoint_schema") != 1
        or m.get("architecture") != "quality_gate_v1"
        or m.get("source_code_sha256") != p6_sources()
        or m.get("candidate_contracts") != contracts
    ):
        raise ValueError("P6 initializer source mismatch")
    if any(contract["seed"] != m["seed"] for contract in contracts.values()):
        raise ValueError("P6 initializer seed mismatch")
    return payload["model"]


def load_refinement(path, contract, scenes, device, *, selection=False):
    payload = completed_payload(Path(path).parent)
    m = payload["metadata"]
    if (
        payload.get("checkpoint_schema") != 1
        or m.get("architecture") != "gate_refinement_v1"
        or m.get("source_code_sha256") != sources()
        or m.get("seed") != contract["seed"]
    ):
        raise ValueError("refinement checkpoint architecture/source/seed mismatch")
    for parent in m["candidate_contracts"].values():
        for name in (
            "seed",
            "gate_config",
            "conditions",
            "preparation_sources",
            "p4_training_identity",
        ):
            if parent[name] != contract[name]:
                raise ValueError("refinement checkpoint contract mismatch")
        if (
            parent["references"]["checkpoint"] != contract["references"]["checkpoint"]
            or parent["base_hashes"]["model_identity"] != contract["base_hashes"]["model_identity"]
        ):
            raise ValueError("refinement frozen candidate source mismatch")
    if selection:
        if (
            m["candidate_contracts"]["train_val"] != contract
            or sorted(scenes) != m["selection_scenes"]
        ):
            raise ValueError("refinement selection source mismatch")
    elif set(scenes) & set(m["train_scenes"] + m["selection_scenes"]):
        raise ValueError("refinement evaluation scene overlap")
    initial = m["p6_checkpoint"]
    if sha256_file(Path(initial["path"])) != initial["sha256"]:
        raise ValueError("refinement initializer changed")
    model = QualityGate().to(device)
    model.load_state_dict(payload["model"], strict=True)
    return model.eval(), m


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cache-root", "p6-training-root", "output-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2027, 2028, 2029])
    parser.add_argument("--config", type=Path, default=Path("configs/models/gate_refinement.yaml"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    torch.set_num_threads(2)
    device = choose_device(args.device)
    settings = load_yaml(args.config)
    for seed in args.seeds:
        loaded = {
            s: load_verified_gate_cache(
                args.cache_root / s / f"seed-{seed}", s, seed, device=device
            )
            for s in ("train", "train_val")
        }
        contracts = {s: v[1]["contract"] for s, v in loaded.items()}
        p6 = args.p6_training_root / f"seed-{seed}" / "best.pt"
        initial = load_initial(p6, contracts)
        metadata = {
            "candidate_contracts": contracts,
            "candidate_manifests": {
                s: {
                    "path": str((args.cache_root / s / f"seed-{seed}" / "manifest.json").resolve()),
                    "sha256": sha256_file(args.cache_root / s / f"seed-{seed}" / "manifest.json"),
                }
                for s in loaded
            },
            "p6_checkpoint": {"path": str(p6.resolve()), "sha256": sha256_file(p6)},
        }
        root = args.output_root / f"seed-{seed}"
        root.mkdir(parents=True, exist_ok=True)
        calibration = {
            "metadata": metadata,
            "source_code_sha256": sources(),
            "train": fit_calibration(RefinementDataset(loaded["train"][0]), device=device),
            "train_val": fit_calibration(RefinementDataset(loaded["train_val"][0]), device=device),
        }
        (root / "calibration.json").write_text(json.dumps(calibration, indent=2) + "\n")
        for name in MODEL_NAMES:
            original = name == "original_gate"
            candidate = "original" if original else "map"
            coefficient = {"consistent_002": 0.02, "consistent_010": 0.1}.get(name, 0.0)
            train_refinement(
                *(
                    RefinementDataset(loaded[s][0], candidate=candidate)
                    for s in ("train", "train_val")
                ),
                metadata,
                root / name,
                initial=None if original else initial,
                seed=seed,
                device=device,
                epochs=settings["original_epochs"] if original else settings["finetune_epochs"],
                patience=settings["original_patience"]
                if original
                else settings["finetune_patience"],
                batch_size=settings["batch_size"],
                learning_rate=settings["learning_rate"],
                revision_weight=coefficient,
            )


if __name__ == "__main__":
    main()

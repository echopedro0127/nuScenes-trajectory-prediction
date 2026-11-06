"""P6.6B identical training trajectories with precision-constrained epoch selection."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from trajectory_prediction.analysis.precision_selection import load_parent
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.context.gate_preparation import load_verified_gate_cache
from trajectory_prediction.online.consistency_training import run_consistency_epoch
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.horizon_gate import HorizonGate, measure_horizon
from trajectory_prediction.online.horizon_training import load_horizon
from trajectory_prediction.online.innovation_consistency import InnovationDataset
from trajectory_prediction.online.refinement_training import load_initial, state_fingerprint
from trajectory_prediction.online.scaled_training import _save_torch
from trajectory_prediction.online.strength_consistency import strength_settings
from trajectory_prediction.online.strength_training import (
    configure_runtime,
    load_strength,
    training_sources,
)
from trajectory_prediction.utils.provenance import sha256_file
from trajectory_prediction.utils.seed import seed_everything

FIELDS = ("ade", "fde", "jitter", "clean_ade", "clean_fde_p95")
ARM = "adaptive_matched_005"
PROJECT = Path(__file__).resolve().parents[3]


def sources():
    return {"parents": training_sources(), "execution": {name: sha256_file(PROJECT / name) for name in (
        "src/trajectory_prediction/online/precision_checkpoint.py",
        "scripts/train_precision_checkpoint.py",
        "docs/superpowers/specs/2026-11-06-causal-context-phase66-checkpoint-design.md",
    )}}


def summarize(metrics):
    values = np.asarray(metrics)
    if (values.ndim != 4 or values.shape[0] != 6 or values.shape[1] < 1
            or values.shape[2:] != (11, 3) or not np.isfinite(values).all() or (values < 0).any()):
        raise ValueError("complete finite nonnegative six-condition metrics required")
    return {"ade": float(values[..., 0].mean()), "fde": float(values[..., 1].mean()),
            "jitter": float(values[..., 2].mean()), "clean_ade": float(values[0, ..., 0].mean()),
            "clean_fde_p95": float(np.quantile(values[0, ..., 1], .95))}


def select_epochs(logs, reference):
    if (not logs or len({row["epoch"] for row in logs}) != len(logs)
            or any(type(row["epoch"]) is not int or row["epoch"] < 1 for row in logs)
            or any(not np.isfinite(row[name]) or row[name] < 0 for row in [reference, *logs] for name in FIELDS)
            or any(reference[name] <= 0 for name in ("ade", "fde", "clean_ade", "clean_fde_p95"))):
        raise ValueError("unique positive epochs and finite metric records required")
    eligible = [{**reference, "epoch": 0}]
    for row in logs:
        if (row["ade"] <= 1.0005 * reference["ade"]
                and row["clean_ade"] <= 1.0005 * reference["clean_ade"]
                and row["fde"] <= 1.0025 * reference["fde"]
                and row["clean_fde_p95"] <= 1.005 * reference["clean_fde_p95"]):
            eligible.append(row)
    return {"ade_best": min(logs, key=lambda row: (row["ade"], row["epoch"]))["epoch"],
            "constrained_best": min(eligible, key=lambda row: (row["jitter"], row["ade"], row["epoch"]))["epoch"]}


def _save_state(root, epoch, model, identity):
    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    fingerprint = state_fingerprint(state)
    name = "reference.pt" if epoch == 0 else f"epoch-{epoch:03d}.pt"
    _save_torch(root / name, {"model": state, "identity": identity, "epoch": epoch})
    return {"file": name, "sha256": sha256_file(root / name), "fingerprint": fingerprint}


def train_path(train, selection, initial, reference_model, output_dir, *, seed=2027,
               epochs=10, batch_size=4096, metadata=None, device=None):
    if set(train.scenes) & set(selection.scenes):
        raise ValueError("training/selection scene overlap")
    root = Path(output_dir)
    if root.exists():
        raise FileExistsError("checkpoint experiment directory already exists")
    if (type(epochs) is not int or not 1 <= epochs <= 10 or type(batch_size) is not int or batch_size < 1
            or train.candidate != selection.candidate or reference_model.variant != "RH"):
        raise ValueError("invalid checkpoint experiment settings")
    device = device or torch.device("cpu")
    if device.type == "cuda":
        configure_runtime()
    executable = sources()
    settings = {"seed": seed, "epochs": epochs, "batch_size": batch_size, "arm": ARM,
                "learning_rate": .001, "weight_decay": .0001, "candidate": train.candidate,
                **strength_settings(ARM, train.train_mean_weight)}
    binding = {"settings": settings, "sources": executable, "metadata": metadata or {},
               "train_content": train.fingerprint(), "selection_content": selection.fingerprint(),
               "initializer": state_fingerprint(initial)}
    identity = digest_json(binding)
    root.mkdir(parents=True, exist_ok=False)
    (root / "run.json").write_text(json.dumps(binding, indent=2) + "\n", encoding="utf-8")
    reference_metrics, _ = measure_horizon(selection, reference_model, device=device, batch_size=batch_size)
    reference = summarize(reference_metrics)
    checkpoints = {"0": _save_state(root, 0, reference_model, identity)}
    seed_everything(seed)
    model = HorizonGate("RH", initial).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    generator = np.random.default_rng(seed)
    logs = []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        losses = run_consistency_epoch(
            model, train, generator.permutation(train.size), batch_size, device, optimizer,
            settings["revision_weight"], settings["revision_kind"], train.train_mean_weight,
        )
        metrics, _ = measure_horizon(selection, model, device=device, batch_size=batch_size)
        row = {"epoch": epoch, **summarize(metrics), **losses,
               "seconds": time.perf_counter() - started}
        logs.append(row)
        checkpoints[str(epoch)] = _save_state(root, epoch, model, identity)
        (root / "progress.json").write_text(json.dumps(logs, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"seed": seed, **row}), flush=True)
    if sources() != executable:
        raise ValueError("checkpoint experiment executable changed during training")
    report = {"schema_version": 1, "identity": identity, "binding": binding,
              "reference": reference, "epochs": logs, "completed_epochs": epochs,
              "selected": select_epochs(logs, reference), "checkpoints": checkpoints,
              "val_run": False, "run_sha256": sha256_file(root / "run.json")}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Re-open checkpoint bytes instead of evaluating an in-memory best-state alias.
    verify_run(root, selection, initial, device, _publishing=True)
    return report


def verify_run(root, selection, initial, device, *, _publishing=False):
    root = Path(root)
    if _publishing:
        if (root / "completed.json").exists():
            raise FileExistsError("checkpoint completion marker already exists")
    else:
        marker = json.loads((root / "completed.json").read_text())
        if marker["report_sha256"] != sha256_file(root / "report.json"):
            raise ValueError("checkpoint report checksum mismatch")
        if marker["metrics_sha256"] != sha256_file(root / "selected_metrics.npz"):
            raise ValueError("checkpoint metrics checksum mismatch")
    report = json.loads((root / "report.json").read_text())
    binding = report["binding"]
    if (digest_json(binding) != report["identity"] or binding["sources"] != sources()
            or binding["selection_content"] != selection.fingerprint()
            or binding["initializer"] != state_fingerprint(initial)
            or sha256_file(root / "run.json") != report["run_sha256"]
            or json.loads((root / "run.json").read_text()) != binding
            or report["val_run"] or len(report["epochs"]) != report["completed_epochs"]
            or [row["epoch"] for row in report["epochs"]] != list(range(1, report["completed_epochs"] + 1))
            or set(report["checkpoints"]) != {str(k) for k in range(report["completed_epochs"] + 1)}):
        raise ValueError("checkpoint input/source identity mismatch")
    for ref in binding["metadata"].get("parents", {}).values():
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("checkpoint parent identity mismatch")
    if report["selected"] != select_epochs(report["epochs"], report["reference"]):
        raise ValueError("checkpoint selection mismatch")
    states = {}
    for epoch, ref in report["checkpoints"].items():
        expected_name = "reference.pt" if epoch == "0" else f"epoch-{int(epoch):03d}.pt"
        if ref["file"] != expected_name or sha256_file(root / expected_name) != ref["sha256"]:
            raise ValueError("checkpoint checksum mismatch")
        payload = torch.load(root / expected_name, weights_only=True, map_location="cpu")
        if (payload["identity"] != report["identity"] or payload["epoch"] != int(epoch)
                or state_fingerprint(payload["model"]) != ref["fingerprint"]):
            raise ValueError("checkpoint state fingerprint mismatch")
        states[int(epoch)] = payload["model"]
    replay = {}
    metrics_path = root / "selected_metrics.npz"
    stored = dict(np.load(metrics_path, allow_pickle=False)) if metrics_path.exists() else None
    for name, epoch in {"RH": 0, **report["selected"]}.items():
        model = HorizonGate("RH", initial)
        model.load_state_dict(states[epoch], strict=True)
        values, _ = measure_horizon(selection, model, device=device)
        expected = report["reference"] if epoch == 0 else report["epochs"][epoch - 1]
        for field in FIELDS:
            np.testing.assert_allclose(summarize(values)[field], expected[field], atol=1e-9, rtol=0)
        if stored is not None:
            np.testing.assert_array_equal(values, stored[name])
        replay[name] = values
    if stored is None:
        np.savez_compressed(metrics_path, **replay)
    if _publishing:
        temporary = root / ".completed.tmp"
        temporary.write_text(json.dumps({"report_sha256": sha256_file(root / "report.json"),
                                         "metrics_sha256": sha256_file(metrics_path)}) + "\n")
        temporary.replace(root / "completed.json")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/context/p66-checkpoint"))
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise FileExistsError("experiment output exists; do not overwrite")
    configure_runtime()
    torch.set_num_threads(2)
    device = choose_device(args.device)
    frame, _ = load_parent(Path("outputs/context/p65/selection"))
    args.output_root.mkdir(parents=True, exist_ok=False)
    reports = []
    for seed in (2027, 2028, 2029):
        cache = Path("outputs/context/p6/cache")
        loaded = {s: load_verified_gate_cache(cache / s / f"seed-{seed}", s, seed, device=device)
                  for s in ("train", "train_val")}
        train, selection = (InnovationDataset(loaded[s][0]) for s in ("train", "train_val"))
        contract = loaded["train_val"][1]["contract"]
        old_path = Path(f"outputs/context/p65/training/seed-{seed}/{ARM}/best.pt")
        old_model, parent = load_strength(old_path, contract, selection.scenes, device, split="train_val",
                                           arm=ARM, train_mean_weight=train.train_mean_weight,
                                           train_content=train.fingerprint())
        if parent["selection_content"] != selection.fingerprint():
            raise ValueError("parent selection content mismatch")
        initial = load_initial(parent["p6_checkpoint"]["path"], parent["candidate_contracts"])
        rh_path = Path(f"outputs/context/p63/training-deterministic/seed-{seed}/RH/best.pt")
        rh, _ = load_horizon(rh_path, contract, selection.scenes, device, split="train_val", variant="RH")
        parents = {"old_arm": {"path": str(old_path.resolve()), "sha256": sha256_file(old_path)},
                   "rh": {"path": str(rh_path.resolve()), "sha256": sha256_file(rh_path)},
                   "selection": {"path": str(Path("outputs/context/p65/selection/selection.json").resolve()),
                                 "sha256": sha256_file(Path("outputs/context/p65/selection/selection.json"))}}
        for split in loaded:
            path = cache / split / f"seed-{seed}" / "manifest.json"
            parents[split] = {"path": str(path.resolve()), "sha256": sha256_file(path)}
        report = train_path(train, selection, initial, rh, args.output_root / f"seed-{seed}",
                            seed=seed, metadata={"parents": parents, "contracts": parent["candidate_contracts"]}, device=device)
        best = report["checkpoints"][str(report["selected"]["ade_best"])]
        if best["fingerprint"] != state_fingerprint(old_model.state_dict()):
            raise ValueError("new ADE-best does not reproduce frozen P6.5 trajectory")
        old_values, _ = measure_horizon(selection, old_model, device=device)
        old_summary = summarize(old_values)
        old_rows = frame[frame.seed.eq(seed) & frame.strategy.eq(ARM)]
        np.testing.assert_allclose([old_summary[f] for f in ("ade", "fde", "jitter")],
                                   old_rows[["ade", "fde", "jitter"]].mean().to_numpy(), atol=1e-9, rtol=0)
        reports.append({"seed": seed, "selected": report["selected"], "parent_trajectory_parity": True,
                        "report_sha256": sha256_file(args.output_root / f"seed-{seed}" / "report.json")})
        print(json.dumps({"seed_completed": reports[-1]}), flush=True)
    result = {"schema_version": 1, "seeds": reports, "val_run": False, "new_epochs": 30,
              "sources": sources(), "scope": "train_val_checkpoint_selection_not_independent_generalization"}
    (args.output_root / "experiment.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0

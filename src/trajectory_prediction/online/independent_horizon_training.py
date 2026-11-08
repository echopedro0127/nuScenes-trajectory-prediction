"""Retrain RH from the independent quality gate, preserving legacy experiments."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.context.independent_candidates import load_verified_candidates
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.horizon_features import HorizonDataset
from trajectory_prediction.online.horizon_gate import HorizonGate, measure_horizon
from trajectory_prediction.online.horizon_training import (
    DEFAULT_SETTINGS,
    configure_runtime,
    train_horizon,
    training_sources,
    validate_role,
)
from trajectory_prediction.online.independent_gate_training import (
    load_independent_gate,
    validate_inputs,
)
from trajectory_prediction.online.independent_map_training import ROLES
from trajectory_prediction.online.quality_gate import CONDITIONS
from trajectory_prediction.online.refinement_training import measure_model, state_fingerprint
from trajectory_prediction.online.scaled_training import _save_torch
from trajectory_prediction.utils.provenance import sha256_file


def sources():
    package = Path(__file__).parents[1]
    return {"horizon": training_sources(), "independent": {name: sha256_file(package / name)
            for name in ("online/independent_horizon_training.py", "online/independent_gate_training.py",
                         "context/independent_candidates.py")}}


def initializer(path, gate_inputs):
    path = Path(path)
    if not path.is_file() or not (path.parent / "completed.json").is_file():
        raise ValueError("independent gate initializer and completion marker required")
    marker = json.loads((path.parent / "completed.json").read_text())
    for name, digest in marker["sha256"].items():
        if sha256_file(path.parent / name) != digest:
            raise ValueError("independent gate initializer checksum mismatch")
    model, payload = load_independent_gate(path, gate_inputs)
    if payload["metadata"]["run_identity"] != marker["run_identity"]:
        raise ValueError("independent gate initializer run mismatch")
    return model, payload


def load_independent_horizon(path, expected_inputs, *, device=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta = payload.get("metadata", {})
    if payload.get("checkpoint_schema") != 1 or meta.get("architecture") != "independent_horizon_gate_v1" or meta.get("variant") != "RH":
        raise ValueError("independent RH architecture/variant mismatch")
    if meta.get("independent_sources") != sources() or meta.get("source_code_sha256") != training_sources():
        raise ValueError("independent RH executable source mismatch")
    if digest_json(meta["independent_inputs"]) != digest_json(expected_inputs):
        raise ValueError("independent RH input identity mismatch")
    inputs = meta["independent_inputs"]
    validate_inputs(inputs["gate_inputs"])
    reference = inputs["gate_checkpoint"]
    if sha256_file(Path(reference["path"])) != reference["sha256"]:
        raise ValueError("independent RH initializer identity mismatch")
    parent, parent_payload = initializer(reference["path"], inputs["gate_inputs"])
    if parent_payload["metadata"]["run_identity"] != inputs["gate_run_identity"]:
        raise ValueError("independent RH initializer run mismatch")
    model = HorizonGate("RH", parent.state_dict())
    if state_fingerprint(model.state_dict()) != meta["initial_state_sha256"] or state_fingerprint(parent.state_dict()) != meta["initial_p6_state_sha256"]:
        raise ValueError("independent RH initialization state mismatch")
    model.load_state_dict(payload["model"], strict=True)
    settings = {**DEFAULT_SETTINGS, "epochs": meta["epochs"], "batch_size": meta["batch_size"], "patience": meta["patience"]}
    validate_role(model, meta, "RH", settings)
    return model.to(device or torch.device("cpu")).eval(), payload


def train_independent_horizon(train, selection, gate_inputs, gate_checkpoint, output_dir, *,
                              epochs=10, batch_size=4096, patience=3, device=None):
    validate_inputs(gate_inputs)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("use a new independent RH output directory")
    for data, key in ((train, "train_scenes"), (selection, "selection_scenes")):
        if data.scenes != gate_inputs[key]:
            raise ValueError("independent RH data scene mismatch")
    parent, parent_payload = initializer(gate_checkpoint, gate_inputs)
    configure_runtime()
    device = device or torch.device("cpu")
    initial = parent.state_dict()
    baseline, scalar = measure_model(selection, parent, batch_size=batch_size, device=device)
    initial_metrics, weights = measure_horizon(selection, HorizonGate("RH", initial), batch_size=batch_size, device=device)
    mask = np.arange(12)[None] >= np.arange(1, 12)[:, None]
    expected = scalar[..., None] * mask[None, None]
    error = float(np.max(np.abs(weights - expected)))
    if error > 1e-6 or not np.allclose(baseline, initial_metrics, atol=1e-5, rtol=1e-5):
        raise ValueError("independent RH must match new gate at initialization")
    reference = Path(gate_checkpoint).resolve()
    inputs = {"gate_inputs": gate_inputs, "gate_checkpoint": {"path": str(reference), "sha256": sha256_file(reference)},
              "gate_run_identity": parent_payload["metadata"]["run_identity"]}
    metadata = {"independent_inputs": inputs, "independent_sources": sources()}
    train_horizon(train, selection, metadata, output, initial=initial, variant="RH", epochs=epochs,
                  batch_size=batch_size, patience=patience, seed=2027, revision_weight=0., device=device)
    payload = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    payload["metadata"]["architecture"] = "independent_horizon_gate_v1"
    _save_torch(output / "best.pt", payload)
    model, payload = load_independent_horizon(output / "best.pt", inputs, device=device)
    metrics, weights = measure_horizon(selection, model, batch_size=batch_size, device=device)
    if not np.isclose(metrics[..., 0].mean(), payload["metadata"]["selection_ade"], atol=1e-12, rtol=0):
        raise ValueError("independent RH checkpoint replay mismatch")
    missing = not np.any(weights[CONDITIONS.index("missing")])
    if not missing:
        raise ValueError("independent RH missing map must force zero weights")
    np.savez_compressed(output / "selection_metrics.npz", rh=metrics, gate=baseline)
    names = ("ade", "fde", "jitter")
    report = {"run_identity": payload["metadata"]["run_identity"], "initialization_max_error": error,
              "initialization_metric_max_error": float(np.max(np.abs(baseline - initial_metrics))),
              "missing_weights_zero": missing, "holdout_evaluated": False, "independent_chain_complete": False,
              "aggregate": {label: dict(zip(names, value.mean((0, 1, 2)).tolist(), strict=True))
                            for label, value in (("gate", baseline), ("RH", metrics))},
              "conditions": {condition: {label: dict(zip(names, value[c].mean((0, 1)).tolist(), strict=True))
                                          for label, value in (("gate", baseline), ("RH", metrics))}
                             for c, condition in enumerate(CONDITIONS)}}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    names = ("run.json", "training.json", "best.pt", "report.json", "selection_metrics.npz")
    (output / "completed.json").write_text(json.dumps({"identity": report["run_identity"],
        "sha256": {name: sha256_file(output / name) for name in names}}, indent=2) + "\n")
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/context/p66-independent"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise FileExistsError("use a new independent RH output directory")
    configure_runtime()
    torch.set_num_threads(2)
    root, device = args.root.resolve(), choose_device(args.device)
    gate_checkpoint = root / "gate-training/seed-2027/best.pt"
    gate_payload = torch.load(gate_checkpoint, map_location="cpu", weights_only=True)
    gate_inputs = gate_payload["metadata"]["independent_inputs"]
    loaded = {role: load_verified_candidates(root / "candidates" / role, root, role, device=device) for role in ROLES}
    for role in ROLES:
        if loaded[role][1]["contract"] != gate_inputs["candidate_contracts"][role] or sha256_file(root / "candidates" / role / "manifest.json") != gate_inputs["candidate_manifest_sha256"][role]:
            raise ValueError("independent RH candidate/gate source mismatch")
    train_independent_horizon(*(HorizonDataset(loaded[role][0]) for role in ROLES), gate_inputs,
                              gate_checkpoint, args.output_root, device=device)


if __name__ == "__main__":
    main()

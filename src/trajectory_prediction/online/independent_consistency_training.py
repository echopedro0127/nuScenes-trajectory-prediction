"""Independent same-start RH control and innovation-weighted consistency."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.context.independent_candidates import load_verified_candidates
from trajectory_prediction.online.consistency_training import (
    DEFAULT_SETTINGS,
    train_consistency,
    training_sources,
    validate_role,
)
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.horizon_gate import HorizonGate, measure_horizon
from trajectory_prediction.online.horizon_training import configure_runtime
from trajectory_prediction.online.independent_horizon_training import (
    initializer,
    load_independent_horizon,
)
from trajectory_prediction.online.independent_map_training import ROLES
from trajectory_prediction.online.innovation_consistency import InnovationDataset
from trajectory_prediction.online.quality_gate import CONDITIONS
from trajectory_prediction.online.refinement_training import state_fingerprint
from trajectory_prediction.online.scaled_training import _save_torch
from trajectory_prediction.utils.provenance import sha256_file

ARMS = ("revision_0", "revision_adaptive")


def sources():
    package = Path(__file__).parents[1]
    return {"core": training_sources(), "independent": {name: sha256_file(package / name)
            for name in ("online/independent_consistency_training.py", "online/independent_horizon_training.py",
                         "online/independent_gate_training.py", "context/independent_candidates.py")}}


def parent_rh(path):
    path = Path(path)
    marker = json.loads((path.parent / "completed.json").read_text())
    for name, digest in marker["sha256"].items():
        if sha256_file(path.parent / name) != digest:
            raise ValueError("independent RH parent checksum mismatch")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    inputs = payload["metadata"]["independent_inputs"]
    model, payload = load_independent_horizon(path, inputs)
    if payload["metadata"]["run_identity"] != marker["identity"]:
        raise ValueError("independent RH parent run identity mismatch")
    return model, payload


def load_independent_consistency(path, expected_inputs, *, arm, device=None):
    if arm not in ARMS:
        raise ValueError("unsupported independent consistency arm")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta = payload.get("metadata", {})
    if payload.get("checkpoint_schema") != 1 or meta.get("architecture") != "independent_consistency_v1" or meta.get("arm") != arm:
        raise ValueError("independent consistency architecture/arm mismatch")
    if meta.get("independent_sources") != sources() or meta.get("source_code_sha256") != training_sources():
        raise ValueError("independent consistency source mismatch")
    if digest_json(meta["independent_inputs"]) != digest_json(expected_inputs):
        raise ValueError("independent consistency input identity mismatch")
    inputs = meta["independent_inputs"]
    reference = inputs["rh_checkpoint"]
    if sha256_file(Path(reference["path"])) != reference["sha256"]:
        raise ValueError("independent consistency RH identity mismatch")
    _, rh_payload = parent_rh(reference["path"])
    if rh_payload["metadata"]["independent_inputs"] != inputs["horizon_inputs"]:
        raise ValueError("independent consistency RH input mismatch")
    horizon = inputs["horizon_inputs"]
    gate, _ = initializer(horizon["gate_checkpoint"]["path"], horizon["gate_inputs"])
    model = HorizonGate("RH", gate.state_dict())
    if state_fingerprint(model.state_dict()) != meta["initial_state_sha256"] or state_fingerprint(gate.state_dict()) != meta["initial_p6_state_sha256"]:
        raise ValueError("independent consistency initial state mismatch")
    model.load_state_dict(payload["model"], strict=True)
    settings = {**DEFAULT_SETTINGS, "epochs": meta["epochs"], "batch_size": meta["batch_size"],
                "patience": meta["patience"], "revision_weight": .02 if arm == "revision_adaptive" else 0.}
    validate_role(model, meta, arm, settings)
    return model.to(device or torch.device("cpu")).eval(), payload


def train_independent_consistency(train, selection, rh_checkpoint, output_dir, *, arm,
                                  epochs=10, batch_size=4096, patience=3, device=None):
    if arm not in ARMS:
        raise ValueError("unsupported independent consistency arm")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("use a new independent consistency output directory")
    reference, rh_payload = parent_rh(rh_checkpoint)
    horizon_inputs = rh_payload["metadata"]["independent_inputs"]
    gate_inputs = horizon_inputs["gate_inputs"]
    for data, key in ((train, "train_scenes"), (selection, "selection_scenes")):
        if data.scenes != gate_inputs[key]:
            raise ValueError("independent consistency data scene mismatch")
    gate, _ = initializer(horizon_inputs["gate_checkpoint"]["path"], gate_inputs)
    configure_runtime()
    device = device or torch.device("cpu")
    path = Path(rh_checkpoint).resolve()
    inputs = {"horizon_inputs": horizon_inputs, "rh_checkpoint": {"path": str(path), "sha256": sha256_file(path)}}
    metadata = {"independent_inputs": inputs, "independent_sources": sources()}
    train_consistency(train, selection, metadata, output, initial=gate.state_dict(), arm=arm,
                      epochs=epochs, batch_size=batch_size, patience=patience, seed=2027, device=device)
    payload = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    payload["metadata"]["architecture"] = "independent_consistency_v1"
    _save_torch(output / "best.pt", payload)
    model, payload = load_independent_consistency(output / "best.pt", inputs, arm=arm, device=device)
    zero_equal = all(torch.equal(value.detach().cpu(), reference.state_dict()[name]) for name, value in model.state_dict().items())
    if arm == "revision_0" and not zero_equal:
        raise ValueError("independent zero regularizer must reproduce RH exactly")
    baseline, _ = measure_horizon(selection, reference, batch_size=batch_size, device=device)
    metrics, weights = measure_horizon(selection, model, batch_size=batch_size, device=device)
    if not np.isclose(metrics[..., 0].mean(), payload["metadata"]["selection_ade"], atol=1e-12, rtol=0):
        raise ValueError("independent consistency checkpoint replay mismatch")
    if np.any(weights[4]) or not np.array_equal(metrics[4], baseline[4]):
        raise ValueError("independent consistency missing map CV fallback mismatch")
    np.savez_compressed(output / "selection_metrics.npz", model=metrics, RH=baseline)
    fields = ("ade", "fde", "jitter")
    report = {"run_identity": payload["metadata"]["run_identity"], "arm": arm,
              "zero_reproduces_rh": zero_equal, "train_mean_weight": train.train_mean_weight,
              "holdout_evaluated": False, "independent_chain_complete": False,
              "aggregate": {label: dict(zip(fields, values.mean((0, 1, 2)).tolist(), strict=True))
                            for label, values in (("RH", baseline), (arm, metrics))},
              "conditions": {condition: {label: dict(zip(fields, values[c].mean((0, 1)).tolist(), strict=True))
                                          for label, values in (("RH", baseline), (arm, metrics))}
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
        raise FileExistsError("use a new independent consistency output root")
    configure_runtime()
    torch.set_num_threads(2)
    root, device = args.root.resolve(), choose_device(args.device)
    checkpoint = root / "horizon-training/seed-2027/RH/best.pt"
    _, rh = parent_rh(checkpoint)
    gate_inputs = rh["metadata"]["independent_inputs"]["gate_inputs"]
    loaded = {role: load_verified_candidates(root / "candidates" / role, root, role, device=device) for role in ROLES}
    for role in ROLES:
        if loaded[role][1]["contract"] != gate_inputs["candidate_contracts"][role] or sha256_file(root / "candidates" / role / "manifest.json") != gate_inputs["candidate_manifest_sha256"][role]:
            raise ValueError("independent consistency candidate/RH source mismatch")
    train, selection = (InnovationDataset(loaded[role][0]) for role in ROLES)
    for arm in ARMS:
        train_independent_consistency(train, selection, checkpoint, args.output_root / arm, arm=arm, device=device)


if __name__ == "__main__":
    main()

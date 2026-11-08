"""Independent fit/select quality gate with frozen, verified candidates."""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.context.independent_candidates import load_verified_candidates
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.gate_dataset import GateDataset
from trajectory_prediction.online.gate_training import run_epoch, train_gate, training_sources
from trajectory_prediction.online.independent_map_training import ROLES
from trajectory_prediction.online.quality_gate import CONDITIONS, QualityGate, blend_forecasts
from trajectory_prediction.online.scaled_training import _save_torch
from trajectory_prediction.utils.provenance import sha256_file


def sources():
    package = Path(__file__).parents[1]
    names = ("online/independent_gate_training.py", "context/independent_candidates.py")
    return {**training_sources(), **{name: sha256_file(package / name) for name in names}}


def validate_inputs(metadata):
    contracts = metadata.get("candidate_contracts", {})
    if set(contracts) != set(ROLES):
        raise ValueError("independent fit/select contracts required")
    partition = metadata.get("partition_identity", "")
    if not re.fullmatch(r"[0-9a-f]{64}", str(partition)):
        raise ValueError("invalid partition identity")
    for role in ROLES:
        contract = contracts[role]
        if contract.get("schema") != "independent_candidates_v1" or contract.get("split") != role or contract.get("partition_identity") != partition or contract.get("seed") != 2027:
            raise ValueError("independent candidate role/partition mismatch")
        digest = metadata.get("candidate_manifest_sha256", {}).get(role, "")
        if not re.fullmatch(r"[0-9a-f]{64}", str(digest)):
            raise ValueError("invalid candidate manifest identity")
    a, b = (contracts[role] for role in ROLES)
    for name in ("map_training_identity", "gate_config", "conditions", "sources"):
        if name not in a or a[name] != b.get(name):
            raise ValueError("independent candidate shared source mismatch")
    if a.get("base_hashes", {}).get("model_identity") != b.get("base_hashes", {}).get("model_identity") or a.get("references", {}).get("checkpoint") != b.get("references", {}).get("checkpoint"):
        raise ValueError("independent candidate frozen model mismatch")
    for groups in ((metadata.get("train_scenes"), metadata.get("selection_scenes")),
                   tuple(metadata.get("role_logs", {}).get(role) for role in ROLES)):
        for values in groups:
            if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values) or values != sorted(set(values)):
                raise ValueError("unique sorted scene/log partitions required")
        if set(groups[0]) & set(groups[1]):
            raise ValueError("independent fit/select scene/log overlap")


def load_independent_gate(path, expected_metadata, *, device=None):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    meta = payload.get("metadata", {})
    if payload.get("checkpoint_schema") != 1 or meta.get("architecture") != "independent_quality_gate_v1":
        raise ValueError("unsupported independent quality gate")
    if meta.get("independent_sources") != sources() or meta.get("source_code_sha256") != training_sources():
        raise ValueError("independent gate executable source mismatch")
    validate_inputs(meta["independent_inputs"])
    if digest_json(meta["independent_inputs"]) != digest_json(expected_metadata):
        raise ValueError("independent gate input identity mismatch")
    model = QualityGate().to(device or torch.device("cpu"))
    model.load_state_dict(payload["model"], strict=True)
    return model.eval(), payload


def diagnose(model, data, *, batch_size=4096, device=None):
    """Select-only diagnostics, equally weighting updates within each condition."""
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    report = {}
    for ci, condition in enumerate(CONDITIONS):
        totals = {name: np.zeros(2, np.float64) for name in ("cv", "original", "map", "gate")}
        count, weight_sum = 0, 0.
        begin, end = ci * data.n * 11, (ci + 1) * data.n * 11
        with torch.inference_mode():
            for start in range(begin, end, batch_size):
                indices = np.arange(start, min(end, start + batch_size))
                inputs, target = data.inputs(indices, device), data.labels(indices, device)
                weight = model(inputs["quality"], inputs["valid"])
                _, rows, updates = data.indices(indices)
                original = torch.from_numpy(np.ascontiguousarray(data.arrays["original"][rows, updates])).to(device)
                forecasts = {"cv": inputs["cv"], "original": original, "map": inputs["map"],
                             "gate": blend_forecasts(inputs["cv"], inputs["map"], weight)}
                mask = inputs["mask"]
                for name, forecast in forecasts.items():
                    error = torch.linalg.vector_norm(forecast.double() - target.double(), dim=-1)
                    totals[name] += (float(((error * mask).sum(1) / mask.sum(1)).sum()), float(error[:, -1].sum()))
                count += len(indices)
                weight_sum += float(weight.double().sum())
        report[condition] = {"updates": count, "mean_gate_weight": weight_sum / count,
                             "strategies": {name: {"ade": float(value[0] / count), "fde": float(value[1] / count)}
                                            for name, value in totals.items()}}
    return report


def train_independent_gate(train, selection, metadata, output_dir, *, epochs=20, batch_size=4096,
                           patience=5, seed=2027, learning_rate=.001, device=None):
    validate_inputs(metadata)
    for data, name in ((train, "train_scenes"), (selection, "selection_scenes")):
        if data.scenes != metadata[name]:
            raise ValueError("independent gate data scene mismatch")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("use a new independent gate output directory")
    if seed != 2027:
        raise ValueError("independent chain currently freezes seed 2027")
    device = device or torch.device("cpu")
    wrapper = {"independent_inputs": metadata, "independent_sources": sources()}
    train_gate(train, selection, wrapper, output, epochs=epochs, batch_size=batch_size,
               patience=patience, seed=seed, learning_rate=learning_rate, device=device)
    payload = torch.load(output / "best.pt", map_location="cpu", weights_only=True)
    payload["metadata"]["architecture"] = "independent_quality_gate_v1"
    _save_torch(output / "best.pt", payload)
    model, payload = load_independent_gate(output / "best.pt", metadata, device=device)
    replay = run_epoch(model, selection, np.arange(selection.size), batch_size, device)
    if not math.isclose(replay["ade"], payload["metadata"]["selection_ade"], rel_tol=0, abs_tol=1e-8):
        raise ValueError("independent gate checkpoint replay mismatch")
    report = {"run_identity": payload["metadata"]["run_identity"], "selection_replay": replay,
              "diagnostics": diagnose(model, selection, batch_size=batch_size, device=device),
              "holdout_evaluated": False, "independent_chain_complete": False}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    names = ("run.json", "training.json", "best.pt", "report.json")
    (output / "completed.json").write_text(json.dumps({"run_identity": report["run_identity"],
        "sha256": {name: sha256_file(output / name) for name in names}}, indent=2) + "\n")
    return model


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/context/p66-independent"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise FileExistsError("use a new independent gate output directory")
    torch.set_num_threads(2)
    device = choose_device(args.device)
    root = args.root.resolve()
    loaded = {role: load_verified_candidates(root / "candidates" / role, root, role, device=device) for role in ROLES}
    parent = json.loads((root / "map-training/seed-2027/run.json").read_text())["input_metadata"]
    metadata = {"partition_identity": parent["partition_identity"], "role_logs": parent["role_logs"],
                "train_scenes": parent["train_scenes"], "selection_scenes": parent["selection_scenes"],
                "candidate_contracts": {role: item[1]["contract"] for role, item in loaded.items()},
                "candidate_manifest_sha256": {role: sha256_file(root / "candidates" / role / "manifest.json") for role in ROLES}}
    train_independent_gate(*(GateDataset(loaded[role][0]) for role in ROLES), metadata, args.output_root, device=device)


if __name__ == "__main__":
    main()

"""Source-bound candidate caches for the independent fit/select chain only."""
import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import IDENTIFIERS, digest_json, key_hash
from trajectory_prediction.context.gate_preparation import (
    audit_candidates,
    generate_candidates,
    preparation_sources,
)
from trajectory_prediction.context.maps import MapConfig
from trajectory_prediction.context.mmap_cache import open_sidecar
from trajectory_prediction.context.preparation import context_contract, load_predictions
from trajectory_prediction.context.scaled_preparation import file_provenance
from trajectory_prediction.data.cache import load_cache
from trajectory_prediction.data.independent_split import verify_prepared
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.gate_dataset import load_gate_cache, save_gate_cache
from trajectory_prediction.online.independent_map_training import (
    ROLES,
    load_pair,
    validate_metadata,
)
from trajectory_prediction.online.quality_gate import CONDITIONS, GateConfig
from trajectory_prediction.utils.provenance import sha256_file


def validate_parent(metadata, arrays, manifest, hashes, role):
    if role not in ROLES:
        raise ValueError("only independent fit/select candidates are allowed")
    validate_metadata(metadata)
    prefix = "train" if role == ROLES[0] else "selection"
    expected = metadata[prefix + "_provenance"]
    if (manifest["contract"].get("split") != role or expected["contract"] != manifest["contract"]
            or expected["context_sha256"] != manifest["npz_sha256"] or expected["base"] != hashes
            or hashes["model_identity"] != metadata["base_model"]
            or sorted(set(arrays["scene_name"].tolist())) != metadata[prefix + "_scenes"]):
        raise ValueError("independent candidate parent provenance mismatch")


def sources():
    package = Path(__file__).parents[1]
    names = ("context/independent_candidates.py", "online/independent_map_training.py",
             "data/independent_split.py", "context/preparation.py", "context/mmap_cache.py")
    return {**preparation_sources(), **{name: sha256_file(package / name) for name in names}}


def _inputs(root, role):
    if role not in ROLES:
        raise ValueError("only independent fit/select candidates are allowed")
    proof = verify_prepared(root)
    config = load_yaml(root / "data.yaml")
    training = root / "map-training/seed-2027"
    marker = json.loads((training / "completed.json").read_text())
    for name, digest in marker["sha256"].items():
        if sha256_file(training / name) != digest:
            raise ValueError("independent model completion checksum mismatch")
    run = json.loads((training / "run.json").read_text())
    identity = run.pop("identity")
    if digest_json(run) != identity or identity != marker["identity"]:
        raise ValueError("independent model run identity mismatch")
    for name, digest in run["source"].items():
        if sha256_file(Path(__file__).parents[1] / name) != digest:
            raise ValueError("independent model executable source mismatch")
    metadata = run["input_metadata"]
    models, payload = load_pair(training / "best.pt", metadata)
    if payload["metadata"]["identity"] != identity or metadata["partition_identity"] != proof["partition_identity"]:
        raise ValueError("independent model partition identity mismatch")
    arrays, trajectory = load_cache(root / "cache" / role, role)
    provenance = file_provenance(config["dataroot"], trajectory["dataset_version"])
    expected = context_contract(arrays, trajectory, provenance, MapConfig())
    map_root = root / "context" / role / "map"
    maps = open_sidecar(map_root, expected)
    map_manifest = json.loads((map_root / "manifest.json").read_text())
    prediction = root / "predictions" / role / "gru_eval/predictions.npz"
    base, hashes = load_predictions(prediction, arrays, expected)
    validate_parent(metadata, arrays, map_manifest, hashes, role)
    paths = {"partition": root / "partitions.json", "split_completion": root / "completed.json",
             "data_config": root / "data.yaml", "trajectory": root / "cache" / role / "manifest.json",
             "map": map_root / "manifest.json", "map_sidecar": map_root / "mmap_manifest.json",
             "base": prediction, "base_metrics": prediction.parent / "metrics.json",
             "base_keys": prediction.parent / "per_sample.csv", "checkpoint": training / "best.pt",
             "training_completion": training / "completed.json", "training_run": training / "run.json"}
    contract = {"schema": "independent_candidates_v1", "split": role, "seed": 2027,
                "partition_identity": proof["partition_identity"], "map_training_identity": identity,
                "base_hashes": hashes, "map_contract": expected, "keys_sha256": key_hash(arrays),
                "gate_config": asdict(GateConfig()), "conditions": list(CONDITIONS),
                "sources": sources(), "references": {name: {"path": str(path.resolve()),
                    "sha256": sha256_file(path)} for name, path in paths.items()},
                "feature_policy": "quality8_observation_and_predicted_candidates_only",
                "holdout_evaluated": False}
    return arrays, maps, base, models, contract


def publish_candidates(output, arrays, maps, base, models, contract, *, device=None):
    if contract.get("split") not in ROLES:
        raise ValueError("only independent fit/select candidates are allowed")
    output = Path(output)
    if output.exists():
        raise FileExistsError("use a new independent candidate directory")
    started = time.perf_counter()
    values = generate_candidates(arrays, maps, base, models["original"], models["map"],
                                 root=output, config=GateConfig(), device=device)
    audit_candidates(arrays, maps, base, models["original"], models["map"], values,
                     config=GateConfig(), device=device)
    if contract["sources"] != sources():
        raise ValueError("candidate executable changed during preparation")
    save_gate_cache(output, values, contract)
    report = {"split": contract["split"], "samples": len(base), "updates": len(base) * 11,
              "condition_updates": len(base) * 11 * len(CONDITIONS),
              "seconds": time.perf_counter() - started, "holdout_evaluated": False,
              "manifest_sha256": sha256_file(output / "manifest.json")}
    (output / "completed.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    return report


def load_verified_candidates(output, root, role, *, device=None):
    arrays, maps, base, models, expected = _inputs(Path(root).resolve(), role)
    output = Path(output)
    marker = json.loads((output / "completed.json").read_text())
    if marker["split"] != role or marker["manifest_sha256"] != sha256_file(output / "manifest.json"):
        raise ValueError("independent candidate completion mismatch")
    data, manifest = load_gate_cache(output, expected)
    for name, value in {**{name: arrays[name] for name in IDENTIFIERS}, "history": arrays["history_xy"],
                        "target": arrays["future_xy"], "base": base}.items():
        if not np.array_equal(data[name], value):
            raise ValueError("independent cached parent arrays mismatch")
    audit_candidates(arrays, maps, base, models["original"], models["map"], data, device=device)
    return data, manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/context/p66-independent"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if not args.verify_only and args.output_root.exists():
        raise FileExistsError("use a new candidate output root")
    torch.set_num_threads(2)
    device = choose_device(args.device)
    for role in ROLES:
        if args.verify_only:
            data, manifest = load_verified_candidates(args.output_root / role, args.root, role, device=device)
            print(json.dumps({"verified_split": role, "samples": len(data["base"]),
                              "contract_sha256": manifest["contract_sha256"]}), flush=True)
        else:
            arrays, maps, base, models, contract = _inputs(args.root.resolve(), role)
            publish_candidates(args.output_root / role, arrays, maps, base, models, contract, device=device)


if __name__ == "__main__":
    main()

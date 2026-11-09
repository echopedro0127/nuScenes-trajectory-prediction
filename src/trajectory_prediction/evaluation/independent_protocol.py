"""Freeze models and statistical rules before any new holdout inference."""
import argparse
import json
import os
import subprocess
from pathlib import Path

import torch

from trajectory_prediction.config import data_config_sha256, load_yaml
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.data.independent_split import ROLES, verify_prepared
from trajectory_prediction.online.independent_consistency_training import (
    ARMS,
    load_independent_consistency,
    parent_rh,
)
from trajectory_prediction.online.independent_horizon_training import initializer
from trajectory_prediction.online.independent_map_training import load_pair
from trajectory_prediction.utils.provenance import sha256_file

PROJECT = Path(__file__).resolve().parents[3]
DESIGN = PROJECT / "docs/superpowers/specs/2026-11-09-independent-holdout-design.md"


def _reference(root, name):
    path = Path(name)
    resolved = (root / path).resolve()
    if path.is_absolute() or not resolved.is_relative_to(root):
        raise ValueError("artifact references must stay within the experiment root")
    return resolved


def publish_lock(root, references, source_files, policy, output):
    root, output = Path(root).resolve(), Path(output)
    if output.exists():
        raise FileExistsError("use a new protocol output directory")
    artifacts = {}
    for name in references:
        path = _reference(root, name)
        key = path.relative_to(root).as_posix()
        if key in artifacts:
            raise ValueError("duplicate protocol artifact")
        artifacts[key] = sha256_file(path)
    for reference in policy.get("models", {}).values():
        path = _reference(root, reference["path"])
        digest = sha256_file(path)
        if digest != reference["sha256"]:
            raise ValueError("selected model checksum mismatch")
        artifacts[path.relative_to(root).as_posix()] = digest
    source = {name: {"relative_path": Path(os.path.relpath(Path(path).resolve(), root)).as_posix(),
                     "sha256": sha256_file(Path(path))} for name, path in source_files.items()}
    body = {"schema": "independent_holdout_protocol_v1", "policy": policy,
            "artifacts": artifacts, "sources": source}
    identity = digest_json(body)
    output.mkdir(parents=True, exist_ok=False)
    (output / "protocol.json").write_text(json.dumps({"identity": identity, "body": body}, indent=2) + "\n")
    marker = {"identity": identity, "protocol_sha256": sha256_file(output / "protocol.json")}
    (output / "completed.json").write_text(json.dumps(marker, indent=2) + "\n")
    verify_lock(root, output)
    return marker


def verify_lock(root, output):
    root, output = Path(root).resolve(), Path(output)
    marker = json.loads((output / "completed.json").read_text())
    if marker.get("protocol_sha256") != sha256_file(output / "protocol.json"):
        raise ValueError("protocol checksum mismatch")
    document = json.loads((output / "protocol.json").read_text())
    if (document["body"].get("schema") != "independent_holdout_protocol_v1"
            or digest_json(document["body"]) != document["identity"] or document["identity"] != marker["identity"]):
        raise ValueError("protocol identity mismatch")
    for name, digest in document["body"]["artifacts"].items():
        if sha256_file(_reference(root, name)) != digest:
            raise ValueError("protocol parent artifact checksum mismatch")
    for reference in document["body"]["policy"].get("models", {}).values():
        path = _reference(root, reference["path"])
        if document["body"]["artifacts"].get(path.relative_to(root).as_posix()) != reference["sha256"]:
            raise ValueError("selected model checksum binding mismatch")
    for reference in document["body"]["sources"].values():
        if sha256_file((root / reference["relative_path"]).resolve()) != reference["sha256"]:
            raise ValueError("protocol executable source checksum mismatch")
    return document


def freeze(root, output):
    root = Path(root).resolve()
    if Path(output).exists():
        raise FileExistsError("use a new protocol output directory")
    for folder in ("predictions/independent_holdout", "context/independent_holdout", "candidates/independent_holdout",
                   "holdout-evaluation", "evaluation", "evaluation_started.json"):
        if (root / folder).exists():
            raise ValueError("holdout artifacts already exist; cannot claim pre-inference freeze")
    prepared = verify_prepared(root)
    gru = torch.load(root / "gru-seed-2027/best.pt", map_location="cpu", weights_only=True)
    expected = {"train_split": "independent_fit", "val_split": "independent_select", "history_steps": 5,
                "data_config_sha256": data_config_sha256(load_yaml(root / "data.yaml"))}
    if gru.get("checkpoint_schema") != 1 or any(gru["metadata"].get(k) != v for k, v in expected.items()):
        raise ValueError("new GRU training identity mismatch")
    map_dir = root / "map-training/seed-2027"
    map_run = json.loads((map_dir / "run.json").read_text())
    load_pair(map_dir / "best.pt", map_run["input_metadata"])
    for name, digest in map_run["source"].items():
        if sha256_file(PROJECT / "src/trajectory_prediction" / name) != digest:
            raise ValueError("map executable identity mismatch")
    gate_path = root / "gate-training/seed-2027/best.pt"
    gate_payload = torch.load(gate_path, map_location="cpu", weights_only=True)
    gate_inputs = gate_payload["metadata"]["independent_inputs"]
    initializer(gate_path, gate_inputs)
    for contract in gate_inputs["candidate_contracts"].values():
        if (contract["base_hashes"]["model_identity"]["checkpoint_sha256"] != sha256_file(root / "gru-seed-2027/best.pt")
                or contract["references"]["checkpoint"]["sha256"] != sha256_file(map_dir / "best.pt")):
            raise ValueError("gate/GRU/map frozen weight identity mismatch")
    rh_path = root / "horizon-training/seed-2027/RH/best.pt"
    rh_model, rh_payload = parent_rh(rh_path)
    if rh_payload["metadata"]["independent_inputs"]["gate_inputs"] != gate_inputs:
        raise ValueError("RH/gate chain identity mismatch")
    if rh_payload["metadata"]["independent_inputs"]["gate_checkpoint"]["sha256"] != sha256_file(gate_path):
        raise ValueError("RH/gate frozen weight identity mismatch")
    if gate_inputs["partition_identity"] != prepared["partition_identity"]:
        raise ValueError("model/data partition identity mismatch")
    model_paths = {"gru": "gru-seed-2027/best.pt", "map_pair": "map-training/seed-2027/best.pt",
                   "quality_gate": "gate-training/seed-2027/best.pt", "RH": "horizon-training/seed-2027/RH/best.pt"}
    references = ["data.yaml", "partitions.json", "report.json", "completed.json", "gru-seed-2027/metrics.csv"]
    references += ["cache/" + role + "/manifest.json" for role in ROLES]
    for relative in ("map-training/seed-2027", "gate-training/seed-2027", "horizon-training/seed-2027/RH"):
        marker = json.loads((root / relative / "completed.json").read_text())
        for name, digest in marker["sha256"].items():
            if sha256_file(root / relative / name) != digest:
                raise ValueError("model completion checksum mismatch")
        references += [relative + "/" + name for name in ("run.json", "best.pt", "report.json", "completed.json", "audit.json")]
    for arm in ARMS:
        relative = "consistency-training/seed-2027/" + arm
        path = root / relative / "best.pt"
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["metadata"]["independent_inputs"]["rh_checkpoint"]["sha256"] != sha256_file(rh_path):
            raise ValueError("consistency/RH frozen weight identity mismatch")
        model, _ = load_independent_consistency(path, payload["metadata"]["independent_inputs"], arm=arm)
        if arm == "revision_0" and any(not torch.equal(value, rh_model.state_dict()[name]) for name, value in model.state_dict().items()):
            raise ValueError("zero control differs from frozen RH")
        marker = json.loads((path.parent / "completed.json").read_text())
        for name, digest in marker["sha256"].items():
            if sha256_file(path.parent / name) != digest:
                raise ValueError("consistency completion checksum mismatch")
        references += [relative + "/" + name for name in ("run.json", "best.pt", "report.json", "completed.json")]
        model_paths[arm] = relative + "/best.pt"
    references.append("consistency-training/seed-2027/audit.json")
    policy = {"date": "2026-11-09", "partition_identity": prepared["partition_identity"],
              "holdout_role": "independent_holdout", "holdout_counts": prepared["roles"]["independent_holdout"],
              "primary_model": "RH", "primary_baseline": "cv", "primary_metric": "ade",
              "strategies": ["cv", "original", "map", "quality_gate", "RH", "revision_adaptive"],
              "models": {name: {"path": path, "sha256": sha256_file(root / path)} for name, path in model_paths.items()},
              "seed": 2027, "history_steps": 5, "future_steps": 12, "frequency_hz": 2,
              "updates": 11, "fixed_original_endpoint": True, "oracle_position_replay": True,
              "conditions": gate_inputs["candidate_contracts"]["independent_fit"]["conditions"],
              "gate_config": gate_inputs["candidate_contracts"]["independent_fit"]["gate_config"],
              "secondary_metrics": ["fde", "jitter"], "aggregation": "equal_conditions_trajectories_updates",
              "paired_bootstrap": {"cluster": "log_token", "replicates": 5000, "seed": 20271109,
                "rng": "numpy_PCG64", "confidence": .95, "quantile_method": "linear",
                "estimator": "pooled_sum_over_count_with_log_multiplicity", "contrast": "RH_minus_cv_ade",
                "superiority": "upper_95_percentile_bound_less_than_zero"},
              "holdout_inference_started": False, "holdout_evaluated": False,
              "post_holdout_tuning": False, "secondary_comparisons": "descriptive_only",
              "git_commit": subprocess.run(["git", "-c", "safe.directory=*", "rev-parse", "HEAD"],
                                           check=True, capture_output=True, text=True).stdout.strip()}
    source = {p.relative_to(PROJECT).as_posix(): p for p in (PROJECT / "src").rglob("*.py")}
    source.update({p.relative_to(PROJECT).as_posix(): p for p in (PROJECT / "configs").rglob("*.yaml")})
    source[DESIGN.relative_to(PROJECT).as_posix()] = DESIGN
    source["scripts/freeze_independent_holdout.py"] = PROJECT / "scripts/freeze_independent_holdout.py"
    return publish_lock(root, references, source, policy, output)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/context/p66-independent"))
    parser.add_argument("--output", type=Path, default=Path("outputs/context/p66-independent/holdout-protocol"))
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    result = verify_lock(args.root, args.output) if args.verify_only else freeze(args.root, args.output)
    print(json.dumps({"identity": result["identity"], "verified": True, "holdout_evaluated_at_freeze": False}), flush=True)


if __name__ == "__main__":
    main()

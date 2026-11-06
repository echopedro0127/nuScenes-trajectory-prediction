"""Log-disjoint resplitting and upstream exposure audit for a new training chain."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from trajectory_prediction.config import data_config_sha256, load_yaml
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.data.cache import load_cache, save_cache
from trajectory_prediction.utils.provenance import sha256_file

ROLES = ("independent_fit", "independent_select", "independent_holdout")
SPLIT_SEED = 20271107


def partition_scene_groups(scene_names, records):
    lookup = {}
    for record in records:
        name, group = record["name"], record["log_token"]
        if (not isinstance(name, str) or not name or not isinstance(group, str) or not group
                or name in lookup):
            raise ValueError("unique scene names and nonempty log tokens required")
        lookup[name] = group
    names = sorted(set(scene_names))
    if not names or any(name not in lookup for name in names):
        raise ValueError("source scene missing from metadata")
    groups = sorted({lookup[name] for name in names}, key=lambda token: (
        hashlib.sha256(f"p66c:{SPLIT_SEED}:{token}".encode()).hexdigest(), token))
    if len(groups) < 7:
        raise ValueError("at least seven log groups required")
    a, b = int(len(groups) * .70), int(len(groups) * .15)
    grouped = (groups[:a], groups[a:a + b], groups[a + b:])
    return {role: [name for name in names if lookup[name] in set(logs)]
            for role, logs in zip(ROLES, grouped, strict=True)}


def _validate_keys(arrays):
    keys = list(zip(arrays["instance_token"].tolist(), arrays["sample_token"].tolist(), strict=True))
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate instance/sample keys in source cache")


def _audit_upstream(manifest, checkpoint, gate):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    meta = payload.get("metadata", {})
    contract = gate.get("contract", {})
    model = contract.get("base_hashes", {}).get("model_identity", {})
    if (payload.get("checkpoint_schema") != 1 or meta.get("train_split") != "train"
            or meta.get("val_split") != "train_val" or meta.get("history_steps") != 5
            or meta.get("data_config_sha256") != manifest["preprocessing_config_sha256"]
            or gate.get("contract_sha256") != digest_json(contract)
            or contract.get("split") != "train" or model.get("model_name") != "gru"
            or model.get("history_seconds") != 2. or model.get("history_steps") != 5
            or model.get("checkpoint_sha256") != sha256_file(Path(checkpoint))
            or model.get("data_config_sha256") != meta.get("data_config_sha256")
            or model.get("model_config_sha256") != meta.get("model_config_sha256")):
        raise ValueError("upstream checkpoint/cache training identity mismatch")


def prepare(source_cache, dataroot, checkpoint, gate_manifest, output_root):
    root = Path(output_root).resolve()
    if root.exists():
        raise FileExistsError("independent output directory already exists")
    source_cache, dataroot = Path(source_cache).resolve(), Path(dataroot).resolve()
    checkpoint, gate_manifest = Path(checkpoint).resolve(), Path(gate_manifest).resolve()
    arrays, original = load_cache(source_cache, "train")
    _validate_keys(arrays)
    scene_path = dataroot / original["dataset_version"] / "scene.json"
    records = json.loads(scene_path.read_text(encoding="utf-8"))
    partitions = partition_scene_groups(arrays["scene_name"].tolist(), records)
    _audit_upstream(original, checkpoint, json.loads(gate_manifest.read_text(encoding="utf-8")))
    input_files = {
        "source_manifest": source_cache / "manifest.json", "scene_metadata": scene_path,
        "old_gru": checkpoint, "gate_manifest": gate_manifest,
    }
    project = Path(__file__).resolve().parents[3]
    input_files.update({name: project / name for name in (
        "src/trajectory_prediction/data/independent_split.py",
        "scripts/prepare_independent_validation.py",
        "docs/superpowers/specs/2026-11-07-causal-context-independent-upstream-design.md",
    )})
    sources = {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in input_files.items()}
    protocol = {"split_seed": SPLIT_SEED, "grouping": "log_token", "fractions": [.70, .15, .15],
                "source_split": "train", "partitions": partitions, "sources": sources}
    identity = digest_json(protocol)
    config = {"version": original["dataset_version"], "dataroot": str(dataroot),
              "splits": list(ROLES), "history_seconds": 2., "future_seconds": 6.,
              "frequency_hz": 2, "cache_root": "cache", "partition_identity": identity}
    config_hash = data_config_sha256(config)
    root.mkdir(parents=True, exist_ok=False)
    (root / "data.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (root / "partitions.json").write_text(json.dumps(protocol, indent=2) + "\n", encoding="utf-8")
    lookup = {row["name"]: row["log_token"] for row in records}
    roles = {}
    for role in ROLES:
        index = np.flatnonzero(np.isin(arrays["scene_name"], partitions[role]))
        subset = {name: values[index] for name, values in arrays.items()}
        manifest = {"schema_version": 1, "split": role, "dataset_version": original["dataset_version"],
                    "nuscenes_devkit_version": original.get("nuscenes_devkit_version"),
                    "preprocessing_config_sha256": config_hash, "partition_identity": identity,
                    "parent_npz_sha256": original["npz_sha256"],
                    "history_seconds": 2., "future_seconds": 6., "frequency_hz": 2}
        save_cache(root / "cache" / role, subset, manifest)
        roles[role] = {"sample_count": len(index), "scene_count": len(partitions[role]),
                       "log_count": len({lookup[name] for name in partitions[role]})}
    exposed = len(set(partitions["independent_holdout"]) & set(arrays["scene_name"]))
    report = {"schema_version": 1, "partition_identity": identity, "partitions": partitions,
              "sources": sources, "source_cache": str(source_cache), "roles": roles,
              "old_gru_exposed_holdout_scenes": exposed, "old_upstream_reusable": exposed == 0,
              "requires_retrain": ["gru", "map_corrector", "quality_gate", "RH", "consistency_heads"],
              "holdout_evaluated": False, "independent_chain_complete": False,
              "scope": "resplit_existing_training_data_not_newly_collected_blind_test"}
    (root / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    _verify_outputs(root, report)
    files = ["data.yaml", "partitions.json", "report.json", *[f"cache/{r}/manifest.json" for r in ROLES]]
    marker = {"sha256": {name: sha256_file(root / name) for name in files}}
    temporary = root / ".completed.tmp"
    temporary.write_text(json.dumps(marker, indent=2) + "\n", encoding="utf-8")
    temporary.replace(root / "completed.json")
    return report


def _verify_outputs(root, report):
    for source in report["sources"].values():
        if sha256_file(Path(source["path"])) != source["sha256"]:
            raise ValueError("independent source checksum mismatch")
    protocol = json.loads((root / "partitions.json").read_text(encoding="utf-8"))
    config = load_yaml(root / "data.yaml")
    if digest_json(protocol) != report["partition_identity"] or config["partition_identity"] != report["partition_identity"]:
        raise ValueError("partition identity mismatch")
    original, manifest = load_cache(Path(report["source_cache"]), "train")
    _validate_keys(original)
    records = json.loads(Path(report["sources"]["scene_metadata"]["path"]).read_text(encoding="utf-8"))
    partitions = partition_scene_groups(original["scene_name"].tolist(), records)
    if partitions != protocol["partitions"] or partitions != report["partitions"]:
        raise ValueError("partition replay mismatch")
    _audit_upstream(manifest, Path(report["sources"]["old_gru"]["path"]),
                    json.loads(Path(report["sources"]["gate_manifest"]["path"]).read_text(encoding="utf-8")))
    exposed = len(set(partitions["independent_holdout"]) & set(original["scene_name"]))
    if (report["old_gru_exposed_holdout_scenes"] != exposed
            or report["old_upstream_reusable"] != (exposed == 0)
            or report["independent_chain_complete"] is not False):
        raise ValueError("upstream exposure conclusion mismatch")
    lookup = {row["name"]: row["log_token"] for row in records}
    used_rows, used_logs = set(), set()
    for role in ROLES:
        index = np.flatnonzero(np.isin(original["scene_name"], partitions[role]))
        logs = {lookup[name] for name in partitions[role]}
        if used_rows & set(index.tolist()) or used_logs & logs:
            raise ValueError("independent row/log overlap")
        used_rows.update(index.tolist())
        used_logs.update(logs)
        actual, meta = load_cache(root / "cache" / role, role, data_config_sha256(config))
        if meta["parent_npz_sha256"] != manifest["npz_sha256"] or meta["partition_identity"] != report["partition_identity"]:
            raise ValueError("independent parent cache mismatch")
        for name in original:
            np.testing.assert_array_equal(actual[name], original[name][index])
        expected = {"sample_count": len(index), "scene_count": len(partitions[role]), "log_count": len(logs)}
        if expected != report["roles"][role]:
            raise ValueError("independent count mismatch")
    if len(used_rows) != len(original["sample_token"]) or report["holdout_evaluated"]:
        raise ValueError("independent coverage/evaluation scope mismatch")


def verify_prepared(root):
    root = Path(root).resolve()
    marker = json.loads((root / "completed.json").read_text(encoding="utf-8"))
    for name, digest in marker["sha256"].items():
        if sha256_file(root / name) != digest:
            raise ValueError("independent output checksum mismatch")
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    _verify_outputs(root, report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source-cache", "dataroot", "upstream-checkpoint", "gate-manifest"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/context/p66-independent"))
    args = parser.parse_args(argv)
    report = prepare(args.source_cache, args.dataroot, args.upstream_checkpoint, args.gate_manifest, args.output_root)
    print(json.dumps({"roles": report["roles"], "exposed_holdout_scenes": report["old_gru_exposed_holdout_scenes"],
                      "old_upstream_reusable": report["old_upstream_reusable"], "partition_identity": report["partition_identity"]}, indent=2))
    return 0

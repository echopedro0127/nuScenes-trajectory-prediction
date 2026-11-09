"""One protocol-bound, read-only holdout inference and complete evaluation."""
import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import IDENTIFIERS, digest_json, key_hash
from trajectory_prediction.context.gate_preparation import audit_candidates, generate_candidates
from trajectory_prediction.context.maps import MapConfig
from trajectory_prediction.context.mmap_cache import open_sidecar
from trajectory_prediction.context.neighbors import NeighborConfig
from trajectory_prediction.context.preparation import context_contract, load_predictions
from trajectory_prediction.context.scaled_preparation import file_provenance, prepare_split
from trajectory_prediction.context.streaming_index import CompactMetadata
from trajectory_prediction.data.cache import load_cache
from trajectory_prediction.data.independent_split import verify_prepared
from trajectory_prediction.evaluation.holdout_statistics import paired_log_bootstrap
from trajectory_prediction.evaluation.independent_protocol import PROJECT, verify_lock
from trajectory_prediction.evaluation.runner import _mapping_sha256, _predict_local
from trajectory_prediction.online.batched_evaluation import forecast_metrics
from trajectory_prediction.online.context_training import choose_device, shared_contract
from trajectory_prediction.online.gate_dataset import GateDataset, load_gate_cache, save_gate_cache
from trajectory_prediction.online.horizon_features import HorizonDataset
from trajectory_prediction.online.horizon_gate import measure_horizon
from trajectory_prediction.online.horizon_training import configure_runtime
from trajectory_prediction.online.independent_consistency_training import (
    load_independent_consistency,
)
from trajectory_prediction.online.independent_gate_training import load_independent_gate
from trajectory_prediction.online.independent_horizon_training import load_independent_horizon
from trajectory_prediction.online.independent_map_training import load_pair
from trajectory_prediction.online.quality_gate import GateConfig
from trajectory_prediction.online.refinement_training import measure_model
from trajectory_prediction.utils.provenance import sha256_file
from trajectory_prediction.utils.seed import seed_everything

ROLE = "independent_holdout"
FIELDS = ("ade", "fde", "jitter")


def execution_sources():
    names = ("src/trajectory_prediction/evaluation/independent_holdout.py",
             "src/trajectory_prediction/evaluation/holdout_statistics.py", "scripts/evaluate_independent_holdout.py")
    return {name: sha256_file(PROJECT / name) for name in names}


def begin_run(output, binding):
    output = Path(output)
    path = output / "evaluation_started.json"
    identity = digest_json(binding)
    if path.exists():
        record = json.loads(path.read_text())
        if record.get("identity") != identity or record.get("binding") != binding:
            raise ValueError("holdout execution binding changed")
        return record
    if output.exists() and any(output.iterdir()):
        raise ValueError("unbound holdout outputs already exist")
    output.mkdir(parents=True, exist_ok=True)
    record = {"identity": identity, "binding": binding}
    temporary = output / ".started.tmp"
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)
    return record


def _scope(root, protocol):
    proof = verify_prepared(root)
    arrays, manifest = load_cache(root / "cache" / ROLE, ROLE)
    policy = protocol["body"]["policy"]
    if proof["partition_identity"] != policy["partition_identity"] or proof["roles"][ROLE] != policy["holdout_counts"]:
        raise ValueError("holdout partition/count identity mismatch")
    gate_path = root / policy["models"]["quality_gate"]["path"]
    gate = torch.load(gate_path, map_location="cpu", weights_only=True)
    inputs = gate["metadata"]["independent_inputs"]
    scenes = set(arrays["scene_name"].tolist())
    records = json.loads(Path(proof["sources"]["scene_metadata"]["path"]).read_text(encoding="utf-8"))
    lookup = {r["name"]: r["log_token"] for r in records}
    logs = np.array([lookup[name] for name in arrays["scene_name"]])
    if (scenes != set(proof["partitions"][ROLE]) or scenes & set(inputs["train_scenes"] + inputs["selection_scenes"])
            or set(logs) & set(inputs["role_logs"]["independent_fit"] + inputs["role_logs"]["independent_select"])
            or len(set(logs)) != policy["holdout_counts"]["log_count"]):
        raise ValueError("holdout scene/log overlap or coverage mismatch")
    return arrays, manifest, inputs, logs


def _models(root, policy, gate_inputs, device):
    paths = {name: root / ref["path"] for name, ref in policy["models"].items()}
    map_meta = json.loads((paths["map_pair"].parent / "run.json").read_text())["input_metadata"]
    pair, _ = load_pair(paths["map_pair"], map_meta)
    gate, _ = load_independent_gate(paths["quality_gate"], gate_inputs, device=device)
    rh_payload = torch.load(paths["RH"], map_location="cpu", weights_only=True)
    rh, _ = load_independent_horizon(paths["RH"], rh_payload["metadata"]["independent_inputs"], device=device)
    adaptive_payload = torch.load(paths["revision_adaptive"], map_location="cpu", weights_only=True)
    adaptive, _ = load_independent_consistency(paths["revision_adaptive"], adaptive_payload["metadata"]["independent_inputs"],
                                               arm="revision_adaptive", device=device)
    return pair, gate, rh, adaptive


def _predictions(root, arrays, trajectory, policy, binding, device):
    output = root / "predictions" / ROLE / "gru_eval"
    marker = output / "completed.json"
    expected = {"protocol_identity": binding["protocol_identity"], "implementation_identity": digest_json(binding)}
    if marker.exists():
        record = json.loads(marker.read_text())
        if any(record.get(k) != v for k, v in expected.items()):
            raise ValueError("holdout prediction binding mismatch")
        for name, digest in record["sha256"].items():
            if sha256_file(output / name) != digest:
                raise ValueError("holdout prediction checksum mismatch")
    else:
        if output.exists():
            raise ValueError("incomplete prediction stage; preserve and investigate before recovery")
        model_config = load_yaml(PROJECT / "configs/models/gru.yaml")
        prediction = _predict_local(model_config["model"], arrays, 5, root / policy["models"]["gru"]["path"], device)
        output.mkdir(parents=True, exist_ok=False)
        np.savez_compressed(output / "predictions.npz", prediction=prediction, target=arrays["future_xy"])
        with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(IDENTIFIERS)
            writer.writerows(zip(*(arrays[name].tolist() for name in IDENTIFIERS), strict=True))
        provenance = {"model_name": "gru", "history_seconds": 2., "history_steps": 5, "split": ROLE,
                      "data_config_sha256": trajectory["preprocessing_config_sha256"],
                      "model_config_sha256": _mapping_sha256(model_config),
                      "checkpoint_sha256": policy["models"]["gru"]["sha256"]}
        (output / "metrics.json").write_text(json.dumps({"provenance": provenance, "static_metrics_not_computed": True}, indent=2) + "\n")
        record = {**expected, "sha256": {name: sha256_file(output / name) for name in ("predictions.npz", "per_sample.csv", "metrics.json")}}
        marker.write_text(json.dumps(record, indent=2) + "\n")
    return output / "predictions.npz"


def _prepare(root, protocol, binding, device):
    arrays, trajectory, gate_inputs, logs = _scope(root, protocol)
    policy = protocol["body"]["policy"]
    config = load_yaml(root / "data.yaml")
    prediction_path = _predictions(root, arrays, trajectory, policy, binding, device)
    source = file_provenance(config["dataroot"], trajectory["dataset_version"])
    expected_map = context_contract(arrays, trajectory, source, MapConfig())
    fit_map = gate_inputs["candidate_contracts"]["independent_fit"]["map_contract"]
    if shared_contract(expected_map) != shared_contract(fit_map):
        raise ValueError("holdout map source differs from frozen training source")
    context_root = root / "context" / ROLE
    if not (context_root / "neighbors/mmap_manifest.json").exists():
        if context_root.exists():
            raise ValueError("incomplete context stage; preserve and investigate before recovery")
        metadata = CompactMetadata(config["dataroot"], trajectory["dataset_version"], root / "context/metadata.sqlite")
        prepare_split(metadata, root / "cache" / ROLE, ROLE, context_root, MapConfig(), NeighborConfig())
    maps = open_sidecar(context_root / "map", expected_map)
    base, hashes = load_predictions(prediction_path, arrays, expected_map)
    if hashes["model_identity"] != gate_inputs["candidate_contracts"]["independent_fit"]["base_hashes"]["model_identity"]:
        raise ValueError("holdout GRU identity differs from frozen training source")
    models = _models(root, policy, gate_inputs, device)
    candidate_root = root / "candidates" / ROLE
    contract = {"schema": "independent_holdout_candidates_v1", "split": ROLE, "seed": 2027,
                "protocol_identity": protocol["identity"], "implementation_identity": digest_json(binding),
                "map_contract": expected_map, "base_hashes": hashes, "keys_sha256": key_hash(arrays),
                "gate_config": policy["gate_config"], "conditions": policy["conditions"],
                "models": policy["models"]}
    pair = models[0]
    if not (candidate_root / "manifest.json").exists():
        if candidate_root.exists():
            raise ValueError("incomplete candidate stage; preserve and investigate before recovery")
        values = generate_candidates(arrays, maps, base, pair["original"], pair["map"], root=candidate_root,
                                     config=GateConfig(**policy["gate_config"]), device=device)
        audit_candidates(arrays, maps, base, pair["original"], pair["map"], values, device=device)
        save_gate_cache(candidate_root, values, contract)
    data, manifest = load_gate_cache(candidate_root, contract)
    for name, value in {**{name: arrays[name] for name in IDENTIFIERS}, "target": arrays["future_xy"],
                        "history": arrays["history_xy"], "base": base}.items():
        if not np.array_equal(data[name], value):
            raise ValueError("holdout candidate parent arrays mismatch")
    audit_candidates(arrays, maps, base, pair["original"], pair["map"], data, device=device)
    references = {"trajectory": root / "cache" / ROLE / "manifest.json",
                  "predictions": prediction_path.parent / "completed.json", "map": context_root / "map/manifest.json",
                  "candidates": candidate_root / "manifest.json"}
    preparation = {"protocol_identity": protocol["identity"], "candidate_contract_sha256": manifest["contract_sha256"],
                   "inputs": {str(path.relative_to(root).as_posix()): sha256_file(path) for path in references.values()}}
    return data, logs, models, preparation


def _measure(data, models, device):
    _, gate, rh, adaptive = models
    metrics = {"cv": np.stack([forecast_metrics(data["cv"], data["target"])] * 6),
               "original": np.stack([forecast_metrics(data["original"], data["target"])] * 6),
               "map": np.stack([forecast_metrics(data["map"][c], data["target"]) for c in range(6)])}
    metrics["quality_gate"], gate_weights = measure_model(GateDataset(data), gate, device=device)
    horizon = HorizonDataset(data)
    metrics["RH"], rh_weights = measure_horizon(horizon, rh, device=device)
    metrics["revision_adaptive"], adaptive_weights = measure_horizon(horizon, adaptive, device=device)
    for name in ("quality_gate", "RH", "revision_adaptive"):
        if not np.array_equal(metrics[name][4], metrics["cv"][4]):
            raise ValueError("holdout missing-map metric fallback mismatch")
    if gate_weights[4].any() or rh_weights[4].any() or adaptive_weights[4].any():
        raise ValueError("holdout missing-map weights must be zero")
    return metrics, {"quality_gate": gate_weights, "RH": rh_weights, "revision_adaptive": adaptive_weights}


def summarize(metrics, logs, policy):
    if set(metrics) != set(policy["strategies"]):
        raise ValueError("all frozen strategies must be reported")
    n = len(logs)
    for values in metrics.values():
        if values.shape != (6, n, 11, 3) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("complete finite holdout metrics required")
    difference = (metrics["RH"][..., 0] - metrics["cv"][..., 0]).mean((0, 2))
    settings = policy["paired_bootstrap"]
    primary, draws = paired_log_bootstrap(difference, logs, replicates=settings["replicates"], seed=settings["seed"])
    if primary["log_count"] != policy["holdout_counts"]["log_count"]:
        raise ValueError("bootstrap log count differs from frozen protocol")
    def means(values):
        return dict(zip(FIELDS, values.mean((0, 1, 2)).tolist(), strict=True))
    report = {"primary": primary, "decision": "RH_improvement_supported" if primary["superior"] else "RH_superiority_not_established",
              "aggregate": {name: means(values) for name, values in metrics.items()},
              "conditions": {condition: {name: dict(zip(FIELDS, values[c].mean((0, 1)).tolist(), strict=True))
                                          for name, values in metrics.items()} for c, condition in enumerate(policy["conditions"])},
              "logs": {log: {"trajectories": int((logs == log).sum()), "strategies":
                      {name: means(values[:, logs == log]) for name, values in metrics.items()}} for log in sorted(set(logs.tolist()))},
              "holdout_counts": policy["holdout_counts"], "holdout_evaluated": True,
              "independent_chain_complete": True, "secondary_comparisons": "descriptive_only"}
    return report, draws


def verify_results(root, output):
    protocol = verify_lock(root, root / "holdout-protocol")
    started = json.loads((output / "evaluation_started.json").read_text())
    if (started["identity"] != digest_json(started["binding"]) or started["binding"]["protocol_identity"] != protocol["identity"]
            or started["binding"]["sources"] != execution_sources()):
        raise ValueError("holdout implementation binding mismatch")
    marker = json.loads((output / "completed.json").read_text())
    if marker["implementation_identity"] != started["identity"]:
        raise ValueError("holdout completion binding mismatch")
    for name, digest in marker["sha256"].items():
        if sha256_file(output / name) != digest:
            raise ValueError("holdout output checksum mismatch")
    preparation = json.loads((output / "preparation.json").read_text())
    for name, digest in preparation["inputs"].items():
        if sha256_file(root / name) != digest:
            raise ValueError("holdout input checksum mismatch")
    arrays, _, _, logs = _scope(root, protocol)
    with np.load(output / "metrics.npz", allow_pickle=False) as archive:
        metrics = {name: archive[name] for name in archive.files}
    rebuilt, draws = summarize(metrics, logs, protocol["body"]["policy"])
    report = json.loads((output / "results.json").read_text())
    for key, value in rebuilt.items():
        if report[key] != value:
            raise ValueError("holdout statistics replay mismatch")
    if len(arrays["sample_token"]) != report["holdout_counts"]["sample_count"]:
        raise ValueError("holdout result sample count mismatch")
    np.testing.assert_array_equal(draws, np.load(output / "bootstrap_draws.npy"))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/context/p66-independent"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    configure_runtime()
    seed_everything(2027)
    torch.set_num_threads(2)
    root, device = args.root.resolve(), choose_device(args.device)
    output = root / "holdout-evaluation"
    protocol = verify_lock(root, root / "holdout-protocol")
    if args.verify_only:
        report = verify_results(root, output)
        binding = json.loads((output / "evaluation_started.json").read_text())["binding"]
        data, _, models, _ = _prepare(root, protocol, binding, device)
        replay, _ = _measure(data, models, device)
        with np.load(output / "metrics.npz") as saved:
            for name, value in replay.items():
                np.testing.assert_array_equal(value, saved[name])
        print(json.dumps({"verified_full_replay": True, "decision": report["decision"], "primary": report["primary"]}), flush=True)
        return
    if (output / "completed.json").exists():
        print(json.dumps({"reused_verified_completion": True, "primary": verify_results(root, output)["primary"]}), flush=True)
        return
    _scope(root, protocol)
    if not (output / "evaluation_started.json").exists() and any((root / folder / ROLE).exists() for folder in ("predictions", "context", "candidates")):
        raise ValueError("unbound holdout inputs already exist")
    binding = {"protocol_identity": protocol["identity"], "sources": execution_sources(),
               "runtime": {"device": str(device), "torch": str(torch.__version__), "numpy": str(np.__version__),
                           "cublas_workspace_config": ":4096:8"}}
    started = begin_run(output, binding)
    begun = time.perf_counter()
    data, logs, models, preparation = _prepare(root, protocol, binding, device)
    metrics, weights = _measure(data, models, device)
    report, draws = summarize(metrics, logs, protocol["body"]["policy"])
    report.update(protocol_identity=protocol["identity"], implementation_identity=started["identity"], seconds=time.perf_counter() - begun)
    np.savez_compressed(output / "metrics.npz", **metrics)
    np.savez_compressed(output / "weights.npz", **weights)
    np.save(output / "bootstrap_draws.npy", draws)
    (output / "preparation.json").write_text(json.dumps(preparation, indent=2) + "\n")
    (output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    if execution_sources() != binding["sources"]:
        raise ValueError("holdout execution source changed during run")
    verify_lock(root, root / "holdout-protocol")
    names = ("evaluation_started.json", "preparation.json", "metrics.npz", "weights.npz", "bootstrap_draws.npy", "results.json")
    (output / "completed.json").write_text(json.dumps({"protocol_identity": protocol["identity"],
        "implementation_identity": started["identity"], "sha256": {name: sha256_file(output / name) for name in names}}, indent=2) + "\n")
    verify_results(root, output)
    print(json.dumps({"completed": True, "decision": report["decision"], "primary": report["primary"], "aggregate": report["aggregate"]}), flush=True)


if __name__ == "__main__":
    main()

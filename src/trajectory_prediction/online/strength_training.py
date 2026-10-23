"""P6.5 identity-bound matched strength training; no validation-set access."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.context.gate_preparation import load_verified_gate_cache
from trajectory_prediction.online.consistency_training import (
    load_consistency,
    run_consistency_epoch,
)
from trajectory_prediction.online.consistency_training import (
    training_sources as consistency_sources,
)
from trajectory_prediction.online.consistency_training import (
    validate_parent_references as validate_horizon_references,
)
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.horizon_evaluation import load_horizon_selection
from trajectory_prediction.online.horizon_gate import (
    VARIANTS,
    HorizonGate,
    blend_horizon,
    feature_mask,
    measure_horizon,
)
from trajectory_prediction.online.horizon_training import load_horizon as load_parent_horizon
from trajectory_prediction.online.horizon_training import training_sources as horizon_sources
from trajectory_prediction.online.innovation_consistency import (
    InnovationDataset,
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
from trajectory_prediction.online.strength_consistency import ARMS, strength_settings
from trajectory_prediction.utils.provenance import sha256_file
from trajectory_prediction.utils.seed import seed_everything

DEFAULT_SETTINGS = {"epochs": 10, "patience": 3, "batch_size": 4096, "learning_rate": .001,
    "weight_decay": .0001, "smoothness_weight": .05, "hidden_dim": 16, "input_dim": 19,
    "revision_weight": 0., "candidate": "map"}
CONFIG = Path(__file__).parents[3] / "configs/models/strength_consistency.yaml"


def configure_runtime():
    current = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if current not in (None, ":4096:8") or (current is None and torch.cuda.is_initialized()):
        raise ValueError("CuBLAS requires frozen :4096:8 configuration before CUDA initialization")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def training_sources():
    package = Path(__file__).parents[1]
    return {"execution": {name: sha256_file(package/name) for name in (
        "online/strength_consistency.py", "online/strength_training.py")},
        "config": sha256_file(CONFIG), "parents": parent_sources(), "horizon": horizon_sources(), "consistency": consistency_sources()}


def validate_role(model, meta, arm, expected):
    if (arm not in ARMS or meta.get("arm") != arm or meta.get("variant") != "RH" or model.variant != "RH"
        or meta.get("parameters") != 337 or any(meta.get(k) != v for k,v in expected.items())
        or not 1 <= meta.get("best_epoch",0) <= meta.get("completed_epochs",0) <= expected["epochs"]
        or not torch.equal(model.feature_mask.cpu(), feature_mask("RH"))):
        raise ValueError("consistency model role/settings mismatch")



def train_strength(
    train,
    selection,
    metadata,
    output_dir,
    *,
    initial,
    arm,
    epochs=10,
    batch_size=4096,
    patience=3,
    seed=2027,
    learning_rate=0.001,
    device=None,
):
    if arm not in ARMS:
        raise ValueError("unknown consistency role")
    variant = "RH"
    train_mean_weight = train.train_mean_weight
    strength = strength_settings(arm, train_mean_weight)
    revision_weight = strength["revision_weight"]
    if set(train.scenes) & set(selection.scenes):
        raise ValueError("refinement scene overlap")
    if train.candidate != selection.candidate:
        raise ValueError("refinement candidate mismatch")
    if (
        any(type(v) is not int or v < 1 for v in (epochs, batch_size, patience))
        or any(not math.isfinite(v) for v in (revision_weight, learning_rate))
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
        "arm": arm,
        "train_mean_weight": train_mean_weight,
        **strength,
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
        validate_role(model, payload["metadata"], arm, settings)
        return model.eval()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
    generator = np.random.default_rng(seed)
    best, bad, best_state, logs = float("inf"), 0, None, []
    for epoch in range(1, epochs + 1):
        started = time.perf_counter()
        training = run_consistency_epoch(
            model,
            train,
            generator.permutation(train.size),
            batch_size,
            device,
            optimizer,
            revision_weight,
            strength["revision_kind"],
            train_mean_weight,
        )
        values, _ = measure_horizon(selection, model, device=device, batch_size=batch_size)
        ade = float(values[..., 0].mean())
        row = {
            "seed": seed,
            "candidate": train.candidate,
        "input_dim": 19,
        "variant": variant,
            "arm": arm,
            "lambda": revision_weight,
            **strength,
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
        "architecture": "matched_strength_v1",
        "run_manifest_sha256": sha256_file(run),
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


def validate_parent_reports(meta):
    from trajectory_prediction.analysis.consistency_report import (
        report_sources as consistency_report_sources,
    )
    from trajectory_prediction.analysis.horizon_report import (
        report_sources as horizon_report_sources,
    )
    from trajectory_prediction.online.consistency_evaluation import load_consistency_selection
    from trajectory_prediction.online.horizon_evaluation import load_horizon_selection
    for name in ("horizon_selection","horizon_report","consistency_selection","consistency_report"):
        ref=meta[name]
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("strength parent reference changed: "+name)
    horizon=load_horizon_selection(Path(meta["horizon_selection"]["path"]).parent)
    selection=load_consistency_selection(Path(meta["consistency_selection"]["path"]).parent)
    old_report=json.loads(Path(meta["horizon_report"]["path"]).read_text())
    report=json.loads(Path(meta["consistency_report"]["path"]).read_text())
    if (horizon["decision"]["accuracy"] != "RH" or old_report["selection"] != horizon
        or not old_report["val_run"] or old_report["sources"] != horizon_report_sources()):
        raise ValueError("strength frozen P6.3 parent selection/report mismatch")
    if (selection["decision"]["balanced"] != "revision_fixed" or selection["decision"]["advance"]
        or report["selection"] != selection or report["val_run"] or report["sources"] != consistency_report_sources()):
        raise ValueError("strength frozen P6.4 parent selection/report mismatch")


def validate_parent_references(meta, device):
    validate_horizon_references(meta, device)
    validate_parent_reports(meta)
    ref=meta["consistency_zero"]
    if sha256_file(Path(ref["path"])) != ref["sha256"]:
        raise ValueError("strength parent reference changed: consistency_zero")
    zero,_=load_consistency(meta["consistency_zero"]["path"],meta["candidate_contracts"]["train_val"],
        meta["selection_scenes"],device,split="train_val",arm="revision_0",
        train_mean_weight=meta["train_mean_weight"],train_content=meta["train_content"])
    parent,_=load_parent_horizon(meta["parent_rh"]["path"],meta["candidate_contracts"]["train_val"],
        meta["selection_scenes"],device,split="train_val",variant="RH")
    if state_fingerprint(zero.state_dict()) != state_fingerprint(parent.state_dict()):
        raise ValueError("strength parent zero/RH state mismatch")


def load_strength(path, contract, scenes, device, *, split, arm, train_mean_weight, train_content):
    path = Path(path)
    payload = completed_payload(path.parent)
    m = payload["metadata"]
    if (not math.isfinite(train_mean_weight) or not 0 < train_mean_weight <= 1
        or m.get("train_mean_weight") != train_mean_weight or m.get("train_content") != train_content
        or m.get("train_weight_stats", {}).get("mean") != train_mean_weight):
        raise ValueError("consistency trusted training mean/content mismatch")
    validate_parent_references(m, device)
    if (m.get("architecture") != "matched_strength_v1" or m.get("source_code_sha256") != training_sources()
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
    model = HorizonGate("RH", initial)
    if (state_fingerprint(initial) != m["initial_p6_state_sha256"]
        or state_fingerprint(model.state_dict()) != m["initial_state_sha256"]):
        raise ValueError("horizon initial state mismatch")
    model.load_state_dict(payload["model"], strict=True)
    expected = {**DEFAULT_SETTINGS, **strength_settings(arm, train_mean_weight),
        "variant": "RH", "arm": arm, "train_mean_weight": m["train_mean_weight"]}
    validate_role(model, m, arm, expected)
    run = path.parent/"run.json"
    recorded = json.loads(run.read_text())
    if (sha256_file(run) != m["run_manifest_sha256"] or recorded["identity"] != m["run_identity"]
        or recorded["settings"] != {k: m[k] for k in recorded["settings"]}
        or not math.isfinite(m["train_mean_weight"]) or not 0 <= m["train_mean_weight"] <= 1):
        raise ValueError("consistency training manifest/role mismatch")
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
    for name in ("cache-root", "p6-training-root", "horizon-training-root", "horizon-selection-root", "horizon-report-root", "consistency-training-root", "consistency-selection-root", "consistency-report-root", "output-root"):
        parser.add_argument("--"+name, type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    configure_runtime()
    if load_horizon_selection(args.horizon_selection_root)["decision"]["accuracy"] != "RH":
        raise ValueError("consistency requires frozen RH selection")
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
        train, selection = (InnovationDataset(loaded[s][0]) for s in ("train", "train_val"))
        metadata["train_weight_stats"] = {"mean": train.train_mean_weight, "count": int(train.n*10),
            "quantiles": np.quantile(train.revision_weights[:,1:],[0,.05,.5,.95,1]).tolist()}
        reference_path = args.horizon_selection_root/"selection.json"
        metadata["horizon_selection"] = {"path": str(reference_path.resolve()), "sha256": sha256_file(reference_path)}
        parent_path = args.horizon_training_root/f"seed-{seed}"/"RH"/"best.pt"
        metadata["parent_rh"] = {"path": str(parent_path.resolve()), "sha256": sha256_file(parent_path)}
        for name,path in (("horizon_report",args.horizon_report_root/"report.json"),
            ("consistency_selection",args.consistency_selection_root/"selection.json"),
            ("consistency_report",args.consistency_report_root/"report.json"),
            ("consistency_zero",args.consistency_training_root/f"seed-{seed}"/"revision_0"/"best.pt")):
            metadata[name]={"path":str(path.resolve()),"sha256":sha256_file(path)}
        validate_parent_references({**metadata,"train_mean_weight":train.train_mean_weight,
            "train_content":train.fingerprint(),"selection_scenes":selection.scenes},device)
        print(json.dumps({"seed": seed, "train_weight_stats": metadata["train_weight_stats"]}), flush=True)
        full_parity = check_initialization(selection, initial, device)
        parent = QualityGate().to(device).eval()
        parent.load_state_dict(initial)
        keys = np.linspace(0, selection.size-1, 256, dtype=np.int64)
        values = selection.inputs(keys, device)
        for arm in ARMS:
            variant = "RH"
            probe = HorizonGate(variant, initial).to(device).eval()
            with torch.inference_mode():
                old = blend_forecasts(values["cv"], values["map"], parent(values["quality"], values["valid"]))
                new = blend_horizon(values["cv"], values["map"], probe(values["features"], values["valid"], values["mask"]))
                torch.testing.assert_close(new[values["mask"]], old[values["mask"]], atol=1e-5, rtol=1e-6)
                initial_difference = float((new[values["mask"]]-old[values["mask"]]).abs().max())
            train_strength(train, selection, {**metadata, "initial_probe_max_difference": initial_difference, "initial_full_metrics": full_parity[variant]},
                args.output_root/f"seed-{seed}"/arm, initial=initial, arm=arm, seed=seed, device=device, **settings)



if __name__ == "__main__":
    main()

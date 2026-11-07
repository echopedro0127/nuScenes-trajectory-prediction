"""Train the new log-disjoint original/map pair without altering legacy contracts."""
import argparse
import json
import math
import re
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.context.maps import MapConfig
from trajectory_prediction.context.neighbors import NeighborConfig
from trajectory_prediction.context.scaled_preparation import load_scaled_bundle
from trajectory_prediction.data.independent_split import verify_prepared
from trajectory_prediction.online.batched_context import BatchedContext
from trajectory_prediction.online.context_training import (
    choose_device,
    shared_contract,
    validate_scene_metadata,
)
from trajectory_prediction.online.corrector import ResidualCorrector
from trajectory_prediction.online.interaction_corrector import InteractionCorrector
from trajectory_prediction.online.scaled_training import _epoch, _save_torch
from trajectory_prediction.utils.provenance import sha256_file
from trajectory_prediction.utils.seed import seed_everything

ROLES = ("independent_fit", "independent_select")


def validate_metadata(metadata):
    roles = tuple(metadata.get(field, {}).get("contract", {}).get("split")
                  for field in ("train_provenance", "selection_provenance"))
    if roles != ROLES:
        raise ValueError("independent fit/select roles required")
    if not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("partition_identity", ""))):
        raise ValueError("invalid partition identity")
    logs = metadata.get("role_logs", {})
    for role in ROLES:
        values = logs.get(role)
        if not isinstance(values, list) or not values or any(not isinstance(v, str) or not v for v in values) or values != sorted(set(values)):
            raise ValueError("unique sorted independent log groups required")
    if set(logs[ROLES[0]]) & set(logs[ROLES[1]]):
        raise ValueError("fit/select log overlap")
    upstream = metadata.get("upstream_training", {})
    if (upstream.get("train_split"), upstream.get("val_split")) != ROLES or upstream.get("history_steps") != 5:
        raise ValueError("upstream GRU must be retrained on independent fit/select")
    if upstream.get("data_config_sha256") != metadata.get("base_model", {}).get("data_config_sha256"):
        raise ValueError("upstream preprocessing mismatch")
    # Reuse source/scene validation only; this private validation view is never saved.
    legacy_view = deepcopy(metadata)
    for field, role in zip(("train_provenance", "selection_provenance"), ("train", "train_val"), strict=True):
        legacy_view[field]["contract"]["split"] = role
    validate_scene_metadata(legacy_view)


def load_pair(path, expected_metadata):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    if payload.get("checkpoint_schema") != 1 or metadata.get("architecture") != "independent_map_pair_v1":
        raise ValueError("unsupported independent map pair")
    validate_metadata(payload["input_metadata"])
    if digest_json(payload["input_metadata"]) != digest_json(expected_metadata):
        raise ValueError("independent input identity mismatch")
    hidden = metadata["settings"]["hidden_dim"]
    models = nn.ModuleDict({"original": ResidualCorrector(hidden_dim=hidden),
        "map": InteractionCorrector(ResidualCorrector(hidden_dim=hidden), "map", hidden)})
    models.load_state_dict(payload["model"], strict=True)
    for name, value in models["original"].state_dict().items():
        if not torch.equal(value, models["map"].fallback.state_dict()[name]):
            raise ValueError("map fallback differs from new original")
    return models.eval(), payload


def train_pair(train, selection, metadata, output_dir, *, epochs=20, batch_size=4096,
               hidden_dim=64, patience=5, seed=2027, learning_rate=.001, device=None):
    validate_metadata(metadata)
    for value in (epochs, batch_size, hidden_dim, patience):
        if type(value) is not int or value < 1:
            raise ValueError("positive training dimensions required")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("positive finite learning rate required")
    seed_everything(seed)
    for data, key in ((train, "train_scenes"), (selection, "selection_scenes")):
        if sorted(set(data.arrays["scene_name"].tolist())) != metadata[key]:
            raise ValueError("training data scene partition mismatch")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError("use a new independent output directory")
    device = device or torch.device("cpu")
    settings = {"epochs": epochs, "batch_size": batch_size, "hidden_dim": hidden_dim, "patience": patience,
                "seed": seed, "learning_rate": learning_rate, "weight_decay": .0001, "smoothness_weight": .05}
    files = [Path(__file__).with_name(name + ".py") for name in
             ("independent_map_training", "scaled_training", "batched_context", "context_training",
              "corrector", "interaction_corrector")]
    files.extend((Path(__file__).parents[1] / "utils/seed.py", Path(__file__).parents[1] / "data/independent_split.py"))
    source = {str(p.relative_to(Path(__file__).parents[1])): sha256_file(p) for p in files}
    run = {"input_metadata": metadata, "settings": settings, "source": source,
           "train_content": train.fingerprint(), "selection_content": selection.fingerprint()}
    identity = digest_json(run)
    output.mkdir(parents=True, exist_ok=False)
    (output / "run.json").write_text(json.dumps({"identity": identity, **run}, indent=2) + "\n")
    models, progress, logs = nn.ModuleDict(), {}, []
    for name in ("original", "map"):
        seed_everything(seed)
        model = (ResidualCorrector(hidden_dim=hidden_dim) if name == "original" else
                 InteractionCorrector(deepcopy(models["original"]), "map", hidden_dim)).to(device)
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                      lr=learning_rate, weight_decay=.0001)
        generator = np.random.default_rng(seed)
        best, best_state, best_epoch, bad = float("inf"), None, 0, 0
        for epoch in range(1, epochs + 1):
            started = time.perf_counter()
            training = _epoch(model, train, generator.permutation(train.size), batch_size, device, optimizer)
            validation = _epoch(model, selection, np.arange(selection.size), batch_size, device)
            row = {"stage": name, "epoch": epoch, "train_loss": training["loss"],
                   "selection_loss": validation["loss"], "selection_ade": validation["ade"],
                   "seconds": time.perf_counter() - started}
            logs.append(row)
            print(json.dumps(row), flush=True)
            (output / "progress.json").write_text(json.dumps(logs, indent=2) + "\n")
            if validation["ade"] < best:
                best, best_epoch, bad = validation["ade"], epoch, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
            if bad >= patience:
                break
        model.load_state_dict(best_state, strict=True)
        models[name] = model.eval()
        progress[name] = {"best_epoch": best_epoch, "completed_epochs": epoch, "selection_ade": best}
        _save_torch(output / "stages" / (name + ".pt"), {"identity": identity, "model": best_state, **progress[name]})
    payload = {"checkpoint_schema": 1, "input_metadata": metadata, "model": models.state_dict(),
               "metadata": {"architecture": "independent_map_pair_v1", "identity": identity,
                            "settings": settings, "source": source, "stage_progress": progress}}
    _save_torch(output / "best.pt", payload)
    restored, _ = load_pair(output / "best.pt", metadata)
    replay = {name: _epoch(model.to(device), selection, np.arange(selection.size), batch_size, device)
              for name, model in restored.items()}
    for name in replay:
        if not math.isclose(replay[name]["ade"], progress[name]["selection_ade"], rel_tol=0, abs_tol=1e-6):
            raise ValueError("independent checkpoint replay mismatch")
    report = {"identity": identity, "stage_progress": progress, "replay": replay, "epochs": logs,
              "holdout_evaluated": False, "independent_chain_complete": False}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    names = ("run.json", "best.pt", "report.json", "stages/original.pt", "stages/map.pt")
    (output / "completed.json").write_text(json.dumps({"identity": identity,
        "sha256": {name: sha256_file(output / name) for name in names}}, indent=2) + "\n")
    return restored


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/context/p66-independent"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args(argv)
    if args.output_root.exists():
        raise FileExistsError("use a new independent output directory")
    root = args.root.resolve()
    report = verify_prepared(root)
    config = load_yaml(root / "data.yaml")
    checkpoint = root / "gru-seed-2027/best.pt"
    upstream = torch.load(checkpoint, map_location="cpu", weights_only=True)["metadata"]
    loaded = [load_scaled_bundle(root / "cache" / role, root / "context" / role, role,
        config["dataroot"], MapConfig(**load_yaml(Path("configs/context/map.yaml"))),
        NeighborConfig(**load_yaml(Path("configs/context/neighbors.yaml"))),
        root / "predictions" / role / "gru_eval/predictions.npz") for role in ROLES]
    train, maps, _, hashes = loaded[0]
    selection, val_maps, _, val_hashes = loaded[1]
    if hashes["model_identity"]["checkpoint_sha256"] != sha256_file(checkpoint):
        raise ValueError("new upstream checkpoint identity mismatch")
    scenes = json.loads(Path(report["sources"]["scene_metadata"]["path"]).read_text(encoding="utf-8"))
    scene_logs = {row["name"]: row["log_token"] for row in scenes}
    metadata = {"partition_identity": report["partition_identity"], "upstream_training": upstream,
                "base_model": hashes["model_identity"], "shared_contract": shared_contract(maps["contract"]),
                "role_logs": {role: sorted({scene_logs[s] for s in report["partitions"][role]}) for role in ROLES}}
    for prefix, bundle, manifest, base in (("train", train, maps, hashes), ("selection", selection, val_maps, val_hashes)):
        values = sorted(set(bundle[0]["scene_name"].tolist()))
        metadata[prefix + "_scenes"] = values
        metadata[prefix + "_scenes_sha256"] = digest_json(values)
        metadata[prefix + "_provenance"] = {"context_sha256": manifest["npz_sha256"],
                                            "contract": manifest["contract"], "base": base}
    torch.set_num_threads(2)
    train_pair(BatchedContext(*train), BatchedContext(*selection), metadata,
               args.output_root, device=choose_device(args.device))


if __name__ == "__main__":
    main()

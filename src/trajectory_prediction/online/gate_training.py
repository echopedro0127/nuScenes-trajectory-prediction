"""Train only a quality gate; candidates are frozen and labels stay separate."""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.context.gate_preparation import load_verified_gate_cache
from trajectory_prediction.online.context_training import choose_device
from trajectory_prediction.online.corrector import smoothness_loss
from trajectory_prediction.online.gate_dataset import GateDataset
from trajectory_prediction.online.quality_gate import QualityGate, blend_forecasts
from trajectory_prediction.online.scaled_training import _save_torch
from trajectory_prediction.utils.provenance import sha256_file
from trajectory_prediction.utils.seed import seed_everything


def training_sources():
    package = Path(__file__).parents[1]
    names = ("online/gate_training.py", "online/gate_dataset.py", "online/quality_gate.py", "online/corrector.py", "utils/seed.py")
    return {name: sha256_file(package / name) for name in names}


def run_epoch(model, data, indices, batch_size, device, optimizer=None):
    model.train(optimizer is not None)
    totals = np.zeros(3, np.float64)
    for start in range(0, len(indices), batch_size):
        selected = indices[start:start + batch_size]
        inputs = data.inputs(selected, device)
        target = data.labels(selected, device)
        mask = inputs["mask"]
        with torch.set_grad_enabled(optimizer is not None):
            weight = model(inputs["quality"], inputs["valid"])
            forecast = blend_forecasts(inputs["cv"], inputs["map"], weight)
            correction = weight[:,None,None] * (inputs["map"] - inputs["cv"])
            loss = F.smooth_l1_loss(forecast[mask], target[mask]) + .05 * smoothness_loss(correction, mask)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite gate loss")
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
        ade = (torch.linalg.vector_norm(forecast.detach()-target, dim=-1)*mask).sum(1)/mask.sum(1)
        totals += (float(loss.detach())*len(selected), float(ade.double().sum()), len(selected))
    return {"loss": float(totals[0]/totals[2]), "ade": float(totals[1]/totals[2])}


def train_gate(train, selection, metadata, output_dir, *, epochs=20, batch_size=4096, seed=2027, patience=5, learning_rate=.001, device=None):
    if set(train.scenes) & set(selection.scenes):
        raise ValueError("gate train/selection scene overlap")
    if any(type(v) is not int or v < 1 for v in (epochs,batch_size,patience)) or not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("invalid gate training settings")
    device = device or torch.device("cpu")
    settings = {"epochs": epochs,"batch_size": batch_size,"seed": seed,"patience": patience,"learning_rate": learning_rate,"weight_decay": .0001,"smoothness_weight": .05,"hidden_dim": 16}
    source = training_sources()
    identity = digest_json({"metadata": metadata, "settings": settings, "source": source, "train_content": train.fingerprint(), "selection_content": selection.fingerprint()})
    output = Path(output_dir)
    output.mkdir(parents=True,exist_ok=True)
    if (output/"run.json").exists() and json.loads((output/"run.json").read_text())["identity"] != identity:
        raise ValueError("gate run identity mismatch")
    (output/"run.json").write_text(json.dumps({"identity": identity,"settings": settings,"source": source},indent=2)+"\n")
    seed_everything(seed)
    model = QualityGate().to(device)
    if (output/"best.pt").exists():
        payload = torch.load(output/"best.pt",map_location="cpu",weights_only=True)
        if payload.get("checkpoint_schema") != 1 or payload["metadata"].get("run_identity") != identity:
            raise ValueError("gate checkpoint identity mismatch")
        model.load_state_dict(payload["model"],strict=True)
        return model.eval()
    optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=.0001)
    generator = np.random.default_rng(seed)
    best, best_state, bad, logs = float("inf"), None, 0, []
    for epoch in range(1,epochs+1):
        started = time.perf_counter()
        training = run_epoch(model,train,generator.permutation(train.size),batch_size,device,optimizer)
        validation = run_epoch(model,selection,np.arange(selection.size),batch_size,device)
        row = {"seed": seed,"epoch": epoch,"train_loss": training["loss"],"selection_loss": validation["loss"],"selection_ade": validation["ade"],"seconds": time.perf_counter()-started}
        logs.append(row)
        print(json.dumps(row),flush=True)
        (output/"progress.json").write_text(json.dumps(logs,indent=2)+"\n")
        if validation["ade"] < best:
            best,best_epoch,bad = validation["ade"],epoch,0
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        else:
            bad += 1
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    final = {**metadata,**settings,"architecture":"quality_gate_v1","run_identity":identity,"source_code_sha256":source,
        "train_scenes":train.scenes,"selection_scenes":selection.scenes,"best_epoch":best_epoch,"completed_epochs":epoch,
        "selection_ade":best,"parameters":sum(p.numel() for p in model.parameters()),"device":str(device),"torch_version":str(torch.__version__)}
    _save_torch(output/"best.pt",{"checkpoint_schema": 1,"model": best_state,"metadata": final})
    (output/"training.json").write_text(json.dumps({"metadata": final,"epochs": logs},indent=2)+"\n")
    return model.eval()


def load_gate_checkpoint(path, contract, scenes, device):
    payload = torch.load(path,map_location="cpu",weights_only=True)
    meta = payload["metadata"]
    if payload.get("checkpoint_schema") != 1 or meta.get("architecture") != "quality_gate_v1" or meta.get("source_code_sha256") != training_sources():
        raise ValueError("gate checkpoint architecture/source mismatch")
    if set(scenes) & set(meta["train_scenes"]+meta["selection_scenes"]):
        raise ValueError("gate evaluation scene overlap")
    for split in ("train", "train_val"):
        parent = meta["candidate_contracts"][split]
        for name in ("seed","gate_config","conditions","preparation_sources","p4_training_identity"):
            if parent[name] != contract[name]:
                raise ValueError("gate checkpoint candidate contract mismatch")
        if parent["base_hashes"]["model_identity"] != contract["base_hashes"]["model_identity"] or parent["references"]["checkpoint"] != contract["references"]["checkpoint"]:
            raise ValueError("gate checkpoint frozen source mismatch")
    model = QualityGate().to(device)
    model.load_state_dict(payload["model"],strict=True)
    return model.eval(), meta


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root",type=Path,required=True)
    parser.add_argument("--output-root",type=Path,required=True)
    parser.add_argument("--config",type=Path,default=Path("configs/models/quality_gate.yaml"))
    parser.add_argument("--seeds",type=int,nargs="+",default=[2027,2028,2029])
    parser.add_argument("--device",default="auto",choices=("auto","cpu","cuda"))
    args=parser.parse_args(argv)
    torch.set_num_threads(2)
    device = choose_device(args.device)
    settings = load_yaml(args.config)
    for seed in args.seeds:
        loaded = {split:load_verified_gate_cache(args.cache_root/split/("seed-"+str(seed)),split,seed,device=device) for split in ("train","train_val")}
        metadata = {"candidate_contracts":{split:item[1]["contract"] for split,item in loaded.items()},
            "candidate_manifest_sha256":{split:sha256_file(args.cache_root/split/("seed-"+str(seed))/"manifest.json") for split in loaded}}
        if metadata["candidate_contracts"]["train"]["p4_training_identity"] != metadata["candidate_contracts"]["train_val"]["p4_training_identity"]:
            raise ValueError("gate train/selection frozen model mismatch")
        train_gate(*(GateDataset(loaded[split][0]) for split in ("train","train_val")),metadata,args.output_root/("seed-"+str(seed)),seed=seed,device=device,**settings)


if __name__ == "__main__":
    main()

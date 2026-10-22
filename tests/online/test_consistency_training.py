import copy
import json

import pytest
import torch
from test_horizon_gate import example_arrays

from trajectory_prediction.online.consistency_training import train_consistency
from trajectory_prediction.online.horizon_training import train_horizon
from trajectory_prediction.online.innovation_consistency import ARMS, InnovationDataset
from trajectory_prediction.online.quality_gate import QualityGate
from trajectory_prediction.online.refinement_training import completed_payload, state_fingerprint


def datasets():
    a,b=example_arrays(2),example_arrays(2)
    a["scene_name"][:]=["t0","t1"]
    b["scene_name"][:]=["s0","s1"]
    return InnovationDataset(a),InnovationDataset(b)


def test_four_roles_completed_reuse_and_input_identity(tmp_path,monkeypatch):
    torch.set_num_threads(1)
    train,selection=datasets()
    initial=copy.deepcopy(QualityGate().state_dict())
    for arm in ARMS:
        p=tmp_path/arm
        model=train_consistency(train,selection,{},p,initial=initial,arm=arm,epochs=1,batch_size=64)
        meta=completed_payload(p)["metadata"]
        assert meta["arm"]==arm and meta["variant"]=="RH"
        assert meta["train_mean_weight"]==train.train_mean_weight
        again=train_consistency(train,selection,{},p,initial=initial,arm=arm,epochs=1,batch_size=64)
        assert state_fingerprint(again.state_dict())==state_fingerprint(model.state_dict())
    train.revision_weights[0,1]=.123
    with pytest.raises(ValueError,match="identity"):
        train_consistency(train,selection,{},tmp_path/ARMS[0],initial=initial,arm=ARMS[0],epochs=1,batch_size=64)
    with pytest.raises(ValueError,match="overlap"):
        train_consistency(train,train,{},tmp_path/"bad",initial=initial,arm=ARMS[0])


def test_zero_arm_exactly_reproduces_parent_training(tmp_path):
    torch.set_num_threads(1)
    train,selection=datasets()
    initial=QualityGate().state_dict()
    a=train_horizon(train,selection,{},tmp_path/"parent",initial=initial,variant="RH",epochs=2,batch_size=64)
    b=train_consistency(train,selection,{},tmp_path/"new",initial=initial,arm="revision_0",epochs=2,batch_size=64)
    assert state_fingerprint(a.state_dict())==state_fingerprint(b.state_dict())


def test_reuse_rejects_role_mask_and_mean_tampering_with_rehashed_checkpoint(tmp_path):
    from trajectory_prediction.utils.provenance import sha256_file
    torch.set_num_threads(1)
    train,selection=datasets(); initial=QualityGate().state_dict()
    train_consistency(train,selection,{},tmp_path,initial=initial,arm="revision_mean",epochs=1,batch_size=64)
    original=torch.load(tmp_path/"best.pt",weights_only=True)
    for field,value in [("arm","revision_fixed"),("train_mean_weight",.123)]:
        payload=copy.deepcopy(original);payload["metadata"][field]=value
        torch.save(payload,tmp_path/"best.pt")
        log=json.loads((tmp_path/"training.json").read_text());log["metadata"]=payload["metadata"]
        (tmp_path/"training.json").write_text(json.dumps(log))
        marker=json.loads((tmp_path/"completed.json").read_text())
        marker.update(checkpoint_sha256=sha256_file(tmp_path/"best.pt"),training_sha256=sha256_file(tmp_path/"training.json"))
        (tmp_path/"completed.json").write_text(json.dumps(marker))
        with pytest.raises(ValueError,match="role"):
            train_consistency(train,selection,{},tmp_path,initial=initial,arm="revision_mean",epochs=1,batch_size=64)


def test_production_loader_requires_trusted_train_mean_and_content(tmp_path,monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import trajectory_prediction.online.consistency_training as engine
    from trajectory_prediction.utils.provenance import sha256_file
    torch.set_num_threads(1)
    train,selection=datasets();initial=QualityGate().state_dict()
    initializer=tmp_path/"initial";initializer.write_text("frozen")
    contract={"seed":2027,"gate_config":{},"conditions":[],"preparation_sources":{},
        "p4_training_identity":"x","references":{"checkpoint":{}},"base_hashes":{"model_identity":"x"}}
    metadata={"candidate_contracts":{"train":contract,"train_val":contract},"candidate_manifests":{},
        "p6_checkpoint":{"path":str(initializer),"sha256":sha256_file(initializer)},
        "train_weight_stats":{"mean":train.train_mean_weight}}
    train_consistency(train,selection,metadata,tmp_path/"run",initial=initial,arm="revision_mean")
    monkeypatch.setattr(engine,"load_initial",lambda *a:initial)
    monkeypatch.setattr(engine,"validate_parent_references",lambda *a:None,raising=False)
    args={"split":"train_val","arm":"revision_mean","train_mean_weight":train.train_mean_weight,"train_content":train.fingerprint()}
    model,meta=engine.load_consistency(tmp_path/"run"/"best.pt",contract,selection.scenes,torch.device("cpu"),**args)
    assert model.variant=="RH" and meta["train_mean_weight"]==train.train_mean_weight
    folder=tmp_path/"run"; payload=torch.load(folder/"best.pt",weights_only=True)
    payload["metadata"]["train_mean_weight"]=.123
    payload["metadata"]["train_weight_stats"]["mean"]=.123
    run=json.loads((folder/"run.json").read_text());run["settings"]["train_mean_weight"]=.123
    (folder/"run.json").write_text(json.dumps(run))
    payload["metadata"]["run_manifest_sha256"]=sha256_file(folder/"run.json")
    torch.save(payload,folder/"best.pt")
    log=json.loads((folder/"training.json").read_text());log["metadata"]=payload["metadata"]
    (folder/"training.json").write_text(json.dumps(log))
    marker=json.loads((folder/"completed.json").read_text())
    marker.update(checkpoint_sha256=sha256_file(folder/"best.pt"),training_sha256=sha256_file(folder/"training.json"))
    (folder/"completed.json").write_text(json.dumps(marker))
    with pytest.raises(ValueError,match="mean|content"):
        engine.load_consistency(folder/"best.pt",contract,selection.scenes,torch.device("cpu"),**args)


def test_parent_reference_hashes_and_frozen_RH_choice_are_checked(tmp_path,monkeypatch):
    import trajectory_prediction.online.consistency_training as engine
    from trajectory_prediction.utils.provenance import sha256_file
    assert hasattr(engine,"validate_parent_references")
    parent=tmp_path/"parent.pt"; parent.write_text("RH")
    selected=tmp_path/"selection.json";selected.write_text("frozen")
    meta={"parent_rh":{"path":str(parent),"sha256":sha256_file(parent)},
        "horizon_selection":{"path":str(selected),"sha256":sha256_file(selected)},
        "candidate_contracts":{"train_val":{}},"selection_scenes":["s0"]}
    monkeypatch.setattr(engine,"load_horizon_selection",lambda *a:{"decision":{"accuracy":"RH"}})
    monkeypatch.setattr(engine,"load_parent_horizon",lambda *a,**k:None)
    engine.validate_parent_references(meta,torch.device("cpu"))
    parent.write_text("changed")
    with pytest.raises(ValueError,match="reference"):
        engine.validate_parent_references(meta,torch.device("cpu"))
    parent.write_text("RH")
    monkeypatch.setattr(engine,"load_horizon_selection",lambda *a:{"decision":{"accuracy":"H"}})
    with pytest.raises(ValueError,match="RH"):
        engine.validate_parent_references(meta,torch.device("cpu"))

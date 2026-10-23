import copy
import json
from pathlib import Path

import pytest
import torch
from test_consistency_training import datasets

from trajectory_prediction.online.quality_gate import QualityGate
from trajectory_prediction.online.refinement_training import completed_payload, state_fingerprint
from trajectory_prediction.online.strength_consistency import ARMS, strength_settings
from trajectory_prediction.online.strength_training import train_strength


def test_all_strength_roles_reuse_trusted_input_identity_and_overlap(tmp_path):
    torch.set_num_threads(1)
    train,selection=datasets();initial=copy.deepcopy(QualityGate().state_dict())
    for arm in ARMS:
        root=tmp_path/arm
        a=train_strength(train,selection,{},root,initial=initial,arm=arm,epochs=1,batch_size=64)
        meta=completed_payload(root)["metadata"]
        for key,value in strength_settings(arm,train.train_mean_weight).items():
            assert meta[key]==value
        b=train_strength(train,selection,{},root,initial=initial,arm=arm,epochs=1,batch_size=64)
        assert state_fingerprint(a.state_dict())==state_fingerprint(b.state_dict())
    train.revision_weights[0,1]=.123
    with pytest.raises(ValueError,match="identity"):
        train_strength(train,selection,{},tmp_path/ARMS[0],initial=initial,arm=ARMS[0],epochs=1,batch_size=64)
    with pytest.raises(ValueError,match="overlap"):
        train_strength(train,train,{},tmp_path/"overlap",initial=initial,arm=ARMS[0])


def rewrite(root,payload):
    from trajectory_prediction.utils.provenance import sha256_file
    run=json.loads((root/"run.json").read_text())
    for name in ("train_mean_weight","target_mean_weight","revision_kind","revision_weight","arm"):
        run["settings"][name]=payload["metadata"][name]
    (root/"run.json").write_text(json.dumps(run))
    payload["metadata"]["run_manifest_sha256"]=sha256_file(root/"run.json")
    torch.save(payload,root/"best.pt")
    log=json.loads((root/"training.json").read_text());log["metadata"]=payload["metadata"]
    (root/"training.json").write_text(json.dumps(log))
    marker=json.loads((root/"completed.json").read_text())
    marker.update(checkpoint_sha256=sha256_file(root/"best.pt"),training_sha256=sha256_file(root/"training.json"))
    (root/"completed.json").write_text(json.dumps(marker))


def test_production_loader_valid_before_rehashed_role_mean_strength_tampering(tmp_path,monkeypatch):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import trajectory_prediction.online.strength_training as engine
    from trajectory_prediction.utils.provenance import sha256_file
    torch.set_num_threads(1)
    train,selection=datasets();initial=QualityGate().state_dict()
    initializer=tmp_path/"initial";initializer.write_text("frozen")
    contract={"seed":2027,"gate_config":{},"conditions":[],"preparation_sources":{},"p4_training_identity":"x",
        "references":{"checkpoint":{}},"base_hashes":{"model_identity":"x"}}
    meta={"candidate_contracts":{"train":contract,"train_val":contract},"candidate_manifests":{},
        "p6_checkpoint":{"path":str(initializer),"sha256":sha256_file(initializer)},"train_weight_stats":{"mean":train.train_mean_weight}}
    root=tmp_path/"run";arm="adaptive_matched_005"
    train_strength(train,selection,meta,root,initial=initial,arm=arm)
    monkeypatch.setattr(engine,"load_initial",lambda *a:initial)
    monkeypatch.setattr(engine,"validate_parent_references",lambda *a:None)
    args={"split":"train_val","arm":arm,"train_mean_weight":train.train_mean_weight,"train_content":train.fingerprint()}
    model,_=engine.load_strength(root/"best.pt",contract,selection.scenes,torch.device("cpu"),**args)
    assert model.variant=="RH"
    original=completed_payload(root)
    for field,value in (("arm","fixed_005"),("target_mean_weight",.1),("revision_weight",.9),("revision_kind","revision_fixed"),("train_mean_weight",.123)):
        payload=copy.deepcopy(original);payload["metadata"][field]=value
        if field=="train_mean_weight":payload["metadata"]["train_weight_stats"]["mean"]=value
        rewrite(root,payload)
        with pytest.raises(ValueError,match="role|mean|content|settings"):
            engine.load_strength(root/"best.pt",contract,selection.scenes,torch.device("cpu"),**args)


def test_both_parent_reports_hash_sources_and_selections_are_bound(tmp_path,monkeypatch):
    import trajectory_prediction.online.strength_training as engine
    from trajectory_prediction.analysis import consistency_report, horizon_report
    from trajectory_prediction.online import consistency_evaluation, horizon_evaluation
    from trajectory_prediction.utils.provenance import sha256_file
    assert hasattr(engine,"validate_parent_reports")
    hs={"decision":{"accuracy":"RH"}}
    cs={"decision":{"balanced":"revision_fixed","advance":False}}
    monkeypatch.setattr(horizon_evaluation,"load_horizon_selection",lambda *a:hs)
    monkeypatch.setattr(consistency_evaluation,"load_consistency_selection",lambda *a:cs)
    meta={}
    for name,value in (("horizon_selection",hs),("consistency_selection",cs),
        ("horizon_report",{"selection":hs,"val_run":True,"sources":horizon_report.report_sources()}),
        ("consistency_report",{"selection":cs,"val_run":False,"sources":consistency_report.report_sources()})):
        path=tmp_path/(name+".json");path.write_text(json.dumps(value))
        meta[name]={"path":str(path),"sha256":sha256_file(path)}
    engine.validate_parent_reports(meta)
    path=Path(meta["horizon_report"]["path"])
    original=path.read_text();path.write_text(original+" ")
    with pytest.raises(ValueError,match="reference"):
        engine.validate_parent_reports(meta)
    meta["horizon_report"]["sha256"]=sha256_file(path)
    changed=json.loads(original);changed["sources"]={}
    path.write_text(json.dumps(changed));meta["horizon_report"]["sha256"]=sha256_file(path)
    with pytest.raises(ValueError,match="report"):
        engine.validate_parent_reports(meta)

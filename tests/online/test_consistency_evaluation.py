import json

import numpy as np
import pandas as pd
import pytest
import torch
from test_horizon_gate import example_arrays

from trajectory_prediction.online.consistency_artifacts import STRATEGIES, publish_run, verify_run
from trajectory_prediction.online.consistency_evaluation import (
    CONDITIONS,
    SEEDS,
    choose_consistency,
)
from trajectory_prediction.online.innovation_consistency import ARMS
from trajectory_prediction.utils.provenance import sha256_file


def matrix():
    values={"RH":(1.,2.,1.,6.),"revision_0":(1.,2.,1.,6.),
        "revision_fixed":(1.001,2.,.98,6.),"revision_adaptive":(1.0005,2.,.97,6.),"revision_mean":(1.001,2.,.985,6.)}
    return pd.DataFrame([{"seed":s,"condition":c,"strategy":n,
        **dict(zip(("ade","fde","jitter","fde_p95"),values.get(n,(1.1,2.,.1,6.)),strict=True))}
        for s in SEEDS for c in CONDITIONS for n in STRATEGIES])


def test_frozen_pool_constraints_ties_and_complete_matrix():
    f=matrix();choice=choose_consistency(f)
    assert choice["balanced"]=="revision_adaptive" and choice["advance"]
    f.loc[f.strategy.eq("revision_adaptive"),"ade"]=1.003
    assert choose_consistency(f)["balanced"]=="revision_fixed"
    f.loc[f.strategy.eq("revision_fixed"),"fde"]=2.02
    assert choose_consistency(f)["balanced"]=="revision_mean"
    for invalid in [f.iloc[:-1],pd.concat([f,f.iloc[:1]])]:
        with pytest.raises(ValueError,match="complete"):
            choose_consistency(invalid)


def test_zero_or_old_RH_winner_never_qualifies():
    f=matrix();f.loc[f.strategy.isin(ARMS[1:]),"ade"]=1.01
    assert choose_consistency(f)["balanced"]=="RH"
    f.loc[f.strategy.eq("revision_0"),"jitter"]=.95
    assert choose_consistency(f)["balanced"]=="revision_0"
    assert not choose_consistency(f)["advance"]


def test_new_artifacts_replay_rejects_numeric_tampering_with_updated_hash(tmp_path):
    metrics={n:np.zeros((6,2,11,3),np.float64) for n in STRATEGIES}
    weights={n:np.zeros((6,2,11,12),np.float32) for n in STRATEGIES}
    tables={"by_condition.csv":pd.DataFrame({"x":[1.]})};meta={"sample_count":2}
    publish_run(tmp_path,metrics,weights,tables,meta)
    verify_run(tmp_path,metrics,weights,tables,meta)
    changed={n:v.copy() for n,v in metrics.items()};changed["revision_adaptive"][0,0,0,0]=1
    np.savez_compressed(tmp_path/"metrics.npz",**changed)
    m=json.loads((tmp_path/"summary.json").read_text());m["sha256"]["metrics.npz"]=sha256_file(tmp_path/"metrics.npz")
    (tmp_path/"summary.json").write_text(json.dumps(m))
    with pytest.raises(ValueError,match="replay"):
        verify_run(tmp_path,metrics,weights,tables,meta)


def test_failed_frozen_decision_does_not_access_val(tmp_path,monkeypatch):
    import trajectory_prediction.online.consistency_evaluation as engine
    (tmp_path/"selection.json").write_text("{}")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    monkeypatch.setattr(engine,"load_consistency_selection",lambda *a:{"decision":{"advance":False}})
    monkeypatch.setattr(engine,"run_split",lambda *a,**k:pytest.fail("must not access val"))
    argv=[v for name in ("cache-root","training-root","p6-training-root","reference-training-root","reference-selection-root","horizon-training-root","horizon-selection-root","horizon-evaluation-root","selection-root","output-root") for v in ("--"+name,str(tmp_path))]
    engine.main([*argv,"--device","cpu"])


def test_all_18_strategies_tables_and_streaming_profile(tmp_path,monkeypatch):
    from test_refinement_evaluation import fixture

    from trajectory_prediction.online.consistency_evaluation import (
        benchmark_consistency,
        derive_tables,
        evaluate_models,
    )
    from trajectory_prediction.online.horizon_features import HorizonDataset
    from trajectory_prediction.online.horizon_gate import VARIANTS, HorizonGate
    old,suite,policy=fixture();observed=example_arrays(2)
    data=HorizonDataset({**old.arrays,**{k:observed[k] for k in ("history","position","velocity","previous_velocity")}})
    initial=suite["models"]["p6_learned"].state_dict()
    models={n:HorizonGate(n if n in VARIANTS else "RH",initial) for n in (*VARIANTS,*ARMS)}
    metrics,weights=evaluate_models(data,suite,policy,models,torch.device("cpu"))
    assert tuple(metrics)==STRATEGIES
    np.testing.assert_array_equal(metrics["revision_0"],metrics["RH"])
    tables=derive_tables(data,metrics,weights,2027)
    assert len(tables["by_condition.csv"])==18*6
    monkeypatch.setattr(data,"inputs",lambda *a,**k:pytest.fail("baseline profile must not construct new features"))
    benchmark_consistency(data,suite,policy,models,tmp_path,torch.device("cpu"))
    monkeypatch.undo()
    from trajectory_prediction.analysis.consistency_report import audit_latency
    audit_latency(tmp_path,data,torch.device("cpu"))

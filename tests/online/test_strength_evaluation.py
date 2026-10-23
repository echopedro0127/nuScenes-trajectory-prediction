import json

import numpy as np
import pandas as pd
import pytest
import torch
from test_horizon_gate import example_arrays

from trajectory_prediction.online.innovation_consistency import ARMS as OLD_ARMS
from trajectory_prediction.online.strength_artifacts import STRATEGIES, publish_run, verify_run
from trajectory_prediction.online.strength_consistency import ARMS
from trajectory_prediction.online.strength_evaluation import (
    CONDITIONS,
    SEEDS,
    choose_strength,
)
from trajectory_prediction.utils.provenance import sha256_file


def matrix():
    values={"RH":(1.,2.,1.,6.),"revision_0":(1.,2.,1.,6.),
        "revision_fixed":(1.001,2.,.998,6.),"fixed_005":(1.001,2.,.98,6.),
        "adaptive_matched_005":(1.0005,2.,.97,6.),"fixed_010":(1.001,2.,.985,6.),
        "adaptive_matched_010":(1.001,2.,.99,6.)}
    return pd.DataFrame([{"seed":s,"condition":c,"strategy":n,
        **dict(zip(("ade","fde","jitter","fde_p95"),values.get(n,(1.1,2.,.1,6.)),strict=True))}
        for s in SEEDS for c in CONDITIONS for n in STRATEGIES])


def test_frozen_pool_constraints_ties_and_complete_matrix():
    f=matrix();choice=choose_strength(f)
    assert choice["balanced"]=="adaptive_matched_005" and choice["advance"]
    f.loc[f.strategy.eq("adaptive_matched_005"),"ade"]=1.003
    assert choose_strength(f)["balanced"]=="fixed_005"
    f.loc[f.strategy.eq("fixed_005"),"fde"]=2.02
    assert choose_strength(f)["balanced"]=="fixed_010"
    for invalid in [f.iloc[:-1],pd.concat([f,f.iloc[:1]])]:
        with pytest.raises(ValueError,match="complete"):
            choose_strength(invalid)


def test_old_winner_and_insufficient_improvement_never_qualify():
    f=matrix();f.loc[f.strategy.isin(ARMS),"ade"]=1.01
    assert choose_strength(f)["balanced"]=="revision_fixed"
    assert not choose_strength(f)["advance"]
    f.loc[f.strategy.eq("revision_fixed"),"ade"]=1.01
    assert choose_strength(f)["balanced"]=="RH"
    f=matrix();f.loc[f.strategy.isin(ARMS),"jitter"]=.997
    assert not choose_strength(f)["advance"]
    f=matrix();f.loc[f.strategy.eq("revision_fixed"),"jitter"]=.96
    assert not choose_strength(f)["advance"]


def test_clean_tail_guard_and_deterministic_tie():
    f=matrix();f.loc[f.strategy.eq("adaptive_matched_005") & f.condition.eq("clean"),"fde_p95"]=6.1
    assert choose_strength(f)["balanced"]=="fixed_005"
    f=matrix();f.loc[f.strategy.eq("adaptive_matched_005"),["ade","jitter"]]=[1.001,.98]
    assert choose_strength(f)["balanced"]=="adaptive_matched_005"


def test_new_artifacts_replay_rejects_numeric_tampering_with_updated_hash(tmp_path):
    metrics={n:np.zeros((6,2,11,3),np.float64) for n in STRATEGIES}
    weights={n:np.zeros((6,2,11,12),np.float32) for n in STRATEGIES}
    tables={"by_condition.csv":pd.DataFrame({"x":[1.]})};meta={"sample_count":2}
    publish_run(tmp_path,metrics,weights,tables,meta)
    verify_run(tmp_path,metrics,weights,tables,meta)
    changed={n:v.copy() for n,v in metrics.items()};changed["adaptive_matched_005"][0,0,0,0]=1
    np.savez_compressed(tmp_path/"metrics.npz",**changed)
    m=json.loads((tmp_path/"summary.json").read_text());m["sha256"]["metrics.npz"]=sha256_file(tmp_path/"metrics.npz")
    (tmp_path/"summary.json").write_text(json.dumps(m))
    with pytest.raises(ValueError,match="replay"):
        verify_run(tmp_path,metrics,weights,tables,meta)


def test_failed_frozen_decision_does_not_access_val(tmp_path,monkeypatch):
    import trajectory_prediction.online.strength_evaluation as engine
    (tmp_path/"selection.json").write_text("{}")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    monkeypatch.setattr(engine,"load_strength_selection",lambda *a:{"decision":{"advance":False}})
    monkeypatch.setattr(engine,"run_split",lambda *a,**k:pytest.fail("must not access val"))
    argv=[v for name in ("cache-root","training-root","p6-training-root","reference-training-root","reference-selection-root","horizon-training-root","horizon-selection-root","horizon-evaluation-root","consistency-training-root","consistency-selection-root","consistency-evaluation-root","consistency-report-root","selection-root","output-root") for v in ("--"+name,str(tmp_path))]
    engine.main([*argv,"--device","cpu"])


def test_all_22_strategies_tables_and_streaming_profile(tmp_path,monkeypatch):
    from test_refinement_evaluation import fixture

    from trajectory_prediction.online.horizon_features import HorizonDataset
    from trajectory_prediction.online.horizon_gate import VARIANTS, HorizonGate
    from trajectory_prediction.online.strength_evaluation import (
        benchmark_strength,
        derive_tables,
        evaluate_models,
    )
    old,suite,policy=fixture();observed=example_arrays(2)
    data=HorizonDataset({**old.arrays,**{k:observed[k] for k in ("history","position","velocity","previous_velocity")}})
    initial=suite["models"]["p6_learned"].state_dict()
    models={n:HorizonGate(n if n in VARIANTS else "RH",initial) for n in (*VARIANTS,*OLD_ARMS,*ARMS)}
    metrics,weights=evaluate_models(data,suite,policy,models,torch.device("cpu"))
    assert tuple(metrics)==STRATEGIES
    np.testing.assert_array_equal(metrics["revision_0"],metrics["RH"])
    tables=derive_tables(data,metrics,weights,2027)
    assert len(tables["by_condition.csv"])==22*6
    monkeypatch.setattr(data,"inputs",lambda *a,**k:pytest.fail("baseline profile must not construct new features"))
    benchmark_strength(data,suite,policy,models,tmp_path,torch.device("cpu"))
    monkeypatch.undo()
    from trajectory_prediction.analysis.strength_report import audit_latency
    audit_latency(tmp_path,data,torch.device("cpu"))


def test_current_execution_sources_and_changes_are_bound(monkeypatch):
    import trajectory_prediction.online.strength_evaluation as engine
    baseline=engine.evaluation_sources()
    expected={"online/strength_artifacts.py","online/strength_evaluation.py"}
    assert set(baseline["execution"])==expected
    original=engine.sha256_file
    for name in expected:
        monkeypatch.setattr(engine,"sha256_file",lambda p,n=name: "changed" if str(p).replace("\\","/").endswith(n) else original(p))
        modified=engine.evaluation_sources()
        assert modified["execution"][name]=="changed" and modified!=baseline
        monkeypatch.undo()

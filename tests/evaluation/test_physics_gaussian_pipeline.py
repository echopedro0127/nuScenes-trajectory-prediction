import json

import numpy as np
import pytest
from test_physics_clean_pipeline import physics_parent as physics_parent  # noqa: PLC0414
from test_query_audit_data import audit_data as audit_data  # noqa: PLC0414
from test_query_audit_pipeline import audit_parents as audit_parents  # noqa: PLC0414

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation.physics_clean import _documents
from trajectory_prediction.evaluation.physics_clean import protocol as clean_protocol
from trajectory_prediction.evaluation.physics_clean_sources import snapshot
from trajectory_prediction.evaluation.track_sources import sources
from trajectory_prediction.utils.provenance import sha256_file


def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def gaussian_parent(physics_parent,tmp_path):
    b3a,b3a_owner=physics_parent;info=snapshot(b3a,b3a_owner)
    docs,_summary=_documents(info);root=tmp_path/'clean';root.mkdir();owner=tmp_path/'clean-owner';(owner/'scripts').mkdir(parents=True)
    script=owner/'scripts/run_realclock_physics_clean.py'
    script.write_text("import argparse,json\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--output-root');p.add_argument('--verify-only',action='store_true');a=p.parse_args();r=Path(a.output_root);s=json.loads((r/'summary.json').read_text());i=json.loads((r/'run.json').read_text())['identity'];print(json.dumps({**s,'identity':i,'physics_clean_semantically_replayed':True,'saved_predictions_independently_rescored':True,'scalar_reference_verified':True}))\n")
    for name,value in docs.items():(root/name).write_bytes(value)
    binding={'schema':'realclock_physics_clean_b3b1_v1','parent':str(b3a),'parent_owner':str(b3a_owner),
             'parent_identity':info['identity'],'sources':sources(owner),'references':info['references'],
             'protocol':clean_protocol(),'neural_training':False,'holdout_evaluated':False,'new_prediction_scores_computed':True}
    write(root/'run.json',{'identity':digest(binding),'binding':binding})
    write(root/'completed.json',{'identity':digest(binding),'sha256':{p.name:sha256_file(p) for p in root.iterdir() if p.name!='completed.json'}})
    return root,owner


def test_publish24_combinations_and_independent_verify(gaussian_parent,tmp_path):
    from trajectory_prediction.evaluation.physics_gaussian import prepare_gaussian, verify_gaussian
    out=tmp_path/'gaussian';s=prepare_gaussian(*gaussian_parent,out)
    assert s['combinations']==24 and s['valid_pairs_per_combination']==58
    assert len(s['seed_results'])==24 and len(s['seed_macro_results'])==8
    result=verify_gaussian(out)
    assert result['gaussian_semantically_replayed'] is True
    assert result['saved_predictions_independently_rescored'] is True
    assert result['scalar_reference_verified'] is True
    with pytest.raises(FileExistsError):prepare_gaussian(*gaussian_parent,out)


@pytest.mark.parametrize('kind',['normal','observation','prediction','gt','summary','aggregate','protocol','proof','extra'])
def test_rehashed_tampering_rejected(gaussian_parent,tmp_path,kind):
    from trajectory_prediction.evaluation.physics_gaussian import prepare_gaussian, verify_gaussian
    out=tmp_path/'gaussian';prepare_gaussian(*gaussian_parent,out)
    paths={'normal':'normals/2027.npz','observation':'gaussian_05/seed_2027/observations.npz',
           'prediction':'gaussian_05/seed_2027/cv_actual_dt.npz','gt':'shared_clean.npz',
           'summary':'summary.json','aggregate':'aggregates/gaussian_05/cv_actual_dt.npz',
           'protocol':'protocol.json','proof':'parent_proof.json','extra':'extra.json'}
    path=out/paths[kind];marker=json.loads((out/'completed.json').read_text())
    if path.suffix=='.npz':
        with np.load(path,allow_pickle=False) as z:values={k:z[k] for k in z.files}
        key={'normal':'normals','observation':'observations_local','prediction':'predictions','gt':'positions_local','aggregate':'mean_error_over_noise_seeds'}[kind]
        values[key].flat[0]+=1;np.savez_compressed(path,**values)
    else:
        value=json.loads(path.read_text()) if path.exists() else {}
        if kind=='summary':value['combinations']=0
        elif kind=='protocol':value['noise_seeds']=[1,2,3]
        elif kind=='proof':value['parent_identity']='forged'
        write(path,value)
    marker['sha256'][paths[kind]]=sha256_file(path);write(out/'completed.json',marker)
    with pytest.raises(ValueError):verify_gaussian(out)


def test_bias_and_interruption_never_complete(gaussian_parent,tmp_path,monkeypatch):
    from trajectory_prediction.evaluation import physics_gaussian as mod
    original=mod.replay_states
    def biased(*args):
        value=original(*args);value['states'][:,:,0]+=.01;return value
    with monkeypatch.context() as m:
        m.setattr(mod,'replay_states',biased)
        with pytest.raises(ValueError,match='scalar|clean parent regression'):mod.prepare_gaussian(*gaussian_parent,tmp_path/'biased')
    assert not (tmp_path/'biased/completed.json').exists()
    def fail(*args):raise RuntimeError('interrupted')
    monkeypatch.setattr(mod,'_write_bytes',fail)
    with pytest.raises(RuntimeError):mod.prepare_gaussian(*gaussian_parent,tmp_path/'partial')
    assert not (tmp_path/'partial/completed.json').exists()


def test_parent_source_change_is_rejected(gaussian_parent,tmp_path):
    from trajectory_prediction.evaluation.physics_gaussian import prepare_gaussian, verify_gaussian
    out=tmp_path/'gaussian';prepare_gaussian(*gaussian_parent,out)
    (gaussian_parent[0]/'windows.json').write_text('[]')
    with pytest.raises(ValueError):verify_gaussian(out)


def test_saved_prediction_rescore_also_checks_noise_seed_macro(gaussian_parent,tmp_path):
    from trajectory_prediction.evaluation import physics_gaussian as mod
    out=tmp_path/'gaussian';mod.prepare_gaussian(*gaussian_parent,out)
    summary=json.loads((out/'summary.json').read_text())
    summary['seed_macro_results'][0]['groups'][0]['mean_window_error_m']+=1
    proof=json.loads((out/'parent_proof.json').read_text())
    info=mod.snapshot(*gaussian_parent,proof)
    with pytest.raises(ValueError,match='macro'):mod._rescore(out,info,summary)


def test_sigma0_regression_checks_frozen_parent_not_only_current_kernel(gaussian_parent,monkeypatch):
    from trajectory_prediction.evaluation import physics_gaussian as mod
    info=mod.snapshot(*gaussian_parent)
    base,*_=mod._load(info)
    original=mod.replay_states
    def biased(*args):
        value=original(*args);value['states'][:,:,0]+=.01;return value
    monkeypatch.setattr(mod,'replay_states',biased)
    with pytest.raises(ValueError,match='clean parent regression'):mod.verify_clean_regression(info,base)

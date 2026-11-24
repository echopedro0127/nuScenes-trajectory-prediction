import json

import numpy as np
import pytest
from test_query_audit_data import audit_data as audit_data  # noqa: PLC0414
from test_query_audit_pipeline import audit_parents as audit_parents  # noqa: PLC0414

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation.query_audit import protocol as parent_protocol
from trajectory_prediction.evaluation.query_audit_data import derive_audit
from trajectory_prediction.evaluation.track_sources import _artifacts, sources
from trajectory_prediction.utils.provenance import sha256_file


def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def physics_parent(audit_data,audit_parents,tmp_path):
    tracks,tracks_owner,events,events_owner=audit_parents
    root=tmp_path/'b3a';root.mkdir();owner=tmp_path/'b3a-owner';(owner/'scripts').mkdir(parents=True)
    script=owner/'scripts/run_realclock_query_audit.py'
    script.write_text("import argparse,json\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--output-root');p.add_argument('--verify-only',action='store_true');a=p.parse_args();r=Path(a.output_root);s=json.loads((r/'summary.json').read_text());i=json.loads((r/'run.json').read_text())['identity'];print(json.dumps({**s,'query_audit_semantically_replayed':True,'identity':i}))\n")
    result=derive_audit(audit_data)
    for name,value in result['documents'].items():write(root/name,value)
    for name,value in result['arrays'].items():np.savez_compressed(root/name,**value)
    refs={};_artifacts(tracks,refs);_artifacts(events,refs)
    binding={'schema':'clock_anchor_audit_b3a_v1','tracks':str(tracks),'tracks_owner':str(tracks_owner),
             'events':str(events),'events_owner':str(events_owner),'dataset_release':'v1.0-trainval',
             'sources':sources(owner),'references':refs,'protocol':parent_protocol(),
             'neural_training':False,'holdout_evaluated':False,'new_prediction_scores_computed':False}
    write(root/'protocol.json',parent_protocol());write(root/'run.json',{'identity':digest(binding),'binding':binding})
    write(root/'completed.json',{'identity':digest(binding),'sha256':{p.name:sha256_file(p) for p in root.iterdir() if p.name!='completed.json'}})
    return root,owner


def test_four_strategies_publish_replay_and_rescore(physics_parent,tmp_path):
    from trajectory_prediction.evaluation.physics_clean import prepare_clean, verify_clean
    out=tmp_path/'clean';summary=prepare_clean(*physics_parent,out)
    assert summary['new_prediction_scores_computed'] is True and summary['valid_pairs_per_strategy']==58
    assert len(summary['strategy_groups'])==4
    assert verify_clean(out)['physics_clean_semantically_replayed'] is True
    assert verify_clean(out)['saved_predictions_independently_rescored'] is True
    with pytest.raises(FileExistsError):prepare_clean(*physics_parent,out)


@pytest.mark.parametrize('kind',['summary','prediction','state','shared','protocol','extra','proof','binding'])
def test_rehashed_tampering_rejected(physics_parent,tmp_path,kind):
    from trajectory_prediction.evaluation.physics_clean import prepare_clean, verify_clean
    out=tmp_path/'clean';prepare_clean(*physics_parent,out);marker=json.loads((out/'completed.json').read_text())
    names={'summary':'summary.json','prediction':'cv_actual_dt.npz','state':'kf_actual_dt.npz','shared':'shared.npz',
           'protocol':'protocol.json','extra':'extra.json','proof':'parent_proof.json','binding':'run.json'}
    path=out/names[kind]
    if kind in ('prediction','state','shared'):
        with np.load(path,allow_pickle=False) as z:values={k:z[k] for k in z.files}
        if kind=='prediction':values['predictions'][0,0,0,0]+=1
        elif kind=='state':values['states'][0,0,0]+=1
        else:values['positions_local'][0,16,0]+=1
        np.savez_compressed(path,**values)
    else:
        value=json.loads(path.read_text()) if path.exists() else {}
        if kind=='summary':value['valid_pairs_per_strategy']=0
        elif kind=='protocol':value['measurement_variance']=0
        elif kind=='proof':value['parent_identity']='forged'
        elif kind=='binding':
            value['binding']['neural_training']=0;value['identity']=digest(value['binding']);marker['identity']=value['identity']
        write(path,value)
    marker['sha256'][path.name]=sha256_file(path);write(out/'completed.json',marker)
    with pytest.raises(ValueError):verify_clean(out)


def test_parent_source_and_interruption(physics_parent,tmp_path,monkeypatch):
    from trajectory_prediction.evaluation import physics_clean as mod
    parent,owner=physics_parent
    def fail(*args):raise RuntimeError('interrupted')
    with monkeypatch.context() as m:
        m.setattr(mod,'_write_bytes',fail)
        with pytest.raises(RuntimeError):mod.prepare_clean(parent,owner,tmp_path/'partial')
    assert not (tmp_path/'partial/completed.json').exists()
    out=tmp_path/'clean';mod.prepare_clean(parent,owner,out)
    (parent/'windows.json').write_text('[]')
    with pytest.raises(ValueError):mod.verify_clean(out)


def test_runtime_records_thread_request_instead_of_unobserved_thread_count(monkeypatch):
    from trajectory_prediction.evaluation.physics_clean import _binding
    monkeypatch.setenv('OPENBLAS_NUM_THREADS','3')
    runtime=_binding({'parent':'p','owner':'o','identity':'i','references':{}})['runtime']
    assert runtime['thread_environment']['OPENBLAS_NUM_THREADS']=='3'
    assert 'threads' not in runtime


def test_shared_vector_kernel_bias_is_rejected_by_independent_scalar_reference(physics_parent,tmp_path,monkeypatch):
    from trajectory_prediction.evaluation import physics_clean as mod
    original=mod.replay_states
    def biased(*args):
        value=original(*args);value['states'][:,:,0]+=.01;return value
    monkeypatch.setattr(mod,'replay_states',biased)
    out=tmp_path/'biased'
    with pytest.raises(ValueError,match='scalar'):mod.prepare_clean(*physics_parent,out)
    assert not (out/'completed.json').exists()

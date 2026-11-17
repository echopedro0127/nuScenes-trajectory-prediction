import json

import numpy as np
import pytest
from test_query_audit_data import audit_data as audit_data  # noqa: PLC0414

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation.track_sources import sources
from trajectory_prediction.utils.provenance import sha256_file


def write(path,value):
    path.write_text(json.dumps(value),encoding='utf-8')


def complete(root):
    run=json.loads((root/'run.json').read_text())
    write(root/'completed.json',{'identity':run['identity'],
                               'sha256':{p.name:sha256_file(p) for p in root.iterdir() if p.name!='completed.json'}})


@pytest.fixture
def audit_parents(audit_data,tmp_path):
    paths=[]
    for kind in ('tracks','events'):
        root=tmp_path/kind;owner=tmp_path/(kind+'-owner');root.mkdir();(owner/'scripts').mkdir(parents=True)
        script=owner/'scripts'/('prepare_scenario_tracks.py' if kind=='tracks' else 'prepare_scenario_events.py')
        flag='track_and_window_replayed' if kind=='tracks' else 'events_semantically_replayed'
        stages=("print(json.dumps({'stage':'offline_events','role':'independent_fit'}))\nprint(json.dumps({'stage':'offline_events','role':'independent_select'}))\n" if kind=='events' else '')
        script.write_text("import argparse,json\nfrom pathlib import Path\np=argparse.ArgumentParser();p.add_argument('--output-root');p.add_argument('--verify-only',action='store_true');a=p.parse_args();r=Path(a.output_root)\ns=json.loads((r/'summary.json').read_text());i=json.loads((r/'run.json').read_text())['identity']\n"+stages+"print(json.dumps({**s,'"+flag+"':True,'identity':i}))\n")
        binding={'schema':'scenario_tracks_b0b_v1' if kind=='tracks' else 'scenario_events_b1a_v1',
                 'scope':'original_fit_select_cache_target_instances','dataset_release':'v1.0-trainval',
                 'references':{},'sources':sources(owner),'neural_training':False,'holdout_evaluated':False}
        if kind=='tracks':
            summary={'scope':binding['scope'],'status':'tracks_and_clock_complete','window_count':2,
                     'canonical_episodes_extracted':False,'new_prediction_scores_computed':False,
                     'neural_training':False,'holdout_evaluated':False}
            for role,values in audit_data['window_arrays'].items():np.savez_compressed(root/(role+'-windows.npz'),**values)
            for role,values in audit_data['track_arrays'].items():np.savez_compressed(root/(role+'-tracks.npz'),**values)
            for file,key in [('windows.json','windows'),('tracks.json','tracks')]:write(root/file,audit_data[key])
        else:
            tracks,tracks_owner=paths
            binding.update(parent=str(tracks),parent_owner=str(tracks_owner),parent_identity=json.loads((tracks/'run.json').read_text())['identity'],new_prediction_scores_computed=False)
            summary={'status':'b1a_rule_candidates_complete','manual_completed':0,'neural_training':False,
                     'holdout_evaluated':False,'new_prediction_scores_computed':False,'rule_candidates_are_human_ground_truth':False}
            write(root/'blocks.json',audit_data['blocks']);write(root/'events.json',audit_data['events'])
        write(root/'run.json',{'identity':digest(binding),'binding':binding});write(root/'summary.json',summary);complete(root)
        paths.extend([root,owner])
    return paths


def test_publish_and_independent_replay(audit_parents,tmp_path):
    from trajectory_prediction.evaluation.query_audit import prepare_audit, verify_audit
    out=tmp_path/'audit'; summary=prepare_audit(*audit_parents,out)
    assert summary['windows']==2 and summary['matching_attempts']==66
    assert verify_audit(out)['query_audit_semantically_replayed'] is True
    with pytest.raises(FileExistsError):prepare_audit(*audit_parents,out)


@pytest.mark.parametrize('kind',['summary','array','protocol','extra','proof','binding'])
def test_rehashed_artifact_tampering_rejected(audit_parents,tmp_path,kind):
    from trajectory_prediction.evaluation.query_audit import prepare_audit, verify_audit
    out=tmp_path/'audit';prepare_audit(*audit_parents,out)
    marker=json.loads((out/'completed.json').read_text())
    names={'summary':'summary.json','array':'query_matches.npz','protocol':'protocol.json',
           'extra':'extra.json','proof':'parent_proofs.json','binding':'run.json'}
    path=out/names[kind]
    if kind=='array':
        with np.load(path,allow_pickle=False) as z:values={k:z[k] for k in z.files}
        values['target_index'][0,0,0]+=1;np.savez_compressed(path,**values)
    else:
        value=json.loads(path.read_text()) if path.exists() else {}
        if kind=='summary':value['queries']=0
        elif kind=='protocol':value['tolerance_us']=50001
        elif kind=='proof':value['events']['parent_identity']='forged'
        elif kind=='binding':
            value['binding']['holdout_evaluated']=0;value['identity']=digest(value['binding']);marker['identity']=value['identity']
        write(path,value)
    marker['sha256'][path.name]=sha256_file(path);write(out/'completed.json',marker)
    with pytest.raises(ValueError):verify_audit(out)


def test_wrong_parent_identity_and_original_source_change(audit_parents,tmp_path):
    from trajectory_prediction.evaluation.query_audit import prepare_audit, verify_audit
    out=tmp_path/'audit';prepare_audit(*audit_parents,out)
    original=audit_parents[0]/'windows.json';original.write_text('[]')
    with pytest.raises(ValueError):verify_audit(out)
    complete(audit_parents[0])
    run=json.loads((audit_parents[2]/'run.json').read_text());run['binding']['parent_identity']='wrong'
    run['identity']=digest(run['binding']);write(audit_parents[2]/'run.json',run);complete(audit_parents[2])
    with pytest.raises(ValueError,match='parent|identity'):prepare_audit(*audit_parents,tmp_path/'other')


def test_interrupted_publication_has_no_completed(audit_parents,tmp_path,monkeypatch):
    from trajectory_prediction.evaluation import query_audit as mod
    def fail(*args):raise RuntimeError('interrupted')
    monkeypatch.setattr(mod,'_write_bytes',fail)
    out=tmp_path/'audit'
    with pytest.raises(RuntimeError):mod.prepare_audit(*audit_parents,out)
    assert not (out/'completed.json').exists()

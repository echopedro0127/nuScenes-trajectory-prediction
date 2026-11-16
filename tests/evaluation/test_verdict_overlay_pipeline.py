import csv
import io
import json
from pathlib import Path

import pytest
from test_candidate_verdicts import source_data as source_data  # noqa: PLC0414

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation import event_review_data
from trajectory_prediction.evaluation.track_sources import sources
from trajectory_prediction.utils.provenance import sha256_file


def write(path,value):path.write_text(json.dumps(value),encoding='utf-8')


@pytest.fixture
def verdict_parent(source_data,tmp_path):
    root,owner=tmp_path/'b2a',tmp_path/'b2a-owner';root.mkdir()
    script=owner/'scripts/prepare_event_review.py';script.parent.mkdir(parents=True)
    script.write_text("import argparse,csv,json\nfrom pathlib import Path\nfrom collections import Counter\n"
      "p=argparse.ArgumentParser();p.add_argument('--output-root');p.add_argument('--verify-only',action='store_true');a=p.parse_args()\n"
      "r=Path(a.output_root);s=json.loads((r/'summary.json').read_text());i=json.loads((r/'run.json').read_text())['identity']\n"
      "c=Counter(x['review_status'] for x in csv.DictReader((r/'review_working.csv').read_text(encoding='utf-8-sig').splitlines()))\n"
      "print(json.dumps({**s,'review_semantically_replayed':True,'identity':i,'working_status_counts':{k:c[k] for k in ['pending','in_progress','reviewed']}}))\n")
    validator=owner/'src/trajectory_prediction/evaluation/event_review_data.py';validator.parent.mkdir(parents=True)
    validator.write_bytes(Path(event_review_data.__file__).read_bytes())
    queue=[{**t,'diagnostic_id':'','review_status':'pending','reviewer':'','notes':''} for t in source_data['queue']]
    for b in source_data['blocks']:
        queue.append({'task_id':'scan-'+b['block_id'],'task_kind':'block_scan','role':b['role'],
            'instance':b['instance'],'block_id':b['block_id'],'event_id':'','diagnostic_id':'',
            'review_status':'pending','reviewer':'','notes':''})
    source_data['queue']=queue;source_data['parent_identity']='fixture-b1a'
    binding={'schema':'event_review_b2a_v1','parent_identity':'fixture-b1a','dataset_release':'v1.0-trainval',
             'sources':sources(owner),'references':{},'human_event_labels_imported':False,
             'neural_training':False,'holdout_evaluated':False,'new_prediction_scores_computed':False}
    identity=digest(binding)
    summary={'status':'review_materials_ready','task_rows':len(queue),'manual_completed':0,
             'human_event_labels_imported':False,'neural_training':False,'holdout_evaluated':False,
             'new_prediction_scores_computed':False,'initial_working_status_counts':{'pending':len(queue),'in_progress':0,'reviewed':0}}
    for name,value in [('run.json',{'identity':identity,'binding':binding}),('summary.json',summary),
                       ('review_data.json',source_data),('queue.json',queue),
                       ('protocol.json',{'mutable_files':['review_working.csv']}),('progress.json',{'state':'complete'})]:write(root/name,value)
    baseline=event_review_data.csv_text(queue)
    for name in ('review_baseline.csv','review_working.csv'):(root/name).write_text(baseline,encoding='utf-8-sig',newline='')
    (root/'review.html').write_text('Synthetic protocol fixture; no scientific result')
    hashes={p.name:sha256_file(p) for p in root.iterdir() if p.name!='review_working.csv'}
    write(root/'completed.json',{'identity':identity,'sha256':hashes,'mutable_files':['review_working.csv']})
    return root,owner


def test_template_empty_overlay_and_mutable_reading_progress(verdict_parent,tmp_path):
    from trajectory_prediction.evaluation.verdict_overlay import (
        import_overlay,
        make_template_pack,
        verify_pack,
    )
    parent,owner=verdict_parent;template=tmp_path/'template';out=tmp_path/'overlay'
    assert make_template_pack(parent,owner,template)['atomic_total']==3
    assert verify_pack(template)['verdicts_semantically_replayed'] is True
    annotations=tmp_path/'annotations.csv';annotations.write_bytes((template/'candidate_verdicts.csv').read_bytes())
    result=import_overlay(parent,owner,annotations,out)
    assert result['status']=='no_human_verdicts' and result['decision_counts']['pending']==3
    assert verify_pack(out)['human_candidate_decisions_imported'] is False
    rows=list(csv.DictReader(io.StringIO((parent/'review_working.csv').read_text(encoding='utf-8-sig'))))
    rows[0].update(review_status='reviewed',reviewer='fixture-reader',notes='reading is not a verdict')
    (parent/'review_working.csv').write_text(event_review_data.csv_text(rows),encoding='utf-8-sig',newline='')
    verified=verify_pack(out)
    assert verified['decision_counts']['pending']==3 and verified['parent_reading_status_counts']['reviewed']==1
    assert json.loads((out/'confirmed_candidates.json').read_text())==[]
    with pytest.raises(FileExistsError):import_overlay(parent,owner,annotations,out)


def test_synthetic_confirmation_and_bound_original_csv(verdict_parent,tmp_path):
    from trajectory_prediction.evaluation.candidate_verdicts import csv_text
    from trajectory_prediction.evaluation.verdict_overlay import (
        import_overlay,
        make_template_pack,
        verify_pack,
    )
    parent,owner=verdict_parent;template=tmp_path/'template';make_template_pack(parent,owner,template)
    rows=json.loads((template/'template.json').read_text())
    for r in rows:
        if r['source_event_id']=='s2':r.update(decision='confirmed',reviewer='fixture-reader',reason='synthetic declaration')
    path=tmp_path/'annotations.csv';path.write_text(csv_text(rows),encoding='utf-8-sig',newline='')
    out=tmp_path/'overlay';result=import_overlay(parent,owner,path,out)
    assert result['decision_counts']['confirmed']==1
    assert json.loads((out/'confirmed_candidates.json').read_text())[0]['source_event']['right_censored'] is True
    assert verify_pack(out)['verdicts_semantically_replayed'] is True
    path.write_text('changed')
    with pytest.raises(ValueError,match='source|checksum'):verify_pack(out)


@pytest.mark.parametrize('kind',['summary','confirmed','protocol','extra','binding','input_copy','proof'])
def test_overlay_rehashed_tamper_rejected(verdict_parent,tmp_path,kind):
    from trajectory_prediction.evaluation.verdict_overlay import (
        import_overlay,
        make_template_pack,
        verify_pack,
    )
    parent,owner=verdict_parent;template=tmp_path/'template';make_template_pack(parent,owner,template)
    path=tmp_path/'annotations.csv';path.write_bytes((template/'candidate_verdicts.csv').read_bytes())
    out=tmp_path/'overlay';import_overlay(parent,owner,path,out);marker=json.loads((out/'completed.json').read_text())
    names={'summary':'summary.json','confirmed':'confirmed_candidates.json','protocol':'protocol.json',
           'extra':'extra.json','binding':'run.json','input_copy':'annotations.csv','proof':'parent_proof.json'}
    p=out/names[kind]
    if kind=='input_copy':p.write_text('changed')
    else:
        value=json.loads(p.read_text()) if p.exists() else {}
        if kind=='summary':value['decision_counts']['confirmed']=1
        elif kind=='confirmed':value=[{'source_event_id':'invented'}]
        elif kind=='protocol':value['reading_progress_is_a_verdict']=True
        elif kind=='binding':
            value['binding']['human_candidate_decisions_imported']=0;value['identity']=digest(value['binding']);marker['identity']=value['identity']
        elif kind=='proof':
            payload=json.loads(value['stdout']);payload['working_status_counts']['pending']=True;value['stdout']=json.dumps(payload)
        write(p,value)
    marker['sha256'][p.name]=sha256_file(p);write(out/'completed.json',marker)
    with pytest.raises(ValueError):verify_pack(out)


def test_invalid_parent_working_and_interruption(verdict_parent,tmp_path,monkeypatch):
    from trajectory_prediction.evaluation import verdict_overlay as mod
    parent,owner=verdict_parent
    def fail(*args):raise RuntimeError('interrupted')
    with monkeypatch.context() as m:
        m.setattr(mod,'_write_bytes',fail)
        with pytest.raises(RuntimeError):mod.make_template_pack(parent,owner,tmp_path/'partial')
    assert not (tmp_path/'partial/completed.json').exists()
    out=tmp_path/'template';mod.make_template_pack(parent,owner,out)
    (parent/'review_working.csv').write_text('broken')
    with pytest.raises(ValueError):mod.verify_pack(out)


@pytest.mark.parametrize('kind',['unknown','bool_count','negative','missing','sum'])
def test_parent_proof_projection_strict(verdict_parent,kind):
    from trajectory_prediction.evaluation.verdict_sources import check_payload
    parent,_=verdict_parent;summary=json.loads((parent/'summary.json').read_text());identity=json.loads((parent/'run.json').read_text())['identity']
    payload={**summary,'identity':identity,'review_semantically_replayed':True,
             'working_status_counts':{'pending':summary['task_rows'],'in_progress':0,'reviewed':0}}
    if kind=='unknown':payload['unknown']=0
    elif kind=='bool_count':payload['working_status_counts']['reviewed']=False
    elif kind=='negative':payload['working_status_counts']['reviewed']=-1
    elif kind=='missing':del payload['working_status_counts']['reviewed']
    else:payload['working_status_counts']['pending']-=1
    with pytest.raises(ValueError):check_payload(payload,summary,identity)

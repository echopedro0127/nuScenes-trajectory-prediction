import json

import pytest
from test_scenario_event_pipeline import CONFIG
from test_scenario_event_pipeline import event_parent as event_parent  # noqa: PLC0414
from test_scenario_tracks import track_parent as track_parent  # noqa: PLC0414

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation import scenario_events
from trajectory_prediction.utils.provenance import sha256_file


@pytest.fixture
def review_parent(event_parent,tmp_path,monkeypatch):
    parent,owner,_=event_parent
    b1_owner=tmp_path/'b1-owner';script=b1_owner/'scripts/prepare_scenario_events.py'
    script.parent.mkdir(parents=True)
    script.write_text("import argparse,json\nfrom pathlib import Path\np=argparse.ArgumentParser()\n"
      "p.add_argument('--output-root');p.add_argument('--verify-only',action='store_true')\n"
      "a=p.parse_args();r=Path(a.output_root);s=json.loads((r/'summary.json').read_text())\n"
      "print(json.dumps({'stage':'offline_events','role':'independent_fit'}))\n"
      "print(json.dumps({'stage':'offline_events','role':'independent_select'}))\n"
      "i=json.loads((r/'run.json').read_text())['identity']\n"
      "print(json.dumps({**s,'events_semantically_replayed':True,'identity':i}))\n")
    output=tmp_path/'b1'
    with monkeypatch.context() as m:
        m.setattr(scenario_events,'PROJECT',b1_owner)
        scenario_events.prepare_events(parent,owner,CONFIG,output)
    return output,b1_owner


def test_review_roundtrip_mutable_notes_not_truth(review_parent,tmp_path):
    from trajectory_prediction.evaluation.event_review import prepare_review, verify_review
    from trajectory_prediction.evaluation.event_review_data import csv_text
    parent,owner=review_parent;out=tmp_path/'review'
    result=prepare_review(parent,owner,out)
    assert result['block_scan_tasks']==result['task_rows']==4
    assert verify_review(out)['review_semantically_replayed'] is True
    baseline=json.loads((out/'queue.json').read_text())
    baseline[0].update(review_status='reviewed',reviewer='reader',notes='reading only')
    (out/'review_working.csv').write_text(csv_text(baseline),encoding='utf-8-sig',newline='')
    verified=verify_review(out)
    assert verified['working_status_counts']['reviewed']==1
    assert verified['manual_completed']==0 and verified['human_event_labels_imported'] is False
    with pytest.raises(FileExistsError): prepare_review(parent,owner,out)


@pytest.mark.parametrize('kind',['data','html','summary','extra','binding','mutable_policy','proof'])
def test_rehashed_static_changes_refused(review_parent,tmp_path,kind):
    from trajectory_prediction.evaluation.event_review import prepare_review, verify_review
    parent,owner=review_parent;out=tmp_path/'review';prepare_review(parent,owner,out)
    marker=json.loads((out/'completed.json').read_text())
    if kind=='mutable_policy':
        marker['mutable_files'].append('review.html')
    else:
        if kind=='html': path=out/'review.html';path.write_text('changed')
        else:
            name={'data':'review_data.json','summary':'summary.json','extra':'extra.json',
                  'binding':'run.json','proof':'parent_proof.json'}[kind]
            path=out/name
            value=json.loads(path.read_text()) if path.exists() else {}
            if kind=='data': value['blocks'][0]['xy'][0][0]+=1
            elif kind=='summary': value['manual_completed']=1
            elif kind=='binding':
                value['binding']['human_event_labels_imported']=0
                value['identity']=digest(value['binding']);marker['identity']=value['identity']
            elif kind=='proof':
                lines=value['stdout'].splitlines();payload=json.loads(lines[-1]);payload['events_semantically_replayed']=1
                value['stdout']='\n'.join([*lines[:2],json.dumps(payload)])
            path.write_text(json.dumps(value))
        marker['sha256'][path.name]=sha256_file(path)
    (out/'completed.json').write_text(json.dumps(marker))
    with pytest.raises(ValueError): verify_review(out)


def test_working_source_identity_and_wrong_owner(review_parent,tmp_path):
    from trajectory_prediction.evaluation.event_review import prepare_review, verify_review
    from trajectory_prediction.evaluation.event_review_data import csv_text
    parent,owner=review_parent
    with pytest.raises(ValueError,match='owner'): prepare_review(parent,tmp_path/'wrong',tmp_path/'bad')
    out=tmp_path/'review';prepare_review(parent,owner,out)
    baseline=json.loads((out/'queue.json').read_text());baseline[0]['role']='other'
    (out/'review_working.csv').write_text(csv_text(baseline),encoding='utf-8-sig')
    with pytest.raises(ValueError): verify_review(out)


def test_review_interruption_without_completed(review_parent,tmp_path,monkeypatch):
    from trajectory_prediction.evaluation import event_review as mod
    parent,owner=review_parent
    def fail(*args): raise RuntimeError('interrupted')
    monkeypatch.setattr(mod,'_write_bytes',fail)
    out=tmp_path/'partial'
    with pytest.raises(RuntimeError): mod.prepare_review(parent,owner,out)
    assert not (out/'completed.json').exists()


@pytest.mark.parametrize('text',['{}','{}\n{}\n{}','not json'])
def test_bad_native_stream_rejected(text):
    from trajectory_prediction.evaluation.review_sources import native_payload
    with pytest.raises(ValueError): native_payload(text)

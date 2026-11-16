import csv
import io
import json

import numpy as np
import pytest

from trajectory_prediction.utils.provenance import sha256_file


@pytest.fixture
def review_fixture(tmp_path):
    parent, b0 = tmp_path/'b1', tmp_path/'b0'
    parent.mkdir(); b0.mkdir()
    common = {'role': 'independent_fit', 'instance': 'i', 'scene': 'scene', 'log': 'log', 'category': 'vehicle.car'}
    blocks = [{**common, 'block_id':'b0', 'segment_id':'seg0', 'first_annotation':'a0',
               'last_annotation':'a2', 'start_us':1000000,'end_us':2000000,'interval_count':2},
              {**common, 'block_id':'b1', 'segment_id':'seg1', 'first_annotation':'a3',
               'last_annotation':'a3', 'start_us':3000000,'end_us':3000000,'interval_count':0}]
    event = {**common,'event_id':'event','block_id':'b0','event_type':'stop_candidate',
             'onset_us':1500000,'confirmation_us':2000000,'end_us':2000000,'right_censored':True}
    documents = {'blocks.json':blocks,'events.json':[event],
                 'diagnostics.json':[{**common,'diagnostic_id':'diagnostic','block_id':'b1','reason':'no_velocity_interval'}],
                 'window_event_links.json':[], 'summary.json':{'coverage':[{'events_total':1}], 'manual_completed':0},
                 'run.json':{'identity':'b1-identity','binding':{'parent':str(b0)}}}
    for name,value in documents.items(): (parent/name).write_text(json.dumps(value),encoding='utf-8')
    refs = {}
    for role in ('independent_fit','independent_select'):
        fit = role=='independent_fit'
        path=b0/(role+'-tracks.npz')
        np.savez_compressed(path,timestamp_us=np.array([1000000,1500000,2000000,3000000] if fit else [],np.int64),
            world_xy=np.array([[0.,0.],[1.,0.],[1.,0.],[2.,0.]] if fit else [],np.float64).reshape(-1,2),
            annotation_token=np.array(['a0','a1','a2','a3'] if fit else [],dtype='U32'),
            sample_token=np.array(['s0','s1','s2','s3'] if fit else [],dtype='U32'),
            instance_token=np.array(['i']*4 if fit else [],dtype='U32'),
            segment_id=np.array(['seg0']*3+['seg1'] if fit else [],dtype='U64'))
        refs[str(path.resolve())]=sha256_file(path)
        path=parent/(role+'-kinematics.npz')
        np.savez_compressed(path,block_id=np.array(['b0','b0'] if fit else [],dtype='U64'),
            left_token=np.array(['a0','a1'] if fit else [],dtype='U32'),
            right_token=np.array(['a1','a2'] if fit else [],dtype='U32'),
            start_us=np.array([1000000,1500000] if fit else [],np.int64),
            end_us=np.array([1500000,2000000] if fit else [],np.int64),
            speed_mps=np.array([2.,0.] if fit else [],np.float64), valid=np.array([True,True] if fit else [],bool))
        refs[str(path.resolve())]=sha256_file(path)
    return parent, refs


def test_complete_queue_and_exact_source_points(review_fixture):
    from trajectory_prediction.evaluation.event_review_data import assemble_review
    data = assemble_review(*review_fixture)
    assert len(data['queue']) == 3
    assert {t['task_kind'] for t in data['queue']} == {'block_scan','event_check'}
    assert data['summary']['blocks_without_candidates'] == 1
    assert data['summary']['diagnostic_rows'] == 1
    block=next(b for b in data['blocks'] if b['block_id']=='b0')
    assert block['xy']==[[0.,0.],[1.,0.],[1.,0.]]
    assert block['time_us']==[1000000,1500000,2000000]
    assert block['speed_mps']==[2.,0.]
    assert data['coverage']==[{'events_total':1}]
    assert data['summary']['manual_completed']==0


@pytest.mark.parametrize('kind',['unbound_points','foreign_event'])
def test_bad_review_source_rejected(review_fixture,kind):
    from trajectory_prediction.evaluation.event_review_data import assemble_review
    parent,refs=review_fixture
    if kind=='unbound_points': refs.clear()
    else:
        value=json.loads((parent/'events.json').read_text());value[0]['block_id']='foreign'
        (parent/'events.json').write_text(json.dumps(value))
    with pytest.raises(ValueError): assemble_review(parent,refs)


def working(rows):
    from trajectory_prediction.evaluation.event_review_data import csv_text
    return csv_text(rows)


def test_working_notes_sort_and_quoted_multiline(review_fixture):
    from trajectory_prediction.evaluation.event_review_data import assemble_review, validate_working
    rows=assemble_review(*review_fixture)['queue']
    rows=[dict(r) for r in reversed(rows)]
    rows[0].update(review_status='reviewed',reviewer='reader',notes='comma, quote "\nand newline')
    counts=validate_working(working(rows),assemble_review(*review_fixture)['queue'])
    assert counts=={'pending':2,'in_progress':0,'reviewed':1}


@pytest.mark.parametrize('kind',['locked','missing','duplicate','status','reviewer','extra_column'])
def test_invalid_working_sheet_rejected(review_fixture,kind):
    from trajectory_prediction.evaluation.event_review_data import assemble_review, validate_working
    baseline=assemble_review(*review_fixture)['queue'];rows=[dict(r) for r in baseline]
    if kind=='locked': rows[0]['instance']='other'
    elif kind=='missing': rows.pop()
    elif kind=='duplicate': rows[1]=dict(rows[0])
    elif kind=='status': rows[0]['review_status']='confirmed_ground_truth'
    elif kind=='reviewer': rows[0].update(review_status='reviewed',notes='read')
    text=working(rows)
    if kind=='extra_column':
        parsed=list(csv.reader(io.StringIO(text)));parsed[0].append('extra')
        out=io.StringIO();csv.writer(out).writerows(parsed);text=out.getvalue()
    with pytest.raises(ValueError): validate_working(text,baseline)

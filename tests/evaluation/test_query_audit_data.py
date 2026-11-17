import copy
import hashlib

import numpy as np
import pytest


@pytest.fixture
def audit_data():
    times=10000000+np.arange(18,dtype=np.int64)*500000
    tokens=np.array(['t'+str(i) for i in range(18)],'U32')
    samples=np.array(['s'+str(i) for i in range(18)],'U32')
    common={'role':'independent_fit','instance':'i','scene':'scene','log':'log','category':'vehicle.car'}
    windows=[]
    for i in range(2):
        sample=str(samples[4+i])
        windows.append({**common,'sample':sample,'window_id':hashlib.sha256(f'independent_fit:i:{sample}'.encode()).hexdigest(),
                        'row':i,'role_window_row':i,'anchor_annotation_token':str(tokens[4+i]),
                        'contiguous_annotation_segment':True,'status':'ok','reason':None})
    arrays={'actual_timestamp_us':np.stack([times[:17],times[1:]]),
            'annotation_token':np.stack([tokens[:17],tokens[1:]]),
            'sample_token':np.stack([samples[:17],samples[1:]]),
            'actual_offset_us':np.stack([times[:17]-times[4],times[1:]-times[5]]),
            'nominal_step':np.arange(-4,13,dtype=np.int16)}
    points={'timestamp_us':times,'annotation_token':tokens,'sample_token':samples,
            'instance_token':np.array(['i']*18,'U32'),'segment_id':np.array(['seg']*18,'U64'),
            'world_xy':np.column_stack([np.arange(18),np.zeros(18)]).astype(np.float64)}
    blocks=[{**common,'block_id':'block','segment_id':'seg','start_us':int(times[0]),'end_us':int(times[-1]),
             'first_annotation':'t0','last_annotation':'t17','interval_count':17}]
    events=[{**common,'event_id':'stop','event_type':'stop_candidate','block_id':'block',
             'onset_us':int(times[9]),'confirmation_us':int(times[11]),'end_us':int(times[-1]),
             'onset_token':'t9','confirmation_token':'t11','end_token':'t17',
             'right_censored':True,'low_end_censored':True}]
    return {'dataset_release':'v1.0-trainval','windows':windows,'window_arrays':{'independent_fit':arrays},
            'track_arrays':{'independent_fit':points},'tracks':[{**common,'role_row_start':0,'role_row_stop':18}],
            'blocks':blocks,'events':events}


def test_query_mapping_event_and_namespace_denominators(audit_data):
    from trajectory_prediction.evaluation.query_audit_data import derive_audit
    result=derive_audit(audit_data)
    s=result['documents']['summary.json']
    assert s['windows']==2 and s['queries']==22 and s['matching_attempts']==66
    assert s['status_counts']['matched']==58 and s['status_counts']['beyond_window_horizon']==8
    assert sum(s['status_counts'].values())==s['matching_attempts']
    assert s['source_packets']==18
    namespace=result['documents']['source_packets.json']
    assert len({p['key_digest'] for p in namespace})==18
    event=result['documents']['event_coverage.json'][0]
    assert event['event_id']=='stop' and event['linked_windows']==2
    assert event['pre_lead_covered_us']==[500000,1000000]
    assert event['right_censored'] is True and event['label_source']=='rule_candidate'
    links=result['documents']['event_query_links.json']
    assert sum(x['pre_lead_us']==500000 for x in links)==2
    assert event['pre_lead_match_counts']['500000']['query_links']==2
    assert event['pre_lead_match_counts']['500000']['unique_source_queries']==1
    assert event['pre_lead_match_counts']['1000000']['confirmation_covered_links']==2
    assert event['coverage_reasons']==['human_review_not_supplied']
    assert s['new_prediction_scores_computed'] is False


def test_event_without_window_is_retained(audit_data):
    from trajectory_prediction.evaluation.query_audit_data import derive_audit
    e=copy.deepcopy(audit_data['events'][0]);e.update(event_id='later',onset_us=20000000,confirmation_us=21000000,end_us=22000000)
    # Make a separate singleton block and a valid later event's points in a different instance.
    # Coverage helper can be tested directly for events that have no eligible windows.
    from trajectory_prediction.evaluation.query_audit_data import event_coverage
    summary,links=event_coverage([e],[])
    assert summary[0]['linked_windows']==0 and summary[0]['pre_lead_covered_us']==[] and links==[]
    assert 'no_existing_anchor' in summary[0]['coverage_reasons']
    assert derive_audit(audit_data)['documents']['summary.json']['events']==1


@pytest.mark.parametrize('kind',['duplicate_window','wrong_id','wrong_clock','wrong_sample','wrong_instance',
                                 'wrong_scene','duplicate_token','nonfinite','gap','wrong_row','bad_event','role_log'])
def test_corrupt_sources_fail_closed(audit_data,kind):
    from trajectory_prediction.evaluation.query_audit_data import derive_audit
    data=audit_data;w=data['windows'][0];a=data['window_arrays']['independent_fit'];p=data['track_arrays']['independent_fit']
    if kind=='duplicate_window':data['windows'][1]=dict(w)
    elif kind=='wrong_id':w['window_id']='forged'
    elif kind=='wrong_clock':a['actual_timestamp_us'][0,0]+=1
    elif kind=='wrong_sample':a['sample_token'][0,4]='foreign'
    elif kind=='wrong_instance':p['instance_token'][2]='other'
    elif kind=='wrong_scene':w['scene']='foreign'
    elif kind=='duplicate_token':p['annotation_token'][1]=p['annotation_token'][0]
    elif kind=='nonfinite':p['world_xy'][0,0]=np.nan
    elif kind=='gap':p['segment_id'][8]='other'
    elif kind=='wrong_row':w['role_window_row']=True
    elif kind=='bad_event':data['events'][0]['confirmation_token']='foreign'
    elif kind=='role_log':
        other=copy.deepcopy(data['tracks'][0]);other.update(role='independent_select');data['tracks'].append(other)
    with pytest.raises(ValueError):derive_audit(data)


def test_namespace_is_window_order_independent(audit_data):
    from trajectory_prediction.evaluation.query_audit_data import derive_audit
    first=derive_audit(audit_data)['documents']['source_packets.json']
    audit_data['windows'].reverse()
    assert derive_audit(audit_data)['documents']['source_packets.json']==first


def test_group_unique_queries_targets_and_conservation(audit_data):
    from trajectory_prediction.evaluation.query_audit_data import derive_audit
    summary=derive_audit(audit_data)['documents']['summary.json']
    groups=[g for g in summary['groups'] if g['log']=='__all__' and g['role']=='independent_fit']
    assert [g['unique_target_annotations'] for g in groups]==[12,11,9]
    assert all(g['unique_query_source_packets']==12 for g in groups)
    all_groups=[g for g in summary['groups'] if g['log']=='__all__']
    assert sum(g['matching_attempts'] for g in all_groups)==summary['matching_attempts']
    for status,value in summary['status_counts'].items():
        assert sum(g['status_counts'][status] for g in all_groups)==value


@pytest.mark.parametrize('kind',['float_event','float_block','bool_interval','blank_release'])
def test_strict_source_integer_types(audit_data,kind):
    from trajectory_prediction.evaluation.query_audit_data import derive_audit
    if kind=='float_event':audit_data['events'][0]['onset_us']=float(audit_data['events'][0]['onset_us'])
    elif kind=='float_block':audit_data['blocks'][0]['start_us']=float(audit_data['blocks'][0]['start_us'])
    elif kind=='bool_interval':audit_data['blocks'][0]['interval_count']=True
    else:audit_data['dataset_release']=''
    with pytest.raises(ValueError):derive_audit(audit_data)

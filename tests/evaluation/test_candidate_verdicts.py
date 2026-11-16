from itertools import product

import pytest


@pytest.fixture
def source_data():
    common={'role':'independent_fit','instance':'i','block_id':'b','scene':'scene','log':'log','category':'vehicle.car'}
    stop={**common,'event_id':'s','event_type':'stop_candidate','onset_us':1000000,'confirmation_us':2000000,
          'end_us':4000000,'right_censored':False,'low_end_censored':False,'low_dwell_end_us':3000000}
    restart={**common,'event_id':'r','event_type':'restart_candidate','onset_us':4000000,'confirmation_us':5000000,
             'end_us':5000000,'right_censored':False,'stop_id':'s'}
    pair={**common,'event_id':'p','event_type':'stop_go_candidate','onset_us':1000000,'confirmation_us':5000000,
          'end_us':5000000,'right_censored':False,'stop_id':'s','restart_id':'r'}
    other={**stop,'event_id':'s2','role':'independent_select','instance':'j','block_id':'c',
           'category':'vehicle.truck','right_censored':True,'low_end_censored':True,'low_dwell_end_us':4000000}
    events=[stop,restart,pair,other]
    queue=[{'task_id':'task-'+e['event_id'],'task_kind':'event_check','role':e['role'],'instance':e['instance'],
            'block_id':e['block_id'],'event_id':e['event_id'],'review_status':'reviewed'} for e in events]
    return {'events':events,'queue':queue,'blocks':[common,{**common,'role':'independent_select','instance':'j',
            'block_id':'c','category':'vehicle.truck'}]}


def test_public_derivation_cannot_bypass_locked_declaration_checks(source_data):
    from trajectory_prediction.evaluation.candidate_verdicts import derive_overlay, make_template
    rows=make_template('review',source_data)
    rows[0].update(role='foreign',source_task_id='forged',decision='confirmed')
    with pytest.raises(ValueError):derive_overlay(source_data,rows,'review')


def mark(rows,key,decision):
    row=next(r for r in rows if r['source_event_id']==key)
    row.update(decision=decision,reviewer='' if decision=='pending' else 'fixture-reader',
               reason='' if decision=='pending' else 'synthetic test declaration')


def test_template_atomic_not_reading_progress(source_data):
    from trajectory_prediction.evaluation.candidate_verdicts import make_template
    rows=make_template('review',source_data)
    assert len(rows)==3 and {r['source_event_id'] for r in rows}=={'s','r','s2'}
    assert all(r['decision']=='pending' and r['reviewer']==r['reason']=='' for r in rows)


@pytest.mark.parametrize('stop,restart',list(product(['confirmed','rejected','pending','needs_correction','uncertain'],repeat=2)))
def test_all_composite_priorities(source_data,stop,restart):
    from trajectory_prediction.evaluation.candidate_verdicts import (
        csv_text,
        derive_overlay,
        make_template,
        parse_verdicts,
    )
    rows=make_template('review',source_data);mark(rows,'s',stop);mark(rows,'r',restart)
    rows=parse_verdicts(csv_text(list(reversed(rows))),make_template('review',source_data))
    result=derive_overlay(source_data,rows,'review')
    expected=next((d for d in ['rejected','needs_correction','uncertain','pending'] if d in (stop,restart)),'confirmed')+'_components'
    assert result['derived_pairs'][0]['component_status']==expected
    assert result['summary']['atomic_total']==3
    assert sum(result['summary']['decision_counts'].values())==3
    assert len(result['confirmed_candidates'])==int(stop=='confirmed')+int(restart=='confirmed')


def test_confirmed_retains_original_censor_and_semantic_order(source_data):
    from trajectory_prediction.analysis.track_timeline import digest
    from trajectory_prediction.evaluation.candidate_verdicts import (
        csv_text,
        derive_overlay,
        make_template,
        parse_verdicts,
    )
    base=make_template('review',source_data);rows=[dict(r) for r in base];mark(rows,'s2','confirmed')
    a=parse_verdicts(csv_text(rows),base);b=parse_verdicts(csv_text(list(reversed(rows))),base)
    assert digest(a)==digest(b)
    result=derive_overlay(source_data,a,'review')
    assert result['confirmed_candidates'][0]['source_event']['right_censored'] is True
    assert next(e for e in source_data['events'] if e['event_id']=='s2')['right_censored'] is True
    a[0]['reason']+='changed';assert digest(a)!=digest(b)


@pytest.mark.parametrize('kind',['missing','duplicate','locked','unknown','pending_note','blank_reviewer','long','extra_column','bad_quotes'])
def test_invalid_verdict_csv_rejected(source_data,kind):
    from trajectory_prediction.evaluation.candidate_verdicts import (
        csv_text,
        make_template,
        parse_verdicts,
    )
    base=make_template('review',source_data);rows=[dict(r) for r in base]
    if kind=='missing':rows.pop()
    elif kind=='duplicate':rows[1]=dict(rows[0])
    elif kind=='locked':rows[0]['role']='other'
    elif kind=='unknown':rows[0]['decision']='reviewed'
    elif kind=='pending_note':rows[0]['reason']=' '
    elif kind=='blank_reviewer':rows[0].update(decision='confirmed',reviewer=chr(65279),reason='test')
    elif kind=='long':rows[0].update(decision='confirmed',reviewer='reader',reason=chr(0x1f697)*8193)
    text=csv_text(rows)
    if kind=='extra_column':text=text.replace('source_review_identity,','extra,source_review_identity,',1)
    elif kind=='bad_quotes':text=text.rstrip('\r\n')+'"unterminated'
    with pytest.raises(ValueError):parse_verdicts(text,base)


@pytest.mark.parametrize('kind',['duplicate_event','missing_task','wrong_pair','bad_time','cross_role','censor_conflict'])
def test_broken_source_relation_rejected(source_data,kind):
    from trajectory_prediction.evaluation.candidate_verdicts import make_template
    if kind=='duplicate_event':source_data['events'].append(dict(source_data['events'][0]))
    elif kind=='missing_task':source_data['queue'].pop(0)
    elif kind=='wrong_pair':source_data['events'][2]['restart_id']='s'
    elif kind=='bad_time':source_data['events'][2]['end_us']+=1
    elif kind=='cross_role':source_data['events'][1]['role']='independent_select'
    else:source_data['events'][0]['right_censored']=True
    with pytest.raises(ValueError):make_template('review',source_data)

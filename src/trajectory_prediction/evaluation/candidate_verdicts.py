"""Candidate-level human declarations; never infer labels from reading progress."""
import copy
import csv
import io
from collections import Counter, defaultdict

from trajectory_prediction.evaluation.event_review_data import (
    MAX_CELL_CODEPOINTS,
    REVIEW_WHITESPACE,
)

LOCKED=('source_review_identity','source_task_id','source_event_id','source_event_type','role','instance','block_id')
COLUMNS=(*LOCKED,'decision','reviewer','reason')
DECISIONS=('pending','confirmed','rejected','needs_correction','uncertain')
PRIORITY=('rejected','needs_correction','uncertain','pending','confirmed')
ATOMIC=('stop_candidate','restart_candidate')


def _unique(rows,key):
    result={r[key]:r for r in rows}
    if len(result)!=len(rows): raise ValueError('duplicate verdict source '+key)
    return result


def source_candidates(data):
    events=_unique(data['events'],'event_id');blocks=_unique(data['blocks'],'block_id')
    tasks=_unique(data['queue'],'task_id');event_tasks=defaultdict(list)
    for e in events.values():
        b=blocks.get(e['block_id'])
        if (b is None or e['event_type'] not in (*ATOMIC,'stop_go_candidate')
            or e['role'] not in ('independent_fit','independent_select')
            or any(e[k]!=b[k] for k in ('role','instance','scene','log','category'))
            or type(e['right_censored']) is not bool
            or any(type(e[k]) is not int for k in ('onset_us','confirmation_us','end_us'))
            or not 0<e['onset_us']<e['confirmation_us']<=e['end_us']):
            raise ValueError('invalid verdict source event/block/boundaries')
        if e['event_type']=='stop_candidate' and (type(e['low_end_censored']) is not bool
            or type(e['low_dwell_end_us']) is not int
            or not e['confirmation_us']<=e['low_dwell_end_us']<=e['end_us']
            or (e['low_end_censored'] and e['low_dwell_end_us']!=e['end_us'])):
            raise ValueError('invalid original stop censoring')
    for t in tasks.values():
        if t['task_kind']=='event_check':
            e=events.get(t['event_id'])
            if e is None or any(t[k]!=e[k] for k in ('role','instance','block_id')):
                raise ValueError('verdict source task foreign key mismatch')
            event_tasks[e['event_id']].append(t)
    if set(event_tasks)!=set(events) or any(len(v)!=1 for v in event_tasks.values()):
        raise ValueError('source event/task one-to-one coverage required')
    atomic={k:e for k,e in events.items() if e['event_type'] in ATOMIC}
    pairs=[e for e in events.values() if e['event_type']=='stop_go_candidate']
    stops,restarts=set(),set()
    for p in pairs:
        s,r=atomic.get(p['stop_id']),atomic.get(p['restart_id'])
        if (s is None or r is None or s['event_type']!='stop_candidate' or r['event_type']!='restart_candidate'
            or r.get('stop_id')!=s['event_id'] or s['event_id'] in stops or r['event_id'] in restarts
            or any(s[k]!=r[k] or s[k]!=p[k] for k in ('role','instance','block_id','scene','log','category'))
            or s['right_censored'] is not False or r['right_censored'] is not False or p['right_censored'] is not False
            or s['end_us']!=r['onset_us'] or p['onset_us']!=s['onset_us']
            or r['confirmation_us']!=r['end_us'] or p['confirmation_us']!=r['confirmation_us']
            or p['end_us']!=r['confirmation_us']):
            raise ValueError('source composite relation/censor/time conflict')
        stops.add(s['event_id']);restarts.add(r['event_id'])
    if restarts!={k for k,e in atomic.items() if e['event_type']=='restart_candidate'}:
        raise ValueError('source restart/pair coverage mismatch')
    if stops!={k for k,e in atomic.items() if e['event_type']=='stop_candidate' and not e['right_censored']}:
        raise ValueError('source complete stop/pair coverage mismatch')
    return atomic,sorted(pairs,key=lambda p:p['event_id']),event_tasks


def make_template(identity,data):
    atomic,_,tasks=source_candidates(data)
    if not isinstance(identity,str) or not identity: raise ValueError('explicit review identity required')
    return [{'source_review_identity':identity,'source_task_id':tasks[key][0]['task_id'],
             'source_event_id':key,'source_event_type':e['event_type'],
             **{k:e[k] for k in ('role','instance','block_id')},'decision':'pending','reviewer':'','reason':''}
            for key,e in sorted(atomic.items())]


def csv_text(rows):
    out=io.StringIO(newline='');writer=csv.DictWriter(out,fieldnames=COLUMNS,lineterminator='\r\n')
    writer.writeheader();writer.writerows(rows);return out.getvalue()


def parse_verdicts(text,baseline):
    reader=csv.DictReader(io.StringIO(text.lstrip('\ufeff'),newline=''),strict=True)
    try: rows=list(reader)
    except csv.Error as error: raise ValueError('malformed verdict CSV') from error
    if reader.fieldnames!=list(COLUMNS): raise ValueError('exact verdict CSV columns required')
    return validate_records(rows,baseline)


def validate_records(rows,baseline):
    expected=_unique(baseline,'source_event_id');seen=set()
    for row in rows:
        key=row.get('source_event_id')
        if (set(row)!=set(COLUMNS) or any(not isinstance(v,str) for v in row.values())
            or key not in expected or key in seen or any(row[k]!=expected[key][k] for k in LOCKED)):
            raise ValueError('verdict source identity/row coverage mismatch')
        if any(len(v)>MAX_CELL_CODEPOINTS for v in row.values()): raise ValueError('verdict cell too long')
        d=row['decision']
        if d not in DECISIONS: raise ValueError('unknown candidate verdict')
        if d=='pending':
            if row['reviewer']!='' or row['reason']!='': raise ValueError('pending verdict must have empty declaration')
        elif not row['reviewer'].strip(REVIEW_WHITESPACE) or not row['reason'].strip(REVIEW_WHITESPACE):
            raise ValueError('non-pending verdict requires reviewer and reason')
        seen.add(key)
    if seen!=set(expected): raise ValueError('missing candidate verdict rows')
    return sorted(rows,key=lambda r:r['source_event_id'])


def _counts(rows):
    counts=Counter(r['decision'] for r in rows)
    return {'atomic_total':len(rows),'decision_counts':{k:counts[k] for k in DECISIONS},
            'non_pending_verdict_rows':len(rows)-counts['pending'],
            'unresolved_rows':sum(counts[k] for k in ('pending','needs_correction','uncertain'))}


def derive_overlay(data,rows,review_identity):
    atomic,pairs,_=source_candidates(data)
    rows=validate_records(rows,make_template(review_identity,data))
    by_id=_unique(rows,'source_event_id')
    if set(by_id)!=set(atomic) or any(r['decision'] not in DECISIONS for r in rows):
        raise ValueError('normalized verdict/source coverage mismatch')
    confirmed,rejected,unresolved=[],[],[]
    for row in rows:
        record={'source_event':copy.deepcopy(atomic[row['source_event_id']]),'verdict':dict(row)}
        target=confirmed if row['decision']=='confirmed' else rejected if row['decision']=='rejected' else unresolved
        target.append(record)
    derived=[]
    for p in pairs:
        s,r=by_id[p['stop_id']]['decision'],by_id[p['restart_id']]['decision']
        status=next(d for d in PRIORITY if d in (s,r))+'_components'
        derived.append({'source_pair':copy.deepcopy(p),'source_pair_id':p['event_id'],
                        'stop_decision':s,'restart_decision':r,'component_status':status})
    groups=[]
    for role in ('independent_fit','independent_select'):
        cats=sorted({b['category'] for b in data['blocks'] if b['role']==role})
        for cat in ['__all__',*cats]:
            for kind in ATOMIC:
                selected=[r for r in rows if r['role']==role and r['source_event_type']==kind
                          and (cat=='__all__' or atomic[r['source_event_id']]['category']==cat)]
                groups.append({'role':role,'category':cat,'event_type':kind,**_counts(selected)})
    summary={'status':'candidate_verdicts_imported' if any(r['decision']!='pending' for r in rows) else 'no_human_verdicts',
             **_counts(rows),'pair_total':len(pairs),
             'derived_pair_counts':{d+'_components':sum(p['component_status']==d+'_components' for p in derived) for d in PRIORITY},
             'confirmed_original_stop_censor_counts':{
                 'right_censored':sum(r['source_event']['event_type']=='stop_candidate' and r['source_event']['right_censored'] for r in confirmed),
                 'low_end_censored':sum(r['source_event']['event_type']=='stop_candidate' and r['source_event']['low_end_censored'] for r in confirmed)},
             'groups':groups,'human_candidate_decisions_imported':any(r['decision']!='pending' for r in rows),
             'exhaustive_human_ground_truth':False,'new_prediction_scores_computed':False,
             'neural_training':False,'holdout_evaluated':False}
    if sum(summary['decision_counts'].values())!=len(atomic) or sum(summary['derived_pair_counts'].values())!=len(pairs):
        raise ValueError('verdict count conservation failed')
    totals=[g for g in groups if g['category']=='__all__']
    if (sum(g['atomic_total'] for g in totals)!=len(atomic)
        or any(sum(g['decision_counts'][d] for g in totals)!=summary['decision_counts'][d] for d in DECISIONS)):
        raise ValueError('verdict role/type denominator conservation failed')
    return {'atomic_verdicts':copy.deepcopy(rows),'confirmed_candidates':confirmed,'rejected_candidates':rejected,
            'unresolved_candidates':unresolved,'derived_pairs':derived,'summary':summary}

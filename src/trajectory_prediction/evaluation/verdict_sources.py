"""Frozen B2a static evidence with separately validated mutable reading progress."""
import json
import os
import subprocess
import sys
from pathlib import Path

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation import event_review_data
from trajectory_prediction.evaluation.event_sources import parse_json, read_json
from trajectory_prediction.evaluation.track_sources import add_reference, check_references, sources
from trajectory_prediction.utils.provenance import sha256_file

MUTABLE=['review_working.csv']
VALIDATOR='src/trajectory_prediction/evaluation/event_review_data.py'


def check_payload(payload,summary,identity):
    if (not isinstance(payload,dict)
        or set(payload)!=set(summary)|{'identity','review_semantically_replayed','working_status_counts'}
        or payload.get('review_semantically_replayed') is not True):
        raise ValueError('B2a native proof fields/flag mismatch')
    counts=payload['working_status_counts']
    if (not isinstance(counts,dict) or set(counts)!=set(event_review_data.STATUSES)
        or any(type(v) is not int or v<0 for v in counts.values())
        or sum(counts.values())!=summary['task_rows']):
        raise ValueError('invalid B2a proof reading counts')
    static={k:v for k,v in payload.items() if k!='working_status_counts'}
    if digest(static)!=digest({**summary,'identity':identity,'review_semantically_replayed':True}):
        raise ValueError('B2a proof static projection mismatch')
    return dict(counts)


def _static_parent(parent,refs):
    marker=read_json(parent/'completed.json')
    if marker.get('mutable_files')!=MUTABLE: raise ValueError('B2a mutable policy mismatch')
    actual={p.relative_to(parent).as_posix() for p in parent.rglob('*') if p.is_file()}
    if actual!=set(marker['sha256'])|set(MUTABLE)|{'completed.json'}:
        raise ValueError('B2a file coverage mismatch')
    if set(marker['sha256']) & set(MUTABLE): raise ValueError('B2a working bytes must not be static')
    for name,expected in marker['sha256'].items():
        path=(parent/name).resolve()
        if not path.is_relative_to(parent): raise ValueError('B2a static file escapes parent')
        add_reference(refs,path,expected)
    add_reference(refs,parent/'completed.json')
    protocol=read_json(parent/'protocol.json')
    if protocol.get('mutable_files')!=MUTABLE: raise ValueError('B2a protocol mutable policy mismatch')
    return marker


def _working(parent,owner,binding):
    if VALIDATOR not in binding['sources']: raise ValueError('B2a working validator version not bound')
    original=(owner/VALIDATOR).read_bytes()
    current=Path(event_review_data.__file__).read_bytes()
    # Python normalizes source CRLF when loading. No code or format changes are allowed.
    if original.replace(b'\r\n',b'\n')!=current.replace(b'\r\n',b'\n'):
        raise ValueError('B2a working validator version changed')
    path=(parent/MUTABLE[0]).resolve()
    if not path.is_relative_to(parent): raise ValueError('B2a mutable file escapes parent')
    return event_review_data.validate_working(path.read_text(encoding='utf-8-sig'),read_json(parent/'queue.json'))


def parent_snapshot(parent,owner,proof=None):
    parent,owner=Path(parent).resolve(),Path(owner).resolve()
    script=owner/'scripts/prepare_event_review.py'
    if not script.is_file(): raise ValueError('explicit B2a owner verifier required')
    refs={};marker=_static_parent(parent,refs)
    run,summary=read_json(parent/'run.json'),read_json(parent/'summary.json');binding=run['binding']
    if (run['identity']!=marker['identity'] or digest(binding)!=run['identity']
        or binding['schema']!='event_review_b2a_v1' or summary['status']!='review_materials_ready'
        or type(summary['task_rows']) is not int or summary['task_rows']<0
        or type(summary['manual_completed']) is not int or summary['manual_completed']!=0
        or any(binding[k] is not False or summary[k] is not False for k in
               ('human_event_labels_imported','neural_training','holdout_evaluated','new_prediction_scores_computed'))):
        raise ValueError('frozen B2a source identity/policy required')
    if sources(owner)!=binding['sources']: raise ValueError('B2a owner source mismatch')
    for name,expected in binding['sources'].items():
        path=(owner/name).resolve()
        if not path.is_relative_to(owner): raise ValueError('B2a source escapes owner')
        add_reference(refs,path,expected)
    for path,expected in binding['references'].items():add_reference(refs,path,expected)
    _working(parent,owner,binding)
    command=[sys.executable,str(script),'--output-root',str(parent),'--verify-only']
    expected={'owner_root':str(owner),'pythonpath':str(owner/'src'),'command':command,
              'parent_identity':run['identity'],'parent_completed_sha256':sha256_file(parent/'completed.json')}
    if proof is None:
        print(json.dumps({'stage':'b2a_owner_verification','state':'started'}),flush=True)
        env={**os.environ,'PYTHONPATH':str(owner/'src'),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        result=subprocess.run(command,cwd=owner,env=env,capture_output=True,text=True,encoding='utf-8',check=False)
        if result.returncode: raise ValueError('B2a native verifier failed: '+result.stderr[-2000:])
        proof={**expected,'stdout':result.stdout}
    if not isinstance(proof,dict) or any(proof.get(k)!=v for k,v in expected.items()):
        raise ValueError('B2a proof owner/command/identity mismatch')
    check_payload(parse_json(proof['stdout']),summary,run['identity'])
    counts=_working(parent,owner,binding);check_references(refs)
    return {'parent':str(parent),'owner':str(owner),'identity':run['identity'],
            'event_identity':binding['parent_identity'],'dataset_release':binding['dataset_release'],
            'working_validator_sha256':binding['sources'][VALIDATOR],
            'references':refs,'proof':proof,'reading_status_counts':counts}

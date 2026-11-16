"""Native frozen B1a proof for review-material construction."""
import json
import os
import subprocess
import sys
from pathlib import Path

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation.event_sources import parse_json, read_json
from trajectory_prediction.evaluation.track_sources import (
    _artifacts,
    add_reference,
    check_references,
    sources,
)
from trajectory_prediction.utils.provenance import sha256_file


def native_payload(stdout):
    lines=stdout.splitlines()
    if len(lines)<3:
        raise ValueError('incomplete B1a native proof stream')
    expected=[{'stage':'offline_events','role':role} for role in ('independent_fit','independent_select')]
    if [parse_json(x) for x in lines[:2]]!=expected:
        raise ValueError('B1a native proof stage mismatch')
    value=parse_json('\n'.join(lines[2:]))
    if not isinstance(value,dict) or value.get('events_semantically_replayed') is not True:
        raise ValueError('B1a native semantic proof required')
    return value


def parent_snapshot(parent,owner,proof=None):
    parent,owner=Path(parent).resolve(),Path(owner).resolve()
    script=owner/'scripts/prepare_scenario_events.py'
    if not script.is_file(): raise ValueError('explicit B1a owner verifier required')
    refs={};marker=_artifacts(parent,refs)
    run,summary=read_json(parent/'run.json'),read_json(parent/'summary.json');binding=run['binding']
    if (run['identity']!=marker['identity'] or digest(binding)!=run['identity']
        or binding['schema']!='scenario_events_b1a_v1'
        or binding['scope']!='original_fit_select_cache_target_instances'
        or summary['status']!='b1a_rule_candidates_complete'
        or any(binding[k] is not False for k in ('neural_training','holdout_evaluated','new_prediction_scores_computed'))
        or any(summary[k] is not False for k in ('neural_training','holdout_evaluated',
               'new_prediction_scores_computed','rule_candidates_are_human_ground_truth'))
        or type(summary['manual_completed']) is not int or summary['manual_completed']!=0):
        raise ValueError('frozen B1a identity/scope/policy required')
    if sources(owner)!=binding['sources']: raise ValueError('B1a owner source mismatch')
    for name,expected in binding['sources'].items():
        path=(owner/name).resolve()
        if not path.is_relative_to(owner): raise ValueError('B1a source escapes owner')
        add_reference(refs,path,expected)
    for path,expected in binding['references'].items(): add_reference(refs,path,expected)
    command=[sys.executable,str(script),'--output-root',str(parent),'--verify-only']
    expected={'owner_root':str(owner),'pythonpath':str(owner/'src'),'command':command,
              'parent_identity':run['identity'],'parent_completed_sha256':sha256_file(parent/'completed.json')}
    if proof is None:
        print(json.dumps({'stage':'b1a_owner_verification','state':'started'}),flush=True)
        env={**os.environ,'PYTHONPATH':str(owner/'src'),'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'}
        result=subprocess.run(command,cwd=owner,env=env,capture_output=True,text=True,encoding='utf-8',check=False)
        if result.returncode: raise ValueError('B1a native verifier failed: '+result.stderr[-2000:])
        proof={**expected,'stdout':result.stdout}
    if not isinstance(proof,dict) or any(proof.get(k)!=v for k,v in expected.items()):
        raise ValueError('B1a proof owner/command/identity mismatch')
    payload=native_payload(proof['stdout'])
    if digest(payload)!=digest({**summary,'events_semantically_replayed':True,'identity':run['identity']}):
        raise ValueError('B1a native summary replay mismatch')
    check_references(refs)
    return {'parent':str(parent),'owner':str(owner),'identity':run['identity'],
            'references':refs,'proof':proof,'dataset_release':binding['dataset_release']}

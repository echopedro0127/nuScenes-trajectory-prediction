"""Frozen metadata-only B3a evidence and its original owner verification."""
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


def snapshot(parent,owner,proof=None):
    parent,owner = Path(parent).resolve(),Path(owner).resolve()
    refs = {};marker = _artifacts(parent,refs)
    run,summary = read_json(parent/'run.json'),read_json(parent/'summary.json')
    binding = run['binding']
    if (binding['schema'] != 'clock_anchor_audit_b3a_v1' or digest(binding) != run['identity']
            or marker['identity'] != run['identity'] or summary['status'] != 'realclock_query_audit_complete'
            or any(binding[k] is not False or summary[k] is not False for k in
                   ('neural_training','holdout_evaluated','new_prediction_scores_computed'))
            or summary['human_labels_imported'] is not False):
        raise ValueError('completed metadata-only B3a parent required')
    if sources(owner) != binding['sources']:raise ValueError('frozen B3a owner source mismatch')
    for name,expected in binding['sources'].items():
        path = (owner/name).resolve()
        if not path.is_relative_to(owner):raise ValueError('B3a source escapes owner')
        add_reference(refs,path,expected)
    for path,expected in binding['references'].items():add_reference(refs,path,expected)
    script = owner/'scripts/run_realclock_query_audit.py'
    if not script.is_file():raise ValueError('explicit original B3a owner required')
    command = [sys.executable,str(script),'--output-root',str(parent),'--verify-only']
    expected = {'owner_root':str(owner),'pythonpath':str(owner/'src'),'command':command,
                'parent_identity':run['identity'],'parent_completed_sha256':refs[str(parent/'completed.json')]}
    if proof is None:
        print('B3a original-owner verification started',flush=True)
        native = subprocess.run(command,cwd=owner,env={**os.environ,'PYTHONPATH':str(owner/'src'),
                                'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2'},capture_output=True,
                                text=True,encoding='utf-8',check=False)
        if native.returncode:raise ValueError('B3a native verifier failed: '+native.stderr[-2000:])
        proof = {**expected,'stdout':native.stdout}
    if (not isinstance(proof,dict) or set(proof) != set(expected)|{'stdout'}
            or any(proof.get(k) != v for k,v in expected.items())):
        raise ValueError('B3a native proof fields/identity mismatch')
    if digest(parse_json(proof['stdout'])) != digest({**summary,'query_audit_semantically_replayed':True,'identity':run['identity']}):
        raise ValueError('B3a native proof semantic payload mismatch')
    check_references(refs)
    return {'parent':str(parent),'owner':str(owner),'identity':run['identity'],'references':refs,'proof':proof}

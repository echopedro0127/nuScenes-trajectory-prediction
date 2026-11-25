"""Original Gaussian-owner proof and transitive clean/source bindings."""
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

APPROVED_PARENT_IDENTITY='6eca3e0cabfd216d00d007ac3f3673a52aedb52866344ef112d3c9595c073551'


def snapshot(parent,owner,proof=None):
    parent,owner=Path(parent).resolve(),Path(owner).resolve();refs={};marker=_artifacts(parent,refs)
    run=read_json(parent/'run.json');summary=read_json(parent/'summary.json');binding=run['binding']
    if run['identity']!=APPROVED_PARENT_IDENTITY:raise ValueError('approved Gaussian identity required')
    if (binding['schema']!='realclock_physics_gaussian_b3b2a_v1' or digest(binding)!=run['identity']
            or marker['identity']!=run['identity'] or summary['status']!='source_gaussian_physics_complete'
            or binding['sources']!=sources(owner) or summary['combinations']!=24
            or summary['noise_seeds']!=[2027,2028,2029]
            or any(summary[k] is not False for k in ('human_labels_imported','neural_training','holdout_evaluated','official_challenge_metrics'))
            or summary['new_prediction_scores_computed'] is not True or summary['synthetic_observation_noise'] is not True):
        raise ValueError('completed original Gaussian parent and unchanged owner required')
    for name,expected in binding['sources'].items():
        path=(owner/name).resolve()
        if not path.is_relative_to(owner):raise ValueError('Gaussian source path escapes owner')
        add_reference(refs,path,expected)
    for path,expected in binding['references'].items():add_reference(refs,path,expected)
    script=owner/'scripts/run_realclock_physics_gaussian.py'
    if not script.is_file():raise ValueError('original Gaussian CLI missing')
    command=[sys.executable,str(script),'--output-root',str(parent),'--verify-only']
    expected={'owner_root':str(owner),'pythonpath':str(owner/'src'),'command':command,'parent_identity':run['identity'],
              'parent_completed_sha256':refs[str(parent/'completed.json')]}
    if proof is None:
        print('B3b2a original-owner verification started',flush=True)
        native=subprocess.run(command,cwd=owner,env={**os.environ,'PYTHONPATH':str(owner/'src'),
            'OMP_NUM_THREADS':'2','MKL_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'},capture_output=True,text=True,encoding='utf-8',check=False)
        if native.returncode:raise ValueError('Gaussian native verification failed: '+native.stderr[-2000:])
        proof={**expected,'stdout':native.stdout}
    payload={**summary,'identity':run['identity'],'gaussian_semantically_replayed':True,
             'saved_predictions_independently_rescored':True,'scalar_reference_verified':True}
    if (not isinstance(proof,dict) or set(proof)!=set(expected)|{'stdout'}
            or any(proof.get(k)!=v for k,v in expected.items()) or digest(parse_json(proof['stdout']))!=digest(payload)):
        raise ValueError('Gaussian native proof mismatch')
    clean=Path(binding['parent']).resolve()
    # Both parent chains must include their completed markers and the original packet list.
    for path in (clean/'completed.json',Path(binding['packets_path'])):
        if str(path) not in refs:raise ValueError('missing transitive clean/source binding')
    check_references(refs)
    return {'parent':str(parent),'owner':str(owner),'identity':run['identity'],'references':refs,
            'proof':proof,'clean':str(clean)}

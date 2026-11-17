"""Validate a B2b1 overlay in its frozen owner before using its declarations."""
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


def parent_snapshot(parent, owner, proof=None):
    parent, owner = Path(parent).resolve(), Path(owner).resolve()
    refs = {}
    marker = _artifacts(parent, refs)
    run = read_json(parent/'run.json')
    binding, summary = run['binding'], read_json(parent/'summary.json')
    if (binding.get('schema') != 'candidate_verdicts_b2b1_v1' or binding.get('mode') != 'overlay'
            or digest(binding) != run['identity'] or marker['identity'] != run['identity']):
        raise ValueError('completed B2b1 overlay parent required')
    if any(binding[k] is not False or summary[k] is not False for k in
           ('neural_training', 'holdout_evaluated', 'new_prediction_scores_computed')):
        raise ValueError('unexpected B2b1 prediction policy')
    if sources(owner) != binding['sources']:
        raise ValueError('frozen verdict owner source mismatch')
    for name, expected in binding['sources'].items():
        path = (owner/name).resolve()
        if not path.is_relative_to(owner):
            raise ValueError('verdict source escapes owner')
        add_reference(refs, path, expected)
    for path, expected in binding['references'].items():
        add_reference(refs, path, expected)
    script = owner/'scripts/import_candidate_verdicts.py'
    if not script.is_file():
        raise ValueError('explicit frozen verdict owner required')
    command = [sys.executable, str(script), '--output-root', str(parent), '--verify-only']
    env = {**os.environ, 'PYTHONPATH':str(owner/'src'), 'OMP_NUM_THREADS':'2', 'MKL_NUM_THREADS':'2'}
    # Re-run the original verifier, never a new clone with a different source identity.
    native = subprocess.run(command, cwd=owner, env=env, capture_output=True, text=True,
                            encoding='utf-8', check=False)
    if native.returncode:
        raise ValueError('B2b1 native verifier failed: '+native.stderr[-2000:])
    payload = parse_json(native.stdout)
    expected_payload = {**summary, 'verdicts_semantically_replayed':True, 'identity':run['identity']}
    if (not isinstance(payload, dict)
            or set(payload) != set(expected_payload)|{'parent_reading_status_counts'}
            or digest({k:v for k,v in payload.items() if k != 'parent_reading_status_counts'}) != digest(expected_payload)):
        raise ValueError('B2b1 native static proof mismatch')
    reading = payload['parent_reading_status_counts']
    task_count = read_json(Path(binding['parent'])/'summary.json')['task_rows']
    if (not isinstance(reading, dict) or set(reading) != {'pending', 'in_progress', 'reviewed'}
            or any(type(v) is not int or v < 0 for v in reading.values())
            or sum(reading.values()) != task_count):
        raise ValueError('invalid native reading count projection')
    current_proof = {'owner_root': str(owner), 'pythonpath': str(owner/'src'), 'command': command,
                         'parent_identity': run['identity'],
                         'parent_completed_sha256': refs[str(parent/'completed.json')],
                         'static_payload': expected_payload}
    if proof is not None and digest(proof) != digest(current_proof):
        raise ValueError('correction parent proof mismatch')
    review = Path(binding['parent'])/'review_data.json'
    if str(review.resolve()) not in refs:
        raise ValueError('review data not bound by verdict parent')
    data = read_json(review)
    verdicts = read_json(parent/'atomic_verdicts.json')
    check_references(refs)
    return {'parent': str(parent), 'owner': str(owner), 'identity': run['identity'],
                'review_identity': binding['source_review_identity'], 'dataset_release': binding['dataset_release'],
                'references': refs, 'proof': current_proof, 'data': data, 'verdicts': verdicts,
                'reading_status_counts': reading}

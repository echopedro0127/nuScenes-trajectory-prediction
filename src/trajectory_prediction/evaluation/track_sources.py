"""Explicit-owner parent proofs for GT-only track preparation."""
import json
import os
import subprocess
import sys
from pathlib import Path

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.utils.provenance import sha256_file


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sources(root):
    root = Path(root).resolve()
    files = [p for folder in ('src', 'scripts') for p in (root / folder).rglob('*.py')]
    files += [p for p in (root / 'configs').rglob('*') if p.is_file()]
    files += [p for name in ('pyproject.toml', 'requirements.txt') if (p := root / name).is_file()]
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(files)}


def add_reference(refs, path, expected=None):
    path = Path(path).resolve()
    key = str(path)
    if key in refs:
        if expected is not None and refs[key] != expected:
            raise ValueError('conflicting source checksum')
        return
    actual = sha256_file(path)
    if expected is not None and actual != expected:
        raise ValueError('source checksum changed: ' + key)
    refs[key] = actual


def check_references(refs):
    for path, expected in refs.items():
        if sha256_file(Path(path)) != expected:
            raise ValueError('source checksum changed: ' + path)


def _artifacts(root, refs, marker='completed.json'):
    root = Path(root).resolve()
    receipt = read(root / marker)
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != set(receipt['sha256']) | {marker}:
        raise ValueError('parent file coverage mismatch')
    for name, expected in receipt['sha256'].items():
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError('parent artifact escapes owner directory')
        add_reference(refs, path, expected)
    add_reference(refs, root / marker)
    return receipt


def parent_snapshot(parent, owner, proof=None):
    parent, owner = Path(parent).resolve(), Path(owner).resolve()
    script = owner / 'scripts/run_scenario_evaluation.py'
    if not script.is_file():
        raise ValueError('explicit parent owner must provide the original verifier')
    refs = {}
    receipt = _artifacts(parent, refs)
    run, summary = read(parent / 'run.json'), read(parent / 'summary.json')
    if (run['identity'] != receipt['identity'] or digest(run['binding']) != receipt['identity']
            or run['binding']['schema'] != 'scenario_phase3a_pipeline_v1'
            or summary['status'] != 'phase3a_verified' or summary['purpose'] not in ('fixture', 'formal_development')
            or summary['formal_full_run'] is not (summary['purpose'] == 'formal_development')
            or summary['neural_training'] is not False or summary['holdout_evaluated'] is not False):
        raise ValueError('invalid completed parent identity/policy')
    freeze = Path(run['binding']['freeze_root']).resolve()
    frozen_receipt = _artifacts(freeze, refs)
    frozen = read(freeze / 'freeze.json')
    if (digest(frozen['binding']) != frozen['identity'] or frozen['identity'] != frozen_receipt['identity']
            or frozen['identity'] != run['binding']['freeze_identity']
            or sha256_file(freeze / 'completed.json') != run['binding']['freeze_marker_sha256']
            or frozen['binding']['purpose'] != summary['purpose']):
        raise ValueError('parent freeze identity mismatch')
    if sources(owner) != frozen['binding']['sources']:
        raise ValueError('original owner frozen source mismatch')
    for name, expected in frozen['binding']['sources'].items():
        path = (owner / name).resolve()
        if not path.is_relative_to(owner):
            raise ValueError('parent source escapes explicit owner')
        add_reference(refs, path, expected)
    preflight = Path(frozen['binding']['preflight']).resolve()
    ready_receipt = _artifacts(preflight, refs, 'preflight_completed.json')
    ready = read(preflight / 'run.json')
    if ready['identity'] != ready_receipt['identity'] or digest(ready['binding']) != ready['identity']:
        raise ValueError('parent preflight identity mismatch')
    for ref in ready['binding']['references'].values():
        add_reference(refs, ref['path'], ref['sha256'])
    phase2 = Path(ready['binding']['parent_path']) / 'run.json'
    if str(phase2.resolve()) not in refs:
        raise ValueError('upstream cache pointer not bound by parent')
    cache = Path(read(phase2)['binding']['source_cache_root']).resolve()
    command = [sys.executable, str(script), '--output-root', str(parent), '--verify-only']
    expected = {'owner_root': str(owner), 'pythonpath': str(owner / 'src'), 'command': command,
                'parent_identity': receipt['identity'], 'parent_completed_sha256': refs[str(parent / 'completed.json')]}
    if proof is None:
        print(json.dumps({'stage': 'parent_owner_verification', 'state': 'started'}), flush=True)
        env = {**os.environ, 'PYTHONPATH': str(owner / 'src'), 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        result = subprocess.run(command, cwd=owner, env=env, capture_output=True, text=True, encoding='utf-8', check=False)
        if result.returncode:
            raise ValueError('parent owner verifier failed: ' + result.stderr[-2000:])
        proof = {**expected, 'stdout': result.stdout}
    if any(proof.get(k) != v for k, v in expected.items()):
        raise ValueError('saved parent proof owner/command/identity mismatch')
    try:
        payload = json.loads(proof['stdout'])
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError('invalid saved native parent proof JSON') from error
    if (not isinstance(payload, dict) or payload.get('top_level_semantic_replay_verified') is not True
            or payload != {**summary, 'top_level_semantic_replay_verified': True}):
        raise ValueError('native parent proof flags or summary mismatch')
    check_references(refs)
    return {'parent': str(parent), 'owner': str(owner), 'identity': receipt['identity'],
            'purpose': summary['purpose'], 'cache_root': str(cache),
            'parent_references': refs, 'proof': proof}

"""Strict B0b provenance, loaded only for offline event evaluation."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import yaml

from trajectory_prediction.analysis.realclock_motion import validate_config
from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation.track_sources import (
    _artifacts,
    add_reference,
    check_references,
    sources,
)
from trajectory_prediction.utils.provenance import sha256_file


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('duplicate mapping key')
        result[key] = value
    return result


def _bad_constant(value):
    raise ValueError('non-finite JSON number: '+value)


def _float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError('non-finite JSON float')
    return result


def parse_json(text):
    return json.loads(text, object_pairs_hook=_pairs, parse_constant=_bad_constant, parse_float=_float)


def read_json(path):
    return parse_json(Path(path).read_text(encoding='utf-8'))


def load_config(path):
    class UniqueLoader(yaml.SafeLoader):
        pass
    def mapping(loader, node):
        return _pairs([(loader.construct_object(k), loader.construct_object(v)) for k, v in node.value])
    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        value = yaml.load(Path(path).read_text(encoding='utf-8'), Loader=UniqueLoader)
    except yaml.YAMLError as error:
        raise ValueError('invalid event YAML') from error
    return validate_config(value)


def parent_snapshot(parent, owner, proof=None):
    parent, owner = Path(parent).resolve(), Path(owner).resolve()
    script = owner/'scripts/prepare_scenario_tracks.py'
    if not script.is_file():
        raise ValueError('explicit B0b owner must provide original verifier')
    refs = {}
    marker = _artifacts(parent, refs)
    run, summary = read_json(parent/'run.json'), read_json(parent/'summary.json')
    binding = run['binding']
    if (run['identity'] != marker['identity'] or digest(binding) != run['identity']
            or binding['schema'] != 'scenario_tracks_b0b_v1'
            or binding['scope'] != 'original_fit_select_cache_target_instances'
            or summary['scope'] != binding['scope'] or summary['status'] != 'tracks_and_clock_complete'
            or any(summary[k] is not False for k in ('canonical_episodes_extracted',
                   'new_prediction_scores_computed', 'neural_training', 'holdout_evaluated'))
            or any(binding[k] is not False for k in ('neural_training', 'holdout_evaluated'))):
        raise ValueError('completed full-cache B0b scope/identity/policy required')
    if sources(owner) != binding['sources']:
        raise ValueError('original B0b owner source mismatch')
    for name, expected in binding['sources'].items():
        path = (owner/name).resolve()
        if not path.is_relative_to(owner):
            raise ValueError('source escapes B0b owner')
        add_reference(refs, path, expected)
    for path, expected in binding['references'].items():
        add_reference(refs, path, expected)
    command = [sys.executable, str(script), '--output-root', str(parent), '--verify-only']
    expected = {'owner_root': str(owner), 'pythonpath': str(owner/'src'), 'command': command,
                'parent_identity': run['identity'], 'parent_completed_sha256': sha256_file(parent/'completed.json')}
    if proof is None:
        print(json.dumps({'stage': 'b0b_owner_verification', 'state': 'started'}), flush=True)
        env = {**os.environ, 'PYTHONPATH': str(owner/'src'), 'OMP_NUM_THREADS': '2', 'MKL_NUM_THREADS': '2'}
        result = subprocess.run(command, cwd=owner, env=env, capture_output=True, text=True, encoding='utf-8', check=False)
        if result.returncode:
            raise ValueError('B0b native verifier failed: '+result.stderr[-2000:])
        proof = {**expected, 'stdout': result.stdout}
    if not isinstance(proof, dict) or any(proof.get(k) != v for k, v in expected.items()):
        raise ValueError('B0b proof owner/command/identity mismatch')
    payload = parse_json(proof['stdout'])
    if (not isinstance(payload, dict) or payload.get('track_and_window_replayed') is not True
            or digest(payload) != digest({**summary, 'track_and_window_replayed': True, 'identity': run['identity']})):
        raise ValueError('B0b native proof flags or summary mismatch')
    check_references(refs)
    return {'parent': str(parent), 'owner': str(owner), 'identity': run['identity'],
            'dataset_release': binding['dataset_release'], 'references': refs, 'proof': proof}

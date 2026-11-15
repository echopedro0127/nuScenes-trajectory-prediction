"""S7 source freeze and top-level acceptance of the complete evaluation pipeline."""
import argparse
import csv
import io
import json
import platform
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

from trajectory_prediction.config import load_yaml
from trajectory_prediction.context.cache import digest_json
from trajectory_prediction.evaluation.scenario_event_report import build_event_report
from trajectory_prediction.evaluation.scenario_preflight import PROJECT
from trajectory_prediction.evaluation.scenario_publication import (
    _artifact_hashes,
    _read,
    _write,
    plan_publication,
    publish_predictions,
)
from trajectory_prediction.evaluation.scenario_replay import time_grid, validate_protocol
from trajectory_prediction.evaluation.scenario_report import build_report
from trajectory_prediction.evaluation.scenario_synthesis import build_synthesis, verify_synthesis
from trajectory_prediction.utils.provenance import sha256_file

STAGES = ('predictions', 'queries', 'events', 'synthesis')
SOURCE_PATHS = ('src', 'scripts', 'configs', 'pyproject.toml', 'requirements.txt')


def _sources():
    files = [p for folder in ('src', 'scripts') for p in (PROJECT / folder).rglob('*.py')]
    files += [p for p in (PROJECT / 'configs').rglob('*') if p.is_file()]
    files += [p for name in ('pyproject.toml', 'requirements.txt') if (p := PROJECT / name).is_file()]
    return {p.relative_to(PROJECT).as_posix(): sha256_file(p) for p in sorted(files)}


def _runtime():
    return {'python': sys.version, 'numpy': str(np.__version__), 'torch': str(torch.__version__),
            'matplotlib': matplotlib.__version__, 'platform': platform.platform(),
            'device': 'cpu', 'threads': 2}


def _git(*args):
    return subprocess.run(['git', '-c', 'safe.directory=*', *args], cwd=PROJECT,
                          check=True, capture_output=True, text=True).stdout.strip()


def freeze_protocol(preflight, output, purpose='formal_development'):
    output, preflight = Path(output).resolve(), Path(preflight).resolve()
    if output.exists():
        raise FileExistsError('use a new freeze directory')
    if purpose not in ('formal_development', 'fixture'):
        raise ValueError('invalid evaluation purpose')
    head = _git('rev-parse', 'HEAD')
    if purpose == 'formal_development' and _git('status', '--porcelain', '--untracked-files=all', '--', *SOURCE_PATHS):
        raise ValueError('commit source/config changes before formal freeze')
    sources = _sources()
    preflight_hash = sha256_file(preflight / 'preflight_completed.json')
    plan = plan_publication(preflight)  # Provenance/resource validation only, no forward inference.
    protocol = validate_protocol(_read(preflight / 'run.json')['binding']['protocol'])
    binding = {'schema': 'scenario_phase3a_freeze_v1', 'purpose': purpose,
               'preflight': str(preflight), 'preflight_sha256': preflight_hash,
               'protocol': protocol, 'resource_plan': plan, 'sources': sources,
               'git_commit': head, 'runtime': _runtime(),
               'neural_training': False, 'holdout_evaluated': False}
    if sources != _sources() or preflight_hash != sha256_file(preflight / 'preflight_completed.json'):
        raise ValueError('source/preflight changed during freeze')
    if purpose == 'formal_development' and (_git('status', '--porcelain', '--untracked-files=all', '--', *SOURCE_PATHS)
                                             or _git('rev-parse', 'HEAD') != head):
        raise ValueError('source Git clean/HEAD changed during freeze')
    output.mkdir(parents=True, exist_ok=False)
    _write(output / 'freeze.json', {'identity': digest_json(binding), 'binding': binding})
    _write(output / 'completed.json', {'identity': digest_json(binding),
                                      'sha256': {'freeze.json': sha256_file(output / 'freeze.json')}})
    return {'identity': digest_json(binding), 'resource_plan': plan, 'purpose': purpose}


def verify_freeze(output):
    output = Path(output).resolve()
    marker_hash = sha256_file(output / 'completed.json')
    marker, run = _read(output / 'completed.json'), _read(output / 'freeze.json')
    _artifact_hashes(output, marker['sha256'])
    binding = run['binding']
    if (binding['schema'] != 'scenario_phase3a_freeze_v1'
            or binding['purpose'] not in ('formal_development', 'fixture')
            or binding['neural_training'] is not False or binding['holdout_evaluated'] is not False):
        raise ValueError('invalid frozen evaluation policy')
    if (run['identity'] != marker['identity'] or run['identity'] != digest_json(binding)
            or binding['sources'] != _sources() or binding['runtime'] != _runtime()):
        raise ValueError('frozen source/runtime identity mismatch')
    protocol = validate_protocol(binding['protocol'])
    preflight = Path(binding['preflight'])
    if sha256_file(preflight / 'preflight_completed.json') != binding['preflight_sha256']:
        raise ValueError('frozen preflight changed')
    if (plan_publication(preflight) != binding['resource_plan']
            or _read(preflight / 'run.json')['binding']['protocol'] != protocol):
        raise ValueError('freeze resource/protocol mismatch')
    if marker_hash != sha256_file(output / 'completed.json') or binding['sources'] != _sources():
        raise ValueError('freeze/source changed during verification')
    return run


def _binding(freeze_root, frozen):
    return {'schema': 'scenario_phase3a_pipeline_v1', 'freeze_root': str(Path(freeze_root).resolve()),
            'freeze_identity': frozen['identity'],
            'freeze_marker_sha256': sha256_file(Path(freeze_root) / 'completed.json'),
            'purpose': frozen['binding']['purpose'], 'stages': list(STAGES),
            'component_markers_are_not_top_level_acceptance': True,
            'neural_training': False, 'holdout_evaluated': False}


def _check_links(root, frozen):
    expected = [('synthesis', 'event_root', root / 'events'),
                ('events', 'query_root', root / 'queries'),
                ('queries', 'publication', root / 'predictions'),
                ('predictions', 'preflight', Path(frozen['binding']['preflight']))]
    for folder, field, path in expected:
        if _read(root / folder / 'run.json')['binding'][field] != str(path.resolve()):
            raise ValueError('pipeline cross-stage path binding mismatch')


def _review_csv(rows):
    output = io.StringIO(newline='')
    fields = ['window_id', 'role', 'instance', 'sample', 'log', 'rule_version', 'rule_label',
              'review_status', 'reviewer', 'review_reason', 'manual_label', 'selection_hash', 'review_history']
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for row in rows:
        writer.writerow({**{k: row[k] for k in fields if k != 'review_history'},
                         'review_history': json.dumps(row['review_history'], allow_nan=False)})
    return output.getvalue()


def _top_documents(root, frozen, binding):
    _check_links(root, frozen)
    proof = verify_synthesis(root / 'synthesis')  # Includes all upstream models, scalar metrics and image replay.
    if (proof.get('semantic_replay_verified') is not True or proof.get('neural_training') is not False
            or proof.get('holdout_evaluated') is not False):
        raise ValueError('invalid component verification proof')
    publication = _read(root / 'predictions' / 'summary.json')
    if publication != frozen['binding']['resource_plan']:
        raise ValueError('full stream/shard conservation mismatch')
    old = _read(Path(frozen['binding']['preflight']) / 'run.json')['binding']
    if old['model_input_fields'] != ['features', 'candidates', 'ca_allowed']:
        raise ValueError('unexpected GT/model input boundary')
    keys = {role: {'path': f'predictions/{role}-keys.json',
                   'sha256': sha256_file(root / 'predictions' / f'{role}-keys.json'),
                   'observation_streams': count}
            for role, count in publication['observation_streams'].items()}
    review = _read(root / 'events' / 'event_review.json')
    if any(r['review_status'] != 'pending' or r['manual_label'] for r in review):
        raise ValueError('unexpected manual review state; frozen run expects pending rules')
    summary = {'status': 'phase3a_verified', 'formal_full_run': binding['purpose'] == 'formal_development',
               'purpose': binding['purpose'], 'unique_windows': publication['unique_windows'],
               'observation_streams': publication['observation_streams'], 'prediction_shards': publication['shards'],
               'forecast_bytes': publication['forecast_bytes'], 'case_count': proof['case_count'],
               'neural_models': 9, 'physical_strategies': 7, 'review_assignments': len(review),
               'manual_completed': 0, 'neural_training': False, 'holdout_evaluated': False,
               'data_scope': 'metadata_positions_static_maps_synthetic_position_degradation',
               'official_nuscenes_challenge_score': False}
    audit = {'freeze_identity': frozen['identity'], 'full_raw_prediction_replay': True,
             'legacy_metrics_exactly_compatible': True, 'new_metrics_scalar_audited': True,
             'case_arrays_images_html_replayed': True, 'all_expected_streams_and_shards_covered': True,
             'model_input_fields': old['model_input_fields'], 'neural_training': False,
             'holdout_evaluated': False, 'manual_review_status': 'pending',
             'statistical_scope': 'descriptive_development_logs_and_training_seed_variation'}
    protocol = {'config': frozen['binding']['protocol'], 'time_grid': time_grid(frozen['binding']['protocol'])}
    return {'run.json': {'identity': digest_json(binding), 'binding': binding}, 'summary.json': summary,
            'audit.json': audit, 'keys.json': keys, 'protocol.json': protocol,
            'event_review.csv': _review_csv(review)}


def _check_freeze_snapshot(binding):
    root = Path(binding['freeze_root'])
    if sha256_file(root / 'completed.json') != binding['freeze_marker_sha256']:
        raise ValueError('pipeline freeze marker changed')
    frozen = verify_freeze(root)
    if frozen['identity'] != binding['freeze_identity']:
        raise ValueError('pipeline freeze identity changed')
    return frozen


def run_pipeline(freeze_root, output):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('use a new pipeline output directory; partial runs are preserved')
    frozen = verify_freeze(freeze_root)
    binding = _binding(freeze_root, frozen)
    output.mkdir(parents=True, exist_ok=False)
    progress = {'state': 'running', 'completed_stages': []}
    _write(output / 'progress.json', progress)
    _write(output / 'run.json', {'identity': digest_json(binding), 'binding': binding})
    hashes = {'run.json': sha256_file(output / 'run.json')}
    operations = [lambda: publish_predictions(frozen['binding']['preflight'], output / 'predictions'),
                  lambda: build_report(output / 'predictions', output / 'queries'),
                  lambda: build_event_report(output / 'queries', output / 'events'),
                  lambda: build_synthesis(output / 'events', output / 'synthesis')]
    try:
        for stage, operation in zip(STAGES, operations, strict=True):
            print(json.dumps({'stage': stage, 'state': 'started'}), flush=True)
            operation()
            hashes.update({p.relative_to(output).as_posix(): sha256_file(p)
                           for p in (output / stage).rglob('*') if p.is_file()})
            progress['completed_stages'].append(stage)
            _write(output / 'progress.json', progress)
            _check_freeze_snapshot(binding)
            print(json.dumps({'stage': stage, 'state': 'written'}), flush=True)
        print(json.dumps({'stage': 'top_level_independent_verification', 'state': 'started'}), flush=True)
        documents = _top_documents(output, frozen, binding)
        for name, value in documents.items():
            if name == 'run.json':
                if _read(output / name) != value:
                    raise ValueError('pipeline run changed')
                continue
            if name.endswith('.csv'):
                (output / name).write_text(value, encoding='utf-8', newline='')
            else:
                _write(output / name, value)
            hashes[name] = sha256_file(output / name)
        _check_freeze_snapshot(binding)
        progress['state'] = 'complete'
        _write(output / 'progress.json', progress)
        hashes['progress.json'] = sha256_file(output / 'progress.json')
        actual = {p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
        if actual != set(hashes) or any(sha256_file(output / n) != h for n, h in hashes.items()):
            raise ValueError('pipeline coverage/artifact changed before completion')
        _write(output / 'completed.json', {'identity': digest_json(binding), 'sha256': hashes})
    except BaseException:
        progress['state'] = 'failed'
        _write(output / 'progress.json', progress)
        raise
    return documents['summary.json']


def verify_pipeline(output):
    output = Path(output).resolve()
    marker_path = output / 'completed.json'
    if not marker_path.is_file():
        raise ValueError('incomplete top-level pipeline')
    marker_hash, marker = sha256_file(marker_path), _read(marker_path)
    _artifact_hashes(output, marker['sha256'])
    run = _read(output / 'run.json')
    frozen = _check_freeze_snapshot(run['binding'])
    binding = _binding(run['binding']['freeze_root'], frozen)
    if run != {'identity': digest_json(binding), 'binding': binding} or marker['identity'] != run['identity']:
        raise ValueError('pipeline identity mismatch')
    documents = _top_documents(output, frozen, binding)
    for name, expected in documents.items():
        actual = (output / name).read_text(encoding='utf-8') if name.endswith('.csv') else _read(output / name)
        if digest_json(actual) != digest_json(expected):
            raise ValueError('pipeline semantic replay mismatch: ' + name)
    if _read(output / 'progress.json') != {'state': 'complete', 'completed_stages': list(STAGES)}:
        raise ValueError('pipeline progress semantic mismatch')
    _check_freeze_snapshot(binding)
    _artifact_hashes(output, marker['sha256'])
    if sha256_file(marker_path) != marker_hash:
        raise ValueError('pipeline completion marker changed')
    return {**documents['summary.json'], 'top_level_semantic_replay_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', type=Path, default=Path('outputs/scenario-evaluation/preflight-v2'))
    parser.add_argument('--freeze-root', type=Path)
    parser.add_argument('--output-root', type=Path)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--parent-project-root', type=Path)
    parser.add_argument('--config', type=Path, default=Path('configs/analysis/scenario_evaluation.yaml'))
    parser.add_argument('--freeze-only', action='store_true')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.freeze_only and args.verify_only:
        parser.error('choose one mode')
    if args.verify_only:
        if args.output_root is None:
            parser.error('--output-root required')
        result = verify_pipeline(args.output_root)
    else:
        if args.freeze_root is None or (not args.freeze_only and args.output_root is None):
            parser.error('--freeze-root required; --output-root also required for execution')
        if not args.freeze_only and verify_freeze(args.freeze_root)['binding']['preflight'] != str(args.preflight.resolve()):
            parser.error('--preflight must match the supplied freeze')
        old = _read(args.preflight.resolve() / 'run.json')['binding']
        if (validate_protocol(load_yaml(args.config)) != old['protocol'] or
                any(value is not None and str(value.resolve()) != old[field] for value, field in (
                    (args.parent, 'parent_path'), (args.parent_project_root, 'parent_project_root')))):
            parser.error('config/parent must match the verified S2 preflight')
        result = freeze_protocol(args.preflight, args.freeze_root) if args.freeze_only else run_pipeline(args.freeze_root, args.output_root)
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)

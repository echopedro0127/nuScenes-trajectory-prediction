"""Immutable token-bound correction templates and explicitly selected snapshots."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation.boundary_corrections import (
    COLUMNS,
    LOCKED,
    csv_text,
    derive_corrections,
    make_template,
    parse_corrections,
)
from trajectory_prediction.evaluation.correction_sources import parent_snapshot
from trajectory_prediction.evaluation.event_sources import read_json
from trajectory_prediction.evaluation.track_sources import add_reference, check_references, sources
from trajectory_prediction.evaluation.verdict_overlay import _files, _json, _write_bytes
from trajectory_prediction.utils.provenance import sha256_file

PROJECT = Path(__file__).resolve().parents[3]
MODES = {'template', 'overlay'}


def _inputs(info, mode, annotations):
    if mode not in MODES:
        raise ValueError('unknown correction mode')
    args = (info['identity'], info['review_identity'], info['data'], info['verdicts'])
    base = make_template(*args)
    path, checksum, raw = None, None, None
    if mode == 'overlay':
        if annotations is None:
            raise ValueError('explicit correction CSV required')
        path = Path(annotations).resolve()
        checksum, raw = sha256_file(path), path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != checksum:
            raise ValueError('correction source changed during read')
        rows = parse_corrections(raw.decode('utf-8-sig'), base)
        add_reference(info['references'], path, checksum)
    else:
        if annotations is not None:
            raise ValueError('template cannot import corrections')
        rows = base
    result = derive_corrections(*args, rows)
    return {'base': base, 'rows': rows, 'raw': raw, 'path': str(path) if path else None,
                'checksum': checksum, 'result': result}


def _binding(info, mode, inputs):
    return {'schema': 'event_boundary_corrections_b2b2a_v1', 'mode': mode,
                'parent': info['parent'], 'parent_owner': info['owner'],
                'source_overlay_identity': info['identity'], 'source_review_identity': info['review_identity'],
                'dataset_release': info['dataset_release'], 'annotations_path': inputs['path'],
                'annotations_sha256': inputs['checksum'], 'template_sha256': digest(inputs['base']),
                'normalized_corrections_sha256': digest(inputs['rows']),
                'references': dict(info['references']), 'sources': sources(PROJECT),
                'runtime': {'python': sys.version, 'numpy': str(np.__version__), 'platform': platform.platform()},
                'human_boundary_declarations_imported': inputs['result']['summary']['human_boundary_declarations_imported'],
                'neural_training': False, 'holdout_evaluated': False, 'new_prediction_scores_computed': False}


def _documents(info, mode, inputs):
    result = inputs['result']
    summary = dict(result['summary'], mode=mode, source_overlay_identity=info['identity'],
                   source_review_identity=info['review_identity'])
    protocol = {'schema': 'event_boundary_corrections_b2b2a_v1', 'mode': mode,
                    'columns': list(COLUMNS), 'locked_columns': list(LOCKED), 'actions': ['pending', 'propose'],
                    'allow_new_events': False, 'allow_repairing_relations': False, 'allow_uncensoring': False,
                    'requires_needs_correction': True, 'official_ground_truth': False, 'mutable_files': []}
    readme = '''# B2b2a 结构化边界修正

本目录全部不可变。复制boundary_corrections.csv为独立草稿，只编辑action、四个token、reviewer、reason。
仅已有needs_correction候选可修正；pending行的token和声明必须全空。
propose是人工边界声明，不是官方GT或新事件。使用同block的真实annotation token，不能填浮点时间。
restart的low_dwell_end_token必须为空；保持原删失，冲突整次拒绝。原规则事件保留，不覆盖来源。
来源、永久输入快照与包内副本均有绑定，不能用草稿路径直接发布后继续编辑。
eligible=0表示父版本尚无待修正候选，不代表复核已完成；本包不计算预测分数。
'''
    docs = {'template.json':_json(inputs['base']), 'summary.json':_json(summary),
            'protocol.json':_json(protocol), 'README.md':readme.encode('utf-8')}
    if mode == 'template':
        docs['boundary_corrections.csv'] = csv_text(inputs['base']).encode('utf-8-sig')
    else:
        docs['annotations.csv'] = inputs['raw']
        for key in ('corrections', 'effective_events', 'derived_pairs'):
            docs[key+'.json'] = _json(result[key])
    return docs, summary


def _check(binding):
    if sources(PROJECT) != binding['sources']:
        raise ValueError('correction source changed')
    check_references(binding['references'])


def _prepare(parent, owner, output, mode, annotations=None):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('use a new correction output directory')
    before = sources(PROJECT)
    info = parent_snapshot(parent, owner)
    inputs = _inputs(info, mode, annotations)
    binding = _binding(info, mode, inputs)
    if binding['sources'] != before:
        raise ValueError('correction source changed during validation')
    output.mkdir(parents=True, exist_ok=False)
    try:
        _write_bytes(output/'progress.json', _json({'state':'assembling'}))
        documents, summary = _documents(info, mode, inputs)
        documents.update({'run.json':_json({'identity': digest(binding), 'binding': binding}),
                          'parent_proof.json':_json(info['proof']),
                          'progress.json':_json({'state':'complete'})})
        for name, value in documents.items():
            _write_bytes(output/name, value)
            if (output/name).read_bytes() != value:
                raise ValueError('correction write/read mismatch')
        _check(binding)
        hashes = {name:sha256_file(output/name) for name in documents}
        _files(output, hashes)
        _write_bytes(output/'completed.json', _json({'identity': digest(binding), 'sha256': hashes}))
    except BaseException:
        (output/'progress.json').write_bytes(_json({'state':'failed'}))
        raise
    return summary


def make_template_pack(parent, parent_owner, output):
    return _prepare(parent, parent_owner, output, 'template')


def import_overlay(parent, parent_owner, annotations, output):
    return _prepare(parent, parent_owner, output, 'overlay', annotations)


def verify_pack(output, expected_mode=None):
    output = Path(output).resolve()
    marker_path = output/'completed.json'
    if not marker_path.is_file():
        raise ValueError('incomplete correction package')
    marker_hash, marker = sha256_file(marker_path), read_json(marker_path)
    if set(marker) != {'identity', 'sha256'}:
        raise ValueError('immutable correction marker fields required')
    _files(output, marker['sha256'], True)
    run, proof = read_json(output/'run.json'), read_json(output/'parent_proof.json')
    binding, mode = run['binding'], run['binding']['mode']
    if mode not in MODES or (expected_mode is not None and mode != expected_mode):
        raise ValueError('correction mode mismatch')
    if digest(binding) != run['identity'] or run['identity'] != marker['identity']:
        raise ValueError('correction run identity mismatch')
    _check(binding)
    info = parent_snapshot(binding['parent'], binding['parent_owner'], proof)
    inputs = _inputs(info, mode, binding['annotations_path'])
    if digest(binding) != digest(_binding(info, mode, inputs)):
        raise ValueError('correction policy/runtime replay mismatch')
    docs, summary = _documents(info, mode, inputs)
    docs.update({'run.json':_json(run), 'parent_proof.json':_json(proof),
                 'progress.json':_json({'state':'complete'})})
    if set(docs) != set(marker['sha256']):
        raise ValueError('correction expected file coverage mismatch')
    for name, value in docs.items():
        if (output/name).read_bytes() != value:
            raise ValueError('correction semantic replay mismatch: '+name)
    _check(binding)
    _files(output, marker['sha256'], True)
    if sha256_file(marker_path) != marker_hash:
        raise ValueError('correction marker changed during replay')
    return dict(summary, corrections_semantically_replayed=True, identity=run['identity'],
                parent_reading_status_counts=info['reading_status_counts'])


def main(mode):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path)
    parser.add_argument('--parent-owner', type=Path)
    if mode == 'overlay':
        parser.add_argument('--annotations', type=Path)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.verify_only:
        result = verify_pack(args.output_root, mode)
    else:
        if args.parent is None or args.parent_owner is None or (mode == 'overlay' and args.annotations is None):
            parser.error('explicit parent, parent-owner and overlay annotations required')
        result = _prepare(args.parent, args.parent_owner, args.output_root, mode, getattr(args, 'annotations', None))
    print(json.dumps(result, indent=2, allow_nan=False), flush=True)

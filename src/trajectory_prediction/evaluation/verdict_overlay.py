"""Immutable candidate verdict templates and explicitly selected import snapshots."""
import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation.candidate_verdicts import (
    COLUMNS,
    DECISIONS,
    LOCKED,
    csv_text,
    derive_overlay,
    make_template,
    parse_verdicts,
)
from trajectory_prediction.evaluation.event_sources import read_json
from trajectory_prediction.evaluation.track_sources import add_reference, check_references, sources
from trajectory_prediction.evaluation.verdict_sources import parent_snapshot
from trajectory_prediction.utils.provenance import sha256_file

PROJECT=Path(__file__).resolve().parents[3]
MODES={'template','overlay'}


def _json(value):return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')


def _write_bytes(path,value):
    temp=path.with_suffix('.tmp');temp.write_bytes(value);temp.replace(path)


def _inputs(info,mode,annotations):
    if mode not in MODES:raise ValueError('unknown verdict mode')
    data=read_json(Path(info['parent'])/'review_data.json');base=make_template(info['identity'],data)
    raw,path,checksum=None,None,None
    if mode=='overlay':
        if annotations is None:raise ValueError('explicit verdict CSV required')
        path=Path(annotations).resolve();checksum=sha256_file(path);raw=path.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=checksum:raise ValueError('verdict input changed during read')
        rows=parse_verdicts(raw.decode('utf-8-sig'),base)
        add_reference(info['references'],path,checksum)
    else:
        if annotations is not None:raise ValueError('template mode cannot import verdicts')
        rows=base
    result=derive_overlay(data,rows,info['identity'])
    return {'base':base,'rows':rows,'result':result,'raw':raw,'path':str(path) if path else None,'checksum':checksum}


def _binding(info,mode,inputs):
    return {'schema':'candidate_verdicts_b2b1_v1','mode':mode,'parent':info['parent'],'parent_owner':info['owner'],
            'source_review_identity':info['identity'],'source_event_identity':info['event_identity'],
            'dataset_release':info['dataset_release'],'working_validator_sha256':info['working_validator_sha256'],
            'annotations_path':inputs['path'],'annotations_sha256':inputs['checksum'],
            'template_sha256':digest(inputs['base']),'normalized_verdicts_sha256':digest(inputs['rows']),
            'references':dict(info['references']),'sources':sources(PROJECT),
            'runtime':{'python':sys.version,'numpy':str(np.__version__),'platform':platform.platform()},
            'human_candidate_decisions_imported':inputs['result']['summary']['human_candidate_decisions_imported'],
            'neural_training':False,'holdout_evaluated':False,'new_prediction_scores_computed':False}


def _documents(info,mode,inputs):
    result=inputs['result'];summary={**result['summary'],'mode':mode,'source_review_identity':info['identity'],
        'source_event_identity':info['event_identity'],'status':'verdict_template_ready' if mode=='template' else
        ('candidate_verdicts_imported' if result['summary']['non_pending_verdict_rows'] else 'no_human_verdicts')}
    protocol={'schema':'candidate_verdicts_b2b1_v1','mode':mode,'decisions':list(DECISIONS),
              'columns':list(COLUMNS),'locked_columns':list(LOCKED),'reading_progress_is_a_verdict':False,
              'boundaries_modified':False,'automatic_human_confirmation':False,'mutable_files':[]}
    readme='''# B2b1 候选判决版本

本目录全部文件不可变；不要直接填写模板包的candidate_verdicts.csv，也不要覆写已导入的源CSV。
请复制模板另存新草稿文件，仅编辑decision/reviewer/reason；每次导入指定新output-root。
pending必须保持reviewer/reason零长度；其他状态须有复核人和理由。模板全pending不代表已完成人工判决。
needs_correction和uncertain不进入confirmed；rejected只是该候选被驳回，不是全分析块阴性。
复合配对按stop/restart组成派生，不是独立人工判决。保留原事件边界及删失；不计算预测分数。
每单元格最多8192个Unicode码点。CSV支持UTF-8 BOM、多行引号，可排序，不得删行或改来源字段。
B2a阅读进度不转成标签；合法阅读表变化不改变本版本，结构损坏会被父校验拒绝。
本版本只验证声明，不认证reviewer身份，也不声称官方GT或穷尽标注。
'''
    docs={'template.json':_json(inputs['base']),'summary.json':_json(summary),
          'protocol.json':_json(protocol),'README.md':readme.encode('utf-8')}
    if mode=='template':docs['candidate_verdicts.csv']=csv_text(inputs['base']).encode('utf-8-sig')
    else:
        for key in ('atomic_verdicts','confirmed_candidates','rejected_candidates','unresolved_candidates','derived_pairs'):
            docs[key+'.json']=_json(result[key])
        docs['annotations.csv']=inputs['raw']
    return docs,summary


def _check(binding):
    if sources(PROJECT)!=binding['sources']:raise ValueError('verdict source changed')
    check_references(binding['references'])


def _files(output,hashes,completed=False):
    actual={p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    if actual!=set(hashes)|({'completed.json'} if completed else set()):raise ValueError('verdict file coverage mismatch')
    for name,expected in hashes.items():
        path=(output/name).resolve()
        if not path.is_relative_to(output) or sha256_file(path)!=expected:raise ValueError('verdict artifact checksum mismatch')


def _prepare(parent,owner,output,mode,annotations=None):
    output=Path(output).resolve()
    if output.exists():raise FileExistsError('use a new verdict output directory')
    before=sources(PROJECT);info=parent_snapshot(parent,owner);inputs=_inputs(info,mode,annotations)
    binding=_binding(info,mode,inputs)
    if binding['sources']!=before:raise ValueError('verdict source changed during parent validation')
    output.mkdir(parents=True,exist_ok=False)
    try:
        _write_bytes(output/'progress.json',_json({'state':'assembling'}))
        documents,summary=_documents(info,mode,inputs)
        documents.update({'run.json':_json({'identity':digest(binding),'binding':binding}),
                          'parent_proof.json':_json(info['proof']),'progress.json':_json({'state':'complete'})})
        for name,value in documents.items():
            _write_bytes(output/name,value)
            if (output/name).read_bytes()!=value:raise ValueError('verdict write/read mismatch')
        _check(binding);hashes={name:sha256_file(output/name) for name in documents};_files(output,hashes)
        _write_bytes(output/'completed.json',_json({'identity':digest(binding),'sha256':hashes}))
    except BaseException:
        (output/'progress.json').write_bytes(_json({'state':'failed'}));raise
    return summary


def make_template_pack(parent,parent_owner,output):return _prepare(parent,parent_owner,output,'template')


def import_overlay(parent,parent_owner,annotations,output):return _prepare(parent,parent_owner,output,'overlay',annotations)


def verify_pack(output,expected_mode=None):
    output=Path(output).resolve();marker_path=output/'completed.json'
    if not marker_path.is_file():raise ValueError('incomplete verdict pack')
    marker_hash=sha256_file(marker_path);marker=read_json(marker_path)
    if set(marker)!={'identity','sha256'}:raise ValueError('immutable verdict marker fields required')
    _files(output,marker['sha256'],True)
    run,proof=read_json(output/'run.json'),read_json(output/'parent_proof.json');binding=run['binding']
    mode=binding['mode']
    if mode not in MODES or (expected_mode is not None and mode!=expected_mode):raise ValueError('verdict mode mismatch')
    if digest(binding)!=run['identity'] or run['identity']!=marker['identity']:raise ValueError('verdict run identity mismatch')
    _check(binding);info=parent_snapshot(binding['parent'],binding['parent_owner'],proof)
    inputs=_inputs(info,mode,binding['annotations_path'])
    if digest(binding)!=digest(_binding(info,mode,inputs)):raise ValueError('verdict policy/runtime replay mismatch')
    documents,summary=_documents(info,mode,inputs)
    documents.update({'run.json':_json(run),'parent_proof.json':_json(proof),'progress.json':_json({'state':'complete'})})
    if set(documents)!=set(marker['sha256']):raise ValueError('verdict expected file coverage mismatch')
    for name,value in documents.items():
        if (output/name).read_bytes()!=value:raise ValueError('verdict semantic replay mismatch: '+name)
    _check(binding);_files(output,marker['sha256'],True)
    if sha256_file(marker_path)!=marker_hash:raise ValueError('verdict marker changed during replay')
    return {**summary,'verdicts_semantically_replayed':True,'identity':run['identity'],
            'parent_reading_status_counts':info['reading_status_counts']}


def main(mode):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path);parser.add_argument('--parent-owner',type=Path)
    if mode=='overlay':parser.add_argument('--annotations',type=Path)
    parser.add_argument('--output-root',type=Path,required=True);parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if args.verify_only:result=verify_pack(args.output_root,mode)
    else:
        if args.parent is None or args.parent_owner is None or (mode=='overlay' and args.annotations is None):
            parser.error('--parent/--parent-owner and overlay --annotations required')
        result=_prepare(args.parent,args.parent_owner,args.output_root,mode,getattr(args,'annotations',None))
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)

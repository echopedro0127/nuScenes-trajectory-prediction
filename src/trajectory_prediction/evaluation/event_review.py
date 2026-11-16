"""Frozen static review pack with one validated mutable reading worksheet."""
import argparse
import csv
import io
import json
import platform
import sys
from pathlib import Path

import numpy as np

from trajectory_prediction.analysis.track_timeline import digest
from trajectory_prediction.evaluation.event_review_data import (
    assemble_review,
    csv_text,
    validate_working,
)
from trajectory_prediction.evaluation.event_review_html import render_html
from trajectory_prediction.evaluation.event_sources import read_json
from trajectory_prediction.evaluation.review_sources import parent_snapshot
from trajectory_prediction.evaluation.track_sources import check_references, sources
from trajectory_prediction.utils.provenance import sha256_file

PROJECT=Path(__file__).resolve().parents[3]
MUTABLE=['review_working.csv']


def _json(value): return (json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n').encode('utf-8')


def _write_bytes(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_bytes(value);temporary.replace(path)


def _documents(data):
    coverage=io.StringIO(newline='')
    fields=sorted({k for row in data['coverage'] for k in row})
    writer=csv.DictWriter(coverage,fieldnames=fields,lineterminator='\r\n')
    writer.writeheader();writer.writerows(data['coverage'])
    readme='''# 离线GT复核资料

打开review.html即可查看全部分析块与规则候选，无需服务或网络。
block_scan包含所有分析块，含无候选上下文；event_check含复合配对，不代表独立驾驶过程数。
规则未触发不是人工阴性，阅读状态不是事件标签。所有诊断与原窗口覆盖均保留，没有预测排名。

review_working.csv是唯一可编辑文件。仅修改review_status/reviewer/notes，可排序，不得删行、增加任务或修改来源ID。
允许pending/in_progress/reviewed；reviewed须填写复核人和备注，仅表示阅读记录，不产生人工GT。
每个单元格最多8192个Unicode码点，Python与浏览器使用相同的长度和空白规则。
每个单元格最多8192个Unicode码点，浏览器与Python使用相同长度与必填空白规则。
在页面导入已有同源CSV继续阅读，再下载更新CSV；浏览器不会自动覆盖本机工作表。
如需用命令验证阅读记录，将下载文件保存为本目录review_working.csv后执行原verify-only命令。
review_baseline.csv及其他文件不可编辑；正式包独立重建时验证全部静态内容与工作表结构。

预测anchor、人工标签导入和停车指标另行冻结。本资料包不训练、不评价holdout、不计算新预测分数。
'''
    return {'review_data.json':_json(data),'queue.json':_json(data['queue']),
            'coverage.csv':coverage.getvalue().encode('utf-8-sig'),
            'review_baseline.csv':csv_text(data['queue']).encode('utf-8-sig'),
            'review.html':render_html(data).encode('utf-8'),'README.md':readme.encode('utf-8'),
            'summary.json':_json(data['summary']),
            'protocol.json':_json({'schema':'event_review_materials_v1','mutable_files':MUTABLE,
                'editable_columns':['review_status','reviewer','notes'],'reading_is_not_human_event_label':True,
                'sampling':'all_blocks_all_candidates_all_diagnostics','new_anchor_selection':False})}


def _binding(info):
    return {'schema':'event_review_b2a_v1','parent':info['parent'],'parent_owner':info['owner'],
            'parent_identity':info['identity'],'dataset_release':info['dataset_release'],
            'references':dict(info['references']),'sources':sources(PROJECT),
            'runtime':{'python':sys.version,'numpy':str(np.__version__),'platform':platform.platform()},
            'human_event_labels_imported':False,'neural_training':False,'holdout_evaluated':False,
            'new_prediction_scores_computed':False}


def _check(binding):
    if sources(PROJECT)!=binding['sources']: raise ValueError('review source changed')
    check_references(binding['references'])


def _files(output,hashes,completed=False):
    actual={p.relative_to(output).as_posix() for p in output.rglob('*') if p.is_file()}
    expected=set(hashes)|set(MUTABLE)|({'completed.json'} if completed else set())
    if actual!=expected: raise ValueError('review file coverage mismatch')
    for name,expected_hash in hashes.items():
        path=(output/name).resolve()
        if not path.is_relative_to(output) or sha256_file(path)!=expected_hash:
            raise ValueError('review static artifact checksum mismatch')


def _working(output,queue):
    path=(output/MUTABLE[0]).resolve()
    if not path.is_relative_to(output): raise ValueError('working file escapes review directory')
    return validate_working(path.read_text(encoding='utf-8-sig'),queue)


def prepare_review(parent,parent_owner,output):
    output=Path(output).resolve()
    if output.exists(): raise FileExistsError('use a new review directory')
    before=sources(PROJECT);info=parent_snapshot(parent,parent_owner)
    binding=_binding(info)
    if binding['sources']!=before: raise ValueError('review source changed during native verification')
    output.mkdir(parents=True,exist_ok=False)
    try:
        _write_bytes(output/'progress.json',_json({'state':'assembling'}))
        data=assemble_review(info['parent'],info['references'])
        documents=_documents(data)
        documents.update({'run.json':_json({'identity':digest(binding),'binding':binding}),
                          'parent_proof.json':_json(info['proof']),'progress.json':_json({'state':'complete'})})
        for name,value in documents.items():
            _write_bytes(output/name,value)
            if (output/name).read_bytes()!=value: raise ValueError('review write/read mismatch')
        _write_bytes(output/MUTABLE[0],documents['review_baseline.csv'])
        _working(output,data['queue']);_check(binding)
        hashes={name:sha256_file(output/name) for name in documents}
        _files(output,hashes)
        _write_bytes(output/'completed.json',_json({'identity':digest(binding),'sha256':hashes,'mutable_files':MUTABLE}))
    except BaseException:
        (output/'progress.json').write_bytes(_json({'state':'failed'}))
        raise
    return data['summary']


def verify_review(output):
    output=Path(output).resolve();marker_path=output/'completed.json'
    if not marker_path.is_file(): raise ValueError('incomplete review pack')
    marker_hash=sha256_file(marker_path);marker=read_json(marker_path)
    if marker.get('mutable_files')!=MUTABLE: raise ValueError('review mutable file policy mismatch')
    _files(output,marker['sha256'],True)
    run,proof=read_json(output/'run.json'),read_json(output/'parent_proof.json');binding=run['binding']
    if digest(binding)!=run['identity'] or marker['identity']!=run['identity']:
        raise ValueError('review run identity mismatch')
    _check(binding);info=parent_snapshot(binding['parent'],binding['parent_owner'],proof)
    if digest(binding)!=digest(_binding(info)): raise ValueError('review policy/runtime replay mismatch')
    data=assemble_review(info['parent'],info['references']);documents=_documents(data)
    documents.update({'run.json':_json(run),'parent_proof.json':_json(proof),'progress.json':_json({'state':'complete'})})
    if set(marker['sha256'])!=set(documents): raise ValueError('review expected file coverage mismatch')
    for name,value in documents.items():
        if (output/name).read_bytes()!=value: raise ValueError('review semantic replay mismatch: '+name)
    counts=_working(output,data['queue']);_check(binding);_files(output,marker['sha256'],True)
    if sha256_file(marker_path)!=marker_hash: raise ValueError('review marker changed during replay')
    return {**data['summary'],'review_semantically_replayed':True,'working_status_counts':counts,'identity':run['identity']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path);parser.add_argument('--parent-owner',type=Path)
    parser.add_argument('--output-root',type=Path,required=True);parser.add_argument('--verify-only',action='store_true')
    args=parser.parse_args()
    if not args.verify_only and (args.parent is None or args.parent_owner is None):
        parser.error('--parent and --parent-owner required')
    result=verify_review(args.output_root) if args.verify_only else prepare_review(args.parent,args.parent_owner,args.output_root)
    print(json.dumps(result,indent=2,allow_nan=False),flush=True)

# Phase 3B B0a Track Timeline Implementation Plan

> **For agentic workers:** 使用superpowers:executing-plans在当前会话顺序实现，测试先行；不默认分发实现任务。完成前运行verification-before-completion并独立只读审查。

**Goal:** 在新worktree把既有461目标track及608窗口装配到真实微秒时钟，保持旧3A源码/产物不变，不抽事件、不评分、不训练。

**Architecture:** 严格流式JSON读取器→纯annotation链/片段构建→缓存窗口几何和时钟映射→带原owner父验证证明的不可变发布。原3A验证器只在其原cwd/PYTHONPATH执行一次，保存证明后核对来源，不按窗口重放模型。

**Tech Stack:** Python标准库、NumPy、pyquaternion、pytest；复用现有Python环境，不下载依赖。

**Design:** [B0设计](../specs/2026-11-16-scenario-episodes-b0-design.md)。仅B0a，1967目标的B0b另作后续范围。

## T1 严格流式读取

Files: `src/trajectory_prediction/data/metadata_stream.py`、`tests/data/test_metadata_stream.py`。

- [x] 先写极小chunk=1/2/7下多对象、跨块转义/Unicode、空数组等价读取测试；再测缺逗号、尾逗号、截断、非对象、尾随内容、重复字段、NaN/Inf及记录长度上限。
- [x] RED命令：`python -m pytest tests/data/test_metadata_stream.py -q --basetemp outputs/pytest-b0-stream-red`。
- [x] 实现接口`iter_json_objects(path, chunk_size=1048576, max_record_chars=1048576)`；顶层严格`[`，value与`,`/`]`交替，JSONDecoder raw_decode跨块重试，已消费前缀丢弃，关闭后仅允许空白。

```python
assert list(iter_json_objects(path, chunk_size=1)) == [{'token': 'a'}, {'token': 'b'}]
with pytest.raises(ValueError):
    list(iter_json_objects(path_with_trailing_comma, chunk_size=2))
```

- [x] GREEN后提交读取器与测试，不把整表读入或引入ijson依赖。

## T2 纯track链、缺口及语义ID

Files: `src/trajectory_prediction/analysis/track_timeline.py`、`tests/analysis/test_track_timeline.py`。

- [x] 先测试按first→next重建、非精确0.5秒、缺失annotation对应的sample链缺口；验证行序改变ID不变、XY语义改变ID改变。
- [x] 再测重复token/instance-sample、循环、断链/非互反、首尾或nbr_annotations不符、非正时间、跨scene/log/role、非有限XY/rotation、缺类别/外键。
- [x] 实现`build_tracks(annotations, samples, instances, scenes, categories, targets_by_role, logs_by_role, dataset_release)`。输出normalized tracks、segments、gaps。source token允许非空<=32字符；时间为正int64，不能bool/float。
- [x] segment只在annotation跳过scene sample时断开；缺口记录真实缺失sample token和时间，不用dt/.5猜缺失帧数。track语义摘要按链顺序规范化，片段ID绑定语义摘要/版本/角色/边界tokens，原文件hash单独存来源。

```python
a = build_tracks(rows, **tables)
b = build_tracks(list(reversed(rows)), **tables)
assert a == b
assert len(a['gaps']) == 1
assert len(a['segments']) == 2
```

- [x] 运行`pytest tests/analysis/test_track_timeline.py -q`，GREEN后提交。

## T3 旧owner证明与B0发布

Files: `src/trajectory_prediction/evaluation/track_sources.py`、`src/trajectory_prediction/evaluation/scenario_tracks.py`、`scripts/prepare_scenario_tracks.py`、`tests/evaluation/test_scenario_tracks.py`。

- [x] 构造两角色、17帧的小型元数据/缓存夹具，使用旧坐标转换生成reference cache；伪父CLI只验证owner调用/证明协议，不冒充真实模型实验。
- [x] 测build→verify、source hash/几何/身份篡改（重算hash也拒绝）、显式owner路径与PYTHONPATH、缺父证明、目录覆盖拒绝及中断无completed。
- [x] `verify_parent_in_owner(parent, owner)`使用显式owner脚本/cwd/PYTHONPATH执行原CLI一次；stdout要求semantic true、no-training/no-holdout，并核对父summary、完成凭据和冻结源码前后快照。保存parent_proof，不在每个窗口重跑父CLI。
- [x] 仅取父predictions的去重窗口/instance，原缓存确定logs；流式过滤目标instance，绑定全部六类元数据和缓存manifest/archive。parent/sample/scene/log既有hash必须一致，不猜owner目录层级。
- [x] 461目标链构建后，608窗口follow prev4/next12映射；原rotation算yaw，历史先按旧提取路径转float32，未来保留原float64输入，再调用既有global_to_agent，要求缓存anchor/history/future精确相同。
- [x] 写入role NPZ（timestamp_us、world_xy、rotation、token、segment_id、actual_dt_us/delta_valid）、tracks/segments/gaps、window映射、audit、protocol、run、parent_proof、summary。先源/代码快照，文件落盘读回比较并记hash；最后复查所有来源和文件集合再写completed。
- [x] verify-only从保存父证明及不变来源重建全部新数组/映射/ID/摘要；不信任单独summary，不重新评分或逐窗口运行旧模型。JSON拒绝NaN。

```powershell
python scripts/prepare_scenario_tracks.py --parent ../scenario-evaluation/outputs/scenario-evaluation/phase3a-v1 --parent-owner ../scenario-evaluation --output-root outputs/scenario-tracks/b0a-v1
python scripts/prepare_scenario_tracks.py --output-root outputs/scenario-tracks/b0a-v1 --verify-only
```

## T4 实际验收与交付

2026-11-16执行检查点：T1–T3代码及夹具已实现，58项定向测试通过（9.60秒），Ruff及diff检查通过；独立只读复审未发现Critical/Important问题。全套回归正在运行，真实B0a装配和独立verify-only尚未执行，因此当前不声明数据验收完成。


- [x] 单元/夹具通过后做只读代码审查，修复必须项再跑全套pytest、Ruff、公开扫描和diff检查；源码先本地提交。
- [x] 运行真实B0a并独立verify-only，核对461 tracks/16010 annotations/608窗口/4缺口，465连续片段不得称episode。显示名义/实际时钟区别，不修改3A分数。
- [x] 验证旧owner的228个源文件及217个产物不变；报告真实计数和输入/输出身份，更新文档，本地提交，不推送。
- [x] B0b、事件抽取、人工overlay、停车评分和新训练均不在本次实现范围。

验收结果：源码09cbe1c；920项测试通过（234.02秒），Ruff/公开扫描/diff检查通过。真实B0a及独立verify-only通过，运行身份d8a904ed958ece5bf797fdf5a21aba6d755680d3f79b8e9809baf03088664102；旧228源码/217产物不变。详见[结果报告](../../SCENARIO_TRACKS_B0A_RESULTS.md)。

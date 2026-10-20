# P6 map packet quality gate Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline with review checkpoints.

**Goal:** 完整三seed地图包退化下固定/学习门控、精确CV回退与可复现报告。
**Architecture:** 已验证P4缓存/冻结模型→过去packet候选+质量→161参数gate→CV混合。
**Tech Stack:** Python3.12、NumPy mmap、PyTorch、pandas、pytest；不下载、不推送。

设计见../specs/2026-10-20-causal-context-phase6-design.md。
使用../trajectory-mvp/.venv/Scripts/python.exe，PYTHONPATH=当前src，测试独占outputs/pytest-GUID。

## Task 1：包时序与门控算子

文件online/quality_gate.py；tests/online/test_quality_gate.py。
- [ ] 写失败测试：delay1s首两次无包、第三次读packet1；drop4/5/6保留packet3；
  stale第6次age2.5无效；未来包变动不影响过去选择；缺失g=0混合严格等于CV。
- [ ] 实现GateConfig、packet_schedule、quality_features、QualityGate、fixed_weights、
  blend_forecasts；输入仅quality/valid/candidates，不含labels。
- [ ] 运行聚焦pytest，确认范围/非法dtype/finite/边界后本地提交。

## Task 2：冻结P4候选准备和独立缓存

文件context/gate_preparation.py、online/gate_dataset.py；scripts/prepare_quality_gate.py；
configs/context/gate.yaml；tests/context/test_gate_preparation.py。
- [ ] 写失败测试：仅选过去P4包；P4原/地图checkpoint scene/base契约；候选不受未释放
  future扰动；cache roundtrip/错文件hash/错mask/质量重算拒绝。
- [ ] load_map_inputs验证trajectory、map sidecar与GRU；load_candidates严格加载P4
  original/map并核对对应split provenance，val须与训练/选模场景无交集。
- [ ] 分批生成CV/filtered/original/map及quality、source/available/age/valid；NPY+manifest
  固定schema+哈希，labels独立读取；不改旧mmap helper的Map schema路径。
- [ ] 先合成parity再完整train/train_val/val三seed；fresh val对照P4结果，提交。

## Task 3：训练与checkpoint

文件online/gate_training.py；scripts/train_quality_gate.py；configs/models/quality_gate.yaml；
tests/online/test_gate_training.py。
- [ ] 失败测试：调用边界、无效门控0、场景重叠拒绝、真实输入变动拒绝复用、best权重回载。
- [ ] 训练仅161参数gate；20epoch/patience5、4096batch、三seed、source身份和逐epoch日志。
- [ ] 合成1epoch通路通过后完整train/train_val固定设置训练并保存实际进度，提交。

## Task 4：评估、统计与延迟

文件online/gate_evaluation.py、analysis/gate_report.py；scripts/evaluate_quality_gate.py、
scripts/report_quality_gate.py；tests/online/test_gate_evaluation.py、tests/analysis/test_gate_report.py。
- [ ] 先失败测试：六条件七策略同key、missing/stale精确CV与gate0、首步jitter、
  dropout恢复连续窗口/右删失、scene-paired CI、漏行/seed/source/诊断/计时字段拒绝。
- [ ] 完整val三seed批量评估，压缩逐行结果、分组、scene sum/count、gate分布、恢复；
  按scene平均固定seed后2000次重采样，记录seed std，不用val调参数。
- [ ] 固定256key/16warmup同步测算子缓存路径延迟，明确不含上游候选forward/地图查询。
- [ ] 全量结果汇总，记录相对CV/filtered/fixed/map正负结果与硬回退计数，提交。

## Task 5：审查与交付

- [ ] requesting-code-review独立只读审查；重要问题先复现失败再修复并复核。
- [ ] 全量pytest、Ruff、公开扫描、diff检查；写docs/CAUSAL_CONTEXT_PHASE6_RESULTS.md。
- [ ] 仅真实完整实验完成后将P6标完成；保留worktree/feature分支、模型与data不提交，不推送。

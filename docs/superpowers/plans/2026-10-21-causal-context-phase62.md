# P6.2 Gate Diagnostics Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline with review checkpoints.

**Goal:** 完成冻结候选/质量特征诊断及完整train_val三seed重放。
**Architecture:** 生产缓存/模型严格加载→oracle与分层置换→完整数值/场景表→跨seed统计与重放。
**Tech Stack:** 现有Python3.12/PyTorch/NumPy/pandas/pytest，CUDA；独立worktree。

设计见[spec](../specs/2026-10-21-causal-context-phase62-design.md)。只读取train_val，不训练、不访问val。

## Task 1：诊断算子与失败测试

文件tests/analysis/test_gate_diagnostics.py、src/trajectory_prediction/analysis/gate_diagnostics.py。
- [ ] 写测试oracle_mix(cv,candidate,target,mask,valid,reference_weights)：cv=0、candidate=2、target=1时g=.5且ADE0；mask外target改变不影响；invalid返回g0；包含实际g保证ADE不差于参照。
- [ ] 写测试permutation_donors(valid,ages,seed)：[6,N,11] donor仅同条件/步/age/valid，invalid为自身；重复可重现，输入不改。
- [ ] 实现上述算子：网格torch逐g前向，按mask均值严格<更新；donor NumPy default_rng固定seed。
- [ ] 运行pytest观察RED→GREEN。

## Task 2：完整指标和审计产物

同模块新增diagnose(data,suite,device)，write_outputs(root,arrays,tables,metadata)，verify_outputs(root,arrays,tables,metadata)。
- [ ] 写合成完整11步数据测试：端点和置换missing精确CV、quality不改、常数特征变化比例0、场景计数/形状完整。
- [ ] 写NPZ篡改且更新manifest hash测试，verify_outputs仍须与重算数值比较拒绝。
- [ ] 实现分块4096预测，forecast_metrics复用，保留所有方法完整指标；oracle单独ADE/g；汇总condition/scene/置换变化；原子manifest绑定key、source、suite、selection、cache。
- [ ] pytest聚焦通过，Ruff通过后本地提交。

## Task 3：三seed运行、统计与重放

新增scripts/diagnose_gate_candidates.py，模块main只接受cache/training/p6/selection/output root和device、--verify。
- [ ] 三seed每次生产load_verified_gate_cache(...,'train_val',seed)、load_suite(...,'train_val',...)，父引用与selection相等，否则拒绝。
- [ ] 输出端点与oracle均值/std，置换先repeat均值再seed均值；176场景bootstrap2000输出66条区间。每seed以及汇总含源哈希，verify模式所有表/NPZ和汇总重算比较。
- [ ] 执行python scripts/diagnose_gate_candidates.py --cache-root outputs/context/p6/cache --training-root outputs/context/p61/training --p6-training-root outputs/context/p6/training --selection-root outputs/context/p61/selection --output-root outputs/context/p62 --device cuda。
- [ ] 同命令加--verify，期望三个seed及汇总audit通过。

## Task 4：验收和结论

- [ ] 全量pytest -q --tb=short（独立outputs/pytest-GUID）、ruff check src scripts tests、scripts/check_public_repo.py、git diff --check。
- [ ] 对比38e346c所有旧绑定源码diff为空；核对报告3seed/6621/66区间与oracle/置换限制。
- [ ] 写docs/CAUSAL_CONTEXT_PHASE62_RESULTS.md，更新roadmap与计划；本地提交，保留branch/worktree，不推送。

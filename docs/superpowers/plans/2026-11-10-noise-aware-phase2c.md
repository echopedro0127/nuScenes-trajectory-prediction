# Phase 2C Implementation Plan

**Goal:** 验证受约束CA/刹停/保持候选对减速和低速失败的作用。
**Architecture:** 只读父缓存+同源packet→三新候选→同输入四组门控→全条件及机动诊断→审计。
**Tech Stack:** NumPy/PyTorch/pandas/pytest，既有trajectory-mvp环境，新worktree内顺序执行。

按[设计](../specs/2026-11-10-noise-aware-phase2c-design.md)执行。

- [ ] 1. tests/robustness/test_motion_experts.py先RED；实现motion_experts.py，测试解析刹停、零速、加速、前缀因果和拒绝非规则流。
- [ ] 2. tests/robustness/test_motion_learning.py先RED；实现motion_learning.py，10头同初始化/严格掩码，baseline7与旧训练权重和日志精确一致，停止候选能被实际学习。
- [ ] 3. tests/robustness/test_phase2c_experiment.py小型发布/篡改先RED；实现phase2c_experiment.py和CLI，原worktree父链验证，复用来源与轨迹键，缓存/训练/评价保存完整记录。
- [ ] 4. 全量pytest/Ruff/公开扫描通过，冻结提交，运行outputs/noise-aware/phase2c。
- [ ] 5. verify-only全指标/分组/权重重算，baseline精确复现和64流候选重建通过；核对报告/路线图，本地提交。

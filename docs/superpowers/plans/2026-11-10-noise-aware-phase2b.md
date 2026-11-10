# Noise-Aware Phase 2B Implementation Plan

**Goal:** 用机动分组、显式时间和专家移除解释Phase 2A收益。
**Architecture:** 只读父缓存→五种固定受控训练→逐更新评价→机动/事件汇总→独立审计。新增模块不改父源码。
**Tech Stack:** NumPy/PyTorch/pandas/pytest，原trajectory-mvp虚拟环境。

按[设计](../specs/2026-11-10-noise-aware-phase2b-design.md)在新worktree顺序执行。

- [x] 1. test_diagnostic_controls.py确认缺少模块的RED，四项测试覆盖时间、协方差查表、严格屏蔽和full精确训练复现。
- [x] 2. diagnostic_controls.py实现控制特征、掩码及相同优化顺序，测试GREEN。
- [x] 3. test_maneuver_diagnostics.py先RED后GREEN；离线标签、重叠组、真实窗口计数及标签/前向隔离通过。
- [x] 4. 小型发布/篡改测试RED→GREEN；phase2b_experiment.py及CLI绑定父实验，在原worktree验证，保存全部模型和逐更新/分组事件指标。
- [x] 5. 625项pytest、Ruff及公开扫描通过，冻结8744ce8。五控制真实实验完成，outputs/noise-aware/phase2b保留93,024条指标；旧产物只读。
- [x] 6. 全指标/分组/事件重算，full权重与5,472流指标精确复现通过；[报告](../../NOISE_AWARE_PHASE2B_RESULTS.md)核对数值并记录减速/低速失败及限制，只本地提交。

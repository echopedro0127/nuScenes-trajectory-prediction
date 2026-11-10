# Phase 2D Implementation Plan

**Goal:** 三模型seed验证CA稳定性和固定因果转弯保护。
**Architecture:** 只读父十候选→原观测重建保护→9个固定预算模型→逐seed/机动/跨seed汇总→审计。
**Tech Stack:** NumPy/PyTorch/pandas/pytest，旧虚拟环境、新worktree顺序执行。

按[设计](../specs/2026-11-10-noise-aware-phase2d-design.md)执行。

- [x] 1. guard缺模块和seed缺参数先RED；直行/静止/转弯/前缀、同seed重现/异seed变化、严格屏蔽及全允许与未保护训练一致通过。
- [x] 2. motion_guard.py及motion_learning.py默认兼容seed/guard实现，旧四控制列表不变，旧2027路径GREEN。
- [x] 3. 跨seed统计与小型发布/篡改RED→GREEN；phase2d_experiment.py/CLI完成父链、保护缓存、9模型和完整评价。
- [x] 4. 636项pytest/Ruff/公开扫描通过，冻结7bc11d9；outputs/noise-aware/phase2d完成180,576条策略记录。
- [x] 5. 全指标/分组/跨seed/权重/覆盖率重算、2027双控制各5,472流精确复现、64流保护重建通过；[报告](../../NOISE_AWARE_PHASE2D_RESULTS.md)保留保护在噪声条件下退化，只本地提交。

# Phase 2D Implementation Plan

**Goal:** 三模型seed验证CA稳定性和固定因果转弯保护。
**Architecture:** 只读父十候选→原观测重建保护→9个固定预算模型→逐seed/机动/跨seed汇总→审计。
**Tech Stack:** NumPy/PyTorch/pandas/pytest，旧虚拟环境、新worktree顺序执行。

按[设计](../specs/2026-11-10-noise-aware-phase2d-design.md)执行。

- [ ] 1. guard和seed回归测试先RED：直行/静止、转弯/前缀、同seed重现/异seed不同、CA严格屏蔽。
- [ ] 2. motion_guard.py实现固定规则；motion_learning.py增加默认兼容seed和guard mask，保持旧四控制列表，旧2027路径不变；测试GREEN。
- [ ] 3. 多seed汇总与小型发布/篡改先RED；实现phase2d_experiment.py与CLI，父实验原worktree校验、保护缓存、9模型与完整评价保存。
- [ ] 4. 全量pytest/Ruff/公开扫描通过，提交冻结后运行outputs/noise-aware/phase2d。
- [ ] 5. 全指标/分组/跨seed/权重重算、2027双控制精确复现、64流保护抽查通过；核对报告/路线图，本地提交。

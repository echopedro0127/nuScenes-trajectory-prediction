# Phase 2C Implementation Plan

**Goal:** 验证受约束CA/刹停/保持候选对减速和低速失败的作用。
**Architecture:** 只读父缓存+同源packet→三新候选→同输入四组门控→全条件及机动诊断→审计。
**Tech Stack:** NumPy/PyTorch/pandas/pytest，既有trajectory-mvp环境，新worktree内顺序执行。

按[设计](../specs/2026-11-10-noise-aware-phase2c-design.md)执行。

- [x] 1. motion_experts.py按RED→GREEN实现；解析刹停/零速/加速、前缀因果及非规则流拒绝通过。
- [x] 2. motion_learning.py按RED→GREEN实现；938名义参数、同初始化/严格屏蔽，baseline训练精确复现和停止候选实际学习通过。
- [x] 3. phase2c_experiment.py/CLI及小型发布/篡改测试RED→GREEN；原worktree父链、来源、轨迹键与完整记录实现。
- [x] 4. 630项pytest/Ruff/公开扫描通过，冻结5c89a99；真实outputs/noise-aware/phase2c完成131,328条策略记录。
- [x] 5. 全指标/分组/权重/标准化重算通过，baseline与父5,472流逐项一致，64流新增候选重建通过；[报告](../../NOISE_AWARE_PHASE2C_RESULTS.md)保留曲率/jitter退化及停止候选无总体增益，只本地提交。

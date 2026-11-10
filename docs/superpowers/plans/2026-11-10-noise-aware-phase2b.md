# Noise-Aware Phase 2B Implementation Plan

**Goal:** 用机动分组、显式时间和专家移除解释Phase 2A收益。
**Architecture:** 只读父缓存→五种固定受控训练→逐更新评价→机动/事件汇总→独立审计。新增模块不改父源码。
**Tech Stack:** NumPy/PyTorch/pandas/pytest，原trajectory-mvp虚拟环境。

按[设计](../specs/2026-11-10-noise-aware-phase2b-design.md)在新worktree顺序执行。

- [ ] 1. 写tests/robustness/test_diagnostic_controls.py，确认缺少模块的RED；覆盖纯时间编码、固定协方差查表、严格专家屏蔽和full训练复现。
- [ ] 2. 实现robustness/diagnostic_controls.py：transform_features、expert_mask、fit_control、predict_control；复用原网络，复制原优化顺序，仅替换特征/屏蔽logits；测试GREEN。
- [ ] 3. 写tests/robustness/test_maneuver_diagnostics.py，RED后实现maneuver_diagnostics.py；由现有describe_motion/tag_maneuvers生成离线标签，唯一轨迹计数与重叠组正确，未来标签不影响控制特征。
- [ ] 4. 写小型发布/篡改测试，RED后实现phase2b_experiment.py和scripts/run_noise_aware_diagnostics.py。绑定并在原worktree验证父实验，保存/重放模型、指标及分组事件。
- [ ] 5. 全量pytest/Ruff/公开扫描通过，提交冻结源码。运行scripts/run_noise_aware_diagnostics.py，默认outputs/noise-aware/phase2b；旧产物只读。
- [ ] 6. verify-only、full与父权重及指标精确一致性、文档数值核对通过，写docs/NOISE_AWARE_PHASE2B_RESULTS.md并更新路线，本地提交。

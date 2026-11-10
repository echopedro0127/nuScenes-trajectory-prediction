# Noise-Aware Phase 2A Implementation Plan

**Goal:** 重建共享带噪观测上的七候选，执行三种可靠性特征消融及常数门控对照。
**Architecture:** packets→候选/诊断特征→fit-only训练→select评价。新文件隔离于Phase 1源码；模型输入与评价标签分离。
**Tech Stack:** NumPy、PyTorch、nuScenes静态地图、pytest。

在现有隔离worktree内按[设计](../specs/2026-11-10-noise-aware-phase2a-design.md)顺序执行，不另行启动子代理。

- [x] Task 1：先写tests/robustness/test_noisy_candidates.py，确认缺少模块的RED；六项测试覆盖未来包前缀、带噪历史、逐步地图查询、共同平移、空图回退、拒绝延迟和非空地图坐标。
- [x] Task 2：实现robustness/noisy_candidates.py；返回candidates[11,7,12,2]和features[11,18]。只读已到包，inference_mode前向，无GT接口，Task 1测试GREEN。
- [x] Task 3：reliability_learning.py按RED→GREEN实现，验证mask/容量/初始化、fit-only统计、凸融合、实际专家学习及目标拒绝。固定20epoch，fit反传；增加全零特征常数门控对照。
- [x] Task 4：phase2_data.py、phase2_experiment.py及scripts/run_noise_aware_gate.py保存完整来源、候选、四门控及指标；小型集成、篡改拒绝测试RED→GREEN。增加JSON路径回归测试，复现并修复验证入口问题。
- [x] Task 5：618项pytest、Ruff及公开扫描通过，最终冻结4936130；outputs/noise-aware/phase2a-v1完成608×3×3条流和65,664条策略指标。旧phase2a保留追溯，参数和最终数值不变。
- [x] Task 6：全部指标/门控前向/标准化重验，两角色候选重建抽查通过；跨Phase 1物理和clean GRU核对及修复前后精确一致性验证通过。[报告](../../NOISE_AWARE_PHASE2A_RESULTS.md)保留负面结果和限制，只本地提交。

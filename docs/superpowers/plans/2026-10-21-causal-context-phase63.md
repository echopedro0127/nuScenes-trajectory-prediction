# P6.3 Causal Feedback Horizon Gate Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline with review checkpoints.

**Goal:** 完成已批准D/R/H八组结构消融、train_val冻结选择及条件推进的val回归。
**Architecture:** 生产P6缓存→独立因果前缀特征→337参数共享gate→身份绑定训练→完整度量与冻结选择→严格重放报告。
**Tech Stack:** Python3.12、PyTorch/CUDA、NumPy mmap/NPZ、pandas、pytest，现有venv；只读旧阶段模块。

设计见[spec](../specs/2026-10-21-causal-context-phase63-design.md)，用户已于本轮批准继续下一步。

## Task 1：因果特征与逐点门控

文件online/horizon_features.py、online/horizon_gate.py、tests/online/test_horizon_gate.py。
- [ ] 先写失败测试：修改未来positions/velocities/labels不影响前缀特征；R第k步只读上一预测的当前点，invalid清空map反馈，首步flag0。
- [ ] 写失败测试：337参数、八variant相同初始有效未来预测、关闭H逐点g相等、缺包精确CV、静止无NaN与样本重排不串状态。
- [ ] 实现HorizonDataset(RefinementDataset)；只从history/position/velocity/previous_velocity/map/cv/quality/valid构建D/R/H，不读取target。
- [ ] 实现HorizonGate(variant,initial)，19→16→1/337参数，前8列初始化P6、其余0；未来mask与valid在输出处hard0；blend_horizon保持精确端点。
- [ ] 聚焦pytest确认RED→GREEN，Ruff；本地提交。

## Task 2：身份绑定三seed训练

文件online/horizon_training.py、scripts/train_horizon_gate.py、configs/models/horizon_gate.yaml、tests/online/test_horizon_training.py。
- [ ] 失败测试：scene重叠拒绝、改输入内容拒绝复用、marker/checkpoint篡改拒绝、角色互换拒绝。
- [ ] 实现train_horizon：B/D/R/H/DR/DH/RH/DRH，同P6初始化、seed/shuffle、10epoch/patience3/batch4096/AdamW.001/.0001；masked SL1+.05 smoothness。
- [ ] 每epoch完整train_val预测计算ADE/FDE/jitter，按ADE选best；绑定实际输入指纹、源码/config/父契约、P6权重和variant；完成后原子marker。
- [ ] 生产main只读取train/train_val，先全部单测及独立只读审查，再冻结源码并执行三个seed共24训练。

## Task 3：度量、选择与推进

文件online/horizon_evaluation.py、scripts/evaluate_horizon_gate.py、tests/online/test_horizon_evaluation.py。
- [ ] 失败测试完整SEEDS×6conditions×全部策略矩阵、角色/场景/来源、future shape、p95和约束/tie、基线胜出和不达推进标准不读val。
- [ ] 复用旧P6.1生产load_suite并绑定frozen selection，计算CV/P6/finetune_0/time_age_gate/consistent_short/consistent_010与八新变体。
- [ ] 每策略保存[6,N,11,3] metrics和[6,N,11,12]weights，分scene/condition/update/horizon/maneuver/p90p95/recovery/weight分布；所有keys与源码/模型哈希绑定。
- [ ] train_val三seed均值选accuracy/stable，FDE<=1.005×reference、clean FDEp95<=1.01×reference等按spec；tie固定，保存selection及所有表。
- [ ] 满足>=.5% ADE或>=2% jitter且优于B才推进val；不满足则明确停止val入口。满足则只对固定方案和全体消融完整val回归，不调参。
- [ ] 固定256key/16warmup/batch1/CUDA同步，计入反馈状态更新/特征/head/blend，地图查询及P4 forward排除并标注；保存原始计时样本。

## Task 4：独立重放、统计与验收

文件analysis/horizon_report.py、scripts/report_horizon_gate.py、tests/analysis/test_horizon_report.py。
- [ ] 失败测试：NPZ更新hash后篡改仍拒绝、漏列/shape/key/source不一致拒绝；selection约束与推进状态可重算。
- [ ] 报告从生产父缓存重建全部D/R/H特征，load所有checkpoint/旧suite，重算所有metrics/weights/分组表与完整选模；不只核对summary。
- [ ] 固定3seed平均后2000次scene bootstrap、seed mean/std及负面证据；若未推进只报告train_val，不冒充val收益。
- [ ] requesting-code-review独立只读复核；全量pytest/Ruff/公开扫描/diff；旧P6/P6.1/P6.2绑定源码diff为空。
- [ ] 写docs/CAUSAL_CONTEXT_PHASE63_RESULTS.md，更新roadmap/spec状态/此计划；本地提交，保留worktree不推送。

执行Python=../trajectory-mvp/.venv/Scripts/python.exe；PYTHONPATH=当前src；pytest独占outputs/pytest-GUID。
本轮只推进结构消融及预设val回归，不自动执行第二轮一致性或SHARP训练。

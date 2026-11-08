# Independent Quality Gate Plan

**Goal:** 在新fit/select候选上训练quality gate并重载核验。
**Architecture:** 新online/independent_gate_training.py验证真实角色、共同上游、scene/log隔离及缓存身份；复用旧train_gate/run_epoch，不改旧源码。独立checkpoint格式independent_quality_gate_v1，绑定新输入及wrapper源码。
**Tech Stack:** PyTorch、NumPy、pytest。

用户已授权继续分阶段重训。沿用旧预算：seed2027，最多20epoch，patience5，batch4096，lr0.001，AdamW weight_decay0.0001，smoothness0.05，hidden16/161参数，六条件均衡、按select平均在线ADE选模。
候选完全冻结，无旧gate权重初始化；地图包无效时gate权重为0并回退CV。只运行fit/select，不训练RH/一致性，不计算holdout指标，不增加看到结果后的预算。

- [x] 先写tests/online/test_independent_gate_training.py并确认RED，覆盖角色/上游/scene/log隔离、短训练及重载、错身份与覆盖拒绝。
- [x] 新增独立入口和脚本，复用训练核心；记录源码、候选manifest与合同、新分组身份，重载复算并发布完成marker。
- [x] 全量pytest、Ruff与公开扫描后冻结源码，执行真实训练。
- [x] 独立重载核验并计算select六条件诊断、CV/map对照，报告限制。更新本地文档，不推送、不改简历。

实现dbbb285，563项测试通过；训练16epoch早停/best11，独立实际数据身份和指标重算通过。见[阶段报告](../../CAUSAL_CONTEXT_INDEPENDENT_GATE_RESULTS.md)。

输出outputs/context/p66-independent/gate-training/seed-2027，拒绝已有目录，无resume。后续RH/一致性仍需独立重训，holdout继续封存。

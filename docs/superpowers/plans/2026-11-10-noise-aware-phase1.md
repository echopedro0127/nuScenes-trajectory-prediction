# Noise-Aware Phase 1 Implementation Plan

**Goal:** 固定退化观测与四物理策略，运行fit/select开发集pilot。
**Architecture:** 新robustness子包分离packet生成、只读物理估计器与含标签评价runner。旧源码不改；开始前记录输入、样本、协议和源码身份，拒绝覆盖。
**Tech Stack:** NumPy、Python、pytest、CSV/JSON。

按[设计](../specs/2026-11-10-noise-aware-phase1-design.md)执行，用户已授权开始实验。使用现有完整环境，PYTHONPATH指向新worktree/src。

- [ ] 先写tests/robustness/test_observation_stream.py和test_filters.py，确认RED；覆盖历史降质、前缀因果性、延迟不重复同化、CV精确轨迹、KF协方差PSD和离群稳健更新。
- [ ] 实现robustness/packets.py、filters.py和benchmark.py，新增scripts/run_noise_aware_baselines.py；采样/缓存角色检查与指标有测试。
- [ ] 全量pytest、Ruff、公开扫描通过，冻结实现与协议。
- [ ] 运行固定三噪声seed、九条件、两角色开发pilot，保存样本/哈希、逐轨迹指标与计时范围；独立重算摘要并报告负面结果。

源cache_root为../causal-context-online/outputs/context/p66-independent/cache；仅fit/select。scene元数据为../../data/nuscenes/v1.0-trainval/scene.json。
输出outputs/noise-aware/phase1，拒绝已有目录。结果只能指导下阶段设计，不成为新holdout成绩。

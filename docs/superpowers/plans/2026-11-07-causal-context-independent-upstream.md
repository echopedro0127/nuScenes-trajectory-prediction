# P6.6C/1 Independent Split and Upstream GRU Implementation Plan

**Goal:** 固定log分组新划分，证明旧上游暴露，并重训第一层GRU。
**Architecture:** 独立缓存准备模块复用旧NPZ schema，生成带分组哈希的data.yaml，调用
冻结的原train.py从头拟合；只使用fit/select，不读取holdout预测指标。
**Tech Stack:** Python、NumPy、PyYAML、PyTorch、pytest。

见[设计](../specs/2026-11-07-causal-context-independent-upstream-design.md)。

- [ ] 先写tests/data/test_independent_split.py，覆盖分组/键/来源和输出拒绝；运行RED。
- [ ] 实现data/independent_split.py及scripts/prepare_independent_validation.py。
- [ ] 聚焦与全量测试、Ruff、公开扫描，冻结实现后运行真实划分准备。
- [ ] 检查真实scene/log/key无交叉、旧GRU暴露和缓存/config身份，不计算holdout误差。
- [ ] 调用原train.py，只用fit/select，从头训练seed2027的GRU，上限80epoch/早停12。
- [ ] 核对新checkpoint元数据和来源，形成阶段报告，标明仅GRU上游已重训，下游未完成。

开发worktree使用原完整环境，PYTHONPATH=src。所有pytest使用outputs下独占UUID临时目录。

```powershell
python scripts/prepare_independent_validation.py --source-cache ../../data/cache/trainval/train --dataroot ../../data/nuscenes --upstream-checkpoint ../../outputs/trainval/gru/best.pt --gate-manifest outputs/context/p6/cache/train/seed-2027/manifest.json
```

真实准备后，使用生成的data.yaml调用原train.py；model-config/output-dir传绝对路径，
确保原CLI的相对路径解析与新配置目录一致。输出根为outputs/context/p66-independent。

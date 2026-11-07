# Independent Map Pair Implementation Plan

**Goal:** 沿新 log 互斥 fit/select 链从头训练 residual original 与 map 两个阶段。
**Architecture:** 新 independent_map_training 模块单独验证新角色、scene/log/GRU 来源；复用既有 BatchedContext、_epoch 与模型，不改旧协议绑定源码。存储 independent_map_pair_v1 checkpoint，禁止旧 suite loader 自动接受。
**Tech Stack:** PyTorch、NumPy、pytest。

用户已授权分阶段继续；本轮设计和执行范围为前述输入准备的直接后续。不训练邻车/fusion/参数匹配完整消融，不运行 holdout。
备选：修改旧验证器会改变历史源码身份；重跑完整七结构套件超出当前上游所需。选择独立 original/map 两阶段，后续门控候选只依赖这两个模型。

冻结预算：seed2027，最多20epoch/阶段，patience5，batch4096，hidden64，AdamW lr0.001、weight_decay0.0001、smoothness0.05、grad_clip5，与既有训练一致；按 select 在线 ADE 最小选 checkpoint。map fallback 从本轮 original 最优权重复制并冻结，无旧权重初始化。每阶段重置随机种子和 PCG64 shuffle。
仅比较新 select 上的诊断指标，不解释成泛化收益；拟合/选模场景与 log 均不得重叠，不改 split seed 或预算。拒绝已有输出目录，不提供 resume。

- [ ] 先写 tests/online/test_independent_map_training.py，覆盖新角色、旧角色/错GRU/log交叉拒绝、真实短训练/重载、fallback一致性和输出覆盖拒绝，运行RED。
- [ ] 实现 online/independent_map_training.py 与 scripts/train_independent_map.py：验证输入、元数据、来源哈希；训练两个阶段，原子保存阶段和总checkpoint、运行记录、最终重载指标及完成marker。
- [ ] 聚焦和全量pytest、Ruff、公开扫描；冻结源码后运行真实训练。
- [ ] 独立重载 checkpoint，复算 select，记录best epoch/实际epoch/误差/耗时；报告完整独立链仍未完成，holdout未评价。

验证时使用完整环境 .worktrees/trajectory-mvp/.venv，PYTHONPATH 指向本 worktree/src，pytest临时目录位于outputs内。真实输出 outputs/context/p66-independent/map-training/seed-2027。

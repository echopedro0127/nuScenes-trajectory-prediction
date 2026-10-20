# P2 Map-only Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, one task at a time with review checkpoints. Steps use checkbox syntax.

**Goal:** 建立可复现、因果、无地图可回退的 Map-only Corrector mini 通路。
**Architecture:** P1 ObservationView → 独立 Map cache → 点 MLP 与 residual head → 新 evaluator；旧代码原样保留。
**Tech Stack:** Python 3.12、NumPy、PyTorch、nuScenes-devkit 1.2、pytest。

设计见 ../specs/2026-10-20-causal-context-phase2-design.md。工作分支 feature/causal-context-online。

## 命令准备

```powershell
$python = (Resolve-Path ../trajectory-mvp/.venv/Scripts/python.exe).Path
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$testRunPath = Join-Path (Get-Location) ('outputs/pytest-' + [guid]::NewGuid().ToString('N'))
```

## Task 1：地图几何和元数据索引

创建 context/maps.py、context/nuscenes_index.py、configs/context/map.yaml；测试 tests/context/test_maps.py。
- [x] 先写测试：前向 +Y 变换、方向不平移、距离/token 稳定排序、空地图、point mask、invalid config。
- [x] 运行 pytest tests/context/test_maps.py -q --basetemp $testRunPath，确认缺少模块。
- [x] 实现 MapConfig、MapView、LaneMap.query(observation, anchor_xy, anchor_yaw)。
  每条离散 poses [x,y,yaw]，用 global_to_agent 转位置，cos/sin 经 anchor rotation 转方向。
  局部查询按 cell 索引候选，再按最小距离/token 过滤、排序，保留最近点周围连续片段。
- [x] 实现 MetadataIndex，从四个 JSON 解析 anchor 并拒绝 instance、scene、anchor 不匹配。
- [x] 聚焦测试通过后提交 feat: add causal lane map context and metadata index。

## Task 2：独立 cache、prediction 绑定与 prepare CLI

创建 context/cache.py、context/preparation.py、scripts/prepare_context.py；测试 tests/context/test_context_cache.py。
- [x] 测试 roundtrip，拒绝 split/config/source hash、key 顺序、checksum、时间不符；原 cache 不变。
- [x] 运行聚焦 pytest 确认缺少模块。
- [x] cache manifest 使用 canonical JSON SHA256 contract 和 content-addressed NPZ，原子发布 manifest。
  每条 key 含 instance/sample/scene、11 个 step/time，保存地图与 mask/token。
- [x] prepare 从旧 cache 只将已揭示 future[:k] 放入 ObservationView；读取静态地图、记录 source hash 与完整场景。
- [x] prediction 读取强制 per_sample.csv 三类 token 顺序一致且 target 与原缓存一致，返回文件/sidecar hashes。
- [x] 新 CLI --trajectory-cache --split --dataroot --config --output-dir 均支持绝对路径。
- [x] 聚焦测试通过并提交 feat: add provenance-checked map context cache。

## Task 3：观测特征、训练数据与 Map Corrector

创建 online/context_features.py、online/context_dataset.py、online/context_corrector.py；测试 tests/online/test_context_corrector.py。
- [x] 测试 11 个 step 与旧 CorrectionDataset features 数值一致、未来扰动不变、mask padding 不影响输出。
- [x] 测试全无地图与原 Corrector 严格相等，真实 predictor 入口没有 label，梯度仅更新 context head。
- [x] 运行聚焦 pytest 确认失败，再实施 residual_features、ContextDataset、MapCorrector。
- [x] masked point MLP pooling 后拼接 8 维 residual，附加 offset 乘地图 availability 与 horizon mask。
- [x] 验证并提交 feat: add observation-only map residual corrector。

## Task 4：场景隔离训练与 checkpoint

创建 online/context_training.py、scripts/train_context_corrector.py、configs/models/context_corrector.yaml；测试 tests/online/test_context_training.py。
- [x] 测试 scene partition 完全不重叠、mini_val 拒绝训练、train/val 重叠拒绝，checkpoint hash 不符拒绝。
- [x] 测试短合成训练可运行并保存可加载模型，metadata 包含训练/选模场景 hash、两个 prediction/context hash、来源/config hash。
- [x] 先训练原 Corrector，在 selection 最佳 ADE 时 deepcopy state；冻结后训练 Map head，同样 selection 选模。
- [x] CLI 支持 --trajectory-cache、--context-cache、--base-predictions、--split、--dataroot、--config、--model-config、--epochs、--device、--output-dir；mini 同 cache 内按场景分割，trainval 另用 --selection-trajectory-cache、--selection-context-cache、--selection-base-predictions、--selection-split。
- [x] 每个 epoch 使用 remaining mask smooth_l1 + 0.05 smoothness，固定 seed，有限 loss、梯度裁剪、patience。
- [x] 提交 feat: train map corrector with scene-isolated selection。

## Task 5：生产 evaluator 与分组报告

创建 online/context_evaluator.py、scripts/evaluate_context_online.py；测试 tests/online/test_context_evaluator.py。
- [x] 测试四策略 key 集合相同、未来扰动不能改变同 step 模型输入、无地图实际 fallback、真实预测入口不接收 label。
- [x] 实现 eval checkpoint config/source/provenance 校验、拒绝训练/selection 场景评估，模型仅接受观察/固定 base/map。
- [x] 输出 per-update CSV、summary JSON、update/horizon/maneuver 分组，包含覆盖/fallback 与分段缓存路径耗时。
- [x] CLI --context-cache --trajectory-cache --split --base-predictions --checkpoint --dataroot --config --device --output-dir。
- [x] 聚焦回归后提交 feat: evaluate map context on matched causal update rows。

## Task 6：mini smoke 与阶段验收

- [x] 在 outputs/context/p2/mini-train 与 mini-val 分别运行 prepare，来源 ../../data/cache/mini 和 ../../data/nuscenes。
- [x] 训练固定 seed 2027 的 2 epoch smoke，使用 mini_train 场景内部划分，固定 ../../outputs/mini/gru_train_eval/predictions.npz。
- [x] 冻结后在 mini_val，使用 ../../outputs/mini/gru_eval/predictions.npz 评估并记录 coverage/fallback/失败与场景清单。
- [x] 独立代码审查，修复问题并运行全量 pytest、ruff check src scripts tests、scripts/check_public_repo.py、git diff --check。
- [x] 更新 DEVELOPMENT_ROADMAP.md 的状态、实际测试数、命令、commit、mini 结果与限制；输出 artifacts 不提交。
- [x] 提交 docs: record map context phase two verification；确认 git status --short 干净。

## 完成记录

2026-10-20：28 项新增测试，全量 339 passed；mini_train prepare/train 与 mini_val 冻结评估通过。
训练 CLI 的实际参数为 --trajectory-cache / --context-cache / --base-predictions / --split；
trainval selection 使用 --selection-trajectory-cache / --selection-context-cache /
--selection-base-predictions / --selection-split train_val。完整实际命令和负向 smoke 结果见路线图。
独立复审通过；固定 GRU 三份 artifact 来源绑定、checkpoint 完整合同及合法 split 组合已补测。

# P3 Neighbors Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, with a review checkpoint after each task.

**Goal:** 完成因果邻车、Map+Neighbors融合与匹配样本消融mini通路。
**Architecture:** 独立neighbor索引/cache，ObservationView-only输入，3个context head和3个宽网络对照，新训练/评估CLI；P2不变。
**Tech Stack:** Python 3.12、NumPy、PyTorch、pytest、现有nuScenes metadata。

设计：../specs/2026-10-20-causal-context-phase3-design.md。
环境沿用 ../trajectory-mvp/.venv/Scripts/python.exe，PYTHONPATH=当前worktree/src。
每个pytest使用 outputs/pytest-<独占GUID> 作为basetemp。

## Task 1：NeighborIndex、配置和只读View

文件 context/neighbors.py、configs/context/neighbors.yaml、tests/context/test_neighbors.py。
- [ ] 编写排序/自身排除/类别/历史mask/真实dt速度/方向旋转/未来不变/延迟过旧测试。
- [ ] 运行聚焦pytest确认模块缺失。
- [ ] 实现NeighborConfig、NeighborView、NeighborIndex.query(observation,target,scene,anchor)。
  可用sample用bisect查找；当前候选只来自当前可用sample，不沿用消失对象。历史槽按
  tk-2s...tk，只选source<=slot且available<=tk，无重复sample，mask明确缺失。
- [ ] 验证通过，提交 feat: add causal oracle neighbor history context。

## Task 2：Neighbor缓存与准备CLI

文件 context/neighbor_cache.py、context/neighbor_preparation.py、scripts/prepare_neighbors.py；
测试 tests/context/test_neighbor_cache.py。
- [ ] 测试roundtrip，拒绝checksum、配置/split/map来源/key顺序、未来source/availability和非法mask。
- [ ] 确认失败后实现content-addressed独立NPZ，manifest绑定P2map与完整metadata/category来源。
- [ ] CLI --trajectory-cache --map-cache --split --dataroot --config --map-config --output-dir，均支持绝对路径。
- [ ] 在mini_train/mini_val生成独立cache，记录coverage/历史velocity masks，验证后提交。

## Task 3：共享encoder、融合模型和训练dataset

文件 online/interaction_corrector.py、online/interaction_dataset.py；tests/online/test_interaction_corrector.py。
- [ ] 测试三模式、非零head下padding不变、单模态/全缺失fallback、past mask、非零encoder梯度、预测签名无label。
- [ ] 确认失败后实现各模态shared point MLP、masked pooling、presence/age编码和head；fallback冻结。
- [ ] dataset复用P2residual/label分离，只显式选择模型输入，验证后提交。

## Task 4：场景隔离消融训练与checkpoint

文件 online/interaction_training.py、scripts/train_interaction_corrector.py、configs/models/interaction_corrector.yaml；
tests/online/test_interaction_training.py。
- [ ] 测试短合成训练、checkpoint恢复、参数差异<=2%、scene overlap/错GRU/错neighbor contract拒绝。
- [ ] 实现original、3个context与3个宽对照的独立选模，每stage同seed预算；记录total/trainable数量。
- [ ] CLI --trajectory-cache --map-cache --neighbor-cache --base-predictions --split --dataroot --output-dir，
  trainval额外传--selection-trajectory-cache/--selection-map-cache/--selection-neighbor-cache/--selection-base-predictions。
  配置--config --map-config --model-config --epochs --device；保存best.pt和training.json。
- [ ] 运行mini_train各stage2epoch smoke，验证后提交。

## Task 5：实际融合评估

文件 online/interaction_evaluator.py、scripts/evaluate_interaction_online.py；tests/online/test_interaction_evaluator.py。
- [ ] 测试9策略同key集合、实际无context回退、未来不变以及真实predictor边界不传label。
- [ ] 实现per-update、summary、update/horizon/maneuver与coverage/fallback/age/分段缓存latency。
- [ ] CLI额外--checkpoint和--split mini_val，对冻结checkpoint执行来源和场景隔离检查。
- [ ] mini_val运行，独立代码审查，修复后重跑受影响检查并提交。

## Task 6：验收记录

- [ ] 全量pytest、ruff check src scripts tests、scripts/check_public_repo.py、git diff --check。
- [ ] 路线图记录新增测试、全量数、场景与参数、smoke效果、命令、commit、来源hash和限制。
- [ ] P3标记工程完成，P4多seed/trainval效果待开始；提交文档并确认worktree干净。

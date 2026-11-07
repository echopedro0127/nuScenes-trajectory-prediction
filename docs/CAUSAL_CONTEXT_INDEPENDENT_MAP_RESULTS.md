# P6.6C/3 独立划分上的残差对照与地图修正器

日期：2026-11-07。接续[新预测与上下文准备](CAUSAL_CONTEXT_INDEPENDENT_CONTEXT_RESULTS.md)。
本阶段沿新 GRU → 新 fit/select 预测 → 新因果上下文，从头训练 original 残差对照和 map 修正器。
只在 independent_select 选模与复算，不生成 holdout 预测。完整质量门控、RH 和一致性链尚未完成。

## 事前范围与实现

预算见[冻结计划](superpowers/plans/2026-11-07-independent-map-training.md)。实现提交 ffde7d9，独立入口 scripts/train_independent_map.py。
seed2027、每阶段最多20epoch、patience5、batch4096、hidden64、AdamW lr0.001/weight_decay0.0001、smoothness权重0.05、梯度裁剪5。
按 select 在线 ADE 选择最优模型；每阶段重新设置随机种子与 PCG64 shuffle。
original 随机初始化，map 从本轮 original 最优状态复制冻结 fallback；没有加载旧协议的修正器权重。
沿用固定原始终点的 11 次更新，仅评价各次更新之后的剩余预测点；各更新等权。

旧 source/scene 校验器只接受 train/train_val。因此新模块先严格检查真实 independent_fit/independent_select、partition identity、log 互斥与新 GRU 训练元数据，再通过不持久化的校验副本复用旧来源检查。
保存到模型的角色始终是真实新角色。没有修改旧校验器、旧模型结构或损失函数。新 checkpoint 使用 independent_map_pair_v1 格式，不能冒充旧完整七结构 suite。
本轮不训练 neighbors/fusion/参数匹配消融；后续质量门控候选仅需要 original/map。

训练使用 fit14,445轨迹/291场景/22log 与 select7,786轨迹/104场景/4log，分别含158,895和85,646次更新。
上下文依然来自标注位置回放与静态地图，未使用 Camera / LiDAR / Radar 感知特征，也不能视为真实跟踪器输出。

## 训练结果

两个阶段都完成20epoch，best epoch均为20，未触发早停。预算未在观察结果后增加。

| 阶段 | select 在线 ADE / m（训练聚合口径） | 重载复算 |
| --- | ---: | --- |
| original | 2.603349208831787 | 与最优epoch完全一致 |
| map | 2.421936273574829 | 与最优epoch完全一致 |

这是同一新 select 上的地图增量诊断，不能和静态GRU完整6秒ADE直接作提升比例，也不能解释为 holdout 泛化结果。
独立复核已验证完成marker的每个文件哈希、run identity重算、所绑定源码哈希、新GRU身份、fallback逐参数一致性，并从磁盘重载后复算85,646次select更新。
训练聚合的float32 ADE与日志完全一致；独立使用double距离和逐更新均值计算如下，微小数值差来自计算与聚合精度。

| 阶段 | select 在线 ADE / m | select 在线 FDE / m |
| --- | ---: | ---: |
| original | 2.603349187 | 4.881749575 |
| map | 2.421936417 | 4.556826117 |

map相对original的select ADE降低 **6.9684%**。这支持地图输入在当前选择集上的增量作用；容量/训练阶段数差异、单seed与select选模偏差尚未排除，不能把该数字当作独立泛化提升。
两阶段训练加每epoch select共 **87.826秒**，不含输入哈希/缓存重载、fingerprint、checkpoint IO和最终独立核验。

## 验证与产物

新增8项测试，先确认RED，完成后全量 **549 passed in 112.15s**；Ruff和公开扫描通过。
测试覆盖新角色原样保存、错角色/log交叉/错GRU配置拒绝、真实短训练与重载、冻结fallback一致性、错partition身份和覆盖拒绝。
正式训练后已通过程序内部的best.pt重载、fallback逐参数一致性与select ADE复算。

输出：outputs/context/p66-independent/map-training/seed-2027，含 run.json、progress.json、stages/original.pt、stages/map.pt、best.pt、report.json、completed.json。
完成marker绑定运行记录、总checkpoint、阶段checkpoint及训练报告；输出和模型不入Git。

| 身份/文件 | SHA256 |
| --- | --- |
| run identity | e5e3e6fa622ea092a9c99c50f2c686f3d9fae9e82ec916d32fcd65be7c39dcaf |
| best.pt | e562d633dcf3518ee316133d91c810a4f77b2131d169d19f2a908ec3db017f46 |
| report.json | 8363d02eb76936c44ab362d2e71caa4dc558bad89c290ab3c3e92ff0752ff473 |
| audit.json | c7c9312efd7a966c5cc301498b82f2b49cdd439949caec4aa92c2645b1ebe03f |

```powershell
python scripts/train_independent_map.py --output-root outputs/context/p66-independent/map-training/seed-2027 --device cuda
```

复现必须使用新的输出路径；入口拒绝覆盖现有目录。本阶段未推送GitHub，未修改简历。

## 下一步及限制

新模型格式的独立候选缓存入口与六条件缓存现已完成，见[P6.6C/4报告](CAUSAL_CONTEXT_INDEPENDENT_CANDIDATES_RESULTS.md)。下一步沿新fit/select重训quality gate、RH和一致性head。
后续超参数和模型选择冻结后才考虑一次性holdout评价。目前仅单seed、select只有4个log，无置信区间或跨seed泛化结论；也未完成容量匹配对照。
旧RH/consistent_010历史推荐不因本阶段select结果而自动替换，P7未启动。

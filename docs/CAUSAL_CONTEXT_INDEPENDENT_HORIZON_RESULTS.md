# P6.6C/6 独立RH重训

日期：2026-11-09。接续[独立质量门控重训](CAUSAL_CONTEXT_INDEPENDENT_GATE_RESULTS.md)。
本轮从新gate初始化RH，在同一新fit/select链上重训。仅单臂RH，不重做八结构消融；一致性head尚未重训，holdout未评价。

## 固定预算与因果输入

[实施计划](superpowers/plans/2026-11-09-independent-horizon.md)，实现6f92374。
新入口scripts/train_independent_horizon.py复用旧HorizonDataset/HorizonGate/train_horizon/measure_horizon，历史绑定源码不改。
337参数、hidden16、19输入、seed2027、最多10epoch、patience3、batch4096、AdamW lr0.001/weight_decay0.0001、smoothness0.05、revision_weight0。
按select平均在线ADE选模；训练前从本轮独立gate（69f49457…87bd8）初始化，不使用旧协议权重。

RH启用R（此前候选在当前已释放位置上的已实现误差反馈）和H（剩余预测时长与候选相对纵/横向差异）；D方向组被feature_mask禁用。
R的特征仅使用当前及此前已到达的位置与先前预测；H依赖当前候选预测而非未来真值。
基础quality8仍保留。新特征对应权重初始化为0，初始RH在整个select上的剩余权重和ADE/FDE/jitter与新gate **逐项完全一致**，初始化最大差为0。
位置仍是每0.5秒释放的标注位置，未使用Camera/LiDAR/Radar感知特征，也未模拟真实跟踪误差。

## 训练及选择集结果

fit14,445轨迹/22log；select7,786轨迹/4log。每角色均回放六条件与11次更新，原始预测终点保持不变。
实际训练10epoch，best10，未触发早停，未增加事后训练预算。内部最优checkpoint重载通过。

| 六条件等权select均值 | 新gate | 新RH |
| --- | ---: | ---: |
| ADE / m | 1.992875340 | 1.977675371 |
| FDE / m | 4.272785093 | 4.259936350 |
| jitter | 1.270521879 | 1.264344878 |

ADE为每次更新之后的剩余轨迹点误差，各更新等权；FDE为原始固定终点误差。
jitter为相邻两次预测在共同剩余时间点的位置修订距离均值，首次更新记0，再沿更新/轨迹/条件平均；不代表车辆加速度、舒适度或真实规划安全。
相对新gate，select ADE降低0.7627%、FDE降低0.3007%、jitter降低0.4862%。
本轮结果是新选择集上的RH延续训练诊断，不能与旧协议RH跨划分比较，也不能把提升单独归因于R/H：模型容量和额外训练轮数尚未匹配控制。

| select条件 | gate ADE | RH ADE | gate FDE | RH FDE | gate jitter | RH jitter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean | 1.941030 | 1.925223 | 4.132484 | 4.121224 | 1.231282 | 1.226969 |
| delay_500ms | 1.994546 | 1.975833 | 4.273266 | 4.255468 | 1.355536 | 1.339548 |
| delay_1000ms | 2.035092 | 2.014872 | 4.374097 | 4.356842 | 1.338052 | 1.328946 |
| dropout | 1.939799 | 1.919529 | 4.123289 | 4.105327 | 1.245455 | 1.242330 |
| missing | 2.097688 | 2.097688 | 4.576311 | 4.576311 | 1.205111 | 1.205111 |
| stale | 1.949096 | 1.932907 | 4.157264 | 4.144447 | 1.247695 | 1.243166 |

完全缺失地图时RH权重严格为0并精确回退CV，其指标与gate完全相同。相关条件是同一批轨迹的重复回放，不是新增独立样本。

## 验收与产物

新增2项端到端测试，先确认RED后实现；覆盖实际新gate初始化、短训练/重载、337参数、初始化等价、缺包回退、错身份/variant和输出覆盖拒绝。
全量 **565 passed in 66.09s**，Ruff与公开扫描通过。训练前核验上游候选、gate完成marker/源码及角色隔离，训练后内部重载复算通过。
独立验收已完成：重验两组候选及上游、完成marker与源码；用实际HorizonDataset fingerprint和初始模型状态重算run identity；重载RH后完整select指标与保存的selection_metrics.npz逐项完全一致，新gate基线也逐项一致。
missing权重全0且全部ADE/FDE/jitter与gate基线精确相同。真实holdout的predictions/context/candidates目录均不存在。
训练加每epoch select共138.284秒，不含上游核验/特征与fingerprint准备、初始化等价检查、checkpoint IO及最终独立验收。

输出outputs/context/p66-independent/horizon-training/seed-2027/RH，包括run.json、progress.json、training.json、best.pt、report.json、selection_metrics.npz、completed.json。
完成marker绑定运行/训练记录、模型、报告及逐更新指标。best.pt使用independent_horizon_gate_v1格式；training.json为复用核心的原始训练记录。

| 身份/文件 | SHA256 |
| --- | --- |
| run identity | baf6ea1956868a53db93c9a0a944b1804b187992573693f33007769a9349a717 |
| best.pt | ee957f2f32a5aeffcbb133689f64d7c07d7c9422f352fdd6ce8206cb1f5d3c05 |
| report.json | 9dee2420b1ca1f890ea2e3e4c4f114f54bdddacd69b16424b2d698e51a5fa3d3 |
| selection_metrics.npz | c4b7d83e223cb73fb66e753da9775d4c274bb3ea4868f8de538247c4bf913e1f |
| audit.json | a4376bf9cada7b58ecc2fd850ee8b66cd8a5aff0956f76c77e57d1202d802812 |

```powershell
python scripts/train_independent_horizon.py --output-root outputs/context/p66-independent/horizon-training/seed-2027/RH --device cuda
```

入口拒绝已有输出目录，复现需换新路径。本轮不推送GitHub，不上传模型/数据，不改简历。

## 下一步

沿新RH与候选重训一致性head，在fit/select冻结完整配置后才考虑一次性holdout评价。
当前只有单seed、4个select log，不能给出独立测试或跨seed置信结论；旧RH/consistent_010历史推荐暂不替换，P7未启动。

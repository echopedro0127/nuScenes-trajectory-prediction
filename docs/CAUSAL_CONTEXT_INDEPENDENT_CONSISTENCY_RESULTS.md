# P6.6C/7 独立创新量加权一致性重训

日期：2026-11-09。接续[独立RH重训](CAUSAL_CONTEXT_INDEPENDENT_HORIZON_RESULTS.md)。
本轮沿新fit/select完成无正则对照和自适应一致性双臂重训。没有运行holdout，也没有新增网络结构。
一致性通过训练损失约束同一RH模型的修订，不是额外预测head；算法复用已有P6.4实现。

## 事前预算与同起点对照

[实施计划](superpowers/plans/2026-11-09-independent-consistency.md)，实现a58bb6d。
337参数RH结构，从本轮新标量gate同起点训练，不从已训练RH继续微调。
两臂均seed2027、最多10epoch、patience3、batch4096、AdamW lr0.001/weight_decay0.0001、smoothness0.05，按select平均ADE选最优。

- revision_0：lambda=0，用于核验与本轮RH完整训练路径一致。
- revision_adaptive：lambda=0.02，修订损失乘以 exp(-(innovation/0.5)^2)。

innovation是当前已释放位置与此前CV对该时刻预测之间的距离；高创新量时减弱一致性约束，避免强迫模型沿用已经不准确的预测。
修订损失比较相邻更新在同一绝对剩余时刻的预测位置。没有未来标签进入推理特征；未来标签仅用于单独监督损失。
fit的自适应权重均值为0.7122181328347162，统计不使用select。模型结构、初始状态、数据顺序和预算相同，不事后追加轮数。
没有固定或均值匹配正则对照，不能据此宣称创新量权重优于其他一致性权重，也不冒用旧consistent_010名字。

## 训练与select结果

fit14,445轨迹/22log，select7,786轨迹/4log，沿同一六条件回放。两臂均完成10epoch，best10。
无正则臂全部模型权重精确复现本轮RH，逐更新ADE/FDE/jitter也完全相同；该对照排除了额外轮数与不同初始化的影响。

| 六条件等权select均值 | RH / revision_0 | revision_adaptive |
| --- | ---: | ---: |
| ADE / m | 1.977675371 | 1.977514377 |
| FDE / m | 4.259936350 | 4.260644418 |
| jitter | 1.264344878 | 1.260875416 |

自适应臂相对zero/RH的ADE降低0.00814%、jitter降低0.27441%、FDE增加0.01662%。这是小幅精度/稳定性权衡，不是全面提升或独立泛化结论，不足以据此替换历史推荐。
更详细的六条件分项保存在各臂report.json与逐更新selection_metrics.npz。missing条件的权重全0，全部指标与RH的CV回退精确一致。
相关条件与更新来自相同轨迹，不能将其数量视为独立样本量；未计算跨log或跨seed置信区间。

## 验收与来源

新增1项完整端到端测试，确认RED后实现：实际新gate/RH、短双臂训练、zero逐参数复现、模型保存/重载、错误身份/arm与覆盖拒绝。
全量 **566 passed in 79.54s**，Ruff、公开扫描通过。旧模型、损失及历史绑定源码未修改。
独立审计已重验两组候选、全部上游、源码与完成marker，用实际InnovationDataset fingerprint及新gate初始状态重算两个run identity；两臂完整逐更新select指标与保存数组完全一致。
zero的全部权重及指标与原RH完全一致，missing条件两臂权重均全0、指标精确等于RH的CV回退。
实际训练加每epoch select：zero136.936秒、adaptive165.798秒，合计302.734秒；不含输入/来源核验、特征/fingerprint、checkpoint IO和最终独立审计。
真实holdout的predictions/context/candidates目录均不存在，未进行holdout预测或误差评价。

产物outputs/context/p66-independent/consistency-training/seed-2027/{revision_0,revision_adaptive}，含run.json、progress.json、training.json、best.pt、report.json、selection_metrics.npz、completed.json。
独立模型格式independent_consistency_v1，绑定新RH、新gate、新候选及源码；training.json保留原核心记录。
报告字段zero_reproduces_rh表示当前臂状态是否与RH完全相同；adaptive中false表示其权重已改变，不表示zero对照失败。

| 身份/文件 | SHA256 |
| --- | --- |
| revision_0 run identity | fd40386de90149512226e997500830978bc46b5029842a4113cce82a033759f4 |
| revision_adaptive run identity | 94def6ba2d9502a0e284eedd3e4b3531e932de3e2681525cd6662f04476f0773 |
| revision_0 best.pt | 1904ae97364b7c23d1de90805cd9bb7c9b6c95552cbb730f3cb703c598f0722e |
| revision_adaptive best.pt | cc8a2ed082ef79a0a4360e5bbd16685cb8179eb9004edcdd82d8f7109d33582b |
| seed-2027/audit.json | 63d97cba2e0517efa33860cf451e60b343400f3026ed5949db9a2bfbf8e8c8dd |

```powershell
python scripts/train_independent_consistency.py --output-root outputs/context/p66-independent/consistency-training/seed-2027 --device cuda
```

拒绝已有输出目录；复现应使用新路径。不推送GitHub，不上传模型/数据，不改简历。

## 后续边界

独立GRU、original/map、gate、RH及本轮一致性双臂均已重训；完整独立评价尚未完成。
下一步应冻结模型选择、指标与一次性holdout评价协议，再准备holdout所需的新预测/上下文/候选；不能看holdout误差后重新调参选模。
本轮不自动替换历史RH/consistent_010推荐。仅单seed、4个select log且为已知数据重划分，研究者已知历史结果；仍需保留该局限。
位置来自每0.5秒释放的标注，未使用Camera/LiDAR/Radar特征或真实跟踪噪声。P7未启动。

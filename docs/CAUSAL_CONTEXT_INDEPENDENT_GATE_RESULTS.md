# P6.6C/5 独立质量门控重训

日期：2026-11-09。接续[独立六条件候选](CAUSAL_CONTEXT_INDEPENDENT_CANDIDATES_RESULTS.md)。
本轮在新fit/select链上从头训练quality gate。候选、GRU及original/map均冻结，未生成holdout预测或误差；RH和一致性head仍未重训。

## 事前预算与实现

[冻结计划](superpowers/plans/2026-11-09-independent-quality-gate.md)，实现dbbb285。
独立入口scripts/train_independent_gate.py复用旧train_gate/run_epoch；旧源码不改。
161参数QualityGate、hidden16、seed2027、最多20epoch、patience5、batch4096、AdamW lr0.001/weight_decay0.0001、smoothness0.05、grad clip5。
六条件等权训练，select平均在线ADE最小选模；没有加载旧gate权重，也没有观察结果后追加训练预算。

模型使用quality8与包valid输出单个混合权重，将因果CV与冻结map候选混合。地图包无效时权重严格为0并回退CV。
“map候选在无效包时回退original”和“gate输出在无效包时回退CV”是不同策略，因此二者的提升比例同时包含回退策略差异。
本轮未新增门控算法，工作是完成隔离上游后的重训与来源核验，不能包装为新模型创新。

新入口验证真实independent_fit/independent_select角色、共同上游、candidate manifest与scene/log隔离，拒绝已有目录。
最终模型格式independent_quality_gate_v1；training.json保留复用核心的原始训练记录，best.pt由独立入口封装并绑定新合同和源码。

## 训练与select诊断

fit14,445轨迹/22log，953,370条件更新；select7,786轨迹/4log，513,876条件更新。
每种条件均为85,646次select更新，ADE为各次更新后剩余轨迹点距离的均值，FDE为原始固定终点误差。
这些更新来自同批轨迹的相关回放，不能当作独立统计样本数量。

训练在第16epoch早停，best epoch11；训练口径平均select ADE **1.992875339269073m**。
内部磁盘重载复算与选模日志完全一致。独立审计已重验两组候选、上游来源及源码哈希，用实际fit/select数组fingerprint重算run identity，并重载gate复算所有select指标；逐条件结果与report.json完全一致。
仅训练及每epoch select共113.759秒，不含输入/候选核验、fingerprint、checkpoint IO与最终独立审计。

| 六条件等权均值 | ADE / m | FDE / m |
| --- | ---: | ---: |
| CV | 2.097687796 | 4.576310661 |
| original | 2.603349188 | 4.881749574 |
| map | 2.510740450 | 4.618880897 |
| gate | 1.992875340 | 4.272785093 |

gate相对CV的select ADE降低4.9966%，相对map降低20.6260%；后者同时包含回退策略差异，不能全部归因于学习到的门控。
诊断使用double距离和均值，训练验证用float32距离、double累加；两者平均ADE约4.1e-10m的差别属于计算精度差异。

| select条件 | gate ADE / m | gate FDE / m | 平均门控权重 |
| --- | ---: | ---: | ---: |
| clean | 1.941030 | 4.132484 | 0.345151 |
| delay_500ms | 1.994546 | 4.273266 | 0.268097 |
| delay_1000ms | 2.035092 | 4.374097 | 0.200651 |
| dropout | 1.939799 | 4.123289 | 0.343605 |
| missing | 2.097688 | 4.576311 | 0 |
| stale | 1.949096 | 4.157264 | 0.270908 |

CV在六条件下的ADE均为2.097687796m、FDE均为4.576310661m；它只依赖释放的位置，不依赖地图包条件。
本轮select上各条件gate均未差于CV，missing严格相等。权重是模型的混合系数，不是校准概率或地图正确率。
不能仅从均值权重推断所有延迟样本都降低权重；stale均值也包含无效包的零权重。

## 验证与身份

新增6项测试，确认RED后实现，全量 **563 passed in 69.98s**，Ruff/公开扫描通过。
覆盖错角色、上游不一致、scene/log交叉、候选身份、真实短训练、磁盘重载、输出覆盖拒绝及源码身份。
缺失地图回退测试确认权重严格0且gate/CV的ADE/FDE完全相同。

产物outputs/context/p66-independent/gate-training/seed-2027：run.json、progress.json、training.json、best.pt、report.json、completed.json，完成marker绑定运行记录/训练日志/模型/报告。

| 身份/文件 | SHA256 |
| --- | --- |
| run identity | 2b034a260a67cf86082572fff813498d19284521cbc5b306b8623d24812c70d3 |
| best.pt | 69f49457c48b29f1e7e646e95eb69d7f9b3eae8052a9fb9398357b02b1887bd8 |
| report.json | f8e8a804bbd595a6b4226ac50040be9738d0384bfd4ff9cacb5b3f8f3a4bf041 |
| audit.json | 4940ff7ffbe2b5b4628bf03876ee77ef97fe66f9fe856954ab0c4626025fd9c3 |

审计另外确认真实holdout的predictions/context/candidates目录均不存在，未执行holdout预测或误差评价。

```powershell
python scripts/train_independent_gate.py --output-root outputs/context/p66-independent/gate-training/seed-2027 --device cuda
```

复现使用新输出目录，不覆盖既有结果；不推送GitHub，不上传模型/数据，不改简历。

## 后续边界

新RH现已重训并通过独立验收，见[P6.6C/6报告](CAUSAL_CONTEXT_INDEPENDENT_HORIZON_RESULTS.md)。下一步推进一致性head；完整配置和模型选择冻结后才能考虑一次性holdout评价。
本轮只做单seed、4个select log的选择集诊断，无容量匹配、跨seed置信区间或独立测试结论。
位置观测仍是每0.5秒释放的标注位置，未加入真实感知误差，也未使用Camera/LiDAR/Radar特征。
旧RH/consistent_010历史推荐不因本轮选择集指标自动替换；P7未启动。

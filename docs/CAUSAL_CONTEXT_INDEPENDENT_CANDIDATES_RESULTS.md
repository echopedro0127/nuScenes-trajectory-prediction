# P6.6C/4 独立候选缓存

日期：2026-11-07，接续[独立地图修正器重训](CAUSAL_CONTEXT_INDEPENDENT_MAP_RESULTS.md)。
本阶段只准备门控训练输入，未训练quality gate、RH或一致性head，未计算holdout预测或误差。
实现ba04d86，计划见[候选准备协议](superpowers/plans/2026-11-07-independent-candidates.md)。

## 输入与固定配置

使用新GRU checkpoint（0804c98a…94757）、新original/map pair（e562d633…17f46）和新划分上下文。
新map训练identity为e5e3e6fa622ea092a9c99c50f2c686f3d9fae9e82ec916d32fcd65be7c39dcaf。
独立入口scripts/prepare_independent_candidates.py保留真实independent_fit/independent_select角色；入口与低层发布函数均拒绝holdout。
没有修改旧P6入口、旧gate schema或历史绑定源码，没有复用旧学习模型的候选预测。

沿用GateConfig：max_age_s2、fixed_scale0.5、decay_s1。六种条件固定为clean、delay_500ms、delay_1000ms、dropout、missing、stale。
每条轨迹11次更新，0.5秒释放一次新增标注位置；地图包按固定时间/缺失规则提供。原始终点保持不变。
保存CV、滤波、original/map候选、quality8、包来源/可用时间和单独target。未来标签只用于逐步位置释放与独立loss/eval，不直接进入预测模型。
位置来自注释，地图为静态HD map；没有使用Camera/LiDAR/Radar特征，也没有模拟真实感知跟踪误差。

## 生成结果

| 角色 | 轨迹 | 更新 | 六条件更新 | 生成+重算+发布 / 秒 |
| --- | ---: | ---: | ---: | ---: |
| independent_fit | 14,445 | 158,895 | 953,370 | 34.241 |
| independent_select | 7,786 | 85,646 | 513,876 | 17.740 |

合计1,467,246个条件更新；条件更新是同一批轨迹的六种回放条件，不是新增独立数据样本。
耗时不含上游来源哈希、模型与地图输入重载和最终独立验收。

| 条件 | fit有效地图更新 | select有效地图更新 |
| --- | ---: | ---: |
| clean | 158,811 | 85,646 |
| delay_500ms | 144,376 | 77,860 |
| delay_1000ms | 129,941 | 70,074 |
| dropout | 158,815 | 85,646 |
| missing | 0 | 0 |
| stale | 72,205 | 38,930 |

有效表示当次可用且未过期的地图包存在覆盖，不能解释成预测成功。dropout可能沿用先前有效包，因此有效数不必逐项低于clean。
地图包无效时map候选回退到本轮original；尚无学习门控结果或新鲁棒性收益结论。

## 核验与产物

新增8项测试，确认RED后实现，覆盖来源/角色/scene拒绝、holdout写入前拒绝、真实候选保存重载、覆盖拒绝及数组篡改检测。
全量 **557 passed in 71.65s**，Ruff、公开扫描通过。生成过程已对全部original/map候选重新前向计算，发布前通过原gate schema的时间、速度、包调度、质量特征和回退检查。
两个角色均已通过独立磁盘重载、父模型/输入/源码核验和全部original/map候选重算。
验收记录保存在candidates/verification.json；真实holdout的candidates/context/predictions目录均不存在。

输出位于outputs/context/p66-independent/candidates/{independent_fit,independent_select}，包括只读加载的NPY数组、manifest.json、completed.json。
合同绑定划分identity、map训练identity、GRU身份、轨迹/地图/预测文件引用及候选生成源码哈希。加载时重验父来源和模型，检查缓存键、history、target、base，并重算候选。

| manifest | SHA256 |
| --- | --- |
| fit | 977b889551479680e569ae5008cc3ae9c17912a2a3e31098043cc2f4b7278055 |
| select | 9aec30512b943eb34f0caaae6fda6b90b54ed8d218790998b5e5f361315bac47 |

验收合同SHA256：fit为f7ec10aa67b27e84c4ad810009d8265f9ab859e62b42324dc6f9a1d98fed30a8；select为f9c4ce17a849ae96f1bc123f011e22494e6611595ca878d160678ee248e3020e。

```powershell
python scripts/prepare_independent_candidates.py --output-root outputs/context/p66-independent/candidates --device cuda
python scripts/prepare_independent_candidates.py --output-root outputs/context/p66-independent/candidates --verify-only --device cuda
```

生成入口拒绝覆盖已有根目录；复现请换新路径。本轮模型/数据/outputs均不入Git，不推送GitHub，不改简历。

## 后续边界

下一步冻结独立quality gate训练预算，适配新角色与候选合同；旧门控训练入口依赖旧P6验证器，不能直接套用。
随后仍需RH和一致性head沿新fit/select训练，再冻结完整系统配置，才可考虑一次性holdout评价。
当前仅seed2027链、select4log、holdout6log，仍不具备独立测试收益或跨seed统计结论；旧RH/consistent_010历史推荐不变。

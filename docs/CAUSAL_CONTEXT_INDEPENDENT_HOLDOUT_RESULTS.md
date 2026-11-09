# P6.6C/9 固定协议holdout评价结果

日期：2026-11-09。按[已冻结协议](CAUSAL_CONTEXT_INDEPENDENT_HOLDOUT_PROTOCOL.md)完成一次新holdout准备与评价，没有训练、重新选模或更改统计规则。
主比较RH相对CV的在线ADE降低 **5.4067%**；log配对bootstrap差值区间低于0，满足事前固定的主结论规则。
同时RH的jitter比CV **增加5.3328%**，因此不宣称全面提升或比CV整体更稳定。

## 数据、角色和执行边界

holdout3,329轨迹、73scene、6log，与新fit/select的scene/log均无交叉。六条件共36,619次基础更新、219,714次条件更新，来自同一批相关轨迹，不是独立样本数量。
history2s/5步、future6s/12步、2Hz；每0.5秒释放标注位置，11次更新，固定原始预测终点。
输入为标注运动与静态地图/包可用信息，没有Camera/LiDAR/Radar特征，也没有真实跟踪器噪声。
GRU、original/map、gate、RH、adaptive均为冻结的新fit/select训练链；没有使用旧协议训练权重或holdout选checkpoint。

新增只读入口scripts/evaluate_independent_holdout.py，实现7ccffc4；原锁内源码没有修改。
第一次推理前写入evaluation_started.json，绑定协议、新执行源码和运行环境；然后生成新GRU预测、地图/邻车上下文和六条件候选。
GRU预测准备仅保存预测/来源，不提前报告静态误差。邻车上下文随既有准备器生成，但本轮固定模型不使用邻车输入。
所有固定策略、条件及log完整报告；revision_0与RH权重相同，不重复计作独立模型。

## 主结论

RH−CV的六条件平均在线ADE绝对差为 **−0.105097267m**。
按log配对cluster bootstrap：5000次、PCG64 seed20271109、按抽中log的误差总和/计数池化，95%线性分位区间为 **[−0.128438650, −0.074846440]m**。
区间上界<0，因此固定决策为 **RH_improvement_supported**：支持RH在该holdout上的ADE改善。
此区间针对固定单seed模型和这6个log的数据分组重采样，不是跨seed不确定性或普遍场景保证。

| 六条件等权holdout均值 | ADE / m | FDE / m | jitter |
| --- | ---: | ---: | ---: |
| CV（主对照） | 1.943822475 | 4.247186126 | 1.107426552 |
| original | 2.447080939 | 4.602740333 | 1.479687230 |
| map | 2.384714175 | 4.367787534 | 1.537443914 |
| quality_gate | 1.854600441 | 3.995681872 | 1.171523452 |
| RH（事前主模型） | 1.838725208 | 3.984548568 | 1.166483295 |
| revision_adaptive（次要） | 1.838218892 | 3.984516888 | 1.162769302 |

RH相对CV：ADE −5.4067%、FDE −6.1838%、jitter +5.3328%。只有ADE主比较预先指定推断规则；其他变化为描述结果。
original和直接map的ADE均差于CV，因此不能把原始学习候选本身描述为强于CV；本轮证据支持的是固定门控/RH混合策略。
adaptive与RH的差异很小且仅为次要比较，不因它在holdout上数值略好而更换主模型。
jitter为相邻更新在共同剩余预测时刻的修订距离均值，首次更新记0，不代表车辆加速度、舒适度或真实规划安全。

## 全条件主模型结果

CV在每种条件的ADE/FDE/jitter相同，为1.943822475/4.247186126/1.107426552，因为它不依赖地图包。

| 条件 | RH ADE / m | RH FDE / m | RH jitter |
| --- | ---: | ---: | ---: |
| clean | 1.790973 | 3.864873 | 1.131935 |
| delay_500ms | 1.839995 | 3.987817 | 1.244967 |
| delay_1000ms | 1.874884 | 4.072940 | 1.228361 |
| dropout | 1.786323 | 3.851431 | 1.145415 |
| missing | 1.943822 | 4.247186 | 1.107427 |
| stale | 1.796354 | 3.883045 | 1.140795 |

missing时gate/RH/adaptive权重全0、完整指标数组与CV精确相同。该回退是实现验收，不是性能收益。
各条件其他模型的全部ADE/FDE/jitter均保存在results.json，未按优劣过滤条件。

## 逐log主比较

| log token | 轨迹 | RH−CV ADE / m |
| --- | ---: | ---: |
| 246e7da6bb344941bac92be421a545e2 | 819 | −0.139226196 |
| 6577357788b24c35a3b0419c138f50db | 182 | −0.063374915 |
| 700b800c787842ba83493d9b2775234a | 693 | −0.117858997 |
| c9b039c47ec54cc7b5c0fcc7a5730e38 | 720 | −0.073872774 |
| efa31cf3cd2f452789ca7f3e7541ea69 | 374 | −0.048700636 |
| f8699afb7a2247e38549e4d250b4581b | 541 | −0.131662834 |

6个log点估计均为负；log规模不同，bootstrap使用池化计数，未将每log简单等权平均。log仍只有6组，区间稳定性有限。

## 验收、身份与产物

新增9项统计/执行测试，先确认RED后实现；验证配对log抽样、池化权重、确定性、无效输入拒绝、started绑定、真实六策略小模型推理及missing精确回退。
全量 **582 passed in 140.11s**，Ruff/公开扫描通过。运行前原协议再次验证成功。
主运行完成后，输出/输入哈希和bootstrap统计重建校验通过；另一次从磁盘重载固定模型与候选的独立全量推理审计已完成，六策略每个指标数组与首次运行逐项完全一致，主结论和缺包回退一致。
审计记录保存在holdout-evaluation/audit.json；未对模型、数据划分或统计规则作事后更改。

准备/推理/统计段 **364.207秒**，不含启动前核验、输出归档和独立审计；不是每次在线更新延迟或p95速度指标。
本地输出outputs/context/p66-independent/holdout-evaluation：evaluation_started.json、preparation.json、metrics.npz、weights.npz、bootstrap_draws.npy、results.json、completed.json。
输入分别保存在predictions/context/candidates的independent_holdout下，数据和模型不入Git。

| 身份/文件 | SHA256 |
| --- | --- |
| 协议identity | adbc5d7c948a57c7e76173a54cb481d551842b8b80416309d8a823327ccb1231 |
| 执行identity | 1fb98ab74e4cb0981d7761adebd10f789fe51e0f10ac2269be099981355cd795 |
| results.json | 9349854fa0d90c84f18b3854f85b5c0e2d52d3bdb91b77a5c4c324a920d65c2b |
| metrics.npz | a694aaaad0c0db1a74983bbc7f1bb51e836eee3b1ed58792ec196cce49d83a39 |
| bootstrap_draws.npy | 85ad42ea12e09499f4d91a1201f505bb00a06cab8be9925d9a285e714499ac20 |
| audit.json | 3856a32814ae11b0fff16f5ab728a656517aac6aff06d9aa1d0b511db696104a |

```powershell
python scripts/evaluate_independent_holdout.py --device cuda
python scripts/evaluate_independent_holdout.py --verify-only --device cuda
```

完成后常规运行只重验并复用完成记录；verify-only重新从固定输入推理六策略并逐项比较保存指标，不重新训练或改变检验。

## 可以与不可以表述的成果

可以表述：在现有nuScenes元数据的log隔离重划分、全部上游重训及事前冻结评价协议下，RH在3,329轨迹/6log holdout上的平均在线ADE较因果CV降低5.41%，主比较log bootstrap区间低于0。
必须同时说明：单seed、6个log、标注位置因果回放、已知数据重划分；研究者已知历史结果，不能称为新采集盲测或nuScenes官方预测benchmark成绩。
不能表述：SOTA、传感器融合提升、真实部署安全、全面稳定性提升或跨seed显著性。RH jitter仍比CV高5.33%。
此holdout已被使用，后续不得在其上调参后继续当作未见测试集；未来模型改进应使用新独立数据/划分或另行定义探索性分析。
本轮不推送GitHub、不改简历，历史公开推荐与成绩不被静默替换。P7未启动。

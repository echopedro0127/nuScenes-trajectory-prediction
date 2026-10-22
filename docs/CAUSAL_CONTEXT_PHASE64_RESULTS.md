# P6.4 创新自适应时序一致性实验

状态：P6.4实验、完整生产重放与验收完成。12模型/120epoch；固定系数组未达推进门槛，本轮未读val，保留P6.3 RH。

## 来源与冻结协议

用户要求继续上一轮；本轮执行已批准P6.3设计第8节，详见[本轮设计](superpowers/specs/2026-10-22-causal-context-phase64-design.md)
和[施工计划](superpowers/plans/2026-10-22-causal-context-phase64.md)。仅固定RH结构，不重新选择H/DRH，不改候选，不新增数据/依赖。
337名义参数，RH有效输入15维/273参数，四组完全相同；创新权重只在训练loss使用，部署时不新增特征或参数。

| 训练组 | lambda | 一致性权重 | 目的 |
| --- | ---: | --- | --- |
| revision_0 | 0 | 不取前序前向 | 同初始化/预算重现P6.3 RH |
| revision_fixed | 0.02 | 1 | 固定一致性 |
| revision_adaptive | 0.02 | exp(-(innovation/0.5m)^2) | 按已兑现CV误差允许修订 |
| revision_mean | 0.02 | 完整train平均权重 | 控制平均正则系数 |

innovation[k]=norm(position[k]-CV[k-1,k])，k>0；第一步无前序，权重0。
误差取上次预测在当前已到达绝对时刻的点，不用当前CV在当前位置的平凡误差，也不取真实未到达future。
双方预测均可导；只比较当前mask内共同未来点。先每update平均L2，再乘权重、对有前序updates等权平均，不除权重和。
监督标签只进入masked SmoothL1+.05 correction smoothness；创新权重不读未来标签，也不通过训练降低自身权重。

train_mean仅用完整train的k>0样本，不在train_val/val重算；六条件只是重复同一创新，不改变均值。
固定均值控制平均系数，不保证梯度范数相同，权重与运动难度的相关性仍是方法机制的一部分。
加权loss数值较低本身不是jitter改善证据，效果只据完整预测指标判断。

三seed2027/2028/2029，train25560轨迹/468场景、train_val6621/176、条件推进val7201/136。
全部从同一P6模型初始化重训，非继续微调P6.3最佳checkpoint。10epoch/patience3/batch4096，AdamW lr.001/wd.0001。
best epoch按完整train_val ADE，四组同shuffle/初始化；实际早停与额外双前向成本如实报告。
无一致性组不执行不需要的前序前向，所以名义预算相同不代表实际计算时间相同。

## 选择与边界

本轮参考固定P6.3 RH，选择池仅含RH及四新组；旧14策略全部作为诊断基线参与评估。
要求总体和clean ADE≤1.002×RH、总体FDE≤1.005×RH、clean FDE p95≤1.01×RH；合格池中选jitter最小，tie依次ADE、名字。
新正则组须比RH降低jitter≥0.5%，且低于新revision_0，才推进读取历史val；选择无论是否推进都会冻结。
旧RH或zero组获选不算正则成功、不运行val。val固定选择后不重选架构、loss或阈值。
研究目标：val jitter相对RH下降≥1%且满足误差/尾部约束；另报相对P6.1准确度/稳定性，不能混淆比较对象。
历史val已经在多轮使用，不是新盲测；将来新holdout须同时避免上游见过相同场景。

位置仍为干净已释放标注观测，地图包按真实source/available时间线选择；缺包/无覆盖/age>2秒硬CV。
创新是时间前缀因果特征，不是形式因果推断。训练一致性不能解决真实跟踪噪声或未建模的传感器异步。
本轮不运行H子特征拆分、SHARP训练、外部下载或P7。

## 身份、重放与测量

父P6/P6.1/P6.2/P6.3绑定源码和旧产物不改；所有新增文件独立。训练身份包含实际输入/创新/均值、
scene/key、配置、源码/父源码、P6初始化、arm/角色与CuBLAS配置。正式CUDA初始化前冻结:4096:8。
生产加载的均值和train指纹由已验证train缓存重建，不能由checkpoint自己证明自己；父RH和冻结选择均核验hash及角色。
完成marker最后原子写入；未完成阶段从相同初始化重启，完成复用仍检查身份与mask。

评估18策略保存[6,N,11,3]指标和[6,N,11,12]权重；报告从生产缓存重建train创新/均值，加载全部模型，
逐元素重算全部metrics/weights/分组及冻结选择。旧14策略对P6.3表核对，zero对RH完整指标/权重与模型状态指纹核对。
每split12对比×7范围×3指标=252个区间，2000次scene bootstrap，先固定三seed平均再配对重采样。
包含四组−RH、三正则−zero、adaptive−fixed/mean、固定方案−P6.1准确度/稳定性/CV；不把seed或条件当独立样本。
逐condition/update/remaining horizon/maneuver/尾部/weight分布/恢复，以及seed mean/std和负面结果完整保留。

在线延迟固定256key/16warmup/batch1/CUDA同步，包含真实单步反馈状态更新、特征、head、blend与返回。
训练创新不在部署计时中；前缀状态重建、地图查询、P4 forward、冷缓存读入排除。
此为缓存候选上的Python门控路径，不是端到端感知预测延迟；微基准受后台负载影响，不与论文延迟直接比较。

审查发现的均值自证和父引用未校验问题已在正式训练前先写失败测试再修复；生产加载测试显式用正式配置和CuBLAS，
先确保正常checkpoint成功加载，再验证同时改均值、元数据、run设置与所有校验和仍拒绝。493项完整回归已通过。

## 复现命令

现有venv，PYTHONPATH=src；正式产物outputs/context/p64，模型和outputs不入git。

```powershell
python scripts/train_innovation_consistency.py --cache-root outputs/context/p6/cache --p6-training-root outputs/context/p6/training --horizon-training-root outputs/context/p63/training-deterministic --horizon-selection-root outputs/context/p63/selection --output-root outputs/context/p64/training --device cuda
python scripts/evaluate_innovation_consistency.py --cache-root outputs/context/p6/cache --training-root outputs/context/p64/training --p6-training-root outputs/context/p6/training --reference-training-root outputs/context/p61/training --reference-selection-root outputs/context/p61/selection --horizon-training-root outputs/context/p63/training-deterministic --horizon-selection-root outputs/context/p63/selection --horizon-evaluation-root outputs/context/p63/evaluation --selection-root outputs/context/p64/selection --output-root outputs/context/p64/evaluation --device cuda
python scripts/report_innovation_consistency.py --cache-root outputs/context/p6/cache --training-root outputs/context/p64/training --p6-training-root outputs/context/p6/training --reference-training-root outputs/context/p61/training --reference-selection-root outputs/context/p61/selection --horizon-training-root outputs/context/p63/training-deterministic --horizon-selection-root outputs/context/p63/selection --horizon-evaluation-root outputs/context/p63/evaluation --selection-root outputs/context/p64/selection --output-root outputs/context/p64/evaluation --device cuda
```

## 实际训练预算

| seed | 训练组 | 完成epoch | best epoch | 训练及逐轮选模秒数 |
| --- | --- | ---: | ---: | ---: |
| 2027 | revision_0 | 10 | 10 | 87.661 |
| 2027 | revision_fixed | 10 | 10 | 150.824 |
| 2027 | revision_adaptive | 10 | 10 | 143.976 |
| 2027 | revision_mean | 10 | 10 | 147.033 |
| 2028 | revision_0 | 10 | 9 | 82.776 |
| 2028 | revision_fixed | 10 | 10 | 156.200 |
| 2028 | revision_adaptive | 10 | 10 | 195.280 |
| 2028 | revision_mean | 10 | 10 | 172.202 |
| 2029 | revision_0 | 10 | 10 | 84.995 |
| 2029 | revision_fixed | 10 | 10 | 154.546 |
| 2029 | revision_adaptive | 10 | 10 | 149.374 |
| 2029 | revision_mean | 10 | 10 | 157.541 |

12模型共120epoch，日志合计1682.409秒（28.04分钟）。不包括缓存核验、最终评估或重放审计。
耗时包含forward/backward/optimizer及完整train_val逐轮选模；不同时间的后台负载会影响实际记录。

完整train有前序更新共255600个（不把六条件重复算成新增观测）；权重均值0.7076048737280716，
p0/p5/p50/p95/p100=[0,0.0215473,0.8617694,0.9999960,1]，三个seed重建核验相同。
固定均值组的等效lambda为0.0141520974745614，不改变按update平均的损失分母。
三个zero组均逐元素重现各自P6.3 RH的完整train_val metrics/weights，并且状态指纹完全相同。

## train_val实测结果（完整重放通过）

冻结选择为revision_fixed，但advance=false；未达到0.5% jitter推进门槛，CLI明确输出val_not_run。
本轮不读取历史val，也没有新的val指标或可宣称的val收益。继续保留P6.3已验证的RH方案，
不能把下述train_val小幅变化写成新的盲测或部署提升。

| 策略 | ADE/m | FDE/m | jitter/m |
| --- | ---: | ---: | ---: |
| sliding | 1.936301 | 4.272695 | 1.021958 |
| p6_learned | 1.800475 | 3.910451 | 1.103900 |
| finetune_0 | 1.798031 | 3.905768 | 1.106225 |
| time_age_gate | 1.805138 | 3.937424 | 1.092249 |
| consistent_short | 1.795673 | 3.907760 | 1.098392 |
| consistent_010 | 1.796937 | 3.917231 | 1.087359 |
| B | 1.798031 | 3.905768 | 1.106225 |
| D | 1.797094 | 3.903378 | 1.105102 |
| R | 1.796491 | 3.904287 | 1.107167 |
| H | 1.773758 | 3.884255 | 1.109866 |
| DR | 1.795766 | 3.904042 | 1.102769 |
| DH | 1.771729 | 3.880709 | 1.111073 |
| RH | 1.773337 | 3.884974 | 1.107842 |
| DRH | 1.770815 | 3.879538 | 1.110789 |
| revision_0 | 1.773337 | 3.884974 | 1.107842 |
| revision_fixed | 1.773126 | 3.885363 | 1.102824 |
| revision_adaptive | 1.773271 | 3.885543 | 1.103979 |
| revision_mean | 1.773149 | 3.885099 | 1.104718 |

### 四组相对RH

| 训练组 | ADE变化 | FDE变化 | jitter变化 |
| --- | ---: | ---: | ---: |
| revision_0 | +0.000000% | +0.000000% | +0.000000% |
| revision_fixed | -0.011896% | +0.010016% | -0.452950% |
| revision_adaptive | -0.003716% | +0.014652% | -0.348705% |
| revision_mean | -0.010631% | +0.003226% | -0.282000% |

固定系数jitter下降不足0.5%，尽管接近门槛，也不据此放宽标准。自适应权重没有超过固定系数组，
不能声称创新自适应是本轮最佳方法。固定均值组控制的是平均系数，而非梯度范数。

### 固定系数逐seed差值（相对各自RH）

| seed | ADE差值/m | FDE差值/m | jitter差值/m |
| --- | ---: | ---: | ---: |
| 2027 | -0.000140 | +0.000178 | -0.007452 |
| 2028 | -0.000543 | -0.000675 | -0.001141 |
| 2029 | +0.000050 | +0.001664 | -0.006460 |

三个seed的jitter差值均为负，但幅度不一致。以下按场景区间描述固定三seed模型的变化，不代表跨随机初始化总体的不确定性。


### 误差与尾部约束

下表为相对RH的比值，均按三个seed平均；clean p95为各seed分位数的均值，并非合并样本后的分位数。

| 组 | 总体ADE比 | clean ADE比 | 总体FDE比 | clean FDE p95比 | 误差约束 |
| --- | ---: | ---: | ---: | ---: | --- |
| revision_0 | 1.00000000 | 1.00000000 | 1.00000000 | 1.00000000 | 通过 |
| revision_fixed | 0.99988104 | 0.99962461 | 1.00010016 | 1.00010929 | 通过 |
| revision_adaptive | 0.99996284 | 0.99981843 | 1.00014652 | 1.00056663 | 通过 |
| revision_mean | 0.99989369 | 0.99971858 | 1.00003226 | 1.00002796 | 通过 |

revision_fixed仍比旧P6.1准确度方案jitter高0.004432m、比旧稳定性方案高0.015465m；不能称新的稳定性冠军。

### 场景配对区间与seed离散度

2000次bootstrap；176场景，先固定三seed平均后按scene配对重采样。以下单位均为m，差值为前者减后者；
这些区间描述当前固定模型和场景分布，未做多重比较校正，也不作为本轮推进阈值的替代。完整252条见本地报告。

| 范围 | 对比 | 指标 | 差值 | 场景95% CI |
| --- | --- | --- | ---: | --- |
| clean | revision_fixed-RH | ade | -0.000641 | [-0.001065, -0.000199] |
| clean | revision_fixed-RH | fde | +0.000084 | [-0.000540, +0.000702] |
| clean | revision_fixed-RH | jitter | -0.005310 | [-0.005736, -0.004902] |
| overall | revision_fixed-RH | ade | -0.000211 | [-0.000566, +0.000153] |
| overall | revision_fixed-RH | fde | +0.000389 | [-0.000156, +0.000941] |
| overall | revision_fixed-RH | jitter | -0.005018 | [-0.005319, -0.004728] |
| overall | revision_adaptive-revision_fixed | ade | +0.000145 | [+0.000028, +0.000265] |
| overall | revision_adaptive-revision_fixed | fde | +0.000180 | [-0.000103, +0.000444] |
| overall | revision_adaptive-revision_fixed | jitter | +0.001155 | [+0.001003, +0.001304] |
| overall | revision_adaptive-revision_mean | ade | +0.000123 | [-0.000054, +0.000288] |
| overall | revision_adaptive-revision_mean | fde | +0.000444 | [+0.000044, +0.000817] |
| overall | revision_adaptive-revision_mean | jitter | -0.000739 | [-0.000888, -0.000597] |

固定系数总体ADE和FDE区间跨0，不能声称显著准确度改善。jitter区间为负，但平均降幅0.453%仍未满足0.5%推进条件。
自适应比固定均值更稳，但相对固定系数的jitter差值区间为正，说明当前设置没有超过固定正则。

| 总体相对RH | ADE差值±seed样本标准差 | FDE差值±seed样本标准差 | jitter差值±seed样本标准差 |
| --- | ---: | ---: | ---: |
| revision_0 | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 | +0.000000 ± 0.000000 |
| revision_fixed | -0.000211 ± 0.000303 | +0.000389 ± 0.001183 | -0.005018 ± 0.003394 |
| revision_adaptive | -0.000066 ± 0.000340 | +0.000569 ± 0.001307 | -0.003863 ± 0.003224 |
| revision_mean | -0.000189 ± 0.000188 | +0.000125 ± 0.000854 | -0.003124 ± 0.003286 |

### 在线延迟与恢复

RTX 3060 Laptop、CUDA同步、batch1、每seed256key/16warmup。下表为三个seed各自p50/p95的均值（ms），
不是合并768样本后的分位数。全部18策略原始key/测量及统计均已核验。

| 策略 | p50均值/ms | p95均值/ms |
| --- | ---: | ---: |
| sliding | 0.3364 | 0.6234 |
| p6_learned | 1.2984 | 1.9819 |
| finetune_0 | 1.2590 | 1.9264 |
| time_age_gate | 1.0794 | 1.7490 |
| consistent_short | 1.8382 | 2.7751 |
| consistent_010 | 1.2589 | 1.9906 |
| B | 1.3848 | 2.2463 |
| D | 1.3829 | 2.2585 |
| R | 1.3710 | 2.2235 |
| H | 1.3715 | 2.0294 |
| DR | 1.3794 | 2.3075 |
| DH | 1.3903 | 2.2265 |
| RH | 1.3863 | 2.1994 |
| DRH | 1.3891 | 2.0480 |
| revision_0 | 1.3723 | 2.0720 |
| revision_fixed | 1.3540 | 1.9935 |
| revision_adaptive | 1.3640 | 1.9847 |
| revision_mean | 1.3568 | 1.9896 |

四新组与RH使用相同推理结构，训练创新不在部署路径；上述差异不证明算法加速。
与上轮测量的后台负载不同，不能把跨轮微基准时间下降称为优化收益。候选读取之后的门控测量不包含感知/地图/P4。

在0.25m相对clean ADE差值阈值下，18策略恢复比例均100%，恢复时间最大值均0秒；
该定义按dropout后连续两个更新比较相对clean误差，不是感知恢复能力或真实跟踪鲁棒性。
缺包/无覆盖/过期硬CV；逐update、remaining horizon、maneuver、分位数、gate与恢复表已完整重放。

### 验收、来源与下一步

12模型/120epoch，18策略×六条件×11更新×三seed，共23,597,244条度量及全部逐点权重完成生产重放。
旧14策略与冻结P6.3表核对atol=1e-9/rtol=0；零组完整指标/权重同容差，三个模型状态指纹完全一致。
每seed初始八变体436986次update的指标/权重差值均0；完整train均值及输入指纹由生产缓存独立重建。
冻结选择重算一致，advance=false、val_run=false；没有val评估目录。研究目标未验收，不添加新的val收益。

493项pytest通过，Ruff、公开扫描和diff检查通过；实现及结果经独立只读审查。旧绑定源码/配置/测试未改。
只读mmap转张量提示沿用冻结路径，已审查当前调用无原地写入，不为消除提示改动旧实验。
实现提交acab2e7（协议）、dcbdd6e（训练）、555c0f2（评估/报告）；模型、数据、outputs不入git，不推送。

冻结选择 SHA256：`8a327337cbf14826c5a2114779a9071a66a8584b7bef2e76af62b5120b9e0f10`。

完整报告 SHA256：`e4576512d4e317417cec4e9fc6e1ecd0d347b995ddd7a1c8522bf41134c762a6`。

252条配对区间 SHA256：`1cfe18f9d48158006b2098068c04e112a916d669132405773a114a4a93248429`。

报告源码 SHA256：`104ac1ec0a80cb62e3291b3dad0100934062aece4284bfee24a7e76281edeaba`。

完整本地产物位于outputs/context/p64/report，report.json绑定全部训练日志、模型、缓存、源码、选择、逐seed summary和报告表格哈希。

下一步建议优先检验正则强度校准：当前adaptive的平均系数只有0.014152，fixed为0.02；
可在独立预冻结协议中比较平均系数匹配的adaptive与fixed，以及有限的固定系数强度。
均值匹配仍不等于梯度匹配，须保留fixed/mean/zero对照并检查误差、转向和尾部。
本轮没有执行该后续实验，也不事后改best epoch或降低推进门槛；当前准确度方案继续使用已验证RH，稳定性保留consistent_010。

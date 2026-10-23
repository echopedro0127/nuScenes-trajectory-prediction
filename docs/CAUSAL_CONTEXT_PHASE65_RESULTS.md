# P6.5 一致性强度校准与平均系数匹配实验

记录日期2026-10-23，分支feature/causal-context-online。
状态：12模型/120epoch与完整train_val/val评估完成。冻结adaptive_matched_010在val jitter下降2.303%，但总体/clean ADE分别增加0.211944%/0.217628%，均超0.2%上限；完整生产重放与验收完成，研究目标不通过，不晋级。

## 冻结协议

见[设计](superpowers/specs/2026-10-23-causal-context-phase65-design.md)与[施工计划](superpowers/plans/2026-10-23-causal-context-phase65.md)。
用户已批准继续本轮。固定RH/P6初始化，不改变上游候选、数据或部署结构。
新四组：fixed_005(alpha/lambda=.05)、fixed_010(.10)、adaptive_matched_005(lambda=.05/mu)、adaptive_matched_010(.10/mu)。
mu仅由完整train k>0权重验证重建，不按batch/epoch/train_val/val归一化。
原权重保留[0,1]，平均系数匹配不等于梯度范数相等；constant-mean等价控制由实际损失/双侧梯度测试核验，不重复训练数学相同的组。

三seed2027/2028/2029，12新模型，最多120epoch；10epoch/patience3/batch4096，AdamW lr.001/wd.0001。
best checkpoint仍按完整train_val ADE，不事后改选模准则。双方预测均可导，首步不配对；剩余点L2先逐update平均、乘w后按有前序update平均。
基础监督masked SmoothL1+.05 correction smoothness；CUDA初始化前固定CuBLAS=:4096:8。
旧18策略严格加载，加新4组共22策略。缺包/无覆盖/age>2秒硬CV。位置为干净已释放标注，不是现实跟踪噪声。
train25560/468场景，train_val6621/176，条件推进历史val7201/136；六地图包条件、11更新、固定6秒终点。

## 选择与统计

只从RH、旧revision_fixed和四新组中选jitter最低者，tie按ADE/name；总体/clean ADE<=1.002*RH，
总体FDE<=1.005*RH，clean FDE p95<=1.01*RH。阈值、矩阵与选择先冻结，再决定是否历史val。
新组须jitter比RH下降>=.5%，并严格优于旧fixed(.02)和zero，才推进。历史val研究目标为jitter下降>=1%且误差约束通过。
历史val已多轮使用，不是盲测；若未推进则本轮不读取val。若推进，旧P6.4四组只作为本轮统一对照，不追溯改变P6.4结论。
每split15对比×7范围×3指标=315条区间，2000次scene bootstrap；先固定三个seed平均，避免seed/condition/update伪重复。
另报seed样本标准差、机动/剩余时域/尾部、gate/恢复及负面结果；区间不替代预设门槛。

## 身份与验收

新增独立模块，旧P6至P6.4源码/配置/测试及产物冻结。训练身份绑定alpha/kind/lambda、actual train mu/content、
模型初始化与RH mask、场景/key、候选契约、父源码、P6.3/P6.4选择及正式报告、CuBLAS与配置。
生产加载由实际train重建mu/content，不接受模型自证；完成marker最后原子发布。
报告从完整生产缓存重算22策略全部指标/权重/分组/选择并核对旧18基线。
在线延迟为固定256key/16warmup/batch1/CUDA同步的实际状态更新、特征/head/blend路径；不含前缀重建、地图查询/P4 forward/冷读。
同推理结构的微基准变化不作加速宣称，恢复相对clean定义不等于感知恢复能力。

训练前506项pytest通过（97.75秒），Ruff、公开扫描和独立只读审查通过。
审查发现当前执行source遗漏与父P6.3报告绑定遗漏，均先RED再GREEN修复，实际父产物加载校验通过后才冻结训练。
协议8562074、施工e4d2023、实现9d1b280。产物独立写outputs/context/p65，不入git，不推送、不合并、不改简历。

## 实际训练预算

| seed | arm | 完成epoch | best epoch | 训练及逐轮选模秒数 |
| --- | --- | ---: | ---: | ---: |
| 2027 | fixed_005 | 10 | 10 | 322.566 |
| 2027 | adaptive_matched_005 | 10 | 10 | 295.820 |
| 2027 | fixed_010 | 10 | 10 | 298.447 |
| 2027 | adaptive_matched_010 | 10 | 10 | 156.923 |
| 2028 | fixed_005 | 10 | 10 | 145.252 |
| 2028 | adaptive_matched_005 | 10 | 10 | 149.785 |
| 2028 | fixed_010 | 10 | 10 | 214.733 |
| 2028 | adaptive_matched_010 | 10 | 10 | 272.422 |
| 2029 | fixed_005 | 10 | 10 | 226.346 |
| 2029 | adaptive_matched_005 | 10 | 10 | 235.457 |
| 2029 | fixed_010 | 10 | 10 | 246.699 |
| 2029 | adaptive_matched_010 | 10 | 10 | 249.568 |

12模型/120epoch，总计2814.017600秒（46.90分钟），仅forward/backward/optimizer与逐轮完整train_val选模；
不含缓存核验、初始化检查、最终评估和生产重放。各时段后台负载不同，实际耗时不作训练加速对照。
三个seed均验证train mean=0.7076048737280716，count=255600（不把六条件重复算成新增观测），
权重p0/p5/p50/p95/p100=[0,0.0215473,0.8617694,0.9999960,1]；两个matched原始lambda为0.07066090392590302/0.14132180785180604。
三个seed完整初始八变体每次436986个update，metrics/weights最大差均0；seed2027已用实际生产load_context成功加载全部16个head（含新四组），未读val。

## train_val冻结结果（完整生产重放通过）

旧18策略与P6.4表完整核对通过，选择池及阈值保持冻结；选择adaptive_matched_010、advance=true。
其train_val jitter相对RH下降2.584072%，ADE增加0.116412%、FDE增加0.204760%，总体/clean/尾部约束均通过。
此为推进依据，不代替历史val固定回归；不在val重选任何其他组。

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
| fixed_005 | 1.773219 | 3.886652 | 1.094270 |
| adaptive_matched_005 | 1.773605 | 3.888250 | 1.092107 |
| fixed_010 | 1.774243 | 3.889093 | 1.082871 |
| adaptive_matched_010 | 1.775402 | 3.892929 | 1.079214 |

| 新组 | 总体ADE比 | clean ADE比 | 总体FDE比 | clean FDE p95比 |
| --- | ---: | ---: | ---: | ---: |
| fixed_005 | 0.99993351 | 0.99928964 | 1.00043198 | 1.00042247 |
| adaptive_matched_005 | 1.00015122 | 0.99958855 | 1.00084329 | 1.00187020 |
| fixed_010 | 1.00051102 | 0.99946393 | 1.00106020 | 1.00146284 |
| adaptive_matched_010 | 1.00116412 | 1.00039312 | 1.00204760 | 1.00371845 |

## 历史val固定回归（完整生产重放通过）

冻结方案adaptive_matched_010，ADE/FDE/jitter=1.770499/3.810212/1.158446m；相对RH分别+0.211944%/+0.304072%/-2.303323%。
总体ADE上限为1.770287936m，实测高0.000211017m；clean ADE增幅0.217628%也超过0.2%上限。
两项ADE约束未通过；总体FDE与clean FDE p95仍合格。按预定规则不晋级，不能因接近阈值而放宽。
jitter降幅超过1%研究幅度目标，但整体研究目标要求误差/尾部约束也通过，因此本轮整体目标未通过。
旧P6.3的14策略val表已独立核对atol1e-9/rtol0。所有对照在本轮目录生成，不修改P6.4未推进的历史结论。

| 策略 | ADE/m | FDE/m | jitter/m |
| --- | ---: | ---: | ---: |
| sliding | 1.937731 | 4.235810 | 1.108377 |
| p6_learned | 1.792485 | 3.831412 | 1.181381 |
| finetune_0 | 1.790014 | 3.826473 | 1.182866 |
| time_age_gate | 1.807564 | 3.871015 | 1.181725 |
| consistent_short | 1.788697 | 3.829622 | 1.175611 |
| consistent_010 | 1.789946 | 3.839947 | 1.164413 |
| B | 1.790014 | 3.826473 | 1.182866 |
| D | 1.789534 | 3.826502 | 1.181484 |
| R | 1.789214 | 3.825125 | 1.184751 |
| H | 1.767457 | 3.800440 | 1.187586 |
| DR | 1.788943 | 3.827156 | 1.180209 |
| DH | 1.765542 | 3.798017 | 1.189635 |
| RH | 1.766754 | 3.798661 | 1.185758 |
| DRH | 1.764619 | 3.796573 | 1.189358 |
| revision_0 | 1.766754 | 3.798661 | 1.185758 |
| revision_fixed | 1.766673 | 3.799103 | 1.180840 |
| revision_adaptive | 1.766892 | 3.799625 | 1.182084 |
| revision_mean | 1.766672 | 3.798916 | 1.182706 |
| fixed_005 | 1.766953 | 3.800365 | 1.172392 |
| adaptive_matched_005 | 1.767737 | 3.803051 | 1.170632 |
| fixed_010 | 1.768467 | 3.803875 | 1.161264 |
| adaptive_matched_010 | 1.770499 | 3.810212 | 1.158446 |

| 新组 | 总体ADE比 | clean ADE比 | 总体FDE比 | clean FDE p95比 |
| --- | ---: | ---: | ---: | ---: |
| fixed_005 | 1.00011224 | 0.99975757 | 1.00044869 | 1.00003065 |
| adaptive_matched_005 | 1.00055628 | 1.00045705 | 1.00115561 | 1.00164917 |
| fixed_010 | 1.00096959 | 1.00050821 | 1.00137251 | 1.00058997 |
| adaptive_matched_010 | 1.00211944 | 1.00217628 | 1.00304072 | 1.00414119 |

四新组完整回归均报告，但不能据val换成其他组来宣告成功。当前推荐仍保留P6.3 RH准确度与旧consistent_010稳定性方案；
候选在某些指标优于旧方案，不等于通过本轮相对RH冻结的整体约束。

### 四新组相对RH

| split | 新组 | ADE变化 | FDE变化 | jitter变化 |
| --- | --- | ---: | ---: | ---: |
| train_val | fixed_005 | -0.006649% | +0.043198% | -1.225074% |
| train_val | adaptive_matched_005 | +0.015122% | +0.084329% | -1.420318% |
| train_val | fixed_010 | +0.051102% | +0.106020% | -2.253958% |
| train_val | adaptive_matched_010 | +0.116412% | +0.204760% | -2.584072% |
| val | fixed_005 | +0.011224% | +0.044869% | -1.127183% |
| val | adaptive_matched_005 | +0.055628% | +0.115561% | -1.275571% |
| val | fixed_010 | +0.096959% | +0.137251% | -2.065692% |
| val | adaptive_matched_010 | +0.211944% | +0.304072% | -2.303323% |

### val冻结方案逐seed差值（相对RH）

| seed | ADE差值/m | FDE差值/m | jitter差值/m |
| --- | ---: | ---: | ---: |
| 2027 | +0.004678 | +0.012546 | -0.030373 |
| 2028 | +0.003302 | +0.012452 | -0.024856 |
| 2029 | +0.003253 | +0.009654 | -0.026707 |

三个seed均呈jitter下降、ADE/FDE增加；不是只由一个seed驱动。场景区间和seed标准差口径另列，不把三seed当场景样本数。

相对旧P6.1准确度方案，冻结候选ADE/FDE/jitter变化分别为-1.017409%/-0.506851%/-1.460085%；
相对旧稳定性方案jitter变化-0.512487%。这不替代本轮相对RH的约束；不能改参考或验收标准后宣称整体达标。
相对同观测CV的ADE/FDE/jitter变化为-8.630303%/-10.047618%/+4.517343%；jitter仍高于CV，不能宣称全面优于CV。

### 单步延迟与恢复（完整重放通过）

RTX 3060 Laptop GPU；每seed同256key、16warmup、batch1、CUDA同步。下表为三个seed各自p50/p95的均值（ms），
不是合并768样本后的分位数；推理结构不变，不把组间/跨轮微基准波动作为算法加速。

| 策略 | train_val p50 | train_val p95 | val p50 | val p95 |
| --- | ---: | ---: | ---: | ---: |
| sliding | 0.9282 | 2.0137 | 0.4993 | 0.9794 |
| p6_learned | 2.5462 | 5.6104 | 1.8011 | 2.7258 |
| finetune_0 | 2.3915 | 4.4256 | 1.7515 | 2.6209 |
| time_age_gate | 2.0983 | 4.3783 | 1.3747 | 2.0481 |
| consistent_short | 3.6235 | 7.0376 | 2.3161 | 2.9178 |
| consistent_010 | 2.2757 | 4.6740 | 1.5829 | 2.2046 |
| B | 2.7716 | 6.8246 | 1.7611 | 2.4633 |
| D | 2.9581 | 7.0745 | 1.8267 | 2.4179 |
| R | 2.5949 | 5.2756 | 1.8429 | 2.4146 |
| H | 2.8904 | 5.2838 | 1.8226 | 2.4055 |
| DR | 2.8137 | 7.0146 | 1.7949 | 2.3552 |
| DH | 2.9280 | 6.4425 | 1.8437 | 2.3827 |
| RH | 3.0532 | 5.9612 | 1.9024 | 2.5155 |
| DRH | 2.9843 | 5.9472 | 1.8815 | 2.3749 |
| revision_0 | 3.0462 | 5.6842 | 1.8666 | 2.3402 |
| revision_fixed | 2.6205 | 5.3173 | 1.7723 | 2.3478 |
| revision_adaptive | 2.8276 | 6.1141 | 1.8306 | 2.3640 |
| revision_mean | 3.0621 | 6.5837 | 1.7241 | 2.1781 |
| fixed_005 | 2.9392 | 4.8791 | 1.8419 | 2.9234 |
| adaptive_matched_005 | 2.7473 | 5.4818 | 2.1680 | 2.9620 |
| fixed_010 | 2.4983 | 6.7797 | 2.1699 | 3.0148 |
| adaptive_matched_010 | 2.5748 | 6.6108 | 2.2932 | 3.1180 |

train_val全部22策略在0.25m相对clean ADE差值阈值下，恢复比例均100%，最大恢复时间均0秒；

val全部22策略在0.25m相对clean ADE差值阈值下，恢复比例均100%，最大恢复时间均0秒；
这是dropout后相对clean的连续两次更新阈值指标，不证明真实传感器、跟踪或感知恢复能力。
单步延迟不含前缀状态重建、地图查询、P4 forward和冷缓存加载，不作为端到端感知预测延迟。

## 场景配对区间与seed离散度

每split315条、合计630条区间；2000次bootstrap，先固定三seed平均后按scene配对重采样。
train_val176场景、val136场景，区间只表示当前固定模型集合下的场景不确定性，不代替跨训练随机性的推断。
未做多重比较校正；区间不替代预设推进幅度或误差阈值。以下差值为前者减后者，单位m。

| split/范围 | 对比 | 指标 | 差值 | 场景95% CI |
| --- | --- | --- | ---: | --- |
| train_val/clean | adaptive_matched_010-RH | ade | +0.000671 | [-0.003049, +0.004339] |
| train_val/clean | adaptive_matched_010-RH | fde | +0.008865 | [+0.002275, +0.015407] |
| train_val/clean | adaptive_matched_010-RH | jitter | -0.028387 | [-0.031077, -0.025865] |
| train_val/overall | adaptive_matched_010-RH | ade | +0.002064 | [-0.001070, +0.005051] |
| train_val/overall | adaptive_matched_010-RH | fde | +0.007955 | [+0.002455, +0.013548] |
| train_val/overall | adaptive_matched_010-RH | jitter | -0.028627 | [-0.030482, -0.026919] |
| train_val/overall | adaptive_matched_005-fixed_005 | jitter | -0.002163 | [-0.002645, -0.001697] |
| train_val/overall | adaptive_matched_010-fixed_010 | ade | +0.001158 | [+0.000046, +0.002188] |
| train_val/overall | adaptive_matched_010-fixed_010 | fde | +0.003836 | [+0.001239, +0.006284] |
| train_val/overall | adaptive_matched_010-fixed_010 | jitter | -0.003657 | [-0.004498, -0.002843] |
| val/clean | adaptive_matched_010-RH | ade | +0.003688 | [+0.000076, +0.007497] |
| val/clean | adaptive_matched_010-RH | fde | +0.013659 | [+0.006872, +0.020579] |
| val/clean | adaptive_matched_010-RH | jitter | -0.025189 | [-0.027210, -0.023031] |
| val/overall | adaptive_matched_010-RH | ade | +0.003745 | [+0.000568, +0.006959] |
| val/overall | adaptive_matched_010-RH | fde | +0.011551 | [+0.005495, +0.017670] |
| val/overall | adaptive_matched_010-RH | jitter | -0.027312 | [-0.028824, -0.025652] |
| val/overall | adaptive_matched_005-fixed_005 | jitter | -0.001760 | [-0.002312, -0.001214] |
| val/overall | adaptive_matched_010-fixed_010 | ade | +0.002031 | [+0.000812, +0.003249] |
| val/overall | adaptive_matched_010-fixed_010 | fde | +0.006337 | [+0.003305, +0.009388] |
| val/overall | adaptive_matched_010-fixed_010 | jitter | -0.002818 | [-0.003794, -0.001842] |

val冻结方案相对RH的jitter差区间为负，ADE/FDE差区间为正：稳定性收益伴随精度代价。
这不是“区间证明超限”的检验；验收使用事前冻结的点估计约束，不能据区间改写阈值。
同平均系数下，自适应比固定系数组更稳，但ADE/FDE也更高；不能只据jitter区间宣称全面优越。

| split | 新组相对RH | ADE差值±seed样本标准差 | FDE差值±seed样本标准差 | jitter差值±seed样本标准差 |
| --- | --- | ---: | ---: | ---: |
| train_val | fixed_005 | -0.000118 ± 0.000580 | +0.001678 ± 0.002099 | -0.013572 ± 0.003725 |
| train_val | adaptive_matched_005 | +0.000268 ± 0.000666 | +0.003276 ± 0.002960 | -0.015735 ± 0.003243 |
| train_val | fixed_010 | +0.000906 ± 0.000971 | +0.004119 ± 0.002114 | -0.024970 ± 0.003990 |
| train_val | adaptive_matched_010 | +0.002064 ± 0.000741 | +0.007955 ± 0.002521 | -0.028627 ± 0.003240 |
| val | fixed_005 | +0.000198 ± 0.000524 | +0.001704 ± 0.000769 | -0.013366 ± 0.003546 |
| val | adaptive_matched_005 | +0.000983 ± 0.000401 | +0.004390 ± 0.001140 | -0.015125 ± 0.002895 |
| val | fixed_010 | +0.001713 ± 0.001004 | +0.005214 ± 0.001240 | -0.024494 ± 0.003882 |
| val | adaptive_matched_010 | +0.003745 ± 0.000809 | +0.011551 ± 0.001644 | -0.027312 ± 0.002808 |

## 完整验收与来源

12新模型/120epoch，22策略×六条件×11更新×三seed；train_val28,841,076条、val31,367,556条，
合计60,208,632条度量及全部逐点权重完成生产重放。每seed均验证完整train创新/mu/指纹、模型/初始化、
候选契约、父P6.3/P6.4选择与正式报告、逐点metric/weight、分组/尾部/恢复和raw latency/keys。
旧18策略train_val与P6.4、旧14策略val与P6.3表核对atol1e-9/rtol0，实际差异均0。
三个zero/RH状态指纹一致；完整初始八变体每seed436986个update的指标/权重差值均0。
冻结选择从重放表重算一致；本轮val_run=true，但固定方案constraints_met=false/research_target_met=false。
不据val重选fixed_010等诊断组，不追溯改变P6.4未进入val的历史结论。

506项pytest、Ruff、公开扫描及独立只读实现/结果审查通过。训练前两项身份绑定问题先RED再GREEN修复。
实验源码相对9d1b280未变；旧P6至P6.4源码/配置/测试未改。只读mmap转张量提示沿用已审查冻结路径，当前调用无原地写入。
保留本地feature分支/worktree，不提交模型/数据/outputs，不推送、不合并、不更新简历。

冻结选择 SHA256：`508bdef16942f9ce4996f9e15649d4f68768d1131557e039a2f9950d40d94a34`。

正式报告 SHA256：`a251942cd0983478483837833565a09ed1c6373c8959c456faef367d611f6ae8`。

train_val315条区间 SHA256：`725f40a0fc9819463d71bcd077e5c2f27da5145388fa3b9c88be49d58af0d56d`。

val315条区间 SHA256：`2818bf8073b5d1fe4cfa9b6d07fd784ee117b86673589876447f13bc69260400`。

报告源码 SHA256：`9093003f984d3a4a0226a77e34ccf8a47b44bd07271e3ef8d720ea5e8f4be37a`。

report.json绑定全部训练日志、模型/缓存/选择/源码及逐seed summary与报告表格哈希；正式产物outputs/context/p65/report。

## 复现命令

现有venv、PYTHONPATH=src、CUDA初始化前CUBLAS_WORKSPACE_CONFIG=:4096:8。

```powershell
python scripts/train_strength_consistency.py --cache-root outputs/context/p6/cache --p6-training-root outputs/context/p6/training --horizon-training-root outputs/context/p63/training-deterministic --horizon-selection-root outputs/context/p63/selection --horizon-report-root outputs/context/p63/report --consistency-training-root outputs/context/p64/training --consistency-selection-root outputs/context/p64/selection --consistency-report-root outputs/context/p64/report --output-root outputs/context/p65/training --device cuda
$phaseArgs = @("--cache-root","outputs/context/p6/cache","--training-root","outputs/context/p65/training","--p6-training-root","outputs/context/p6/training","--reference-training-root","outputs/context/p61/training","--reference-selection-root","outputs/context/p61/selection","--horizon-training-root","outputs/context/p63/training-deterministic","--horizon-selection-root","outputs/context/p63/selection","--horizon-evaluation-root","outputs/context/p63/evaluation","--consistency-training-root","outputs/context/p64/training","--consistency-selection-root","outputs/context/p64/selection","--consistency-evaluation-root","outputs/context/p64/evaluation","--consistency-report-root","outputs/context/p64/report","--selection-root","outputs/context/p65/selection","--output-root","outputs/context/p65/evaluation","--device","cuda")
python scripts/evaluate_strength_consistency.py @phaseArgs
python scripts/report_strength_consistency.py @phaseArgs
```

## 下一步建议（本轮未执行）

有限强度校准已验证降低抖动的路径，但最小jitter选模会消耗精度余量。下一轮宜先在独立协议中
检验更保守的train_val精度余量与精度约束下的checkpoint/Pareto选择，避免继续单纯增大正则。
所有新选择必须在train/train_val冻结；历史val已反馈到设计，不能再称盲测，若据本轮val选择fixed_010或其他方案，
必须标明它是事后候选，并用上游训练也未见的新场景验证，才能形成新的独立收益证据。
本轮不自动执行新协议、H拆分、SHARP或P7，当前推荐保持RH准确度与consistent_010稳定性方案。

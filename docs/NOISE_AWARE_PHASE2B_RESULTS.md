# V2 Phase 2B：时间、候选与机动分组诊断

日期2026-11-10。分支`feature/noise-aware-diagnostics`，独立worktree`.worktrees/noise-aware-diagnostics`；实现冻结提交`8744ce8`。依据[Phase 2A](NOISE_AWARE_PHASE2A_RESULTS.md)继续执行[设计](superpowers/specs/2026-11-10-noise-aware-phase2b-design.md)及[计划](superpowers/plans/2026-11-10-noise-aware-phase2b.md)。

本轮确认：规则时间流中的固定Q/R协方差特征仅是时间编码；加入残差信息仍有有限开发增量。神经候选整体有帮助，地图候选仅有小幅且不一致的收益。更明确的失败项是减速和低速目标，不应只依据总体均值宣传全面提升。

## 固定协议与控制

只读复用父实验的352 fit窗口/22log、256 select窗口/4log，三条件clean/noise_05/noise_outlier，三噪声seed。仍只有608个不同轨迹窗口，窗口可能重叠，不能当独立实体；三个噪声seed不是三次模型训练。

全部五组训练保持839个名义参数、18→32→7网络、同初始化seed2027、同样本顺序、fit-only标准化、20epoch/每组700更新、batch1024和原Adam配置；无早停和看select调参。

| 新控制名 | 改变内容 |
| --- | --- |
| full | 精确重放父innovation门控，所有特征和七候选不变 |
| motion_time | motion6 + t至t六次幂编码6维，残差通道置零；t=step/11 |
| time_innovation | motion6 + 显式时间6 + 原innovation6 |
| physical_candidates | 原全部特征不变，只开放四个物理候选，神经候选logit屏蔽 |
| no_map_candidate | 原全部特征不变，只屏蔽map候选，保留GRU/original |

移除候选采用负无穷logit后softmax，失活权重严格为零，不用复制物理专家填空。名义参数量不变，但可用输出及有效参数量改变。输入特征保持一致，仍含地图覆盖率，所以最后两组检验的是候选价值，不能称为完全移除地图信息。

机动标签由已有配置固定生成：未来平均速度<2m/s为低速，净航向变化15/45度分转弯/高曲率，平均加减速度阈值1m/s²。标签只用于离线评价，不进入训练特征或在线前向。方向三组互斥，其余标签可重叠；这些是整体未来轨迹的描述，不是每步实时意图真值。

## Select总体ADE

单位m，越低越好。旧motion/covariance为父模型对照，其余五行为本轮固定预算训练。

| 策略 | clean | noise_05 | noise_outlier |
| --- | ---: | ---: | ---: |
| gate_motion | 2.311269 | 3.163847 | 3.955041 |
| gate_covariance | 2.208481 | 3.156777 | 3.949921 |
| motion_time | 2.217627 | 3.158643 | 3.887043 |
| time_innovation | 2.146166 | 3.150235 | 3.842254 |
| full | 2.151427 | 3.144192 | 3.845871 |
| physical_candidates | 2.282920 | 3.154126 | 3.965660 |
| no_map_candidate | 2.180021 | 3.152340 | 3.845176 |

完整FDE/jitter、全部17策略（旧12+新5）、按log/seed结果见本地`summary.json`。full与旧gate_innovation是同一个模型的重放，不能算独立训练重复。所有三条件、策略和机动组均保留，没有按结果过滤。

1. **协方差并未识别实际噪声。** 用零位置的合成流、固定初始P和Q/R，只依据更新步就能逐元素重建父缓存中的全部六维协方差特征。在本轮规则零延迟协议下，它是确定性时间函数。显式时间组与covariance组的clean/noise_05 ADE只差约0.41%/0.06%，离群条件还低1.59%；不同编码基底会改变拟合，不能把两者差值解释为观测质量估计收益。该结论不直接外推到延迟/丢包或自适应Q/R。
2. **残差增量比先前说法更有限。** time_innovation相对motion_time的三条件ADE分别下降3.22%/0.27%/1.15%，支持残差信息的开发价值。time_innovation和full很接近，不能据这一次开发均值改称新最佳模型或作泛化结论。
3. **不能简单移除所有神经候选。** full相对physical_candidates的三条件ADE分别低5.76%/0.31%/3.02%；但单独GRU候选仍弱，当前没有逐个移除GRU和original，不能归因于某一个网络。
4. **地图候选收益较小。** full相对no_map_candidate在clean/noise_05约改善1.31%/0.26%；离群条件反而高0.018%，近乎持平。不能宣传地图在所有条件下提高鲁棒性。

纯高斯噪声的原短板仍存在：full ADE3.144192 m，普通KF为3.101417 m；本轮没有解决这项退化。

## Clean机动分组：总体小收益掩盖哪些失败

以下为ADE（m）。每组均覆盖4个select log，但减速/加速各只有15个不同窗口，样本很少；不能把三个相同clean seed重复计成独立轨迹。

| 标签 | 不同窗口数 | CV | full | physical_candidates | no_map_candidate |
| --- | ---: | ---: | ---: | ---: | ---: |
| straight | 189 | 1.925414 | 1.933652 | 2.026394 | 1.948799 |
| turn | 39 | 2.571235 | 2.395155 | 2.718600 | 2.469210 |
| high_curvature | 28 | 3.230352 | 3.281932 | 3.407634 | 3.337968 |
| low_speed | 38 | 1.211682 | 1.381542 | 1.303588 | 1.343844 |
| acceleration | 15 | 3.932052 | 3.553286 | 4.234693 | 3.666685 |
| deceleration | 15 | 4.238193 | 4.633937 | 4.591063 | 4.671046 |

full相对CV：转弯改善6.85%、加速改善9.63%；减速退化9.34%、低速退化14.02%、高曲率退化1.60%、直行退化0.43%。不能说模型已普遍改善真实机动。

方向三组189+39+28=256，可精确分解总体clean ADE差：直行贡献+0.006081 m、转弯贡献−0.026825 m、高曲率贡献+0.005642 m，合计−0.015102 m。因此总体仅约0.70%的收益主要由转弯组贡献，其他方向组有抵消。低速/加减速与方向标签重叠，不能再把它们的贡献相加。

## NIS诊断：报警不等于当前观测有错

普通KF的NIS>9.21触发率及full给NIS-KF的平均权重：

| 条件/更新 | NIS触发率 | NIS-KF平均权重 |
| --- | ---: | ---: |
| clean全部更新 | 0.78% | 17.07% |
| noise_05全部更新 | 1.83% | 56.62% |
| noise_outlier注入时刻4/8 | 99.09% | 82.94% |
| noise_outlier其余更新 | 26.16% | 71.76% |

clean减速组触发率6.06%，高于全体clean的0.78%。没有添加位置噪声时真实运动也可能使CV模型失配；这不是已经确认的传感器异常误报。离群条件下未注入的其他时刻仍高频报警，说明状态污染/恢复过程会影响残差，不能把一次高NIS直接解释成当前观测错误。它们提示需要区分运动失配与异常观测，但没有证明所有减速误差都由该机制导致。

## 审计与复现

运行身份：`ee0b81a7c965d69b41fbc9f95b7471818bbe9a32eee9fa1687e81c4d14a2bca7`。

输出`outputs/noise-aware/phase2b/`：父引用、输入NPZ、记录、五门控PT、训练日志、93,024条策略级记录、逐更新NPZ、全部机动/事件JSON及完成哈希。五组训练加评价/分组约32.44秒CPU，不包含父实验验证和输入复制，不是在线端到端延迟。

- 625项pytest、Ruff和公开文件扫描通过。
- 全部指标、逐更新指标、fit-only标准化、机动标签/计数/均值、事件统计和摘要重算通过。
- full权重、标准化及5,472条观测流的指标逐项精确复现父模型。
- 父实验在原worktree重新验证，包括其原源码清单和带噪候选抽查；旧worktree仍是0b03bdd、无修改。新源码单独绑定，不改写父实验哈希。
- 本轮没有holdout评价，没有重训GRU/map，没有下载新数据。

在新worktree运行：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/run_noise_aware_diagnostics.py --verify-only
# 复现实验必须使用新输出目录。
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/run_noise_aware_diagnostics.py --output-root outputs/noise-aware/phase2b-reproduction
```

## 下一步

优先处理减速和低速组：评估带速度约束的加速度/停止候选，再检验持续性残差能否区分真实运动变化与离群恢复。需要更细的专家移除、多训练seed及未参与开发的数据验证；本轮不依据select换掉历史推荐或调阈值。

只有4个select log、单训练seed、合成位置退化及准确航向/地图参考系。机动标签是离线启发式标签，不能当成在线输入或真实行为识别成绩。本轮只本地提交，不推送GitHub、不修改简历，不把已有KF/NIS/融合方法说成首创。

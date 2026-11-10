# V2 Phase 2A：带噪候选重建与可靠性门控

日期2026-11-10，分支`feature/noise-aware-online`。最终实现冻结于`4936130`，初始实现`d4cbce4`。已经实际重建七候选、训练四组门控并完成开发集评价与审计。

加入残差/NIS信息的门控，比仅运动或协方差特征门控有更低的三条件平均ADE；离群点条件下也优于最强单候选NIS-KF。但0.5 m高斯噪声下仍略差于普通KF。不能声称全面超越物理模型，不能视为新holdout或论文SOTA成绩。

## 实现与实验控制

依据[Phase 1结果](NOISE_AWARE_PHASE1_RESULTS.md)，执行[设计](superpowers/specs/2026-11-10-noise-aware-phase2a-design.md)与[计划](superpowers/plans/2026-11-10-noise-aware-phase2a.md)：

1. 共享观测流重建CV、α-β、KF、NIS-KF、固定初始GRU、original在线修正、map在线修正七候选；不读旧干净预测或地图sidecar生成新候选。
2. GRU从带噪历史重新前向，以带噪t0位置为网络平移原点。在线修正使用同一新增位置；地图每步以带噪当前位置重新查询，几何也转到带噪原点。缺图严格回退original。
3. 比较constant（全零输入，只学全局混合）、motion（6维）、covariance（12维）、innovation（18维）四组门控，保留七单候选及等权融合。候选API没有GT标签参数；残差/NIS按时间更新，改变未来包不会改变已发布前缀。

GRU/original/map沿用旧独立fit训练且select选择的冻结权重，**仅重算带噪输入，没有重训这些上游模型**。新增训练仅限门控。本轮包含上游模型的噪声分布偏移，不能等同于完整鲁棒训练。

| 项目 | 固定设置 |
| --- | --- |
| 实际轨迹/log | fit 352条/22log；select 256条/4log，角色间不重叠 |
| 三条件 | clean、0.5 m XY高斯噪声、该噪声叠加第4/8源时刻5 m离群点 |
| 噪声重复 | seed2027/2028/2029；clean重复结果相同，不是独立重复 |
| 观测流 | fit 3,168；select 2,304，共5,472；不同轨迹仍只有608条 |
| 时间协议 | 2Hz、5历史点、12未来点、11更新、原固定终点 |
| 网络 | 四组均18→32→7 softmax、839个名义参数；不用的输入置零 |
| 训练控制 | 单训练seed2027，同初始化及批次顺序，固定20epoch，每组700次更新 |
| 优化器 | Adam lr0.001、weight_decay0.0001、batch1024 |
| 标准化 | 只用fit统计，不向模型提供condition、真实sigma、seed、log或GT误差 |
| 选模 | 最终epoch，无早停/按select选epoch/结果后调q、R、NIS阈值 |

损失为每次更新剩余点的平均欧氏距离，各轨迹/条件/噪声seed/更新等权。相同名义参数量不等于相同有效参数量或信息量；常数对照的输入连接不携带信息。

## Select结果

ADE均值，单位m，越低越好；完整保留三条件与全部策略。

| 策略 | clean | noise_05 | noise_outlier |
| --- | ---: | ---: | ---: |
| raw_cv | 2.166529 | 4.367043 | 9.852065 |
| alpha_beta | 3.313826 | 3.531762 | 4.418871 |
| kf_cv | 2.616950 | 3.101417 | 4.722299 |
| kf_nis | 2.623073 | 3.111897 | 3.947242 |
| gru_base | 7.171525 | 11.506883 | 11.506883 |
| original | 2.718510 | 5.124232 | 9.097036 |
| map | 2.526186 | 4.801882 | 8.679424 |
| equal | 2.770023 | 4.033577 | 5.932114 |
| gate_constant | 2.565228 | 3.110364 | 4.010208 |
| gate_motion | 2.311269 | 3.163847 | 3.955041 |
| gate_covariance | 2.208481 | 3.156777 | 3.949921 |
| gate_innovation | 2.151427 | 3.144192 | 3.845871 |

完整FDE/jitter和按log/噪声seed均值保留在`summary.json`，逐轨迹记录在`per_trajectory.csv`。固定GRU在各更新中保持原预测，jitter为零，不能据此认为预测更好。

主要结论：

- **残差信息有开发增量证据。** innovation相对covariance的clean/noise_05/outlier ADE约再降2.58%/0.40%/2.63%。按log分别在4/4、3/4、4/4个select log改善；只有4log，不作显著性结论。
- **干净条件平均精度接近CV且略好。** innovation ADE 2.151427 m，比CV的2.166529 m低约0.70%；FDE 4.541759 m，CV 4.691643 m；jitter 1.111489，CV 1.264446。相对CV只在2/4个log改善，不能声称普遍或显著提升。
- **离群条件有有限改善。** 相对NIS-KF，innovation ADE约降2.57%，FDE从7.005555降到6.804962 m，jitter从3.486859降到3.318437。四个log均改善；相比motion门控ADE约降2.76%。
- **纯高斯噪声仍有退化。** innovation ADE 3.144192 m，高于KF 3.101417 m，约退化1.38%，四个log都没有赢过KF；jitter 2.411196也高于KF 2.030010。相对常数门控3.110364 m同样没有优势。
- **更复杂的候选并非更强。** map在三个条件下都好于original，但二者都差于对应强物理候选；冻结GRU对带噪历史敏感。尚未做专家移除消融，不能将门控的全部收益归于地图模块。

innovation平均权重：clean下Raw CV约51.9%、NIS-KF约17.1%、map约23.1%；离群条件下NIS-KF约73.8%、Raw CV约4.2%、map约8.0%。这是可检查的行为描述，不等于证明模型已正确识别真实机动。

**协方差解释边界：** 固定Q/R和规则时间流下，普通KF的六维协方差特征只随更新步变化，不随轨迹或噪声变化，已在全部候选特征中核对。因此covariance组的收益可能来自时间阶段信息，不能说它已经估计真实观测质量；下一阶段需要显式时间步对照。

## 验证、工程问题与复现

最终输出`outputs/noise-aware/phase2a-v1/`，运行身份：

`f529852532c5125207038f3c52305c53371ce60077e721e145cb724c9aea3fc3`

保存run/preparation/records、两角色候选特征NPZ、四门控PT、training、summary、per_trajectory及completed哈希清单。65,664条策略记录=5,472条观测流×12策略。纯CPU门控训练加评价约25.12秒，排除前面的地图加载、候选生成和候选缓存写入，不是端到端延迟。

- 618项pytest、Ruff和公开文件扫描通过。
- 文件/父数据哈希、源码身份、fit/select log隔离及fit-only标准化通过；全部门控重新前向、全部65,664条指标/摘要/平均权重重算通过。
- 每角色一条离群观测流重新执行GRU、地图查询及七候选构建，共两条抽查通过；没有把这称作上游候选全量重放。
- 与Phase 1核对21,888条共同物理策略记录，ADE/FDE/jitter最大绝对差约0.000002 m，来自float32候选缓存精度；608条clean GRU初始预测与旧预测核对，最大绝对差约0.0000153 m。
- 旧holdout冻结链仍通过；本轮没有用其误差训练/选择，也没有新增holdout评价。

实际问题：旧配置身份采用规范化内容哈希，不能拿YAML字节哈希比较，已使用旧项目同一规则。初次实验结束后，复核入口读回JSON字符串路径，触发`load_yaml`的Path类型错误；回归测试先复现，再补转换，以相同参数从头运行到`phase2a-v1`。原`phase2a`保留追溯，不作为最终推荐产物；修复前后候选/特征/标签、四门控权重/标准化和逐项指标完全一致，见本地`cross_checks.json`。没有因结果调整训练配置。

当前worktree内使用既有环境：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/run_noise_aware_gate.py --verify-only
# 重跑须给新目录，已有目录拒绝覆盖。
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/run_noise_aware_gate.py --output-root outputs/noise-aware/phase2a-reproduction
```

## 限制与下一步

这是已经用于开发的fit/select，仅4个select log、单模型训练seed；三个噪声seed不是独立训练重复。输入仍是metadata位置合成退化与静态地图，没有Camera/LiDAR/Radar特征。航向和地图参考系准确，不能称完整定位噪声鲁棒性或真实感知实验。本轮没有延迟/丢包/乱序，也没有重训上游候选。

下一阶段先做真实机动/异常观测分组诊断、显式时间步及移除神经/地图候选的对照，再考虑持续性残差特征和CA/转弯专家。还需多训练seed及未参与本轮开发的数据评价；不能再在已用holdout上调参后包装为盲测。当前属于有限开发证据，KF/NIS/在线融合都不是本项目首创。

本轮只本地提交，未推送GitHub，未修改简历。

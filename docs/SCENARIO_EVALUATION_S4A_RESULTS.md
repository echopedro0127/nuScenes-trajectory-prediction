# Phase 3A S4a：同目标修正增益、尾部误差与Miss

日期：2026-11-12。分支`feature/scenario-evaluation`，实现前HEAD为`9e0e4af`。本轮实现M2及M5相关纯指标和S2子集集成；不是S7正式全量成绩，不重新训练、选模型或改阈值。

## 1. 用什么方法解决什么问题

只看新预测误差下降，可能混入不同目标时刻或剩余区间缩短的影响。新增`evaluation/revision_metrics.py::revision_gains()`，对同一个目标列j比较前次和当前预测：

```text
k>=2，j=k+m-1，k+m<=12
G = ||P[:,k-2,j,:]-Y[:,j,:]|| - ||P[:,k-1,j,:]-Y[:,j,:]||
```

G>0表示当前更新更准确；>0.05m为有益、<−0.05m为有害，等于±0.05m及中间区间均为中性，不加epsilon改变边界。0.5/1/2s lead分别有10/9/7个有效更新对；k=1没有前次输出，不生成伪增益。

固定1s lead时，比较的是前次预测对同一目标的1.5s外推与当前的1s外推，目标位置不变。G描述已部署算法两次输出的净变化，包含观测、模型状态及时间特征变化；没有“禁止新观测”的反事实对照，不能把G单独当作新增观测的纯因果效果。

为保留S3已绑定源码，新增API放在revision_metrics.py，没有修改streaming_metrics.py。S0–S3冻结数值及其来源仍可验证。

## 2. 分母、聚合与尾部定义

- `mean_gain_m`：查询先窗口内均值，同窗口noise重复均权，再对不同窗口均权。
- beneficial/harmful/neutral比例：池化有效更新对，共用pair_count分母，计数总和必须守恒。
- `window_macro_*_fraction`：每窗口先处理重复再均权，与池化比例分名；重复噪声不能增加窗口权重。
- `median_pair_gain_m`：更新对池化中位数，明确不同于窗口均值的中位数。
- P50/P95/P99：固定lead逐查询误差池化，NumPy linear分位数；窗口均值P95另命名`p95_window_mean_error`，先合并窗口内noise重复。
- 固定lead miss：严格E>2m，等于2不算miss；分母是有效查询数。
- `online_MR2_remaining`：每个查询k，在尚未观测的原目标列k至11中，任一点误差>2m即miss；逐k给分母和比例，不把不同剩余长度平均成官方6s挑战成绩。

空组返回null/status/reason及零计数，不填0成功率。重复window/noise key、错误维度、非有限数、负位置误差等拒绝；汇总溢出不得输出Infinity。指标和比较均为float64，不改变父预测及旧指标精度。

测试明确区分逐查询P95=8.5与窗口均值P95≈3.9667的同一夹具，并验证noise重复不膨胀窗口权重、阈值等值、尾部和已观测列排除。

## 3. 真实子集核验

输入仍为S2 `preflight-v2`，identity `12e2584ecf0a0a0fd0d3e847cc9279f124d084f346ea2954f221a0e7ec9b9fab`；新增入口`scripts/check_scenario_revisions.py`。

| 范围 | 数量/结果 |
| --- | --- |
| 输入 | 32个分片，2角色×16策略，每策略746条S2邻域流 |
| 同目标增益 | 310,336个，三个lead有效更新对数10+9+7 |
| 固定lead误差 | 346,144个，查询数11+10+8 |
| remaining miss布尔标记 | 131,296个，逐查询不跨k平均 |
| 角色/条件/策略/lead汇总 | 144行；remaining汇总48组×11查询 |
| 独立标量参考 | 距离、增益、分类、分位数、窗口均权均值及remaining miss均通过 |
| 最大参考绝对差 | 1.0658141036401503e-14m；预设atol=rtol=1e-12 |

另从保存数组独立复算全部144组的类别计数、更新对中位数及窗口宏平均比例，并核验224个数组与文件哈希。

S2切片不是均衡三条件队列：fit邻域全部outlier，select有47条noise_05、326条outlier，噪声重复也不完整。后续正式评分必须按冻结全量队列汇总；本轮不能据跨条件均值推断噪声影响或据子集排名选模型。

## 4. 子集诊断：增益更大不代表预测更好

以下保留事先已有对照，展示select/outlier、1s lead的共同窗口子集；门控均为训练seed2027，未挑最优训练seed。该组326条观测流、256个窗口、4log；G有2,934个更新对，尾部/Miss有3,260个查询，分母不同。

| 策略 | 窗口均权G/m | 有益对/% | 有害对/% | 中性对/% | 逐查询P95/m | E>2m/% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw CV | 2.830578 | 66.3940 | 33.1629 | 0.4431 | 16.317866 | 74.8160 |
| NIS-KF | 0.883087 | 64.0082 | 34.1513 | 1.8405 | 5.995464 | 42.3313 |
| baseline7 | 0.853137 | 63.2924 | 35.0034 | 1.7042 | 5.873375 | 42.9755 |
| plus_ca | 0.841948 | 63.1220 | 35.0375 | 1.8405 | 5.879754 | 41.9939 |
| guarded_ca | 0.846098 | 63.2924 | 35.0375 | 1.6701 | 5.872052 | 42.2393 |

原始CV修正增益更大，同时绝对误差尾部也更差；更差的旧预测可能留下更大的改善空间。因此不能以最大G或最高有益比例单独排名，必须联合绝对误差、尾部与有害比例。

约35%的门控更新对按0.05m阈值属于有害变化；这不是35%独立驾驶事件、传感器误报率或事故率。该组包含注入离群与其他更新，还没有事件级恢复/机动分类，不能从这些比例判断具体失败原因。

没有根据本轮P95微小差异替换推荐、调整门控或修改保护规则。正式全量及三训练seed统计留S6/S7。

## 5. 审查、测试与来源

独立审查发现一个Important：源码哈希若在耗时S2验证后才采集，验证期间的文件变化可能与已导入实现不一致。先添加失败回归，再将三份S4a源码哈希提前冻结，并在验证后、发布前复查。复核确认关闭，无新增Critical/Important；未修改冻结S3来迁就本轮实现。

新增27项测试，最终完整套件**744 passed in 108.49s**，Ruff通过；公开扫描及diff检查通过。小型文件测试不依赖真实nuScenes；真实集成另使用本机冻结缓存和模型。

本轮identity：`8f008fbb2567a59ef8413a1e369b0cadb4c874f2627105c9a1f5fe164a1edcbc`。输出`outputs/scenario-evaluation/s4a-integration-v1/`含run/summary/groups/remaining_groups JSON、values.npz及completed。schema `revision_s4a_subset_v1`，明确`full_phase3a_evaluation=false`。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/check_scenario_revisions.py `
  --preflight outputs/scenario-evaluation/preflight-v2 `
  --output-root outputs/scenario-evaluation/s4a-integration-reproduction
```

复现使用新目录，已有结果拒绝覆盖。模型、数据和详细简历进展文档保留本地，不推送GitHub。

## 6. 下一步

S4b加入共用GT参考方向的纵向/横向误差和独立方向有效mask，区分沿运动方向外推偏差与横向偏差；再推进S4c额外位置误差恢复、S4d窗口事件标签。当前没有这些分解/恢复/标签结果，没有新holdout或真实感知输入实验。

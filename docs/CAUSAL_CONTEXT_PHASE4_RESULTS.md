# P4：完整 trainval 三种子消融

状态：完整三 seed 训练、val评估和延迟/统计报告已完成。训练代码锁定于 `a975c14`；下述协议在查看完整 val 结果前确定。

## 数据与协议

| split | 轨迹 | 场景 | 每策略更新 |
| --- | ---: | ---: | ---: |
| train | 25,560 | 468 | 281,160 |
| train_val | 6,621 | 176 | 72,831 |
| val | 7,201 | 136 | 79,211 |

三组场景两两不相交。固定 GRU、原始轨迹缓存和基础预测均复用已有产物；加载绑定
NPZ/CSV/metrics、token 顺序、target、模型配置及预处理。val 是已有历史固定验证集回归，
不称新盲测。target 与 neighbor 位置来自 annotation oracle，未读取或下载 sensor。

流式解析 annotation 建 SQLite 场景索引，逐场景 query；NumPy NPY 只读映射配合原格式
NPZ。加载分块验证全部语义，并流式核对 NPZ 成员与独立 NPY 完全相同。mini 的全部
缓存数组与 P2/P3 精确一致；向量化邻车 query 测试覆盖旋转、缺失与延迟。

固定 seeds=2027/2028/2029、hidden=64、batch=4096、最多20 epoch、patience=5；
AdamW lr=0.001、weight_decay=0.0001，SmoothL1 + 0.05 smoothness。
只用 train_val ADE 选 checkpoint；每个 seed 训练 original、map、neighbors、fusion
及三种 matched 宽度对照，评估再加 frozen/sliding，共九策略。

上下文模型继承并冻结同 seed 的 original fallback；宽度对照匹配 total 参数、全部参数
可训练。因此不是 trainable 参数或总训练计算量匹配，比较不能单独证明上下文的因果价值。
训练输入与标签由不同方法取得，预测函数只接收可见输入。缓存/配置/实际数组内容及模型
源码共同绑定 stage 复用身份；已完成 stage 可复用，未完成 stage 从头运行。
复用要求相同源码与训练记录中的 git revision，变更 revision 时使用新输出目录。

指标沿用 P3：只评估每次更新后的剩余轨迹，终点固定为 t0+6s；首步 jitter=0，后续只
比较相邻预测重叠的绝对未来时刻。批量九策略数值与 P3 逐条真实预测路径测试一致。

## 统计与计时

每策略报告三 seed 均值与样本标准差（ddof=1）。先对固定三 seed 的场景 sum 取平均，
以场景为配对 cluster 做2000次 bootstrap（统计随机种子0），用抽中场景的 sum/count
计算样本加权均值，报告策略自身及相对 original、对应 matched 对照的95%区间。
区间只反映固定 seed 集合下的场景不确定性；训练 seed 波动单独列示，不把相邻 update
或 seed×scene 当作独立样本。

报告检查实际压缩逐行 CSV、scene sum/count、summary、分组文件的哈希与一致性，
三 seed 九策略必须有完全相同 key 顺序；frozen/sliding 跨 seed 指标相同，漏行不丢弃。
输出按 update、remaining horizon、maneuver 分组；maneuver 仅为标签侧事后诊断。

批量 forward 耗时仅表示吞吐。单次延迟另固定随机抽256个 sample/update key（三 seed
同一集合，抽样种子2027），每策略16次 warmup 后走 P3 实际 predictor，CUDA 前后同步。
总时间包含预缓存上下文取值、view 构造、特征/传输、head 和输出，单列 feature、head
及总计 p50/p95。它不包含 metadata/sensor 冷读，也不表示传感器端到端延迟。

## 复现

在开发 worktree 执行；所有路径参数支持绝对路径，以下共享路径为本仓库布局。

```powershell
$python = (Resolve-Path ../trajectory-mvp/.venv/Scripts/python.exe).Path
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& $python scripts/prepare_scaled_context.py --trajectory-root ../../data/cache/trainval --dataroot ../../data/nuscenes --version v1.0-trainval --output-root outputs/context/p4/trainval --splits train train_val val
& $python scripts/train_scaled_interactions.py --trajectory-root ../../data/cache/trainval --context-root outputs/context/p4/trainval --prediction-root ../../outputs/trainval --dataroot ../../data/nuscenes --output-root outputs/context/p4/training --device cuda
& $python scripts/evaluate_scaled_interactions.py --trajectory-root ../../data/cache/trainval --context-root outputs/context/p4/trainval --prediction-root ../../outputs/trainval --dataroot ../../data/nuscenes --training-root outputs/context/p4/training --output-root outputs/context/p4/evaluation --device cuda
& $python scripts/report_context_ablation.py --runs outputs/context/p4/evaluation/seed-2027 outputs/context/p4/evaluation/seed-2028 outputs/context/p4/evaluation/seed-2029 --output outputs/context/p4/report
```

`--mini --epochs 1 --device cpu` 的三 seed 训练，加 mini_val 完整评估及报告已通过 smoke；
正式实验使用完整 train/train_val/val 与上述固定训练设置。原始数据、NPY/NPZ、模型和
逐行结果全部位于被忽略的 data/outputs，不进入代码提交，不推送远程。

## 工程验收与审查

批量执行与统计入口完成时，全套 pytest 最终为402通过；Ruff、公开仓库扫描及 diff 检查通过。
独立审查发现 NPY 与 NPZ 未逐成员绑定、缓存 schema/语义校验不足、stage 复用只信任
声明 provenance、训练源哈希遗漏原 Corrector。相应修复已加入：NPZ 成员流式比较、
复用原 canonical validator 的分块语义检查、实际输入/标签数组指纹、完整训练依赖哈希。
先复现失败再修复；实际 base 改变、原 Corrector 源码哈希改变均拒绝复用。

后续新增评估/统计由本代理完成逐行路径复核、P3 数值对照与 mini 三 seed 全流程。
再次请求独立复核时审查代理达到服务使用上限，未取得该轮复核意见；如后续独立审查
提出问题，仍需据此回归。此限制不替代已执行的测试或实际完整实验结果。

## 实际结果

完整三个 seed、九策略、136个 val 场景、7,201条轨迹；每策略每 seed 79,211次更新，
总计2,138,697条评估记录，失败0。402项 pytest 通过，Ruff、公开扫描、diff检查通过。

### 三 seed 指标

误差/抖动单位为米；表内是 seed 均值 ± 样本标准差（ddof=1），并非置信区间。
各策略自身的场景95% CI保留在 report.json / metrics.csv。

| 策略 | ADE | FDE | jitter | fallback更新/seed |
| --- | ---: | ---: | ---: | ---: |
| frozen | 6.297287 ± 0.000000 | 8.949827 ± 0.000000 | 0.000000 ± 0.000000 | 0 |
| sliding | 1.937731 ± 0.000000 | 4.235810 ± 0.000000 | 1.108377 ± 0.000000 | 0 |
| original | 2.328250 ± 0.012864 | 4.240271 ± 0.024515 | 1.555624 ± 0.006592 | 0 |
| map | 2.170413 ± 0.007376 | 3.983093 ± 0.012787 | 1.493390 ± 0.016681 | 0 |
| neighbors | 2.334388 ± 0.013140 | 4.225095 ± 0.025971 | 1.562015 ± 0.008086 | 6216 |
| fusion | 2.184295 ± 0.011948 | 3.970296 ± 0.003591 | 1.495179 ± 0.002922 | 0 |
| matched_map | 2.295116 ± 0.008563 | 4.173823 ± 0.007708 | 1.555123 ± 0.001518 | 0 |
| matched_neighbors | 2.297971 ± 0.005045 | 4.182251 ± 0.006561 | 1.554015 ± 0.005998 | 0 |
| matched_fusion | 2.289864 ± 0.016413 | 4.176605 ± 0.009476 | 1.545420 ± 0.012819 | 0 |

覆盖率（全体更新）：map=100.000000%、neighbors=92.152605%、either=100.000000%。

### 配对场景差值

负值表示上下文模型更低；95%区间按场景配对，条件于固定三个 seed。

| 对比 | ADE差值 [95% CI] | FDE差值 [95% CI] | jitter差值 [95% CI] |
| --- | ---: | ---: | ---: |
| map-original | -0.157837 [-0.177403, -0.137334] | -0.257178 [-0.292288, -0.224095] | -0.062234 [-0.072194, -0.052856] |
| map-matched_map | -0.124703 [-0.143528, -0.104740] | -0.190730 [-0.219252, -0.163217] | -0.061733 [-0.073579, -0.049945] |
| neighbors-original | 0.006138 [0.003130, 0.009101] | -0.015176 [-0.024616, -0.006752] | 0.006391 [0.004602, 0.008093] |
| neighbors-matched_neighbors | 0.036417 [0.031302, 0.042046] | 0.042843 [0.031354, 0.053585] | 0.008000 [0.001529, 0.014401] |
| fusion-original | -0.143955 [-0.165055, -0.121554] | -0.269974 [-0.312076, -0.230512] | -0.060445 [-0.072232, -0.049121] |
| fusion-matched_fusion | -0.105569 [-0.126184, -0.083069] | -0.206308 [-0.243199, -0.170074] | -0.050242 [-0.063252, -0.037762] |

### 实际选模进度

单元格为 best_epoch/completed_epochs；早停基于 train_val ADE，未使用 val 选模。

| 模型 | seed2027 | seed2028 | seed2029 | total/trainable参数 |
| --- | ---: | ---: | ---: | ---: |
| original | 20/20 | 20/20 | 19/20 | 4866/4866 |
| map | 15/20 | 19/20 | 17/20 | 14212/9346 |
| neighbors | 3/8 | 4/9 | 4/9 | 14468/9602 |
| fusion | 18/20 | 18/20 | 20/20 | 23108/18242 |
| matched_map | 17/20 | 19/20 | 20/20 | 14127/14127 |
| matched_neighbors | 17/20 | 17/20 | 20/20 | 14366/14366 |
| matched_fusion | 17/20 | 19/20 | 20/20 | 23070/23070 |

### 单次延迟与批量吞吐

下表是三个 seed 各自百分位的最小–最大范围（毫秒），没有把百分位当成均值。
每策略每 seed 固定256次测量、16次 warmup，GPU同步且 batch=1；逐 seed 与分段详情
在 report/latency.csv、evaluation/seed-*/latency*.csv/json。

| 策略 | 总计p50范围/ms | 总计p95范围/ms | 批量forward耗时范围/s |
| --- | ---: | ---: | ---: |
| frozen | 0.7352–0.7536 | 1.2835–1.8081 | 0.0036–0.0065 |
| sliding | 1.6930–1.8785 | 3.1364–4.5312 | 0.0416–0.0521 |
| original | 1.6269–1.7125 | 2.3477–5.3970 | 0.3023–0.4067 |
| map | 3.4348–3.5803 | 4.3132–5.3316 | 0.6820–0.7695 |
| neighbors | 3.3568–3.6221 | 4.2944–4.6130 | 0.3857–0.4984 |
| fusion | 4.1272–4.2316 | 5.1133–5.2619 | 0.7095–0.8056 |
| matched_map | 1.5726–1.6384 | 2.3348–2.4059 | 0.1206–0.1475 |
| matched_neighbors | 1.6151–1.6912 | 2.4018–2.4741 | 0.1044–0.1181 |
| matched_fusion | 1.6240–1.6285 | 2.3888–2.5311 | 0.1160–0.1274 |

设备：NVIDIA GeForce RTX 3060 Laptop GPU，PyTorch 2.6.0+cu124。
批量耗时仅计 forward 路径；与单次缓存输入预测延迟口径不同，不包括离线准备或指标/CSV写出。

### maneuver 分组诊断

下表为三 seed 平均 ADE，所有预先定义的标签均列出；标签可重叠，不应相加为总样本。

| 标签 | 更新数/seed | sliding | original | map | neighbors | fusion |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| acceleration | 4686 | 3.877075 | 3.826590 | 3.669003 | 3.812035 | 3.689479 |
| deceleration | 3498 | 3.789476 | 4.361835 | 3.957080 | 4.388488 | 3.961867 |
| high_curvature | 8833 | 3.518293 | 3.780212 | 3.467818 | 3.778144 | 3.499617 |
| low_speed | 11902 | 1.472415 | 1.615644 | 1.610829 | 1.621757 | 1.623514 |
| straight | 61182 | 1.674521 | 2.084273 | 1.947891 | 2.091739 | 1.958486 |
| turn | 9196 | 2.170723 | 2.556810 | 2.404686 | 2.561996 | 2.423231 |

### 产物与哈希

完整来源在训练 metadata、每 seed summary、各 split manifest 与 report.json。

| 产物/字段 | 值 / SHA256 |
| --- | --- |
| train/map NPZ | `8553cfb72bdb2e80a1d3b88ac70488bbf6a2bb0db5b2dfed0851e22f56ec37b0` |
| train/neighbors NPZ | `d2e416e7cefd1e4e761037e0c836a394eacc24f42322370f99582e2a49be34b0` |
| train_val/map NPZ | `7859376a0d4bff9f50c3f92a78723d4bb8ba56bb07a3038d145f6fe342a49f64` |
| train_val/neighbors NPZ | `a0432bb2c4b4b217f16b4886aabf659b06e2e9ba1a8399d4c598d0cca0040578` |
| val/map NPZ | `f48691539575842e751aab5c548a10206bbea107ac7dfe9dfc16cd2a419bd8da` |
| val/neighbors NPZ | `33c2085c197ecff16dc5dca0385bae27939152022b5185ecebf4f603cd4fb802` |
| seed2027 best.pt | `a7196688ae3cc30c45467df85f59f0173e5d2520a9d0059f6bceace8be44bc93` |
| seed2028 best.pt | `690b61f160b229f20b4c56c554c82f4cd79380473ad786b2147ac00beea012ce` |
| seed2029 best.pt | `8fb800ae3ed70e2fd478e1ed86ce74443dca563d7991604a476ba4a5b9fe027d` |
| 固定GRU model_name | `gru` |
| 固定GRU checkpoint_sha256 | `ec253a446d0bca3205221a27777442ecffd038dac11add3dedb949cb3b947ae3` |
| 固定GRU model_config_sha256 | `88d7d5b77bcdc950b53345720f04a8bc132812c182574de447e883ac74974d24` |
| 固定GRU data_config_sha256 | `b9297cd438e6cbea6dd7083d7bc48b28763994a633e0858536d93ea0a1c69340` |
| 固定GRU history_seconds | `2.0` |
| 固定GRU history_steps | `5` |

报告目录：outputs/context/p4/report；逐行结果/分组/延迟：outputs/context/p4/evaluation；
全部checkpoint、逐epoch日志：outputs/context/p4/training。所有模型和数据产物均未提交。

### 结果解读

地图模型相对 original 的 ADE 降低0.157837m（约6.78%），对应配对场景95% CI
为[-0.177403,-0.137334]m；相对 matched_map 也降低0.124703m，区间不跨0。
FDE、jitter 相对两种对照也降低。这支持在本固定设置与历史验证集上的地图增益，
并不能消除继承 original 预训练及 trainable 参数不匹配的归因限制。

融合模型相对 original 的 ADE 降低0.143955m，95% CI为[-0.165055,-0.121554]m，
相对 matched_fusion 也有下降；但其 ADE 2.184295m 高于 map 的2.170413m，
FDE略低、jitter略高。本次没有证明在地图上继续加入邻车会带来额外收益。

neighbors 相对 original 的 ADE 增加0.006138m，95% CI为[0.003130,0.009101]m，
jitter也增加；FDE略降。相对 matched_neighbors 的三个指标均更高。因此保留退化
结果，不把 annotation neighbor oracle 的较高覆盖率解释成有效信息或真实检测性能。

sliding 恒速基线 ADE=1.937731m、jitter=1.108377m，仍优于全部 learned 模型；
地图/融合仅在 FDE 上优于 sliding。本次不能声称 learned 方案整体优于滑动恒速基线。
地图模型的 low_speed ADE=1.610829m，也高于 matched_map=1.569315m，
说明总体提升不能外推到全部子组。分组存在标签重叠，仅作事后诊断；阈值未据 val 调整。

P4 工程与完整实验已完成；下一阶段 P5 是 mini Radar 与位置退化对照，需另行细化
计划。P6门控仍取决于对应场景中的实证价值，不因本次地图结果自动推进其他模态。

# P6：地图查询包质量门控完整实验

记录日期：2026-10-21。开发分支 `feature/causal-context-online`，沿用独立 worktree。
设计与计划分别见 [设计](superpowers/specs/2026-10-20-causal-context-phase6-design.md)、
[计划](superpowers/plans/2026-10-20-causal-context-phase6.md)。

## 结论与限制

冻结 P4 original/map 和固定 GRU，仅训练 161 参数质量门控，以 CV 为混合锚点。
完整 val 六条件等权、三个固定 seed 平均 ADE：学习门控 **1.792485 m**、固定门控
1.913447 m、CV 1.937731 m、ungated_map 2.256548 m。学习门控相对 CV 平均减少
0.145246 m（约7.50%）；clean 从1.937731降至1.723315 m（11.07%），1秒延迟为
1.839944 m（5.05%）。这说明在本协议中，按观测状态和候选分歧选择地图候选比例有价值。

准确度增益伴随稳定性代价：学习门控 jitter 六条件平均1.181381 m，高于CV的
1.108377 m；所有非missing条件均高于CV，500ms/1000ms/stale还高于固定门控。
clean学习门控−CV jitter差0.040679 m，场景95% CI=[0.025426,0.055446]，故不能
声称全面优于CV或提高了轨迹稳定性。filtered的jitter更低0.812519 m，但ADE更高
2.846076 m；固定参数滤波只是预定强对照，未针对本阶段干净位置观测另调参。

本实验只退化**局部地图查询结果包**；位置观测始终干净，使用已释放的目标标注位置，
继承P4 oracle定位限制。地图几何是静态数据，age表示按过去位置选择的局部查询包时龄。
不将这些数字解释为真实地图服务网络、定位噪声或Radar系统性能。历史val已用于此前
阶段固定回归，本次未据val调门控结构/超参数，但它仍不是新的盲测测试集。

## 数据、训练与候选来源

完整 train 25,560轨迹/468场景，train_val 6,621轨迹/176场景，val 7,201轨迹/136场景，
场景互斥。三个种子2027/2028/2029分别继承同种子的P4 checkpoint，只训练gate；
train六条件混合训练，train_val ADE选模。hidden16、batch4096、max20epoch、patience5，
AdamW lr0.001、weight_decay0.0001；masked SmoothL1 + 0.05修正平滑损失。

| seed | best epoch | completed epochs | train_val ADE/m | trainable params |
| --- | --- | --- | --- | --- |
| 2027 | 20 | 20 | 1.796018071 | 161 |
| 2028 | 12 | 17 | 1.798810745 | 161 |
| 2029 | 7 | 12 | 1.806595566 | 161 |

gate输入8维为map点覆盖比例、clip(age/2)、当前观测与GRU位置偏差、速度偏差、速度、
过去差分加速度、剩余horizon比例，以及map/CV候选在剩余horizon上的预测分歧。
最后一项只读冻结候选预测；未来真实轨迹只通过独立labels接口进入损失/评估。
固定gate=.5·exp(−age_s)，学习gate=sigmoid(MLP8→16→1)，初始末层bias=−2。
预测=CV+g·(map预测−CV)，是新的CV锚点混合器，不能等同于对原Corrector缩放offset。

每0.5秒更新一次，共11次；剩余终点固定为t0+6秒。source=t0+k·0.5秒，available=
source+delay，只选已交付最新包，不虚构t0地图包，不用当前位置重新选择旧包几何。
条件：clean、500ms/1000ms延迟、dropout丢第4/5/6包、missing全部缺失、stale仅第1包。
缺包、无map点或age>2秒时，ungated_map回退original，两种gate强制为0、预测精确等于CV。
过期包仍保留source/availability/age诊断，所有失效行参与评估。

生产加载校验轨迹、固定GRU、P4 checkpoint、全部父地图NPZ/NPY成员及metadata哈希；
缓存绑定schema、配置、源码、场景和实际数组。父时间线必须逐项相等，并分批重新前向
冻结P4模型，对照original/map候选，防止用真实future伪造候选后重算quality的自洽绕过。
报告重读完整缓存并再次验证；所有逐行指标、权重、组统计和恢复记录均重新计算核对。

clean val的CV/original/map三项指标与对应P4产物最大差小于1e−15；冻结模型来源未改变。

## 六条件七策略完整指标

每格为三个固定seed的均值±样本标准差（ddof=1），单位m；不是独立样本置信区间。
每条件每策略79,211更新，七策略×六条件×三seed合计**9,980,586行**；失败0。
ADE/FDE只评估尚未释放的剩余horizon；jitter比较连续更新的相同绝对预测时间，首步0。

| condition | strategy | ADE mean ± std | FDE mean ± std | jitter mean ± std |
| --- | --- | --- | --- | --- |
| clean | frozen | 6.297287 ± 0.000000 | 8.949827 ± 0.000000 | 0.000000 ± 0.000000 |
| clean | sliding | 1.937731 ± 0.000000 | 4.235810 ± 0.000000 | 1.108377 ± 0.000000 |
| clean | filtered | 2.846076 ± 0.000000 | 5.564660 ± 0.000000 | 0.812519 ± 0.000000 |
| clean | original | 2.328250 ± 0.012864 | 4.240271 ± 0.024515 | 1.555624 ± 0.006592 |
| clean | ungated_map | 2.170413 ± 0.007376 | 3.983093 ± 0.012787 | 1.493390 ± 0.016681 |
| clean | fixed_gate | 1.922040 ± 0.004056 | 3.918236 ± 0.005386 | 1.256448 ± 0.007032 |
| clean | learned_gate | 1.723315 ± 0.005751 | 3.646701 ± 0.012002 | 1.149056 ± 0.008005 |
| delay_500ms | frozen | 6.297287 ± 0.000000 | 8.949827 ± 0.000000 | 0.000000 ± 0.000000 |
| delay_500ms | sliding | 1.937731 ± 0.000000 | 4.235810 ± 0.000000 | 1.108377 ± 0.000000 |
| delay_500ms | filtered | 2.846076 ± 0.000000 | 5.564660 ± 0.000000 | 0.812519 ± 0.000000 |
| delay_500ms | original | 2.328250 ± 0.012864 | 4.240271 ± 0.024515 | 1.555624 ± 0.006592 |
| delay_500ms | ungated_map | 2.225312 ± 0.010913 | 3.994978 ± 0.015066 | 1.568398 ± 0.014273 |
| delay_500ms | fixed_gate | 1.921504 ± 0.003180 | 4.067504 ± 0.003625 | 1.240036 ± 0.003039 |
| delay_500ms | learned_gate | 1.788836 ± 0.005345 | 3.818537 ± 0.011707 | 1.264908 ± 0.007552 |
| delay_1000ms | frozen | 6.297287 ± 0.000000 | 8.949827 ± 0.000000 | 0.000000 ± 0.000000 |
| delay_1000ms | sliding | 1.937731 ± 0.000000 | 4.235810 ± 0.000000 | 1.108377 ± 0.000000 |
| delay_1000ms | filtered | 2.846076 ± 0.000000 | 5.564660 ± 0.000000 | 0.812519 ± 0.000000 |
| delay_1000ms | original | 2.328250 ± 0.012864 | 4.240271 ± 0.024515 | 1.555624 ± 0.006592 |
| delay_1000ms | ungated_map | 2.323348 ± 0.017965 | 4.052791 ± 0.019711 | 1.618323 ± 0.012464 |
| delay_1000ms | fixed_gate | 1.926254 ± 0.002047 | 4.150731 ± 0.002198 | 1.190964 ± 0.001366 |
| delay_1000ms | learned_gate | 1.839944 ± 0.004599 | 3.950234 ± 0.009689 | 1.243370 ± 0.006641 |
| dropout | frozen | 6.297287 ± 0.000000 | 8.949827 ± 0.000000 | 0.000000 ± 0.000000 |
| dropout | sliding | 1.937731 ± 0.000000 | 4.235810 ± 0.000000 | 1.108377 ± 0.000000 |
| dropout | filtered | 2.846076 ± 0.000000 | 5.564660 ± 0.000000 | 0.812519 ± 0.000000 |
| dropout | original | 2.328250 ± 0.012864 | 4.240271 ± 0.024515 | 1.555624 ± 0.006592 |
| dropout | ungated_map | 2.195883 ± 0.007699 | 3.954251 ± 0.010061 | 1.570393 ± 0.013281 |
| dropout | fixed_gate | 1.931303 ± 0.002899 | 3.965271 ± 0.003446 | 1.259690 ± 0.004372 |
| dropout | learned_gate | 1.722902 ± 0.005003 | 3.639320 ± 0.011487 | 1.162116 ± 0.007505 |
| missing | frozen | 6.297287 ± 0.000000 | 8.949827 ± 0.000000 | 0.000000 ± 0.000000 |
| missing | sliding | 1.937731 ± 0.000000 | 4.235810 ± 0.000000 | 1.108377 ± 0.000000 |
| missing | filtered | 2.846076 ± 0.000000 | 5.564660 ± 0.000000 | 0.812519 ± 0.000000 |
| missing | original | 2.328250 ± 0.012864 | 4.240271 ± 0.024515 | 1.555624 ± 0.006592 |
| missing | ungated_map | 2.328250 ± 0.012864 | 4.240271 ± 0.024515 | 1.555624 ± 0.006592 |
| missing | fixed_gate | 1.937731 ± 0.000000 | 4.235810 ± 0.000000 | 1.108377 ± 0.000000 |
| missing | learned_gate | 1.937731 ± 0.000000 | 4.235810 ± 0.000000 | 1.108377 ± 0.000000 |
| stale | frozen | 6.297287 ± 0.000000 | 8.949827 ± 0.000000 | 0.000000 ± 0.000000 |
| stale | sliding | 1.937731 ± 0.000000 | 4.235810 ± 0.000000 | 1.108377 ± 0.000000 |
| stale | filtered | 2.846076 ± 0.000000 | 5.564660 ± 0.000000 | 0.812519 ± 0.000000 |
| stale | original | 2.328250 ± 0.012864 | 4.240271 ± 0.024515 | 1.555624 ± 0.006592 |
| stale | ungated_map | 2.296079 ± 0.008680 | 4.057144 ± 0.011024 | 1.681200 ± 0.001138 |
| stale | fixed_gate | 1.841848 ± 0.000768 | 3.998075 ± 0.000215 | 1.102985 ± 0.000846 |
| stale | learned_gate | 1.742183 ± 0.003642 | 3.697872 ± 0.007875 | 1.160461 ± 0.003253 |

## 场景配对不确定性

先对固定三个seed按同scene平均metric sum，再按136个scene整体有放回抽样2000次，
bootstrap seed0，按实际更新count加权；相同scene/key对照，未把三次seed当独立场景。
以下为learned减对照的ADE差（负值更好）。missing−CV/fixed区间为[0,0]是精确回退的结果。
完整ADE/FDE/jitter共72条对比保存在 `outputs/context/p6/report/paired_scene_intervals.json`。

| condition | contrast | ADE delta/m | scene bootstrap 95% CI/m |
| --- | --- | --- | --- |
| clean | learned−sliding | -0.214416 | [-0.263619, -0.168983] |
| clean | learned−filtered | -1.122760 | [-1.260096, -0.991800] |
| clean | learned−fixed_gate | -0.198725 | [-0.216162, -0.182701] |
| clean | learned−ungated_map | -0.447098 | [-0.469976, -0.425222] |
| delay_500ms | learned−sliding | -0.148895 | [-0.186233, -0.113686] |
| delay_500ms | learned−filtered | -1.057240 | [-1.182861, -0.935041] |
| delay_500ms | learned−fixed_gate | -0.132668 | [-0.150209, -0.116365] |
| delay_500ms | learned−ungated_map | -0.436476 | [-0.464894, -0.408552] |
| delay_1000ms | learned−sliding | -0.097787 | [-0.125859, -0.071698] |
| delay_1000ms | learned−filtered | -1.006132 | [-1.123750, -0.893001] |
| delay_1000ms | learned−fixed_gate | -0.086310 | [-0.101788, -0.071443] |
| delay_1000ms | learned−ungated_map | -0.483404 | [-0.522124, -0.445620] |
| dropout | learned−sliding | -0.214829 | [-0.264925, -0.168394] |
| dropout | learned−filtered | -1.123174 | [-1.262147, -0.990353] |
| dropout | learned−fixed_gate | -0.208401 | [-0.230367, -0.188020] |
| dropout | learned−ungated_map | -0.472981 | [-0.495096, -0.451967] |
| missing | learned−sliding | 0.000000 | [0.000000, 0.000000] |
| missing | learned−filtered | -0.908345 | [-1.005054, -0.814602] |
| missing | learned−fixed_gate | 0.000000 | [0.000000, 0.000000] |
| missing | learned−ungated_map | -0.390519 | [-0.445834, -0.333732] |
| stale | learned−sliding | -0.195548 | [-0.240554, -0.154663] |
| stale | learned−filtered | -1.103893 | [-1.239642, -0.974727] |
| stale | learned−fixed_gate | -0.099665 | [-0.128180, -0.072743] |
| stale | learned−ungated_map | -0.553896 | [-0.581140, -0.525276] |

## 硬回退与门控分布

以下计数是每seed、每策略；三seed使用同包时序和map覆盖，计数相同。gate均值与p95列
先各seed计算再平均。完全缺包两种门控全部指标精确等于CV。stale第6次起包age2.5秒，
后6次强制回退；切换首步jitter还受上一次非CV预测影响，不要求当步jitter也等于CV。

| condition | updates | hard fallback | missing | expired | learned mean g | learned p95 g |
| --- | --- | --- | --- | --- | --- | --- |
| clean | 79211 | 0 | 0 | 0 | 0.391346 | 0.924538 |
| delay_500ms | 79211 | 7201 | 7201 | 0 | 0.306423 | 0.874028 |
| delay_1000ms | 79211 | 14402 | 14402 | 0 | 0.226547 | 0.784262 |
| dropout | 79211 | 0 | 0 | 0 | 0.385980 | 0.924536 |
| missing | 79211 | 79211 | 79211 | 0 | 0.000000 | 0.000000 |
| stale | 79211 | 43206 | 0 | 43206 | 0.302779 | 0.925187 |

dropout第7包开始恢复，定义为相对同策略clean的ADE差<=0.25m连续两次，首个窗口起点
距第7次的秒数；未恢复按2秒右删失。本实验全部策略/样本/seed恢复时间均0秒、删失0。
这是因为策略没有跨包状态，第7包恢复后候选与quality立即等于clean，是协议的构造性质，
不能作为学习到快速恢复能力的证据。

## 延迟与候选准备耗时

固定256个update key（选择seed2027）、每策略16次warmup、batch1、CUDA前后同步。
范围仅为mmap候选/quality读取、传输、gate/混合与返回；**不含地图查询和P4候选模型
forward**，也不含冷缓存加载，不能称完整在线系统端到端延迟。设备RTX3060 Laptop，
PyTorch2.6.0+cu124。单次原始计时、key和范围字段均保存并核对。

| seed | fixed p50/ms | fixed p95/ms | learned p50/ms | learned p95/ms |
| --- | --- | --- | --- | --- |
| 2027 | 1.021400 | 6.713925 | 0.936150 | 1.475650 |
| 2028 | 1.188750 | 1.636500 | 1.184700 | 1.502775 |
| 2029 | 1.612450 | 2.419050 | 1.157800 | 1.749125 |

首seed评估与后续训练曾并行，fixed p95存在较大尾延迟；这些是该次运行的原始记录，
不将学习gate比固定gate更快的表面差异当模型效率结论。GPU同步不消除背景负载干扰。
批量预测+metric耗时另记summary，不能拿批量吞吐代替上述单update延迟。

以下为离线生成六条件候选并写入mmap的实测时间，包含冻结original/map前向与候选
quality生成；不含事前父缓存/metadata完整校验和事后manifest校验。吞吐是离线处理量，
不是在线服务吞吐；完整原始来源保存在各cache manifest。

| seed | split | candidate preparation/s | candidate updates/s |
| --- | --- | --- | --- |
| 2027 | train | 12.840 | 131382.1 |
| 2027 | train_val | 4.255 | 102710.7 |
| 2027 | val | 5.183 | 91688.5 |
| 2028 | train | 11.967 | 140964.0 |
| 2028 | train_val | 3.710 | 117795.0 |
| 2028 | val | 4.693 | 101279.0 |
| 2029 | train | 12.752 | 132288.2 |
| 2029 | train_val | 5.527 | 79060.0 |
| 2029 | val | 5.092 | 93336.6 |

## 复现、来源与验收

```powershell
$python = (Resolve-Path ../trajectory-mvp/.venv/Scripts/python.exe).Path
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& $python scripts/prepare_quality_gate.py --trajectory-root ../../data/cache/trainval --context-root outputs/context/p4/trainval --prediction-root ../../outputs/trainval --dataroot ../../data/nuscenes --p4-training-root outputs/context/p4/training --output-root outputs/context/p6/cache --device cuda
& $python scripts/train_quality_gate.py --cache-root outputs/context/p6/cache --output-root outputs/context/p6/training --device cuda
& $python scripts/evaluate_quality_gate.py --cache-root outputs/context/p6/cache --training-root outputs/context/p6/training --output-root outputs/context/p6/evaluation --device cuda
& $python scripts/report_quality_gate.py --evaluation-root outputs/context/p6/evaluation --output-root outputs/context/p6/report --device cuda
& $python -m pytest -q --tb=short --basetemp (Join-Path (Get-Location) ('outputs/pytest-' + [guid]::NewGuid().ToString('N')))
& $python -m ruff check src scripts tests
& $python scripts/check_public_repo.py
git diff --check
```

报告重放须选择与原评估相同的计算设备。本次CUDA候选门控与CPU重放最多约8.1e−6m
浮点差异，曾触发严格1e−6逐行阈值；已修复报告强制CPU的问题，保留原阈值和原评估。
逐行CSV使用round-trip精度解析，避免默认解析器对coverage产生约5e−17的变化而破坏
原始诊断哈希；非整覆盖率fixture已先复现失败再验证修复，未放宽任何哈希校验。
缓存源码/内容身份变化会拒绝复用；需要新实验输出目录，不覆盖旧P4/P5来源。

| seed | gate best.pt SHA256 | evaluation summary.json SHA256 |
| --- | --- | --- |
| 2027 | 453215fb37c1677220985a22aed1eed623bed92b7ca11d5301d9360f788334e0 | 8b7f1493ad14876381e6bb968bc39191151ba0b275d047fd43800b310bb53f5c |
| 2028 | ea5becc2ed1dea97e10daa9780e0dcbdb7080485ea1da7dc3728d4761ea41bb7 | 2b3be856f0032486f7fa87298e3df85c7ac2a96b77ba34d0c4240e4bbb8e4bc4 |
| 2029 | e1bdb9f4d4c026f2885cc856aba390dd161f785e9399759d641a0f958ab03db0 | 45002c43f2c443568abb511e9474299e66dd257aac4f8af60d34113165c18ad3 |

固定GRU checkpoint SHA256：`ec253a446d0bca3205221a27777442ecffd038dac11add3dedb949cb3b947ae3`。
各cache manifest还绑定P4 checkpoint、轨迹/map缓存、GRU预测、metadata与实际执行源码。
原始数据、候选、模型和压缩逐行产物均在gitignore排除的outputs/context/p6下，不提交。

独立只读审查覆盖包时序、缓存来源、训练/选模隔离与复用、评估/恢复、流式统计和延迟。
审查发现的候选伪造、时间线平移、父NPY未校验、报告契约未绑定和表格缺列问题均已修复；
另独立核对三seed实际小表、P4 parity和硬回退，未把小表审查称为全行审计。

验收状态：三个seed共9,980,586行已完成生产来源验证与逐行重算审计，
42个condition×strategy汇总、72个场景配对区间及全部分组/恢复/延迟校验通过。
434项pytest通过；Ruff、公开仓库扫描和diff检查通过。
报告report.json SHA256：`9b6f43c8153f95bd58879e4a442fe36d71f6e0f3138b752f80186adddf1ffa4c`。
本地实现提交4c94fd6、d7aef75；审计回归修复a937bda、e2727f8。
P7仍可选；现有证据不自动支持新增Camera/LiDAR训练或下载完整sensor。

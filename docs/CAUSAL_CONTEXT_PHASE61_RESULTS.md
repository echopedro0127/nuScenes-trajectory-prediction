# P6.1：门控归因、短时收缩与跨更新一致性完整实验

记录日期2026-10-21，分支feature/causal-context-online，基线af9e45c。
设计见 [spec](superpowers/specs/2026-10-21-causal-context-phase61-design.md)，
施工计划见 [plan](superpowers/plans/2026-10-21-causal-context-phase61.md)。

## 实际结论

固定三seed、六条件等权平均，P6 ADE/FDE/jitter为1.792485/3.831412/1.181381m。
train_val提前选定的准确度方案为consistent_002+tau1.5短时收缩，val指标为
**1.788697/3.829622/1.175611m**。ADE降低0.003788m（约0.21%），属于小幅改进。
稳定性方案为consistent_010、不再收缩，val为**1.789946/3.839947/1.164413m**；
ADE降低0.002540m，jitter降低0.016968m（约1.44%），FDE增加0.008534m。
稳定性方案满足预设全条件与clean ADE最多增加1%的约束，但不能称所有指标同时改善。

与CV比较，两个方案的jitter仍高0.067234/0.056037m；本阶段没有消除CV稳定性优势。
对预算匹配的finetune_0，稳定性方案ADE只改善0.000068m，主要变化是jitter降低
0.018452m，同时FDE增加0.013474m。额外训练本身贡献了部分收益，不能把P6到新方案
的全部ADE下降归因于一致性损失。准确度方案的FDE也没有优于finetune_0。

归因对照：CV+original学习门控全条件ADE1.821912m，已经优于CV的1.937731m；
使用地图候选的P6 gate再改善到1.792485m。因此CV混合本身贡献显著，地图提供额外
价值。clean original_gate−P6 ADE差0.041917m，场景区间[0.030630,0.054793]m。
两种gate头均161参数，但冻结候选编码器容量不同，此对比不能等同于严格只更换地图信息。

简单基线也不可忽略：train_val选的常数g在三个seed均为0.2，ADE1.900254m；train拟合
时长/包龄查表为1.807564m，距P6仅0.015079m。当前质量MLP仍有收益，但该结果不支持
仅通过扩大gate规模就能取得很大增益的判断。查表55格中每seed实际覆盖33格，其余0。

## 实验协议与冻结选择

沿用完整train25560轨迹/468场景、train_val6621/176、val7201/136；三seed2027/2028/2029。
冻结全部GRU/P4候选与P6源码/缓存。位置仍为干净已释放目标标注，地图包仅模拟交付
延迟、缺失、丢第4/5/6包、仅首包；并非真实定位/跟踪噪声实验。历史val持续作为固定
回归集，不称新的盲测。训练、校准与超参数选择均未使用本轮val结果。

完整十二策略×六条件×三seed×7201轨迹×11更新，共**17,109,576条指标记录**，每seed
5,703,192条。它们是同一组轨迹在不同条件/策略/更新上的重复记录，不是独立样本数。
逐更新指标为metrics.npz中[condition,sample,update,metric] float64；权重为同key float32。
ID、绝对时间和包诊断通过shape、顺序、sample-key hash及完整P6父缓存契约绑定。

选择顺序：训练全部完成→train_val三seed共同确定cap和一致性模型→选准确度/稳定性
方案→发布selection.json→运行val。准确度按全条件ADE最小；稳定性在全条件及clean
ADE<=1.01×P6时，选全条件jitter最小；所有tie规则固定，val不重新选择。
既有四对照若获选不当作P6.1新方法成功，报告分别记录train_val与固定val的目标成败。

冻结参数：short_gate tau=1.5；consistent_short采用lambda=.02、tau1.5；accuracy为
consistent_short，stable为lambda=.1的consistent_010。selection SHA256：
`3720fdeffbd13c3339776dade31e5e7b819ac33e7cceed2ef6a03fcb19aa07a7`。

## 十二策略

- sliding、ungated_map、p6_fixed、p6_learned：原P6四个冻结对照。
- constant_gate：train_val在0:.05:1选择标量权重，乘原硬有效mask。
- time_age_gate：train上每update×age bin拟合ADE网格最优值；非train_val查表。
- original_gate：相同161参数头、从头20epoch/patience5；原候选质量分歧相应重算。
- short_gate：P6权重乘cap=clip((remaining_seconds−.5)/tau,0,1)，tau0为恒1。
- finetune_0、consistent_002、consistent_010：同一P6初始化、同seed/shuffle、10epoch/
  patience3，lambda分别0/.02/.1，其他优化器/损失与P6相同。
- consistent_short：仅train_val三seed平均选择一致性模型与tau组合。

训练loss=masked SmoothL1+.05预测内部修正平滑+lambda跨更新修订损失。修订项比较
同轨迹/同条件当前与上一次预测，在当前仍未释放的同一绝对未来时间上计算L2，双方
可导；首步无配对，不跨样本边界。pair项不读真实未来标签，监督标签仍经独立接口。
无效包仍强制gate=0，混合精确CV；过期切换首步jitter包含上次有效预测的修订，不能
要求当步jitter也等于CV。missing全部指标精确CV，stale后六步ADE/FDE精确CV。
所有策略无状态，dropout恢复0秒仍是第7包回到clean输入的构造性质。

## 全条件平均与seed标准差

下表先每seed对六条件等权平均，再计算三个seed均值±样本std(ddof1)，单位m。

| strategy | ADE mean ± std | FDE mean ± std | jitter mean ± std |
| --- | --- | --- | --- |
| sliding | 1.937731 ± 0.000000 | 4.235810 ± 0.000000 | 1.108377 ± 0.000000 |
| ungated_map | 2.256548 ± 0.009948 | 4.047088 ± 0.015275 | 1.581221 ± 0.009324 |
| p6_fixed | 1.913447 ± 0.002135 | 4.055938 ± 0.002453 | 1.193083 ± 0.002539 |
| p6_learned | 1.792485 ± 0.004053 | 3.831412 ± 0.008786 | 1.181381 ± 0.005486 |
| constant_gate | 1.900254 ± 0.001461 | 4.108686 ± 0.001519 | 1.155443 ± 0.001224 |
| time_age_gate | 1.807564 ± 0.003207 | 3.871015 ± 0.006325 | 1.181725 ± 0.003632 |
| original_gate | 1.821912 ± 0.004575 | 3.913723 ± 0.009866 | 1.158237 ± 0.002824 |
| short_gate | 1.790584 ± 0.003578 | 3.831105 ± 0.008266 | 1.177959 ± 0.005386 |
| finetune_0 | 1.790014 ± 0.003715 | 3.826473 ± 0.005966 | 1.182866 ± 0.002572 |
| consistent_002 | 1.789577 ± 0.003661 | 3.829240 ± 0.005868 | 1.178192 ± 0.003019 |
| consistent_010 | 1.789946 ± 0.003690 | 3.839947 ± 0.007148 | 1.164413 ± 0.003153 |
| consistent_short | 1.788697 ± 0.003123 | 3.829622 ± 0.005396 | 1.175611 ± 0.003208 |

## clean指标

| strategy | ADE mean ± std | FDE mean ± std | jitter mean ± std |
| --- | --- | --- | --- |
| sliding | 1.937731 ± 0.000000 | 4.235810 ± 0.000000 | 1.108377 ± 0.000000 |
| ungated_map | 2.170413 ± 0.007376 | 3.983093 ± 0.012787 | 1.493390 ± 0.016681 |
| p6_fixed | 1.922040 ± 0.004056 | 3.918236 ± 0.005386 | 1.256448 ± 0.007032 |
| p6_learned | 1.723315 ± 0.005751 | 3.646701 ± 0.012002 | 1.149056 ± 0.008005 |
| constant_gate | 1.882792 ± 0.001788 | 4.056418 ± 0.001952 | 1.153041 ± 0.002239 |
| time_age_gate | 1.741973 ± 0.004023 | 3.698875 ± 0.007027 | 1.140755 ± 0.004285 |
| original_gate | 1.765233 ± 0.006076 | 3.762920 ± 0.013161 | 1.133443 ± 0.003396 |
| short_gate | 1.720174 ± 0.005171 | 3.646045 ± 0.011269 | 1.143628 ± 0.007635 |
| finetune_0 | 1.720351 ± 0.004878 | 3.639624 ± 0.009140 | 1.153193 ± 0.002538 |
| consistent_002 | 1.719576 ± 0.004935 | 3.642030 ± 0.009012 | 1.148181 ± 0.002728 |
| consistent_010 | 1.719503 ± 0.005344 | 3.652942 ± 0.010464 | 1.132983 ± 0.003458 |
| consistent_short | 1.717765 ± 0.004202 | 3.642382 ± 0.008267 | 1.143711 ± 0.002816 |

## 预先冻结两方案的六条件回归

| condition | strategy | ADE/m | FDE/m | jitter/m |
| --- | --- | --- | --- | --- |
| clean | p6_learned | 1.723315 | 3.646701 | 1.149056 |
| clean | consistent_short | 1.717765 | 3.642382 | 1.143711 |
| clean | consistent_010 | 1.719503 | 3.652942 | 1.132983 |
| delay_500ms | p6_learned | 1.788836 | 3.818537 | 1.264908 |
| delay_500ms | consistent_short | 1.783833 | 3.815289 | 1.256922 |
| delay_500ms | consistent_010 | 1.785431 | 3.828258 | 1.241979 |
| delay_1000ms | p6_learned | 1.839944 | 3.950234 | 1.243370 |
| delay_1000ms | consistent_short | 1.835423 | 3.949998 | 1.234063 |
| delay_1000ms | consistent_010 | 1.836722 | 3.964178 | 1.217930 |
| dropout | p6_learned | 1.722902 | 3.639320 | 1.162116 |
| dropout | consistent_short | 1.717078 | 3.636829 | 1.154441 |
| dropout | consistent_010 | 1.718832 | 3.650846 | 1.140871 |
| missing | p6_learned | 1.937731 | 4.235810 | 1.108377 |
| missing | consistent_short | 1.937731 | 4.235810 | 1.108377 |
| missing | consistent_010 | 1.937731 | 4.235810 | 1.108377 |
| stale | p6_learned | 1.742183 | 3.697872 | 1.160461 |
| stale | consistent_short | 1.740354 | 3.697426 | 1.156152 |
| stale | consistent_010 | 1.741455 | 3.707648 | 1.144339 |

稳定方案在所有五个非missing条件的FDE均略有退化。准确度方案和稳定方案仍均未将
jitter降低到CV水平。报告不以选定方案替代全部消融结果，其余策略原始统计均保留。

## 配对场景区间

按同一136个scene，先平均固定三个seed，再按scene整体重采样2000次(seed0)、按更新
count加权。完整180条区间含八个新策略−P6及预选accuracy/stable−CV，三种metric、
六条件；位于outputs/context/p61/report/paired_scene_intervals.json。未把update或seed
重复当独立场景。以下为clean的部分区间；多项探索性对比未做多重检验校正。

| contrast | metric | delta/m | scene95% CI/m |
| --- | --- | --- | --- |
| time_age_gate-p6_learned | ade | 0.018658 | [0.011784, 0.025606] |
| time_age_gate-p6_learned | fde | 0.052174 | [0.036542, 0.069993] |
| time_age_gate-p6_learned | jitter | -0.008301 | [-0.014242, -0.001597] |
| original_gate-p6_learned | ade | 0.041917 | [0.030630, 0.054793] |
| original_gate-p6_learned | fde | 0.116220 | [0.092366, 0.143631] |
| original_gate-p6_learned | jitter | -0.015613 | [-0.019937, -0.010774] |
| consistent_010-p6_learned | ade | -0.003813 | [-0.005508, -0.002057] |
| consistent_010-p6_learned | fde | 0.006241 | [0.002535, 0.010132] |
| consistent_010-p6_learned | jitter | -0.016073 | [-0.017262, -0.014781] |
| consistent_short-p6_learned | ade | -0.005551 | [-0.006457, -0.004613] |
| consistent_short-p6_learned | fde | -0.004319 | [-0.006001, -0.002641] |
| consistent_short-p6_learned | jitter | -0.005345 | [-0.006401, -0.004206] |

## 尾部误差

下表均为clean条件，是每seed内逐update误差的分位数再取三seed平均，不是轨迹级尾部风险，亦非把
三个seed当独立样本混合。准确度的平均ADE改善没有带来全面的尾部改善；稳定性方案
clean P95 FDE更差。

| strategy | metric | p90/m | p95/m |
| --- | --- | --- | --- |
| consistent_010 | ade | 4.142369 | 5.535568 |
| consistent_010 | fde | 9.360679 | 12.631645 |
| consistent_short | ade | 4.127828 | 5.522763 |
| consistent_short | fde | 9.323348 | 12.594119 |
| finetune_0 | ade | 4.125286 | 5.518010 |
| finetune_0 | fde | 9.313650 | 12.579291 |
| p6_learned | ade | 4.131802 | 5.525015 |
| p6_learned | fde | 9.334176 | 12.592158 |

## 训练进度与成本

所有训练仅更新161参数gate。下列epoch秒数包含训练前向/反向/优化器及完整train_val
选择度量，不能称纯配对前向成本；阶段完成原子marker绑定best.pt与完整训练日志。

| seed | strategy | best epoch | completed | selectionADE/m | sum epoch/s | mean epoch/s |
| --- | --- | --- | --- | --- | --- | --- |
| 2027 | original_gate | 9 | 14 | 1.8265409 | 107.89 | 7.71 |
| 2027 | finetune_0 | 9 | 10 | 1.7932777 | 68.72 | 6.87 |
| 2027 | consistent_002 | 9 | 10 | 1.7927262 | 111.36 | 11.14 |
| 2027 | consistent_010 | 10 | 10 | 1.7928524 | 111.32 | 11.13 |
| 2028 | original_gate | 19 | 20 | 1.8277690 | 151.41 | 7.57 |
| 2028 | finetune_0 | 10 | 10 | 1.7975116 | 67.72 | 6.77 |
| 2028 | consistent_002 | 10 | 10 | 1.7966863 | 110.07 | 11.01 |
| 2028 | consistent_010 | 10 | 10 | 1.7959160 | 104.48 | 10.45 |
| 2029 | original_gate | 7 | 12 | 1.8383974 | 85.68 | 7.14 |
| 2029 | finetune_0 | 10 | 10 | 1.8033038 | 65.33 | 6.53 |
| 2029 | consistent_002 | 10 | 10 | 1.8025469 | 105.82 | 10.58 |
| 2029 | consistent_010 | 10 | 10 | 1.8020412 | 105.76 | 10.58 |

独立配对前向benchmark采用同一个P6模型、同4096个固定keys、16warmup和64次正式测量，
CUDA同步；single与paired分别测候选/quality读取、gate和混合，paired额外执行上次
输入前向及revision。排除标签、反向、优化器；不是完整训练step，也不是单update推理。

| seed | single p50/ms | single p95/ms | paired p50/ms | paired p95/ms | p50 ratio |
| --- | --- | --- | --- | --- | --- |
| 2027 | 4.1944 | 6.3630 | 10.4184 | 14.2150 | 2.48× |
| 2028 | 4.1240 | 4.9813 | 11.2036 | 13.2025 | 2.72× |
| 2029 | 4.5263 | 5.3831 | 10.8760 | 12.6751 | 2.40× |

单update推理另使用256keys、16warmup、batch1、CUDA同步。仍仅覆盖缓存候选读取、
必要质量重算、gate/cap/混合及返回，排除P4 forward、地图查询与冷缓存加载。
配对正则仅在训练时使用，推理无额外历史前向；consistent_short多一个cap算子。

| seed | strategy | p50/ms | p95/ms |
| --- | --- | --- | --- |
| 2027 | p6_learned | 2.0213 | 2.5011 |
| 2027 | consistent_short | 2.8611 | 3.3655 |
| 2027 | consistent_010 | 1.9957 | 2.4819 |
| 2028 | p6_learned | 2.0897 | 2.7247 |
| 2028 | consistent_short | 2.9199 | 3.7123 |
| 2028 | consistent_010 | 2.0546 | 2.5820 |
| 2029 | p6_learned | 2.0951 | 2.7440 |
| 2029 | consistent_short | 3.0582 | 3.7373 |
| 2029 | consistent_010 | 2.0854 | 2.7697 |

设备RTX3060 Laptop、PyTorch2.6.0+cu124；OS缓存已热，机器背景负载未作为受控实验变量，
不将小幅计时差异当架构优越性证据。所有12策略原始推理样本和独立pair样本均保留。

## 来源、复现和验收

新增实现独立于原P6已绑定模块。生产加载核对父轨迹/map/GRU/P4来源与原候选重新前向；
新模型按策略核对candidate、lambda、初始化hash、训练配置、scene与完成标记。
报告重新拟合train及train_val校准、完整重算cap sweep和冠军选择，重新预测每个NPZ
指标/权重，再核对所有分组、分位数、recovery、P6六条件parity和两套benchmark样本。
仅修改NPZ并更新文件hash仍会在预测重放处拒绝。CSV以round-trip精度读取。

| seed | val summary SHA256 | lambda.02 best.pt SHA256 | lambda.1 best.pt SHA256 |
| --- | --- | --- | --- |
| 2027 | fe651ec0d1029328f86ce359624ad97dd42b46488b003d520c7b883d15d1fbef | 3ee35bb36d756477d213515c319ab00876742d1b93ce7423fb8cf0bd931ce203 | 2f4e4ff0cc87175df0037e1cd36005d6f0e3d8fef3298a95b470727a844ab585 |
| 2028 | 2d8af913e79c809e326658cdff316c0c30dc29470930cc2777dcc9f53cd81525 | 50579c30974ff3b0e21eff63e5fa2779797b2eefa88e620d96bb885f544f52bf | f623352cf33ea10358049edf26f65a154e9a8980227f4e8d1732f798b46a80a8 |
| 2029 | 727ecec1a2661928152953bdc06265c1db5193b62fd1eb478d12a9175f93299e | 2b6e95d3f2d9fe10dcb4a2553127ed15ddbf45940f71b5cc33ba1e6e568d3738 | bc5a5f923007a9d96904de6a7f5ad246046701e7eded34c81d0952ea08ad7e4d |

```powershell
$python = (Resolve-Path ../trajectory-mvp/.venv/Scripts/python.exe).Path
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& $python scripts/train_gate_refinement.py --cache-root outputs/context/p6/cache --p6-training-root outputs/context/p6/training --output-root outputs/context/p61/training --device cuda
& $python scripts/evaluate_gate_refinement.py --cache-root outputs/context/p6/cache --training-root outputs/context/p61/training --p6-training-root outputs/context/p6/training --selection-root outputs/context/p61/selection --output-root outputs/context/p61/evaluation --device cuda
& $python scripts/report_gate_refinement.py --evaluation-root outputs/context/p61/evaluation --selection-root outputs/context/p61/selection --output-root outputs/context/p61/report --device cuda
& $python -m pytest -q --tb=short --basetemp (Join-Path (Get-Location) ('outputs/pytest-' + [guid]::NewGuid().ToString('N')))
& $python -m ruff check src scripts tests
& $python scripts/check_public_repo.py
git diff --check
```

源码/数据/配置身份改变会拒绝复用；不完整stage缺marker则从同初始化重训该stage。
独立只读审查覆盖算子、训练、选择/角色、完整性、统计和实际小表；审查发现的问题
均先复现再修复。449项pytest、Ruff、公开扫描通过。旧P6来源模块diff为空。

验收状态：2026-10-21完整三seed训练、val与严格重放审计均已通过；共17,109,576条
策略更新记录、180条配对场景置信区间。train_val与固定val上的稳定性预设目标均满足，
此目标不包含FDE或尾部误差全面改善。
最终report.json SHA256：`4e0e164b3ed13fe713b2242b3ecf024c17a27e8059ffb89468abbe322eddf8b0`。
实现提交aba6235、4f14e1a、d586306；数据、模型、所有outputs均未加入git，不推送。

## 后续入口

本轮表明小型门控与时序正则有可复现的小幅收益，提升幅度及FDE/尾部代价必须保留。
接下来优先诊断运动候选和质量特征的增量价值，不能把进一步扩大gate或增加传感器
当作已被证明有效的方案。新留出若从P4训练scene划分，必须同步重训上游；历史val
仍只用于固定回归。P7不自动开始，也不自动下载sensor。

# P6.2：冻结运动候选与质量特征诊断

记录日期2026-10-21；分支feature/causal-context-online，基线38e346c。
设计见[spec](superpowers/specs/2026-10-21-causal-context-phase62-design.md)，
施工见[plan](superpowers/plans/2026-10-21-causal-context-phase62.md)。

## 结论与适用范围

本轮只诊断完整train_val：6621条轨迹/176场景、seeds2027/2028/2029；未训练新模型、
未读取历史val。下列数字不能与P6.1的val表直接相减，不能作为新的val提升。

P6 gate六条件平均ADE1.800475m，使用未来标签逐更新选择的map理想混合为1.615440m，
差0.185034m。当前候选的有限混合空间仍有差距，但这个标签使用型参考不能证明因果
特征或可部署gate可以获得该幅度收益，也不是连续g的精确最优解。

冻结gate下，运动四特征整组置换使ADE增加0.018670m，场景95%区间[0.014096,0.023438]；
全部八特征置换增加0.019713m。运动组依赖较强，但相关性、冗余和置换引入的不一致
可能影响解释，不能称这就是特征重训消融的增量价值。

当前filtered候选端点jitter低、ADE高；即使理想混合，其ADE1.855430m仍高于map理想混合。
本轮不支持优先扩大现有filter权重来改善ADE；此结论只针对当前固定滤波参数和候选。

## 固定协议与计数

全部GRU/P4/P6/P6.1模型冻结，生产load_verified_gate_cache验证真实父数据、地图、候选
前向及时间线；load_suite/load_selection绑定相同模型与train_val场景。三个seed的样本
key及dataset identity相同，源码快照在运行结束前再次核对。

六个实际参照/端点，加11组特征×3次置换，共39个完整度量方法。六条件×6621轨迹×
11更新×39方法×3seed，共51,127,362条ADE/FDE/jitter记录；另有三个理想混合的
3,932,874组ADE/g。每seed45个数值数组。所有重复记录不是独立样本数。

CV端点始终可用。map、original、filtered端点和所有混合统一乘原地图包valid mask，
invalid精确退回CV。这是人为控制可用性，original/filter本身并不依赖地图包。
map_endpoint也不同于旧P6缺包时回original的ungated_map，不能直接混用历史指标。

理想混合使用尚未释放的未来target，在0:.05:1及实际参照g中选择当前更新最小ADE，
不看mask外标签。map参考P6 gate，original参考original_gate，filtered使用P6权重
移植至filtered候选。filtered的reference_ade不是实际map gate ADE。g=0/1保证包含
CV及候选端点；网格外实际g也被包含。oracle不输出部署jitter/延迟或新的可执行策略。

置换仅改quality列，donor为同condition/update/实际age/valid=true内的轨迹排列；invalid
不变。repeat0/1/2共用各组donor，组内列整体复制。仍保留原CV/map/valid，预测完整11步
后计算同绝对未来时间上的jitter。保存donor哈希与实际特征/权重变化比例。

## 实际端点与门控：六条件等权平均

先每seed六条件平均，再三seed均值±样本std(ddof1)，单位m。

| method | ADE mean ± std | FDE mean ± std | jitter mean ± std |
| --- | --- | --- | --- |
| cv | 1.936301 ± 0.000000 | 4.272695 ± 0.000000 | 1.021958 ± 0.000000 |
| map_endpoint | 2.151397 ± 0.011039 | 4.156946 ± 0.018916 | 1.394720 ± 0.006802 |
| original_endpoint | 2.219821 ± 0.015641 | 4.343576 ± 0.029626 | 1.390708 ± 0.005947 |
| filtered_endpoint | 2.640205 ± 0.000000 | 5.303236 ± 0.000000 | 0.900191 ± 0.000000 |
| p6_learned | 1.800475 ± 0.005482 | 3.910451 ± 0.012311 | 1.103900 ± 0.006394 |
| original_gate | 1.830902 ± 0.006520 | 3.992213 ± 0.014405 | 1.089136 ± 0.004174 |

## 标签使用型混合参考

ADE/reference/gap先每seed六条件平均，再固定三seed平均；ADE另列seed std。
末两列只按valid更新数加权，排除无效包；不是各条件比例简单平均。

| candidate | oracle ADE ± std | 同候选参考ADE | 参考−oracle | 端点优于CV/valid | oracle g=0/valid |
| --- | --- | --- | --- | --- | --- |
| map | 1.615440 ± 0.003191 | 1.800475 | 0.185034 | 37.49% | 37.84% |
| original | 1.654159 ± 0.007473 | 1.830902 | 0.176744 | 34.68% | 39.00% |
| filtered | 1.855430 ± 0.000000 | 2.204033 | 0.348604 | 17.45% | 72.50% |

map各条件差距：

| condition | P6 ADE | map oracle ADE | gap |
| --- | --- | --- | --- |
| clean | 1.736780 | 1.472929 | 0.263851 |
| delay_500ms | 1.800403 | 1.580396 | 0.220007 |
| delay_1000ms | 1.848752 | 1.665759 | 0.182993 |
| dropout | 1.734521 | 1.468291 | 0.266229 |
| missing | 1.936301 | 1.936301 | 0.000000 |
| stale | 1.746091 | 1.568966 | 0.177126 |

差距不是不同模型在新验证集上的收益。missing没有有效包，全部策略/混合精确CV；
原始oracle_summary中valid_count=0时条件比例记0，不能解释为有样本的零概率。

## 冻结特征依赖

Δ=置换−未置换，正数表示置换使误差变差。区间先平均三个repeat，再固定平均三个
模型seed，按176个scene配对bootstrap2000次(seed0)，以scene sum/count重建轨迹更新
均值。三seed标准差另存在feature_importance.csv。所有66区间未作多重比较校正，
按探索性诊断解释；固定三seed场景区间不覆盖任意重训seed的全部不确定性。

| feature/group | clean实际变化比例 | overall ΔADE [95% CI] | clean ΔADE [95% CI] | overall Δjitter |
| --- | --- | --- | --- | --- |
| coverage | 98.40% | +0.004730 [+0.002191, +0.007346] | +0.006673 [+0.003250, +0.010391] | +0.006784 |
| age | 0.00% | +0.000000 [-0.000000, +0.000000] | +0.000000 [-0.000000, +0.000000] | -0.000000 |
| position_residual | 99.99% | -0.000874 [-0.001843, +0.000116] | -0.000850 [-0.002240, +0.000585] | -0.000843 |
| velocity_residual | 99.99% | +0.004542 [+0.002517, +0.006664] | +0.006633 [+0.003915, +0.009436] | +0.013469 |
| speed | 99.99% | +0.002870 [+0.000128, +0.005749] | +0.001813 [-0.001814, +0.005628] | +0.009665 |
| acceleration | 99.97% | +0.002424 [+0.001491, +0.003423] | +0.004248 [+0.002741, +0.005914] | -0.002505 |
| remaining_time | 0.00% | +0.000000 [-0.000000, +0.000000] | +0.000000 [-0.000000, +0.000000] | -0.000000 |
| disagreement | 99.99% | +0.001611 [-0.000444, +0.003670] | +0.001295 [-0.001249, +0.003986] | -0.007024 |
| timing_coverage | 98.40% | +0.004730 [+0.002191, +0.007346] | +0.006673 [+0.003250, +0.010391] | +0.006784 |
| motion | 99.99% | +0.018670 [+0.014096, +0.023438] | +0.024552 [+0.018602, +0.030646] | +0.031510 |
| all | 99.99% | +0.019713 [+0.014053, +0.025507] | +0.027581 [+0.020052, +0.035409] | +0.022393 |

age与remaining_time在本分层内是常数，实际变化比例0，其零差值无法衡量时间特征
重要性。timing_coverage与coverage的结果相同也是分层常数造成，不能称时间特征无用。

position_residual的ADE置换均值略降低，但区间跨0，FDE反而增加0.003488m；不能据此
直接删特征。disagreement置换ADE增加0.001611m且区间跨0，FDE增加0.009585m，
jitter却降低0.007024m；acceleration置换也出现ADE变差/jitter变好的取舍。
这进一步说明只按jitter排名删特征可能损害终点准确度。

## map理想混合差距的预测时长分布

下表为六条件×固定三seed均值，按当前剩余预测时长汇总，单位m。

| update | remaining/s | P6 ADE | map oracle ADE | gap |
| --- | --- | --- | --- | --- |
| 1 | 5.5 | 3.841304 | 3.592862 | 0.248443 |
| 2 | 5.0 | 3.307308 | 3.019300 | 0.288008 |
| 3 | 4.5 | 2.825322 | 2.527947 | 0.297374 |
| 4 | 4.0 | 2.407753 | 2.113418 | 0.294334 |
| 5 | 3.5 | 2.037195 | 1.775804 | 0.261391 |
| 6 | 3.0 | 1.667724 | 1.494728 | 0.172996 |
| 7 | 2.5 | 1.310930 | 1.173350 | 0.137579 |
| 8 | 2.0 | 0.982685 | 0.871532 | 0.111153 |
| 9 | 1.5 | 0.694949 | 0.609277 | 0.085672 |
| 10 | 1.0 | 0.457381 | 0.388703 | 0.068677 |
| 11 | 0.5 | 0.272672 | 0.202923 | 0.069750 |

早中期绝对误差与差距更大，应优先检查这些更新上的方向、转向与候选偏差；
这不是长时段相对改善比例或实现收益的证明。

## 下一项训练建议：独立P6.3协议

优先检验因果方向特征，暂不直接增加gate宽度、传感器或改上游候选。候选四维为：
已观测加速度在当前速度方向的有符号纵/横分量，及map−CV修正在该方向的纵/横分量。
仅用已释放位置/速度和候选预测；低速方向、归一化与裁剪规则必须在运行前冻结。
这些是待验证假设，本阶段没有实现或测得新增特征收益。

建议保持16隐藏单元，把输入统一扩至12维：8旧特征+4零填充作为匹配对照，8旧特征+
4方向特征作为实验，另配旧特征冗余拷贝对照；三者均225名义参数。旧P6的8列权重与
其余层复制，新列初始权重0，初始预测相同；同epoch/优化器/seed/shuffle预算。零填充
与冗余对照的有效输入自由度仍不同，参数量相等不代表所有有效容量完全相同。

仅train训练、train_val选方案；同时保留ADE/FDE/jitter、尾部与早中期分组，选模约束
提前冻结。只有明确的预算匹配增益才冻结并回归历史val。当前不启动这项训练，P7
也不自动开始；新盲测需独立holdout，若拆P4已见场景必须同步重训上游。

## 来源、命令与验收

原始产物outputs/context/p62，未加入git。每seed的diagnostics.npz、condition/scene表、
oracle_summary、feature_change、donor哈希被manifest绑定；总manifest绑定三个子manifest
和候选/特征/区间汇总。源码、父cache、P6/P6.1 checkpoint及selection均带哈希。

| seed | manifest SHA256 | diagnostics.npz SHA256 |
| --- | --- | --- |
| 2027 | 0aee4e3109af53051a165797bf0dd122474dbfcb42c0c76122f4d377c4a02683 | 4530b3795ec1f43544642e88fe77c3301d1c310fcb4a331ccf79d50fccea7327 |
| 2028 | ad39804a2eb3d025a2a89dca5e147e01a3f655b56ad3272c3bf99d1a9c795a8a | 4ea47ab5eafc86e88ab0bf46d8280ed3cde73aae0c79b80352d6192c345a336a |
| 2029 | f66176849a6615776e04d269eef9af84c231d5f7ddfd3a549226906147970fd1 | dde5d2033676e4a07e0899c9e1f2cf6af7b49cb6dceef05a48f04dfca3d0709a |

总manifest SHA256：`bd2d3f1251a0b99d6cfaad630552734509f1fd0fcfbf841f7dcc862dcfaed0f6`。
P6.1冻结selection SHA256仍为`3720fdeffbd13c3339776dade31e5e7b819ac33e7cceed2ef6a03fcb19aa07a7`。

```powershell
$python = (Resolve-Path ../trajectory-mvp/.venv/Scripts/python.exe).Path
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& $python scripts/diagnose_gate_candidates.py --cache-root outputs/context/p6/cache --training-root outputs/context/p61/training --p6-training-root outputs/context/p6/training --selection-root outputs/context/p61/selection --output-root outputs/context/p62 --device cuda
& $python scripts/diagnose_gate_candidates.py --cache-root outputs/context/p6/cache --training-root outputs/context/p61/training --p6-training-root outputs/context/p6/training --selection-root outputs/context/p61/selection --output-root outputs/context/p62 --device cuda --verify
& $python -m pytest -q --tb=short --basetemp (Join-Path (Get-Location) ('outputs/pytest-' + [guid]::NewGuid().ToString('N')))
& $python -m ruff check src scripts tests
& $python scripts/check_public_repo.py
git diff --check
```

verify重新从生产父来源加载，重新计算所有45个数组与表；仅更新产物hash也会被数值
比较拒绝。root汇总/66区间同样重算；CSV按round_trip精度读取，检查列和顺序。

独立只读源码及三seed小表/manifest复核未发现重要问题；其未读取完整NPZ，逐元素
重放由主会话完成。已按建议增加不同seed、
repeat及不等scene样本量的手算案例，期望ΔADE1.6/seed std.8/CI[.4,2]，以验证统计加权。
旧P6/P6.1绑定源码不改；新增9项诊断测试，包括oracle mask、分层、硬回退、篡改与统计。

验收状态：2026-10-21完整三seed train_val诊断与全部数值/表/区间生产重放均通过。
458项pytest（49.04秒）、Ruff、公开扫描与diff检查通过；旧绑定源码diff为空。
设计提交a0d6d32，实现提交7e74ee3；保留feature分支与worktree，不推送、不合并。

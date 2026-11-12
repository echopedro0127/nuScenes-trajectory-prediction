# Phase 3A S4c：首离群事件的额外位置误差与恢复

日期：2026-11-12。分支`feature/scenario-evaluation`，实现前HEAD为`f486d83`。本轮完成M3纯指标与完整配对pilot，新增源码由run哈希绑定；不是S7正式全量评价，没有训练或新holdout。

## 1. 为什么需要新配对pilot

原S2保存的是原推理批次邻域。fit邻域全部为noise_outlier；select虽然有noise_05和noise_outlier，但并非同seed的完整参考/受扰对，不能直接相减或只保留某个交集冒充共同队列。

因此从原608个pilot窗口中，按`SHA256('s4c-pilot-v1:'+role+':'+instance+':'+sample)`排序，每角色最多8个窗口；每个窗口保留三个noise seed和两条件的全部配对。选择与运行身份在新的冻结推理前持久化，不根据本轮结果更换窗口。

| 角色 | 窗口 | log | noise重复 | 配对事件/策略 | 推理观测流/策略 |
| --- | ---: | ---: | ---: | ---: | ---: |
| fit | 8 | 7 | 3 | 24 | 48 |
| select | 8 | 3 | 3 | 24 | 48 |

这是原开发窗口的有界配对诊断，不是新增数据或盲测；与S4a/b切片队列不同，不能把其均值跨阶段相减。正式3A仍使用原冻结全量队列。

## 2. 方法：先配对，再计算额外误差

新增`evaluation/event_metrics.py`：

- 以role/instance/sample/noise seed/策略身份配对noise_outlier与noise_05，核对完整GT、scene/log和原row；不以行号直接配对，不丢弃缺对事件。
- 新pilot按原`replay_streams`重放9门控和7物理策略，保留父4096平坦批次与精度路径，旧指标仍精确对父兼容。
- 每个窗口同seed重建两条件观测包，验证历史和非注入步的高斯噪声完全共享，source/arrival一致，step4/8的增量均为5m（预设幅值容差1e-10m）。这核验packet生成协议，不声称重新完整生成所有GRU/map候选。
- 仅观察首注入step4至7，分别为注入后0/0.5/1/1.5s；固定1s lead，目标列5至8。查询输入不包含step8第二次注入。

```text
A(k) = max(0, E_outlier,1s(k) - E_noise05,1s(k))
excess_AUC = 对上述四个A点做梯形积分，单位m·s
```

目标时间可以晚于step8，不等于读取了step8观测。未来GT可以用于离线误差评价，不参与冻结预测输入。

## 3. 恢复判定与统计分母

- 整窗A<=0.5m：unaffected，表示没有明显额外位置误差，不表示绝对预测准确。
- 存在A>0.5m：affected；首次受损后需至少两个连续合格更新，且到窗末无再次超阈值，才记recovered，否则not_recovered。
- 第二个合格更新为确认step；从注入step4起计延迟，当前只能1.0或1.5s。未受损/未恢复延迟为null，不填0秒。
- 无复发条件需离线检查到观察窗末；记录的确认step不是在线保证“之后不会再复发”的时刻。“1.5s内未恢复”也不表示永久失败。
- AUC包含所有共同eligible事件；先noise重复均权、再窗口均权，不只计算受损或已恢复事件。
- `eligible=affected+unaffected`，`affected=recovered+not_recovered`；恢复率/未恢复率以affected为分母，未受损率以eligible为分母。
- 延迟均值只在recovered事件上计算，字段明确命名`mean_confirm_delay_s_recovered_events`，同时报告分母；不能靠条件延迟单独排名。

各策略共同eligible键一致，但affected/recovered集合依模型变化。漏对、错GT、重复、非有限数和短窗缺点均报错，不作正常删失跳过；空统计组和零条件分母返回null。

联合报告参考/受扰的绝对E_1s，以及预测间距离D作为诊断。GT=(0,0)、两预测(3,0)/(0,3)时A=0而D≈4.243m，说明额外误差恢复不等于预测重合或物理状态恢复。

## 4. 真实pilot与独立验证

| 项目 | 数量/结果 |
| --- | --- |
| 不同窗口 | 16；仍有相关窗口，不当独立采集数 |
| 完整window/noise配对 | 48 |
| 策略 | 9门控（3控制×3训练seed）+7物理策略 |
| 策略级事件记录 | 768；不是768个独立驾驶事件 |
| 汇总 | 32个role/strategy组，每组24个eligible事件 |
| 额外误差曲线点 | 3,072；参考/受扰E各3,072个 |
| 原始预测 | 32个分片，每分片48条双条件观测流 |
| 独立标量最大绝对差 | 5.329070518200751e-15m，预设atol=rtol=1e-12 |

标量参考独立按时间计算E/A/D、梯形积分及最后一次超阈值的位置，核验恢复状态和确认时刻。另通过`--verify-only`重新加载模型、重放全部pilot，逐项比较原始P、曲线、状态、事件JSON和汇总，并再次核对来源与自身产物完成标记。

## 5. 子集诊断：恢复率和AUC都不能单独排名

以下为select pilot的24个eligible配对（8窗口/3log/3noise）；门控统一展示训练seed2027，不挑最佳seed。参考/受扰E为四个1s lead查询的均值，再按完整noise/窗口队列平均。

| 策略 | AUC/m·s | affected | recovered | 恢复率/% | 已恢复延迟均值/s | 参考E/m | 受扰E/m |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Raw CV | 7.154184 | 24 | 24 | 100.00 | 1.5000 | 2.560642 | 7.693025 |
| NIS-KF | 1.272772 | 24 | 17 | 70.83 | 1.1471 | 1.480180 | 2.496590 |
| baseline7 | 1.105173 | 24 | 17 | 70.83 | 1.2059 | 1.548265 | 2.355688 |
| plus_ca | 1.140475 | 24 | 16 | 66.67 | 1.2188 | 1.513395 | 2.345202 |
| guarded_ca | 1.135088 | 24 | 17 | 70.83 | 1.2059 | 1.511251 | 2.356342 |
| hold_position | 0.431607 | 13 | 13 | 100.00 | 1.2308 | 8.283928 | 8.447755 |

Raw CV虽在短窗末全部恢复，期间额外误差代价和受扰绝对误差仍高。位置保持策略AUC较小且条件恢复率100%，但本来就预测得较差；其11个unaffected事件也不能挪进恢复率分母。

同样，CA在该小队列的参考绝对误差略低，但额外AUC略高；这只是可追问的准确性/异常响应取舍，不是新优胜模型或泛化结论。24个重复事件中相差1次恢复就变化4.17个百分点，当前不作显著性或跨seed稳定性判断，三训练seed结果全部保留在产物。

## 6. 实现与可追溯性

本轮新增31项测试，覆盖持续确认/复发/延后受损/末点不足/等阈值/未受损、完整配对、模型身份/GT错配、分母与noise重复、原始预测重放和产物篡改。先失败再修复了AUC可表示但中间求和溢出、verify-only期间自身完成标记变化两个边界；独立只读审查及复核未发现Critical/Important。

最终完整套件**798 passed in 237.69s**，Ruff通过；真实pilot的verify-only完整重放通过。另独立核验768条事件、32组状态/分母/AUC/noise权重/共同eligible、32个原始预测分片及352个曲线/状态数组，文档链接/编码、公开扫描与diff检查通过。

原S0–S4b源码未修改。S4c入口在耗时验证前冻结源码与S2快照，核验原预检及GT来源，再保存selection/run，最后发布completed；中途失败没有完成标记，不能冒充成功。

本轮identity：`bf5fc3e6d032f11b3abf6a511f957fc9cf6a2b3706d54218dbbfe065a8db9dc4`。

输出`outputs/scenario-evaluation/s4c-pilot-v1/`：run/summary/groups/events JSON、forecasts.npz、curves.npz、completed.json。schema为`recovery_s4c_complete_pair_pilot_v1`，明确不是原S2邻域切片，也不是完整3A成绩。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
# 重跑须换新目录，不覆盖已有产物。
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/check_scenario_recovery.py `
  --preflight outputs/scenario-evaluation/preflight-v2 `
  --output-root outputs/scenario-evaluation/s4c-pilot-reproduction
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/check_scenario_recovery.py `
  --output-root outputs/scenario-evaluation/s4c-pilot-v1 --verify-only
```

本轮不训练、不评价新holdout、不推送GitHub、不上传模型/数据。下一步S4d GT窗口事件标签及人工抽查清单，之后再完成S5/S6/S7全量发布、分层统计与正式报告。

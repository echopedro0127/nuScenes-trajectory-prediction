# Phase 3A S3：固定提前量误差与独立标量验证

日期：2026-11-12。分支`feature/scenario-evaluation`，实现前HEAD为`0c44854`；新增实现以本轮run源码哈希绑定。[执行计划](superpowers/plans/2026-11-10-scenario-evaluation-phase3a.md)中的S3已完成，S4及以后尚未实施。

## 解决的问题

旧在线ADE在每次更新后评价剩余未来点，剩余区间不断缩短；观察到误差下降，不能直接解释为新增观测使预测改善。本阶段实现固定0.5/1/2秒提前量误差，使每次查询评价相同lead。

固定lead仍不是“单次修正的净收益”：比较更新前后同一目标时间的增益属于下一步M2。也没有用本轮数值重新训练、挑选模型或调整阈值。

## 实现与数值契约

新增`evaluation/streaming_metrics.py::fixed_lead_errors()`，纯函数、无文件IO：

```text
k = 1..11，数组row = k-1
m = 1/2/4，对应h = 0.5/1/2s
j = k+m-1，要求k+m<=12
error = hypot(P[:,k-1,j,0]-Y[:,j,0], P[:,k-1,j,1]-Y[:,j,1])
```

返回values、query_steps/query_rows、target_columns、相对查询/目标时间、有效mask及被排除尾部step。三个lead每条流分别11/10/8个有效查询；不足lead时不补零、不重复终点、不外推GT。

输入只接受有限float32/float64、完整`P[N,11,12,2]`和`Y[N,12,2]`，N>=1。包括未被当前lead评分的槽位也必须有限。先转换float64再做差和hypot，避免大float32坐标差溢出；超出float64可表示范围明确拒绝。float64是本次新指标精度，不改变父预测float32及旧ADE/FDE/jitter算法。

单元测试包含区分查询行和目标列的3–4–5距离夹具、随机数组、两种精度、尾部、未知lead/布尔值、错shape、空输入、NaN/Inf、整数数组及极大坐标边界。独立参考按查询时间加lead确定目标时间，再逐样本用Python math.hypot计算，不调用向量实现。

## 真实集成范围与结果

复用[S0–S2预检](SCENARIO_EVALUATION_S0_S2_RESULTS.md)的`preflight-v2`，未改动其源码。新增入口`scripts/check_scenario_fixed_lead.py`先执行旧预检verify-only，再在已保存预测上计算S3。

| 项目 | 实际数量/结果 |
| --- | --- |
| 父预检 | identity `12e2584ecf0a0a0fd0d3e847cc9279f124d084f346ea2954f221a0e7ec9b9fab` |
| 预测来源 | 2角色×16策略，共32个S2预测分片 |
| 每策略观测流 | fit/select各373，共746；仍为S2批次邻域子集 |
| h=0.5s误差 | 746×16×11 = 131,296个 |
| h=1s误差 | 746×16×10 = 119,360个 |
| h=2s误差 | 746×16×8 = 95,488个 |
| 总计 | 346,144个逐查询位置误差 |
| 独立参考比较 | 全部通过，预先固定atol=rtol=1e-12 |
| 最大绝对差 | 7.105427357601002e-15m |

容差只用于两个float64数值实现的参考比对，**没有放宽S2旧指标精确兼容要求**。误差值写入96个role/strategy/lead数组，与完整时间网格一起保存；本阶段不生成正式全量窗口/log/训练seed排名，也未实现P95、M2修正增益或M4方向分解。

S2邻域不是三条件均衡全量评测；noise seed和相关窗口也不是独立样本。上述误差计数不是新增轨迹量，最大参考差不是模型预测误差或性能提升。

## 来源审计修复

独立只读审查指出：若先验证S2产物A，再于读取run/marker前将目录切为B，原实现可能对B计算却记A的身份。已先添加“错误proof身份”“验证后marker切换”两项失败回归，再修复：

- 验证前冻结marker哈希，核对run哈希及run/marker/规范化binding身份。
- 验证后核对proof身份与快照，GT装配父identity及全部references必须等于已验证S2绑定。
- 发布前重新核验全部S2文件、GT来源、新源码及marker快照。

审查复核确认原Important关闭，无新增Critical/Important。测试不依赖真实nuScenes下载；真实集成再核验本机数据。

最终验证：新增26项测试，完整套件**717 passed in 115.44s**；Ruff通过。96个真实误差数组的dtype、有限性、总计数及产物哈希另行检查通过，文档链接/编码与公开文件扫描、diff检查通过。原S0–S2源码未修改，其verify-only在本次集成入口中再次通过。

## 产物与复现

本轮identity：`d9a2cf4f0a2cb572671e632f160a38b72dba2481beae71a2c360740bc65d632c`。

输出`outputs/scenario-evaluation/s3-integration-v1/`：run.json、summary.json、time_grid.json、fixed_lead_errors.npz、completed.json。schema为`fixed_lead_s3_integration_v1`，`full_phase3a_evaluation=false`、`neural_training=false`、`holdout_evaluated=false`。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
# 换新目录重算，原目录拒绝覆盖。
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/check_scenario_fixed_lead.py `
  --preflight outputs/scenario-evaluation/preflight-v2 `
  --output-root outputs/scenario-evaluation/s3-integration-reproduction
```

## 后续

S4a优先实现同目标时间修正增益和有益/有害/中性比例，随后再接方向分解、额外位置误差恢复与窗口标签。S5/S6汇总发布和S7正式全量评分仍待实现。当前不新增模型训练或读取已用holdout调参，不推送GitHub，不上传实验数据和模型。

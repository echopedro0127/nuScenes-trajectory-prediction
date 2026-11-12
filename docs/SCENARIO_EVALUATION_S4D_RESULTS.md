# Phase 3A S4d：GT窗口事件标签与待复核案例册

日期：2026-11-12。分支`feature/scenario-evaluation`，实现前HEAD为`432451d`。本轮完成窗口规则标签、固定哈希抽查清单及GT图表；没有训练、调模型、计算新的预测成绩或新增holdout。

**人工复核完成数为0。** 下文均为规则候选，不是已经确认的真实行为标签，也不是独立episode数量。

## 1. 用什么方法解决什么问题

旧分组以整体净航向和平均加速度描述未来轨迹，可能漏掉方向抵消的S形运动或局部走停。新增`analysis/scenario_events.py`，使用原GT历史5点和未来12点生成局部事件证据，与旧maneuvers标签共存，不修改旧阈值、旧报告或模型输入。

轨迹共有17个点、16个0.5s速度区间，覆盖[-2,6)。历史只提供上下文；阳性事件onset位于[0,6)，跨左边界的急刹仅保存删失证据。S弯/掉头按未来anchor+12点的最大连续可靠方向段判定，不搜索任意子段拼出候选。

| 标签 | 冻结规则 |
| --- | --- |
| hard_brake_candidate | onset前4个完整速度区间可用且曾>=2m/s；平滑加速度连续两段<=−2m/s² |
| stop_event | prior四区间曾>=1m/s；平滑速度<=0.3m/s持续两个完整区间；初始已静止不新建stop |
| stop_go_candidate | 本窗口有可确认stop，之后速度>=1m/s持续两区间；0.3至1之间爬行不算重启 |
| s_turn_candidate | 同一可靠段累计绝对角>=45°、净角绝对值<=15°、正负各累计>=20° |
| u_turn_candidate | 同一可靠段净转角绝对值>=135°；只是运动方向规则候选 |

速度来自相邻GT位移，三点中心中值平滑，只有序列最外端重复最近值；无效mask传播，不填GT padding。加速度a_i=(s_i−s_(i−1))/0.5归于区间起点。事件end取真实最后区间终点，不利用平滑边界重复制造额外持续时间。

方向来自未平滑位移，速度>=0.5m/s才可靠，角差wrap到[-π,π)。低速/缺失立即断段，不将停车前左转与停车后右转拼成S弯。

这些是离线GT规则，可以使用中心平滑和未来轨迹，**不意味着因果在线事件识别**；模型预测、专家屏蔽和门控均不读取本模块。

## 2. 边界与状态

每个窗口/标签保留candidate、absent或not_evaluable，以及reason和不完整候选数；原始/平滑速度、加速度、有效性、区间索引和相对时间完整保存。

- 初始状态根据anchor前最后一个平滑速度区间判定；已静止不冒充新停车。
- 不满足已知prior速度门槛，不因窗尾时长未知而伪造潜在阳性。
- prior历史不足、窗尾单次低速/起步不足持续时间、事件onset在左边界外分别记录。
- 即使低速停车未满足prior>=1的规则，观察到低速dwell后再持续起步，也不能直接当成stop-go阴性；记录缺少本窗口已确认stop的删失状态。
- 窗尾同一次未确认起步只输出一次删失记录，避免两个边界分支重复计数。
- 没有可靠方向段时S/U标签不可评价；absent只表示在规则可观察部分没有候选，不是人工判定“绝不发生”。

没有合并跨窗口canonical episode，没有改变352 fit/256 select的采样，没有新增合成驾驶场景或停车位置/时间误差指标。

## 3. 全部既有窗口的规则覆盖

输入仍从S2预检追溯父数据并用S1装配GT；按role/instance/sample去除条件和noise重复。608个GT窗口只各标注一次，而不是对5,472条观测流重复统计。

| 规则标签 | fit候选/352 | fit不可评价 | select候选/256 | select不可评价 |
| --- | ---: | ---: | ---: | ---: |
| 急刹 | 13 | 20 | 8 | 12 |
| 停车 | 18 | 3 | 12 | 2 |
| 走停 | 2 | 36 | 4 | 18 |
| S弯 | 8 | 0 | 2 | 0 |
| 掉头 | 8 | 0 | 0 | 0 |

62个窗口有至少一个候选，共75个窗口×标签组合，类别可重叠。共167条窗口内事件/删失记录，其中76条candidate、91条not_evaluable；一个窗口可能有同类多个局部段，不能把这些记录当独立驾驶事件。

select掉头候选为0，保留零覆盖，不降低阈值或换样本凑案例。标签数量描述当前规则与样本覆盖，不是行为识别准确率、召回率或新预测性能提升。

## 4. 人工抽查清单与图表

按role/log/tag使用固定SHA256排序，每组最多20个阳性窗口；不足全取，不看模型排名。当前75个阳性组合全部进入清单，无组触及截断上限。

- `review.json`及`event_review.csv`：75条，全部pending，reviewer/manual_label为空，人工完成0。
- 每条保存原rule_label、rule_version、selection_hash、GT证据哈希及图像路径。
- `update_review()`纯API支持confirmed/rejected/corrected，要求复核人及相应原因，保留原规则和review_history；本轮未实际执行人工标签修订。
- 62张独立窗口PNG复用于多标签行，包含等比例GT轨迹、原始/平滑速度、加速度、阈值和候选时间区间。
- `review.html`为离线案例册，明确标注“GT离线规则候选，待人工复核”；没有模型预测或排名供选择。

图表布局抽看两例通过，**不计人工行为标签复核**。当前只抽阳性候选，不能据其估计漏检率。人工意见应另存修订版本，不覆盖冻结JSON/CSV和原规则；直接修改冻结产物会被verify-only拒绝。

## 5. 验证、审查和来源

新增27项测试，覆盖净航向抵消、停车断段、U转阈值/角度wrap、急刹/走停、初始静止/爬行/短停/尾部、缺历史、非法GT、固定抽查与人工修订保真、文件流水线和源码/输入变化。

独立审查复现并关闭两项状态机缺口：未满足prior条件的低速停车后起步被误作阴性，以及同一窗尾删失起步重复输出。均先补失败回归再修复；复核未发现新增Critical/Important。

最终完整套件**825 passed in 98.59s**，Ruff通过。真实verify-only重算全部608个标签、抽查清单、统计、CSV与HTML，核验图像和全部产物哈希；图像本身不重新渲染。另独立按math.hypot/statistics.median重建608个速度/平滑trace，核对阳性持续时间、prior/重启条件、10行覆盖计数及75条哈希抽样状态。

入口在耗时验证前冻结源码/config与S2快照；GT身份与父引用一致，图表生成后再次检查全部GT/S2/source，最后发布completed。保留旧maneuvers字段，不改原S0–S4c冻结实现。标签不进入模型，不新增预测分数；启动时重验的只是已冻结S2预检。

## 6. 产物与复现

本轮identity：`285ab3b9fb8e4c9ed53e66f59b6f27f6785b754f364b6101f996274095598f22`。

本机输出`outputs/scenario-evaluation/s4d-labels-v1/`：run、windows、events、coverage、review、summary JSON，event_review.csv、review.html、figures目录和completed.json。schema `scenario_s4d_window_labels_v1`；scope=window_only，manual_review_completed=0。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/check_scenario_labels.py `
  --preflight outputs/scenario-evaluation/preflight-v2 `
  --output-root outputs/scenario-evaluation/s4d-labels-reproduction
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/check_scenario_labels.py `
  --output-root outputs/scenario-evaluation/s4d-labels-v1 --verify-only
```

重跑使用新目录；不推送GitHub、不上传数据/模型/GT案例图，本地详细简历进展文档单独保留。

下一步S5开发全策略分片重放与原始P发布器，然后S6分层统计/案例和S7冻结后的正式全量评测。待复核标签可明确作为rule candidates描述，不因生成清单就声称人工审核已完成。

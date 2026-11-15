# S6b：恢复事件配对与离线机动标签报告

日期：2026-11-15。完成S6b开发及小夹具验证；完整S6仍待训练seed汇总、确定性案例及整体报告装配。未运行真实608窗口全量新评分，未训练或评价holdout。

## 解决的问题与实现

S6a只输出查询指标，不能回答“同一观测异常相对无异常参考造成多少额外误差、何时恢复、哪些机动更容易失败”。本轮新增 `scenario_event_report.py` 与 `scripts/build_scenario_event_report.py`，消费已验证的S6a报告及其S5原始预测，分别装配M3事件和离线GT标签组，不向预测网络新增GT输入。

M3严格按role、instance、sample、noise_seed和策略身份配对noise_outlier与noise_05。策略身份绑定发布身份及完整策略登记，物理策略train_seed保持null。配对还要求scene/log/source row和整段GT完全一致；缺参考、重复或不足3个预期噪声种子直接失败，不跳过。标量审计从原始记录独立重建预期配对列表，核对生产配对的全部索引，避免“错误索引和错误数值彼此一致”掩盖跨种子配对。

## M3时间轴与状态

- 固定1秒lead，异常源步k=4，评价查询k=4/5/6/7，对应目标列5/6/7/8；相对异常释放时间为0/0.5/1/1.5秒。
- 每个事件保存参考绝对误差E_ref、受扰绝对误差E_out、额外误差A=max(0,E_out−E_ref)，以及预测之间的距离D_pred。D_pred只是诊断，不能代替GT误差。
- 任一点A>0.5m为affected；全窗未超过阈值为unaffected。首次受损后需连续两次A≤0.5m，并在剩余观察窗内无复发，才算recovered；其余为not_recovered。
- 确认延迟从k=4计时，仅recovered事件有数值，其余内部NaN、汇总null。AUC采用1.5秒窗内梯形积分，单位m·s。
- 保证eligible=affected+unaffected、affected=recovered+not_recovered；恢复率分母为affected，零affected时为null。无样本组明确not_evaluable/empty_group。

无复发条件使用完整评价窗，属于离线评测定义，不能宣称车端在该时刻已能确定不会复发。低AUC或高恢复率也可能来自参考与受扰预测同时很差，因此报告始终附绝对误差。

## 标签与分组

从共享GT按唯一窗口运行冻结的旧机动描述/阈值和S4d规则。保留低速、直行、转弯、高曲率、加速、减速，以及急刹、停车、启停、S弯、U弯规则候选。规则标签互不排他，不能将各组计数相加当作独立窗口总数。

所有新规则保留candidate/absent/not_evaluable及原因；覆盖表按role/log统计唯一GT窗口，三类状态之和等于窗口数。报告分别计算all、legacy和rule_candidate组的M3与M1/M2/M4/M5；空组保留原因，不制造零误差。候选抽查清单沿用固定哈希选择，review_status均为pending、manual_label为空、manual_completed=0；本轮不导入或冒称完成人工复核。

M3主均值先查询/事件内部、噪声重复、窗口等权；逐log结果单列，另给log_macro均值和valid_log_count。恢复率的log宏平均仅包含affected>0的log，延迟仅包含有恢复事件的log，并分别给affected_log_count/recovered_log_count，避免分母混淆。4个select log仍仅支持描述统计。

## 产物与可复算性

每个角色×策略保存一个事件NPZ，共32个；pair_keys.json关联原始参考/受扰流索引与策略身份，event_time_grid.json固定查询和目标列。另存windows.json、label_coverage.json、event_groups.json、maneuver_query_groups.json、event_review.json、summary.json和run.json。

开始先验证S6a来源并重放上游冻结预测。标量Python循环独立核对配对键、逐查询误差/预测距离、恢复状态、AUC、计数、条件比例和延迟；机动查询分组复用S6a已独立审计的汇总器。verify-only从原始P/GT重新派生全部产物，对比查询、事件、标签和分组，而非只检查哈希。即使改写统计并同步更新哈希，仍会被语义重算拒绝。

身份绑定当前源码、阈值、S6a凭据及原始父实验引用。运行末尾再次检查S6a、S5、S2文件和原始输入引用，防止计算期间上游变化；已写NPZ立即读回并快照校验，最后写completed.json。中断不写完成凭据，已有目录拒绝覆盖。完成凭据只代表S6b，full_s6_complete仍为false。

## 验证与限制

小夹具包含2个GT窗口、18条观测流、32个策略/角色分片，对应96个“策略×窗口×噪声”配对评价条目。96不是独立物理事件数量。本轮只验证实现，无新模型精度提升结论。

测试覆盖三类恢复状态、绝对误差、缺配对/错GT、错误恢复汇总器、相同预测下跨seed错误配对、空组、不等窗口数的log宏平均、停车候选pending、端到端保存/独立复算、重哈希统计篡改、覆盖拒绝、中断及原始引用变化。审查重点修复了配对索引缺少独立审计、遗漏M3 log宏平均两项问题。

最终验收：8项新增测试通过；全套851项pytest通过（362.75秒），Ruff、公开仓库扫描、命令行help及git diff检查通过。独立只读复审确认本轮范围内无剩余必须修复项。

下一步S6c：按训练模型先独立评价后做ddof=1汇总，物理策略不复制训练seed；固定共同案例并绘制同轴GT/预测/观测时序。随后完成统一报告验收，S7冻结后再执行正式全量评分，不将开发夹具成绩包装成实验提升。

# P6.6A 保守精度余量与 Pareto 选模诊断

日期：2026-11-06。父提交：6c6c396；父实验：P6.5。用户要求继续实验。

## 目的和范围

P6.5 冻结候选降低 jitter，但历史 val 的总体与 clean ADE 超过原有 0.2% 约束。
本轮先检验保守选模的可行空间：复用原有 22 策略、3 seed、6 条件的 train_val
汇总与尾部表，不重训、不改变 checkpoint，不运行或读取 val 指标。

这是受先前结果启发的回顾性选择诊断，不是新模型实验或新盲测。它不会替换 P6.5
冻结选择，不晋级任何新方法，不把选模改动包装成准确率或部署性能提升。

## 冻结规则

候选池沿用 RH、revision_fixed、fixed_005、adaptive_matched_005、fixed_010、
adaptive_matched_010。按三个 seed 和六条件等权计算总体均值；clean 单独计算。
参考始终为 RH。

主规则在读取细粒度结果前固定：总体/clean ADE <= RH*1.0005（0.05% 余量），
总体 FDE <= RH*1.0025，clean FDE p95 的逐 seed 均值 <= RH*1.005。
合格池中 jitter 最小，tie 依次 ADE、策略名字典序。

另列固定 ADE 余量网格 [0, 0.0005, 0.001, 0.0015, 0.002]，FDE/尾部约束同上。
网格只用于敏感性诊断，不能看完结果后换主规则。逐 seed 的精度代价分别报告，
不是额外选择条件。

Pareto 定义：在候选池的总体 ADE/FDE/jitter、clean ADE/clean FDE p95 五维中，
没有另一个候选所有指标不大于它且至少一个严格更小。报告被支配候选及其支配者。
不把 Pareto 成员资格等同于满足误差约束。

输出仅标记 meets_diagnostic_gate：选中四新组之一、jitter 比 RH 至少下降 0.5%，
并严格低于 revision_fixed 与 revision_0。无论此值真假，advance=false、val_run=false。
不存在从诊断规则到历史 val 的自动分支。

## 输入身份与输出

输入仅接受 P6.5 selection.json 的固定 SHA256：
508bdef16942f9ce4996f9e15649d4f68768d1131557e039a2f9950d40d94a34。
调用旧 load_strength_selection 验证当前父源码、完整矩阵和父 summary 绑定。
额外逐 seed 核对来源 summary 的 split/seed、by_condition.csv、quantiles.csv 哈希，
从两表重建 selection rows，再与冻结 selection_conditions.csv 逐行比较。

本轮验证的是冻结汇总表的身份和统计一致性，不重跑逐 update 模型生产重放。
P6.5 的逐点重放结论仍以其原报告为准。

新增独立 analysis/precision_selection.py、scripts/audit_precision_selection.py、
tests/analysis/test_precision_selection.py。旧 P1 至 P6.5 源码、配置、测试与输出冻结。
输出到全新 outputs/context/p66-selection：候选表、逐 seed 代价表、余量网格、JSON
报告及完成 marker。记录协议、脚本与父输入 SHA256；已存在输出目录则拒绝覆盖。
所有文件写完后再写 completion.json；该 marker 不代替模型或原始数据验证。

## 验收和后续

测试覆盖：总体/clean/尾部约束、tie、无新组可行时回退、Pareto 支配与相等、完整
矩阵/重复/非有限值拒绝、输入篡改、train_val 来源重建、已有输出拒绝及不访问 val。
完成后跑全量 pytest、Ruff、公开扫描，记录实际诊断结果与局限。

无新增训练、延迟或恢复指标。若诊断显示有可行空间，下一轮再独立设计保存 epoch
候选并在保守约束下选 checkpoint 的新训练；若缺乏空间，保留 RH 与 consistent_010。
任何独立泛化声明需要上游训练也未见的新场景。保持本地分支，不推送、不改简历。

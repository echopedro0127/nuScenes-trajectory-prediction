# P6.6A 保守精度余量与 Pareto 选模诊断结果

日期：2026-11-06；分支 feature/causal-context-online。
状态：真实 train_val 汇总核验和诊断完成，没有新训练或 val 评估。

## 实验边界与来源

父提交6c6c396/P6.5。协议03531f5先冻结规则，实现0e77d95通过验收后运行。
复用22策略×3 seed×6条件，共396行train_val汇总；重建逐seed条件表和FDE p95并与
原冻结选择表精确核对。父selection SHA256、源码、summary与表格身份检查全部通过。

本轮受已知历史结果启发，是回顾性选择诊断。原checkpoint已按完整train_val选过，
这里没有新的独立选择集，没有重跑逐update模型生产重放，也没有独立泛化收益结论。
P6.5原选择、失败结论及RH/consistent_010推荐保持原样。

## 主规则与实际结果

候选池：RH、revision_fixed及P6.5四新组。固定参考RH；总体/clean ADE余量0.05%，
总体FDE余量0.25%，clean FDE p95余量0.5%。合格池中jitter最小，tie依次ADE、名称。
三个seed等权聚合，总体还对六条件等权；clean FDE p95为三个seed各自p95的均值。

主规则选中 **adaptive_matched_005**。

| train_val指标 | RH | 候选 | 相对RH变化 |
| --- | ---: | ---: | ---: |
| overall ADE/m | 1.773337 | 1.773605 | +0.015122% |
| overall FDE/m | 3.884974 | 3.888250 | +0.084329% |
| overall jitter/m | 1.107842 | 1.092107 | -1.420318% |
| clean ADE/m | 1.706452 | 1.705750 | -0.041145% |
| clean FDE p95/m | 12.666380 | 12.690068 | +0.187020% |

四项精度约束通过，jitter下降超过诊断门槛0.5%，且严格低于revision_fixed和revision_0。
输出meets_diagnostic_gate=true，但 **advance=false、val_run=false**。这只表明旧候选中
存在保守的train_val权衡，不能替换推荐模型或宣称新的val收益。

## 预设余量网格

| overall/clean ADE允许增加 | 选中候选 | 合格数 |
| --- | --- | ---: |
| 0% | fixed_005 | 3 |
| 0.05%（主规则） | adaptive_matched_005 | 4 |
| 0.10% | fixed_010 | 5 |
| 0.15% | adaptive_matched_010 | 6 |
| 0.20% | adaptive_matched_010 | 6 |

FDE与尾部约束在各行不变；网格只作敏感性诊断，不能事后换主规则。零ADE退化要求下
fixed_005可行，扩大精度余量则逐渐选择更强正则，说明必须显式定义可接受的精度代价。

## Pareto与逐seed检查

在overall ADE/FDE/jitter、clean ADE及clean FDE p95五维中，**六候选均位于Pareto前沿**。
因此Pareto筛选没有淘汰候选，不能单靠前沿成员资格确定赢家。

选中候选相对RH的逐seed变化：

| seed | overall ADE | clean ADE | overall FDE | clean FDE p95 | jitter |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2027 | +0.026112% | -0.020794% | +0.057069% | +0.206188% | -1.653191% |
| 2028 | -0.026713% | -0.089822% | +0.025633% | +0.122848% | -1.094100% |
| 2029 | +0.045877% | -0.012859% | +0.169955% | +0.231913% | -1.513059% |

三seed各自也未超过主规则误差上限；这不是额外选择条件，不等于独立场景泛化验证。
本轮不新增置信区间，不重新解释P6.5区间，不据已知历史val结果重选。

## 工程验收与产物

新增14项测试，完整 **520 passed in 86.84s**；Ruff、公开扫描通过。覆盖父身份、
表格重建、数值篡改、完整矩阵、精度约束、tie/回退、Pareto、输出拒绝覆盖及不访问val。
旧P1–P6.5源码、配置、测试和输出保持原内容。

输出位于outputs/context/p66-selection：candidates.csv、per_seed.csv、margins.csv、
report.json与最后发布的completion.json。report绑定父输入、协议和执行源码哈希。

| 文件 | SHA256 |
| --- | --- |
| 父selection.json | 508bdef16942f9ce4996f9e15649d4f68768d1131557e039a2f9950d40d94a34 |
| candidates.csv | 4e449bd74a462d32a5c991620996fbb47d8865ef237b1d1a6167355d4969807d |
| per_seed.csv | 298a4861aefb327dddaf85f602429a96b2890e9bf6b297d3558898a1de9e68e4 |
| margins.csv | ae24803288b10d8e6719d5c9e2628f0acc4b2c51e21b568816e6ead7047d57eb |
| report.json | a6ec74d5b824a02236ecb716936f68c60bd944a6c008d3dd51a21e069fdbe94c |

无新增训练、延迟、恢复或传感器测量。产物不入Git；不推送、不合并、不改简历。

## 复现与后续

在开发worktree设置PYTHONPATH=src，使用原完整依赖环境：

```powershell
python scripts/audit_precision_selection.py --selection-root outputs/context/p65/selection --output-root outputs/context/p66-selection-rerun
```

输出目录必须尚不存在。工具拒绝覆盖已完成结果；再次运行须核对相同父身份和源码。

下一轮可独立设计新的epoch候选保存与精度约束checkpoint选择。需要上游也未见的新场景
支持独立泛化判断；不能根据已知历史val反馈宣告本轮成功。P7及其他传感器扩展未启动。

见[冻结协议](superpowers/specs/2026-11-06-causal-context-phase66-selection-design.md)与
[实施计划](superpowers/plans/2026-11-06-causal-context-phase66-selection.md)。

# P6.1 Gate Refinement Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline with review checkpoints.

**Goal:** 完成门控归因、短时收缩和跨更新一致性的完整三seed可审计实验。
**Architecture:** 冻结P6缓存→只读候选适配→校准/配对训练→train_val冻结选择→val重放报告。
**Tech Stack:** Python3.12、PyTorch、NumPy mmap/NPZ、pandas、pytest；现有venv/GPU。

设计：[spec](../specs/2026-10-21-causal-context-phase61-design.md)。所有设置以该spec为准。
执行python=../trajectory-mvp/.venv/Scripts/python.exe，PYTHONPATH=当前src，测试独占outputs/pytest-GUID。

## Task 1：适配、cap与配对算子

文件src/trajectory_prediction/online/gate_refinement.py；tests/online/test_gate_refinement.py。
- [x] 写失败测试：original输入candidate/quality分歧重建、不改原缓存；labels不进模型。
- [x] 写失败测试：tau0等于原gate、h=.5 cap0、无效包精确CV；pair只同轨迹过去update。
- [x] 实现RefinementDataset(inputs/labels/fingerprint)、shrink_weights、paired_revision。
  公共调用为 inputs=data.inputs(indices,device)，candidate可选map/original，steps从索引派生；
  shrink_weights(g,remaining_s,tau)要求finite/tau>=0；paired_revision(now,previous,mask,has_previous)
  用mask&has_previous[:,None]求L2均值，空集返回可导0。
- [x] pytest聚焦RED→GREEN；Ruff；本地提交。

## Task 2：校准和身份绑定训练

文件online/refinement_training.py、scripts/train_gate_refinement.py、configs/models/gate_refinement.yaml；
tests/online/test_refinement_training.py。
- [x] 失败测试：constant/lookup拟合不读val；scene overlap拒绝；缓存实际变更拒绝复用。
- [x] 实现constant21网格和lookup11×5×21有界batch拟合；缺格0；写calibration.json含fit split。
- [x] 实现train_refinement：原gate训练、三种lambda匹配微调、逐epoch监督损失/配对损失，
  同seed初始化/顺序；配对indices-1仅k>0，不跨trajectory边界。
- [x] 保存best.pt、完整settings/source/实际输入fingerprint/run.json，完成产物原子写入。
  Checkpoint加载核对architecture/seed/P6权重/cache contract/源码和评估scene无交集。
- [x] 合成2epoch及best checkpoint复用测试通过；完整train/train_val三seed；本地提交。

## Task 3：选模、全序列度量和新产物

文件online/refinement_evaluation.py、scripts/evaluate_gate_refinement.py；tests/online/test_refinement_evaluation.py。
- [x] 失败测试：12策略同keys/shape、原P6 parity、missing/stale硬回退、跨更新jitter对齐。
- [x] 实现forecast_variant与evaluate_refinements，输入已验证cache与对应checkpoint，
  输出12个[6,N,11,3]指标与gateweights，逐update分块前向后按轨迹计算jitter。
- [x] 在train_val按固定网格选择tau、consistent_short lambda/tau，三seed均值选共同设置；
  accuracy和stable规则写selection.json并hash所有selection表/模型，不访问val。
- [x] 冻结selection后完整val三seed，metrics.npz/weights.npz+strict manifest；scene sum/count、
  condition/update/horizon/maneuver/quantile/recovery表；完整源身份与原数组shape/keyhash。
- [x] 固定256keys/16warmup同步gate路径benchmark，独立原始计时scope和hash；本地提交。

## Task 4：严格重放报告

文件analysis/refinement_report.py、scripts/report_gate_refinement.py；tests/analysis/test_refinement_report.py。
- [x] 写失败测试：错shape/hash/漏策略/源契约/选择文件/指标篡改（即使更新hash）拒绝。
- [x] 生产load_verified_gate_cache+load全部模型重算所有NPZ元素；strict列集合和round-trip
  CSV读取；重新计算所有scene/group/quantile/recovery表、对照parity和benchmarkkeys。
- [x] 固定三seed平均后scene bootstrap2000(seed0)，输出全条件seed mean/std、完整差值区间，
  明确accuracy/stable是否实现预期，保留退化结果。完整运行与来源固定后提交。

## Task 5：审查与验收

- [x] requesting-code-review独立只读审查各核心模块；发现问题先复现再修复并回归。
- [x] 全量pytest、Ruff、公开扫描、diff检查；检查旧P6所有sourcehash不变、旧报告仍可验证。
- [x] 写docs/CAUSAL_CONTEXT_PHASE61_RESULTS.md，更新roadmap和此计划；本地提交，保留worktree不推送。

验收：2026-10-21，449项pytest、Ruff、公开扫描及diff检查通过；完整三seed
17,109,576条val记录已严格重放，180条场景区间已生成。结果见
[实验报告](../../CAUSAL_CONTEXT_PHASE61_RESULTS.md)。保留本地分支/worktree，不推送。

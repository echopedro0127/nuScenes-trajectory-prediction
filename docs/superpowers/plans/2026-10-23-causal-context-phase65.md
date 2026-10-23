# P6.5 Matched Consistency Strength Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline, task by task, with independent read-only review before training and after reporting. Do not delegate implementation.

**Goal:** 执行已批准的四组强度校准、三seed训练、冻结选择与条件历史val回归。
**Architecture:** 复用冻结InnovationDataset/weighted_revision/RH，新增独立系数、训练身份、22策略产物/评估与315区间报告模块。旧P6至P6.4代码与产物不改。
**Tech Stack:** 现有Python/PyTorch/CUDA/NumPy/pandas/pytest，无新依赖。

协议见[设计](../specs/2026-10-23-causal-context-phase65-design.md)，用户已确认继续。
既有worktree与feature/causal-context-online分支复用，不新建worktree、不改简历或D盘文件。

## Task 1：系数规则与均值控制

Files: 新增src/trajectory_prediction/online/strength_consistency.py；tests/online/test_strength_consistency.py。

- [ ] 先写失败测试，精确覆盖四arm和非法arm/mu（0、负、>1、NaN、inf）。
- [ ] 新接口strength_settings(arm, mu)返回target_mean_weight、revision_weight、revision_kind；fixed取alpha，adaptive取alpha/mu，不改变[0,1]权重。

```python
ARMS = ("fixed_005", "adaptive_matched_005", "fixed_010", "adaptive_matched_010")
# alpha=.05 or .10; kind="fixed" or "adaptive"
# returned settings: target_mean_weight=alpha, revision_weight=alpha/mu if adaptive else alpha
```

- [ ] 写实际weighted_revision损失及双方梯度等价测试：alpha/mu * constant(mu) == alpha * constant(1)，另测adaptive训练平均系数匹配。
- [ ] 运行聚焦测试确认RED，再实现、运行GREEN/Ruff；不修改旧loss或数据模块。

## Task 2：独立训练与可信生产加载

Files: 新增online/strength_training.py、configs/models/strength_consistency.yaml、scripts/train_strength_consistency.py、tests/online/test_strength_training.py。

- [ ] 从测试起覆盖四arm完成复用、actual input/mean/arm/lambda篡改拒绝、场景重叠、同初始化。
- [ ] 以P6.4训练文件为只读参考建立独立实现；导入Task1系数规则，沿用run_consistency_epoch作为只读原语，kind映射到revision_fixed/revision_adaptive；不修改其globals。
- [ ] 元数据新增alpha/kind/raw lambda/actual mu/训练指纹，独立architecture、source/config与run manifest；完成marker原子最后发布。
- [ ] 从完整train验证缓存重建mu；生产load_strength必须显式接收trusted mu/content，核对lambda=alpha/mu及所有默认设置、角色/mask、父初始化与scene契约。
- [ ] 加入P6.4冻结selection/report引用校验（hash+source+advance=false），核对旧zero/RH状态指纹，父RH与P6.3选择沿用严格角色检查。
- [ ] CLI仅train/train_val，四arm同P6初始化、seed/shuffle、10epoch/patience3/batch4096；初始完整八变体parity保存。无新zero重复训练。
- [ ] 实际production loader正常加载再篡改mu、alpha、lambda、arm和所有checksums；必须仍拒绝。测试正式默认预算及显式CuBLAS。
- [ ] 聚焦GREEN/Ruff后本地提交训练代码，不启动正式训练。

训练CLI参数：--cache-root、--p6-training-root、--horizon-training-root、--horizon-selection-root、--consistency-selection-root、--consistency-report-root、--output-root、--device。

## Task 3：22策略评估及冻结选择

Files: 新增online/strength_artifacts.py、online/strength_evaluation.py、scripts/evaluate_strength_consistency.py、tests/online/test_strength_evaluation.py。

- [ ] 先写失败矩阵测试：22策略×6条件×3seed，选择池仅RH/revision_fixed/新4；ADE/FDE/clean p95限制、tie、旧组获选和未达门槛禁止val。
- [ ] 用独立artifact模块保存22策略metrics/weights/分组表，严格ordered keys/shape/dtype/mask，atomic summary最后；NPZ篡改且重算hash重放仍拒绝。
- [ ] parent load_context通过复制args把training_root设为P6.4，并调用冻结P6.4 loader；新增四组严格load_strength，实际train mu/content绑定。
- [ ] parent evaluate_models先执行旧18，再追加新4；derive_tables独立覆盖22，benchmark包含实际stream_inputs状态更新，旧六基线用RefinementDataset。
- [ ] 固定选择实现choose_strength(frame)：eligible四误差约束；sort(jitter,ADE,name)；advance仅新组、jitter<=.995*RH且严格优于revision_fixed/zero。
- [ ] 冻结selection保存完整22矩阵、来源、summary引用；fit_selection核对旧18的完整P6.4 train_val表，不能复用旧selection决定新选择。
- [ ] CLI extra roots：--consistency-training-root、--consistency-selection-root、--consistency-evaluation-root、--consistency-report-root；保留P6.3/P6.1/cache/output roots。
- [ ] 完整合成22策略预测、tables、真实streaming profile与后续audit_latency通过，旧基线不构造额外特征。未晋级不读val；晋级本轮评22策略不改旧结果。

## Task 4：全量生产重放与315条区间

Files: 新增analysis/strength_report.py、scripts/report_strength_consistency.py、tests/analysis/test_strength_report.py。

- [ ] 先写失败测试，三个seed固定平均、配对scene sum/count，15对比×7范围×3指标=315；缺scene/seed/condition拒绝。
- [ ] 15对比按spec第5节构造；报告arm effects均值/std，研究目标只按固定方案和RH守门。
- [ ] audit_training由实际train重建mu/权重stats/fingerprint，重载12模型、run log与父引用，核验source/config/alpha/lambda/kind。
- [ ] audit_one重算全部22策略metrics/weights/表/选择、旧18对照parity；val旧18因P6.4无val不查不存在目录，而查本轮重算、契约和冻结模型身份。
- [ ] 验证全部raw latency/keys/hash、真实prefix状态输入，汇总训练实际epoch和时间、分组/尾部/恢复/权重、不利结果。
- [ ] report.json绑定完整文件/源码/模型/cache/selection/运行引用，val_run按本轮冻结决定，写入outputs/context/p65/report。

## Task 5：审查、正式实验和收尾

- [ ] 全量pytest/Ruff、公开扫描、diff；git diff a3af54a --diff-filter=M -- src scripts tests configs为空。
- [ ] 请求独立只读审查训练、选择、加载身份与统计边界，修复问题需先RED→GREEN；训练前提交冻结源码。
- [ ] 运行12模型，完成复用须重新校验；正式训练后源码变化不允许继续复用旧身份。
- [ ] 完整train_val评估→冻结选择→条件val→独立生产重放；失败不放宽检查，不追加系数。
- [ ] 编写docs/CAUSAL_CONTEXT_PHASE65_RESULTS.md，更新roadmap/spec/plan，明确是否晋级及历史val口径。
- [ ] 最终只读结果审查、来源/哈希/行数/区间/模型身份/旧源不变核验；本地提交，不推送、merge或清理worktree。

## 执行命令

```powershell
$python = (Resolve-Path ../trajectory-mvp/.venv/Scripts/python.exe).Path
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
& $python -m pytest tests/online/test_strength_consistency.py tests/online/test_strength_training.py tests/online/test_strength_evaluation.py tests/analysis/test_strength_report.py -q --tb=short --basetemp (Join-Path (Get-Location) ('outputs/pytest-'+[guid]::NewGuid().ToString('N')))
& $python -m pytest -q --tb=short --basetemp (Join-Path (Get-Location) ('outputs/pytest-'+[guid]::NewGuid().ToString('N')))
& $python -m ruff check --no-cache .
& $python scripts/check_public_repo.py
```

正式路径：cache outputs/context/p6/cache；P6初始化outputs/context/p6/training；P6.1 training/selection在outputs/context/p61；P6.3 training-deterministic/selection/evaluation在outputs/context/p63；P6.4 training/selection/evaluation/report在outputs/context/p64；本轮outputs/context/p65/training、selection、evaluation、report。

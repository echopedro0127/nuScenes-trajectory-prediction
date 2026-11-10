# Phase 3A：冻结模型场景评测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: 使用superpowers:executing-plans顺序执行；代码步骤遵循superpowers:test-driven-development，完成前使用superpowers:verification-before-completion。保留阶段检查点，不默认启动子代理。

**Goal:** 在不训练新模型的前提下，回答修正是否有效、异常恢复多快以及哪些机动失败。

**Architecture:** 在原worktree验证父实验；从冻结候选/特征/checkpoint重建预测；纯指标函数与GT离线标签分别处理；最后合并逐查询/事件、log、训练seed结果。模型输入构建不能导入新GT标签模块。

**Tech Stack:** 既有Python/NumPy/PyTorch/pandas/pytest/Matplotlib环境，不新增大型依赖。

**Status:** 仅计划，下面列出的新源码、配置、测试和CLI尚未创建；本次只写文档。

完整协议见[开发设计](../../SCENARIO_EVALUATION_DEVELOPMENT.md)。后续在当前`feature/scenario-evaluation`独立worktree执行；父`noise-aware-stability`只读。以下所有新CLI命令均是拟实现接口，当前不能当作已可运行命令。

## 文件职责

| 拟创建文件 | 单一职责 |
| --- | --- |
| `configs/analysis/scenario_evaluation.yaml` | v1指标/事件/模型选择和时间索引配置 |
| `src/trajectory_prediction/evaluation/streaming_metrics.py` | 固定提前量、同时间点增益、方向/尾部指标纯函数 |
| `src/trajectory_prediction/evaluation/event_metrics.py` | 首个离群事件配对、AUC和持续恢复定义 |
| `src/trajectory_prediction/analysis/scenario_events.py` | GT事件标签、候选标签、人工抽查清单及episode键 |
| `src/trajectory_prediction/evaluation/scenario_replay.py` | 原环境父验证、冻结模型预测重建、来源绑定；不训练 |
| `src/trajectory_prediction/evaluation/scenario_report.py` | 分组/逐log/逐训练seed统计、独立verify、确定性案例图 |
| `scripts/run_scenario_evaluation.py` | 统一入口与verify-only |
| `tests/evaluation/test_streaming_metrics.py` | 时间索引/增益/无效方向及尾部边界 |
| `tests/evaluation/test_event_metrics.py` | 未受损、延后受损、恢复/反复超阈值/缺失配对 |
| `tests/analysis/test_scenario_events.py` | 事件标签、S弯抵消及去重 |
| `tests/evaluation/test_scenario_replay.py` | 冻结父链/前缀/模型不训练/原指标兼容 |
| `tests/evaluation/test_scenario_report.py` | 小型完整产物、重算、篡改拒绝及统计分母 |

不修改旧`analysis/maneuvers.py`、旧阈值文件或旧评测定义；新标签共存。停车时刻/超调、真实道路越界和邻车风险指标放在后续阶段，不塞进首轮实现。

## Task 1：冻结配置和时间契约

- [ ] 将以下初始配置写入新YAML，并记录schema `scenario_evaluation_v1`：

```yaml
schema: scenario_evaluation_v1
dt: 0.5
query_steps: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
lead_steps: [1, 2, 4]
primary_lead_steps: 2
revision_tolerance_m: 0.05
miss_distance_m: 2.0
heading_min_speed_mps: 0.5
quantiles: [0.5, 0.95, 0.99]
recovery:
  condition: noise_outlier
  paired_reference: noise_05
  source_step: 4
  last_query_step: 7
  excess_threshold_m: 0.5
  consecutive_updates: 2
  require_no_relapse: true
roles: [independent_fit, independent_select]
conditions: [clean, noise_05, noise_outlier]
model_seeds: [2027, 2028, 2029]
gate_variants: [baseline7, plus_ca, guarded_ca]
physical_candidates: [raw_cv, alpha_beta, kf_cv, kf_nis, bounded_ca, brake_stop, hold_position]
event_labels:
  speed_median_window: 3
  braking_acceleration_mps2: -2.0
  braking_duration_steps: 2
  braking_prior_speed_mps: 2.0
  stop_speed_mps: 0.3
  stop_duration_steps: 2
  stop_prior_speed_mps: 1.0
  prior_window_steps: 4
  restart_speed_mps: 1.0
  restart_duration_steps: 2
  turn_min_speed_mps: 0.5
  s_turn_absolute_yaw_deg: 45.0
  s_turn_max_net_yaw_deg: 15.0
  s_turn_each_direction_deg: 20.0
  u_turn_net_yaw_deg: 135.0
  manual_review_cap_per_log_tag: 20
```

- [ ] 在`test_scenario_replay.py`先写无效lead/dt、缺失schema、未知角色/条件、重复模型seed拒绝测试，确认RED；在`scenario_replay.py`实现`validate_protocol(config)`后GREEN。只允许fit/select，holdout角色直接拒绝；纯指标模块不执行文件IO。
- [ ] 输出协议中保存整数查询/目标列，不用浮点相等寻找时间点。人工核算h=0.5/1/2s分别11/10/8个有效查询，h=1s增益9对。
- [ ] 提交配置和校验；后续若改定义必须升schema或metric版本，不覆盖旧协议。

## Task 2：纯在线指标，先测试再实现

- [ ] 在`test_streaming_metrics.py`加入以下数值夹具（函数名为拟定API），确认缺模块的RED：

```python
import numpy as np
from trajectory_prediction.evaluation.streaming_metrics import (
    fixed_lead_errors, revision_gains,
)

def test_one_second_lead_uses_absolute_target_column():
    p = np.zeros((1, 11, 12, 2), np.float32)
    y = np.zeros((1, 12, 2), np.float32)
    p[0, 0, 2, 0] = 1.0
    r = fixed_lead_errors(p, y, lead_steps=2)
    assert r['values'].shape == (1, 10)
    assert r['query_steps'].tolist() == list(range(1, 11))
    assert r['values'][0, 0] == 1.0

def test_gain_compares_the_same_target_time():
    p = np.zeros((1, 11, 12, 2), np.float32)
    y = np.zeros((1, 12, 2), np.float32)
    p[0, 0, 3, 0], p[0, 1, 3, 0] = 3.0, 1.0
    r = revision_gains(p, y, lead_steps=2, tolerance_m=0.05)
    assert r['values'].shape == (1, 9)
    assert r['values'][0, 0] == 2.0
```

- [ ] 实现该纯函数API：预测/GT shape及有限性校验；返回values/query_steps/target_columns；无效尾部排除，不生成伪GT。增益的values指有符号G，另返回beneficial/harmful/neutral布尔数组。
- [ ] 增加GT当前参考方向投影测试：同一误差在固定d/n下有已知纵横向分量，零速无历史方向记invalid，不得取未来方向补齐。
- [ ] 增加Miss边界、空/少样本quantile分母测试。阈值使用严格`>2m`，同值不算miss；P95/P99始终附带窗口/log/更新数量。
- [ ] GREEN后保持原ADE/FDE/jitter函数调用不变，提交纯指标模块。

## Task 3：首个离群事件的配对与恢复

- [ ] 写`test_event_metrics.py`并先RED；接口`recovery_from_excess(excess, dt, threshold_m, consecutive_updates)`只处理已配对的非负A数组，输出affected/recovered/confirm_delay_s/excess_auc。

```python
import numpy as np
from trajectory_prediction.evaluation.event_metrics import recovery_from_excess

def test_recovery_needs_sustained_confirmation():
    a = np.array([[1.0, .3, .2, .1],
                  [1.0, .1, .1, .8],
                  [.1, .8, .2, .1],
                  [.1, .2, .1, .3]])
    r = recovery_from_excess(a, dt=.5, threshold_m=.5,
                            consecutive_updates=2)
    assert r['recovered'].tolist() == [True, False, True, False]
    assert r['affected'].tolist() == [True, True, True, False]
    assert r['confirm_delay_s'][0] == 1.0
    assert r['confirm_delay_s'][2] == 1.5
    assert np.isnan(r['confirm_delay_s'][1])
    assert np.isclose(r['excess_auc'][0], .525)
```

- [ ] 配对按role/instance/sample/noise_seed/模型身份匹配，拒绝缺失/重复配对或GT不一致，不能仅按行号相减。
- [ ] 构造A的上层逻辑只使用固定1s提前量；主事件step4至7，第8步不加入主窗口。窗口不足报告不可评价，不填0。
- [ ] 恢复必须在首次受损后，至少两次连续合格且窗内不再复发；未受损事件单列。确认延迟不是首次低误差时刻，更不是预测剩余时长。
- [ ] 输出每事件A序列、配对键、AUC、恢复状态和确认step，原始证据可独立重算；GREEN后提交。

## Task 4：离线事件标签与抽查清单

- [ ] 写`test_scenario_events.py`，测试净航向接近0的S弯仍有累计转角，低速乱向不能触发转弯标签，平均加速度接近0的走停过程仍可出现急刹事件。

```python
import numpy as np
from trajectory_prediction.analysis.scenario_events import turn_shape_tags

def test_s_turn_cannot_be_lost_by_net_yaw_cancellation():
    angle = np.radians([0, 10, 20, 30, 20, 10, 0, 0, 0, 0, 0, 0])
    future = np.cumsum(np.column_stack((np.cos(angle), np.sin(angle))), axis=0)
    tags = turn_shape_tags(future, dt=.5)
    assert 's_turn_candidate' in tags
```

- [ ] 实现新事件表：原时间戳、onset/end、instance/scene/log、事件类型、规则版本与用于判定的速度/方向证据。旧净航向标签和旧报告保持不变。
- [ ] 实现同log/instance/type、onset相距<=1s的相邻窗口episode合并；事件ID由合并后的身份/时间确定性哈希产生，窗口ID保留。新标签不可加入predict_motion输入。
- [ ] 只对已有窗口打标签，不改变3A采样；生成每log/tag最多20例哈希抽查清单，人工完成字段为空或pending时报告未复核，不自动声称标签正确。
- [ ] 图中标注“GT离线标签”，图像/未来标签不传给模型；GREEN后提交。

## Task 5：冻结预测重放与来源绑定

- [ ] 写`test_scenario_replay.py`：小型checkpoint重放、缺参数/改checkpoint拒绝、未来观测改变不影响已发布前缀、推理阶段无optimizer或backward。
- [ ] 原worktree调用父`run_noise_aware_stability.py --verify-only`；绑定父completed/run、样本键、数据NPZ、9个checkpoint及所有metric/scenario配置和新源码哈希。
- [ ] 9模型用父`predict_motion`和相同角色/批次路径重建预测；7物理候选从父缓存取值。禁止把`per_update.npz`里的旧指标当新预测，也不复制物理策略为三个训练seed。
- [ ] GT传给纯评测层；模型重放只接features/candidates/ca_allowed。运行身份含`neural_training=false`、`holdout_evaluated=false`、CPU/线程/版本、Git提交。
- [ ] 导出字段使用`noise_seed`和可空`train_seed`，策略有明确type/identity；读取旧CSV显式dtype并设置low_memory=False，不能把旧空variant当新模型。
- [ ] 数值兼容检查：所有9门控既有ADE/FDE/jitter与父逐项一致；7物理候选对照父引用，说明因batch/dtype造成的差异，不能更新父成绩来迁就新实现。

## Task 6：完整发布、统计与确定性案例

- [ ] 写`test_scenario_report.py`小型端到端：文件/哈希齐全、拒绝覆盖、篡改拒绝、物理候选train_seed为空、漏配对有明确状态、重复噪声不膨胀独立窗口数。
- [ ] 报告至少含逐更新M1/M2、逐事件M3、新旧机动组、M4/M5、逐log、逐训练seed及有效/不可评价计数。禁止把三类合成/真实数据混为单个总分。
- [ ] 先窗口内汇总，再逐策略统计；模型seed先逐模型评价再汇总，样本std不当CI。4个select log下只给描述统计与逐log结果。
- [ ] 预先选取按标签哈希排序的案例及各模型共同的top-error案例，固定策略/时刻集合；图含GT、预测、观测到达时刻、异常位置、同一坐标轴和评价时间点，不只画获胜案例。
- [ ] verify-only独立从预测与GT重算全部纯指标及事件汇总；旧指标只调用旧函数，新指标验证不只复读summary。提交验收代码和冻结协议。

## Task 7：验证后运行与交付

- [ ] 运行新测试，确认RED→GREEN证据；执行全量`pytest`、`ruff check src scripts tests`、`scripts/check_public_repo.py`及`git diff --check`。测试临时目录先创建在outputs下。
- [ ] 审阅是否存在GT标签进入输入、缺包补GT、不同目标时间相减、缩短horizon冒充恢复、模型seed混为noise seed等问题。
- [ ] 冻结源码/配置再运行以下拟实现接口；此处是接口约定，文档阶段不执行：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/run_scenario_evaluation.py `
  --parent ../noise-aware-stability/outputs/noise-aware/phase2d `
  --config configs/analysis/scenario_evaluation.yaml `
  --output-root outputs/scenario-evaluation/phase3a-v1
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/run_scenario_evaluation.py `
  --output-root outputs/scenario-evaluation/phase3a-v1 --verify-only
```

- [ ] 最终产物：`protocol.json`、`run.json`、`keys.json`、逐查询/事件CSV或NPZ、`summary.json`、`event_review.csv`、案例图、`audit.json`、`completed.json`。未复核标签明确标注，不能宣称人工完成。
- [ ] 写`docs/SCENARIO_EVALUATION_PHASE3A_RESULTS.md`，记录所有模型及退化/缺失，不把“新指标有了数字”说成模型提升；更新路线，本地提交。GitHub推送不在本次范围。

## 阶段完成标准

可以解释每个指标的分母、单位、对应时间点、有效性及来源；原成绩不变，新指标和配对可重算；冻结推理无训练；所有模型共享观测；缺失/未复核/不可评价有独立状态。达到这些标准才进入3B场景扩展，不以分数是否更好作为完成条件。

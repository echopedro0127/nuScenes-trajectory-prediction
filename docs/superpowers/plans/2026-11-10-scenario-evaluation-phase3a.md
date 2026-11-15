# Phase 3A：冻结模型场景评测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: 使用superpowers:executing-plans顺序执行；代码步骤遵循superpowers:test-driven-development，完成前使用superpowers:verification-before-completion。保留阶段检查点，不默认启动子代理。

**Goal:** 在不训练新模型的前提下，回答修正是否有效、异常恢复多快以及哪些机动失败。

**Architecture:** 在原worktree验证父实验；从冻结候选/特征/checkpoint重建预测；纯指标函数与GT离线标签分别处理；最后合并逐查询/事件、log、训练seed结果。模型输入构建不能导入新GT标签模块。

**Tech Stack:** 既有Python/NumPy/PyTorch/pandas/pytest/Matplotlib环境，不新增大型依赖。

**Status:** S0–S6开发验收已实现；S7统一入口和冻结机制正在验收，见[S7协议](../../SCENARIO_EVALUATION_S7_PROTOCOL.md)。人工复核仍pending。正式全量新评分尚未开始，需先通过入口与来源时序检查并提交源码。

**审查修订：2026-11-11；进度更新：2026-11-15。** 已按[R1–R9](../../SCENARIO_EVALUATION_REVIEW.md)修订；S0–S6b阶段验证逐步落实，人工复核单独pending。3A限定窗口标签与相对时钟，canonical episode及停车误差指标属于3B。

完整协议见[开发设计](../../SCENARIO_EVALUATION_DEVELOPMENT.md)。在当前`feature/scenario-evaluation`独立worktree执行；父`noise-aware-stability`只读。S7的完整评测CLI仍是拟实现接口；已可用的S0–S2入口为`scripts/run_scenario_preflight.py`。

## 文件职责

| 拟创建文件 | 单一职责 |
| --- | --- |
| `configs/analysis/scenario_evaluation.yaml` | v1指标/事件/模型选择和时间索引配置 |
| `src/trajectory_prediction/evaluation/streaming_metrics.py` | 固定提前量、同时间点增益、方向/尾部指标纯函数 |
| `src/trajectory_prediction/evaluation/event_metrics.py` | 首个离群事件配对、AUC和持续恢复定义 |
| `src/trajectory_prediction/analysis/scenario_events.py` | GT窗口事件标签、相对onset/end及人工抽查清单；不做episode归并 |
| `src/trajectory_prediction/evaluation/scenario_inputs.py` | 父资源预检、GT历史/未来和窗口身份/时钟装配 |
| `src/trajectory_prediction/evaluation/scenario_replay.py` | 原环境父验证、冻结模型预测重建、来源绑定；不训练 |
| `src/trajectory_prediction/evaluation/scenario_preflight.py` | S0–S2装配/重放/来源证明、就绪产物及独立重算 |
| `src/trajectory_prediction/evaluation/scenario_report.py` | 分组/逐log/逐训练seed统计、独立verify、确定性案例图 |
| `scripts/run_scenario_evaluation.py` | 统一入口与verify-only |
| `scripts/run_scenario_preflight.py` | 已实现的S0–S2专用入口，不运行新指标 |
| `tests/evaluation/test_streaming_metrics.py` | 时间索引/增益/无效方向及尾部边界 |
| `tests/evaluation/test_event_metrics.py` | 未受损、延后受损、恢复/反复超阈值/缺失配对 |
| `tests/analysis/test_scenario_events.py` | 事件标签、S弯抵消、停车断段及持续时间边界 |
| `tests/evaluation/test_scenario_inputs.py` | GT来源、身份错配、历史mask及资源缺失 |
| `tests/evaluation/test_scenario_replay.py` | 冻结父链/前缀/模型不训练/原指标兼容 |
| `tests/evaluation/test_scenario_report.py` | 小型完整产物、重算、篡改拒绝及统计分母 |

不修改旧`analysis/maneuvers.py`、旧阈值文件或旧评测定义；新标签共存。停车时刻/超调、真实道路越界和邻车风险指标放在后续阶段，不塞进首轮实现。

## 依赖顺序与检查点

| 检查点 | 必须依赖 | 交付与放行条件 |
| --- | --- | --- |
| S0协议 | 本次审查 | 索引、分母、异常和字段冻结；不计算正式成绩 |
| S1输入装配 | S0 | 父链一次验证、GT历史及相对时钟、来源哈希 |
| S2真实重放 | S1 | 原批次路径单窗口及旧ADE/FDE/jitter精确兼容 |
| S3固定lead | S0，S2集成夹具 | M1数值/索引/有效mask |
| S4a增益/尾部 | S3 | M2/M5及分位数层级 |
| S4b方向 | S1+S3 | 公共GT方向及独立mask |
| S4c额外误差恢复 | S1+S3 | M3配对、状态和计数守恒 |
| S4d窗口标签 | S0+S1 | candidate标签、pending抽查清单 |
| S5发布器 | S2+S4接口 | 分片P及来源产物，小夹具通过 |
| S6报告器 | S4a–d+S5 | 统计与独立复算，小夹具通过 |
| S7正式运行 | S0–S6测试通过 | 冻结后全量重放/复算/结果报告 |

S4a–d可独立推进，标签不阻塞M1/M2数值测试；执行仍使用当前会话，无须启动子代理。S5/S6是开发验收，正式全量实验只在S7运行。

## S0：冻结配置和时间契约

- [x] 将以下初始配置写入新YAML，并记录schema `scenario_evaluation_v1`：

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
quantile_method: linear
tail_unit: query_error
mean_reduction: query_then_noise_then_window
time_basis: window_relative
event_scope: window
missing_expected_pair: error
nonfinite_prediction: error
json_nonfinite: reject
noise_seeds: [2027, 2028, 2029]
inference:
  dtype: float32
  device: cpu
  threads: 2
  flat_batch_size: 4096
  compatibility: exact
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

- [x] 在`test_scenario_replay.py`先写无效lead/dt、缺失schema、未知角色/条件、重复模型seed拒绝测试，确认RED；在`scenario_replay.py`实现`validate_protocol(config)`后GREEN。只允许fit/select，holdout角色直接拒绝；纯指标模块不执行文件IO。
- [x] 输出协议中保存整数查询/目标列，不用浮点相等寻找时间点。核算h=0.5/1/2s分别11/10/8个有效查询，h=1s增益9对。
- [x] 配置和校验纳入本次本地实现提交；后续若改定义必须升schema或metric版本，不覆盖旧协议。

## S1：输入装配与父资源预检

**Files:** 新建`scenario_inputs.py`及`test_scenario_inputs.py`，扩展`test_scenario_replay.py`。

- [x] 先RED：对小型父manifest/records/NPZ及原GT缓存夹具，测试按role/instance/sample匹配；相同行号换instance、重复键、GT未来错配、role/log交集、缺文件/统计/checkpoint、非规则时间和非有限输入均拒绝。纯测试不需要nuScenes数据或真实checkpoint。
- [x] GREEN：实现`assemble_inputs(parent, parent_project_root)`，以`ScenarioInputs`返回datasets/records/gt_windows/references等字段；GT含window_key、历史mask、未来、坐标/单位、相对时间及来源，不从门控速度反推。
- [x] 固化owner roots映射：直接父root显式传入，上游parent_path链经Git解析真实worktree root；不按目录深度猜测。当前冻结原入口布局已通过预检，未修改父源码。
- [x] 原worktree执行父`run_noise_aware_stability.py --verify-only`一次并保存证明；绑定completed/run、records、角色NPZ、9checkpoint/统计、GT缓存manifest和数据哈希。缺资源列出路径及再生入口，不自动训练或下载。
- [x] 检查模型输入与评价GT分离：回放仅features/candidates/ca_allowed；GT历史/未来/标签由独立评价层消费。3A只输出相对时钟；绝对timestamp/episode不作前置条件。
- [x] 真实窗口装配核对键、历史、12点未来、11个查询和候选来源；父完整模型历史/未来意外缺失直接失败，纳入本次本地提交。

## S2：提前验收真实冻结重放及旧指标

**Files:** 新建`scenario_replay.py`核心适配及`test_scenario_replay.py`相关测试。

- [x] RED→GREEN：小型checkpoint重放、checkpoint被改拒绝、未来观测改变不影响输出前缀、无optimizer/backward；调用原forecast_metrics，未重写旧指标。
- [x] 冻结9门控使用父predict_motion、归一化统计、float32/model.eval/CPU两线程及原4096平坦批次路径。从完整role前向提取选定窗口批次邻域，未改成独立batch=1。7物理策略读取已核验父候选。
- [x] 两角色各373条批次邻域流：9门控逐更新ADE/FDE/jitter、7物理策略窗口均值与父结果精确相等。物理策略父引用不都含逐更新数组，明确只主张窗口均值兼容；完整新评测兼容仍留S7。
- [x] 记录字段/身份/父证明/shape，保存所选P并独立磁盘重载重放；纳入本次本地实现提交。下一步为S3。

## S3：固定lead已完成（2026-11-12）

- [x] fixed_lead_errors纯函数：有限输入、零基行列、11/10/8个有效查询及独立mask/相对时钟。
- [x] 先RED再GREEN，覆盖标量参考、随机float32/64、明确行列哨兵、尾部及非有限/数值范围边界。
- [x] 新指标统一float64；独立math.hypot比对预设atol/rtol=1e-12，旧指标仍精确兼容。
- [x] 已保存的32个真实S2预测分片、三个lead共346,144个误差逐项对比通过；这是子集集成验证，不是S7正式评分。
- [x] 来源快照切换回归与审查修复通过；不改S0–S2冻结源码。实现及结果纳入本次本地提交。

## S4a增益/尾部与S4b方向已完成

- [x] 新增revision_metrics.py，保持S3源码冻结；同目标增益、严格阈值分类及更新对/窗口宏平均分母分名。
- [x] 固定lead分位数、窗口均值分位数、严格E>2 miss及按k的online_MR2_remaining。
- [x] 27项新测试、完整744项回归通过；真实S2子集310,336个增益、346,144个误差、131,296个remaining标记独立核对通过。
- [x] S4a来源快照审查问题修复，子集结果和局限见阶段报告；S4b/c/d随后已实现，人工标签复核仍pending。

以下保留原联合任务逐项追踪，S4a/b均已完成；结果口径以各阶段报告为准。

- [x] 在`test_streaming_metrics.py`及`test_revision_metrics.py`覆盖以下数值夹具，确认RED后实现（API分模块以保留S3冻结）：

```python
import numpy as np
from trajectory_prediction.evaluation.streaming_metrics import fixed_lead_errors
from trajectory_prediction.evaluation.revision_metrics import revision_gains

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

- [x] 实现该纯函数API：预测/GT shape及有限性校验；返回values/query_steps/target_columns；无效尾部排除，不生成伪GT。增益的values指有符号G，另返回beneficial/harmful/neutral布尔数组。
- [x] 唯一索引为`j=k+m-1; current=P[:,k-1,j,:]; previous=P[:,k-2,j,:]`，previous仅k>=2。输出query_rows/query_times_s/target_times_s；另用逐k慢速标量参考测试全部h及随机小夹具，不复用同一向量实现证明自己。
- [x] 增加GT当前参考方向投影测试：同一误差在固定d/n下有已知纵横向分量，零速无历史方向记invalid，不得取未来方向补齐。实现位于directional_metrics.py，保留冻结S3模块。
- [x] 增加Miss边界、空/少样本quantile分母测试。严格`>2m`，同值不算miss；P95/P99附带窗口/查询数，集成分组另给log数。
- [x] M2正负0.05恰等值均neutral，三类同分母；用float64阈值夹具区分语义与存储精度。
- [x] M4低速只沿用过去最后有效方向，无方向仅该查询M4无效，不删整窗口；23项新测试及真实子集标量/范数守恒检查通过，来源与mask跨模型一致。
- [x] 验证误差`[[0,0,10],[4,4,4]]`的`p95_query_error=8.5`与`p95_window_mean_error≈3.9666667`分别命名，quantile method=linear；计数覆盖空组、单窗口与noise重复。
- [x] GREEN后保持原ADE/FDE/jitter函数调用不变，纯指标模块纳入本次本地提交。

## S4c：首个离群事件的配对与额外位置误差恢复

已完成，API位于event_metrics.py，完整配对pilot及独立verify-only入口为scripts/check_scenario_recovery.py。原S2邻域没有完整同seed配对，故预先固定哈希每角色cap8窗口、三个noise seed，从父完整缓存冻结重放双条件；这不改变S7原全量队列。详见阶段报告。

- [x] 写`test_event_metrics.py`并先RED；接口`recovery_from_excess(excess, dt, threshold_m, consecutive_updates)`只处理已配对的非负A数组，输出affected/recovered/confirm_delay_s/excess_auc。

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

- [x] 配对按role/instance/sample/noise_seed/模型身份匹配，拒绝缺失/重复配对或GT不一致，不仅按行号相减。
- [x] 构造A只用固定1s提前量；主事件step4至7，step8不加入。v1完整预测/配对缺失即终止；未来短窗协议未实现，不以不可评价跳过本轮错误。
- [x] 恢复在首次受损后至少两次连续合格且窗末无复发；未受损单列。确认延迟不是首次低误差或预测剩余时长，最终无复发状态需离线窗末复核。
- [x] 输出每事件A、配对键、AUC、状态、确认step及原始P；独立标量与全预测重放通过，纳入本次本地提交。
- [x] 计数守恒；AUC含未受损，恢复率分母affected，未受损率分母eligible，延迟仅recovered并给计数，零分母null；策略共同eligible，affected可不同。
- [x] A=0但预测相距sqrt(18)夹具通过；联合绝对E与D诊断，不称状态恢复。确认延迟仅1.0/1.5s或未确认。

## S4d：离线窗口标签与抽查清单

- [x] 写`test_scenario_events.py`，覆盖净航向抵消的S弯、低速断段、走停中的局部急刹及删失边界；先RED再实现。

```python
import numpy as np
from trajectory_prediction.analysis.scenario_events import turn_shape_tags

def test_s_turn_cannot_be_lost_by_net_yaw_cancellation():
    angle = np.radians([0, 10, 20, 30, 20, 10, 0, 0, 0, 0, 0, 0])
    future = np.cumsum(np.column_stack((np.cos(angle), np.sin(angle))), axis=0)
    tags = turn_shape_tags(future, dt=.5)
    assert 's_turn_candidate' in tags
```

- [x] 窗口事件表包含相对onset/end、身份、规则版本、状态/原因和速度/方向证据；旧标签共存，旧报告不变，无canonical episode ID。
- [x] 速度/平滑/加速度时间和持续条件、prior有效性、方向断段、爬行、短停/窗尾、角度wrap等通过测试；两项stop-go审查缺口已修复。
- [x] 仅窗口规则；3B再定完整track canonical事件，未进行链式归并；标签不进入predict_motion。
- [x] 608个既有窗口去noise重复标注，固定每role/log/tag最多20个阳性窗口，75条清单全部pending，不改采样。
- [x] 离线GT三联图及HTML明确pending，不传入模型；62张图已生成，825项回归通过，代码纳入本次本地提交。
- [x] 保留rule_label、manual_label/reviewer/reason，支持保留history的人工修订API；本轮无人工修订，pending不计完成。
- [ ] 人工复核75条候选，另存带复核人/原因的版本；当前完成0条，不以规则生成替代人工完成。

## S5：全策略分片重放与原始预测发布器

- [ ] 写`test_scenario_replay.py`：小型checkpoint重放、缺参数/改checkpoint拒绝、未来观测改变不影响已发布前缀、推理阶段无optimizer或backward。
- [ ] 复用S1已完成的父验证证明，不每策略/窗口递归验证；将样本键、数据NPZ、9个checkpoint及所有metric/scenario配置和新源码哈希绑定到新run。
- [ ] 9模型用父`predict_motion`和相同角色/批次路径重建预测；7物理候选从父缓存取值。禁止把`per_update.npz`里的旧指标当新预测，也不复制物理策略为三个训练seed。
- [ ] GT传给纯评测层；模型重放只接features/candidates/ca_allowed。运行身份含`neural_training=false`、`holdout_evaluated=false`、CPU/线程/版本、Git提交。
- [ ] 导出字段使用`noise_seed`和可空`train_seed`，策略有明确type/identity；读取旧CSV显式dtype并设置low_memory=False，不能把旧空variant当新模型。
- [ ] 按role/strategy保存原始P分片、行键、query_step/query_row/target_column、dtype/shape/hash及共享GT引用。全部float32 P约92,454,912字节，不含其他产物及峰值RAM；按角色读取缓存一次，禁止循环重复解压和一次堆叠全部模型到GPU。
- [ ] 小型夹具验证分片覆盖、拒绝重复键/缺分片、拒绝覆盖未匹配身份的输出；写completed前必须全部文件哈希及数量守恒，不完整输出不能冒充完成。
- [ ] 完整旧指标兼容在S7执行；当前只开发/验证发布器，不提前跑全量成绩，也不修改父成绩。

## S6：统计、确定性案例及独立审计器

执行拆分（2026-11-12）：S6a先实现M1/M2/M4/M5查询数组、condition/log分组和独立标量核对；S6b依赖S6a装配M3与机动/规则标签；S6c再汇总训练seed、确定性案例及完整报告验收。S6a使用小夹具，不代表以下整个清单已完成；证据见[S6a报告](../../SCENARIO_EVALUATION_S6A_RESULTS.md)。

- [x] 小型端到端已在S6a/b/c三个测试模块落实：文件/哈希、拒绝覆盖、篡改拒绝、物理train_seed为空、缺配对失败、无方向/空组状态、噪声重复不膨胀窗口数。
- [x] S6a逐查询M1/M2/M4/M5、S6b逐事件M3及新旧机动组、逐log有效计数、S6c逐训练模型统计串联；condition分别报告，不生成混池总分。
- [x] query→noise→window均值、linear查询分位数、M2/M5分母、M3状态守恒、独立log宏均值及有效log数均已实现并标量核对；训练seed先独立评价再ddof=1。select log仅描述统计。
- [x] 合法不可评价以mask/null/status/reason保留，JSON拒绝非有限数；坏预测/GT错配/缺配对失败，不转成统计缺失。
- [x] 标签哈希案例与各策略共同top-error并集，固定展示5策略及k3/4/6；图含GT/预测、到达时间色标、异常位置、共享坐标和1秒lead评价点。夹具6案例/90面板通过生成与重绘核对。
- [x] verify链从P/GT重算指标/汇总/计数，复用原旧指标函数，新指标与汇总独立标量核对；上游模型采用全部冻结预测重放，比计划的固定样本更严格；案例NPZ/PNG/HTML另做语义重算。正式全量执行留S7。

## S7：冻结后正式全量运行与交付

- [ ] 运行新测试，确认RED→GREEN证据；执行全量`pytest`、`ruff check src scripts tests`、`scripts/check_public_repo.py`及`git diff --check`。测试临时目录先创建在outputs下。
- [ ] 审阅是否存在GT标签进入输入、缺包补GT、不同目标时间相减、缩短horizon冒充恢复、模型seed混为noise seed等问题。
- [ ] 先按[S7协议](../../SCENARIO_EVALUATION_S7_PROTOCOL.md)验收并提交源码，再独立冻结，最后执行以下已实现接口；未验收前不运行真实评分：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/run_scenario_evaluation.py `
  --freeze-root outputs/scenario-evaluation/freeze-phase3a-v1 --freeze-only
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/run_scenario_evaluation.py `
  --freeze-root outputs/scenario-evaluation/freeze-phase3a-v1 `
  --parent ../noise-aware-stability/outputs/noise-aware/phase2d `
  --parent-project-root ../noise-aware-stability `
  --config configs/analysis/scenario_evaluation.yaml `
  --output-root outputs/scenario-evaluation/phase3a-v1
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/run_scenario_evaluation.py `
  --output-root outputs/scenario-evaluation/phase3a-v1 --verify-only
```

- [ ] 全部9门控/7物理策略既有ADE/FDE/jitter逐项精确复现父结果；从保存P全量重算新指标/计数，再执行固定样本模型重放。任一失败不写完成标记，不事后改容差。
- [ ] 最终产物：`protocol.json`、`run.json`、`keys.json`、按role/strategy原始P分片及共享GT/来源引用、逐查询/窗口/事件CSV或NPZ、逐log/model-seed统计、`summary.json`、`event_review.csv`、案例图、`audit.json`、`completed.json`。未复核标签明确标注，不能宣称人工完成。
- [ ] 写`docs/SCENARIO_EVALUATION_PHASE3A_RESULTS.md`，记录所有模型及退化/缺失，不把“新指标有了数字”说成模型提升；更新路线，本地提交。GitHub推送不在本次范围。

## 阶段完成标准

以下为实现时必须验收的边界矩阵，文档阶段不勾选通过：

| 边界/异常 | 预期行为 | 检查点 |
| --- | --- | --- |
| k=1、末尾lead不足、dt非0.5或乱序 | 无前次增益；尾部mask；非法时钟拒绝 | S0/S3 |
| 同row不同instance、错GT/角色、重复键/seed、缺父产物 | 终止，列路径与原因 | S1/S2 |
| NaN/Inf预测、shape错误、非米制或不同坐标 | 完整性失败，不当统计缺失跳过 | S1/S3/S5 |
| G=±0.05，E=2，正负零 | neutral；E等于2不算miss | S4a |
| 全窗未受损、延后受损、只有末点合格、恢复后复发 | 独立状态，两次确认且无复发 | S4c |
| step8再注入、目标时间晚于step8 | 主窗只用4至7的输入；远目标本身不构成泄漏 | S2/S4c |
| affected/recovered为零、空组、单log | null及计数/原因，不输出0成功率或显著性结论 | S6 |
| 无方向、低速抖动、±pi跨界 | M4独立mask，过去方向回填；转弯标签低速断段 | S4b/S4d |
| 初始静止、creeping、短停、prior不足、尾部持续时间不足 | 区分状态与不可评价，不制造停车/重启 | S4d |
| 停车前左转/停车后右转、净转角抵消 | 连续可靠段独立判定S弯，不跨停车拼接 | S4d |
| 人工pending、人工与规则不同 | 保留原标签及修订来源，不宣称复核完成 | S4d/S6 |
| 窗口重叠、0/0.8/1.6s链式事件 | 3A仅报窗口数，3B另定canonical episode | S4d/S6 |
| 预测方向改变而GT误差相等、参考和受扰都差 | 联合报告绝对误差，不声称状态恢复 | S4c/S6 |
| 分片缺失/被改、运行中断、输出路径已有结果 | 拒绝发布completed和覆盖已有结果 | S5/S6 |
| 后端/批次变化、父目录移动、资源缺失 | 预检身份与原批次；新容差需先改协议，不迁就失败 | S1/S2/S7 |

可以解释每个指标的分母、单位、对应时间点、有效性及来源；原成绩不变，新指标和配对可重算；冻结推理无训练；所有模型共享观测；缺失/未复核/不可评价有独立状态。达到这些标准才进入3B场景扩展，不以分数是否更好作为完成条件。

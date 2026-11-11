# Phase 3A：冻结模型场景评测 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: 使用superpowers:executing-plans顺序执行；代码步骤遵循superpowers:test-driven-development，完成前使用superpowers:verification-before-completion。保留阶段检查点，不默认启动子代理。

**Goal:** 在不训练新模型的前提下，回答修正是否有效、异常恢复多快以及哪些机动失败。

**Architecture:** 在原worktree验证父实验；从冻结候选/特征/checkpoint重建预测；纯指标函数与GT离线标签分别处理；最后合并逐查询/事件、log、训练seed结果。模型输入构建不能导入新GT标签模块。

**Tech Stack:** 既有Python/NumPy/PyTorch/pandas/pytest/Matplotlib环境，不新增大型依赖。

**Status:** 仅计划，下面列出的新源码、配置、测试和CLI尚未创建；本次只写文档。

**审查修订：2026-11-11。** 已按[R1–R9](../../SCENARIO_EVALUATION_REVIEW.md)修订；本计划所有实现复选框仍未完成。3A限定窗口标签与相对时钟，canonical episode及停车误差指标属于3B。

完整协议见[开发设计](../../SCENARIO_EVALUATION_DEVELOPMENT.md)。后续在当前`feature/scenario-evaluation`独立worktree执行；父`noise-aware-stability`只读。以下所有新CLI命令均是拟实现接口，当前不能当作已可运行命令。

## 文件职责

| 拟创建文件 | 单一职责 |
| --- | --- |
| `configs/analysis/scenario_evaluation.yaml` | v1指标/事件/模型选择和时间索引配置 |
| `src/trajectory_prediction/evaluation/streaming_metrics.py` | 固定提前量、同时间点增益、方向/尾部指标纯函数 |
| `src/trajectory_prediction/evaluation/event_metrics.py` | 首个离群事件配对、AUC和持续恢复定义 |
| `src/trajectory_prediction/analysis/scenario_events.py` | GT窗口事件标签、相对onset/end及人工抽查清单；不做episode归并 |
| `src/trajectory_prediction/evaluation/scenario_inputs.py` | 父资源预检、GT历史/未来和窗口身份/时钟装配 |
| `src/trajectory_prediction/evaluation/scenario_replay.py` | 原环境父验证、冻结模型预测重建、来源绑定；不训练 |
| `src/trajectory_prediction/evaluation/scenario_report.py` | 分组/逐log/逐训练seed统计、独立verify、确定性案例图 |
| `scripts/run_scenario_evaluation.py` | 统一入口与verify-only |
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

- [ ] 在`test_scenario_replay.py`先写无效lead/dt、缺失schema、未知角色/条件、重复模型seed拒绝测试，确认RED；在`scenario_replay.py`实现`validate_protocol(config)`后GREEN。只允许fit/select，holdout角色直接拒绝；纯指标模块不执行文件IO。
- [ ] 输出协议中保存整数查询/目标列，不用浮点相等寻找时间点。人工核算h=0.5/1/2s分别11/10/8个有效查询，h=1s增益9对。
- [ ] 提交配置和校验；后续若改定义必须升schema或metric版本，不覆盖旧协议。

## S1：输入装配与父资源预检

**Files:** 新建`scenario_inputs.py`及`test_scenario_inputs.py`，扩展`test_scenario_replay.py`。

- [ ] 先RED：对小型父manifest/records/NPZ及原GT缓存夹具，测试按role/instance/sample匹配；相同行号换instance、重复键、GT未来错配、role/log交集、缺文件/统计/checkpoint、非规则时间和非有限预测全部抛ValueError或FileNotFoundError。纯测试不需要nuScenes数据或真实checkpoint。
- [ ] GREEN：实现`assemble_inputs(parent, parent_project_root)`，返回`datasets/records/gt_windows/references`；gt_windows含window_key、history_xy/history_mask/future_xy、坐标/单位、query_times及来源引用。GT数组从父run的source_cache_root经原load_cache加载，不能从门控速度反推。
- [ ] 显式owner roots映射供父链验证使用：每个父身份对应其project root及Python入口，路径迁移需重新核对哈希；不按parent.parents固定层数猜源码。当前Phase 2D原入口内部目录约束需在预检中确认，不修改其绑定源码绕过失败。
- [ ] 原worktree执行父`run_noise_aware_stability.py --verify-only`一次并保存证明；绑定completed/run、records、角色NPZ、9checkpoint/统计、GT缓存manifest和数据哈希。缺资源列出路径及设计§4.3再生入口，不自动训练或下载。
- [ ] 检查模型输入与评价GT无共享字段接口：回放仅features/candidates/ca_allowed；GT历史/未来/标签由独立评价层消费。3A只输出相对时钟；绝对timestamp/episode不作前置条件。
- [ ] 真实窗口装配核对键、历史、12点未来、11个查询和候选来源后提交；父完整模型历史/未来意外缺失直接失败。

## S2：提前验收真实冻结重放及旧指标

**Files:** 新建`scenario_replay.py`核心适配及`test_scenario_replay.py`相关测试。

- [ ] RED：小型checkpoint重放、checkpoint被改拒绝、未来观测改变不影响输出前缀、无optimizer/backward；测试调用原forecast_metrics，不能重写旧指标。
- [ ] GREEN：冻结9门控使用父predict_motion、归一化统计、float32/model.eval/CPU两线程及原4096平坦批次路径。选定真实窗口仍从其原批次上下文抽取，不改成独立batch=1后要求逐位相等。7物理策略直接读取已核验父候选。
- [ ] 先对固定真实窗口及其所属批次核对原ADE/FDE/jitter与父逐项精确相等，保留证据；完整9门控/7物理全数据兼容留S7。失败先查dtype/批次/后端，不事后放宽容差。
- [ ] 记录输入字段、模型身份、父验证证明及输出shape；此检查点通过后再推进新指标集成，提交。

## S3：固定lead；S4a增益/尾部；S4b方向

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
- [ ] 唯一索引为`j=k+m-1; current=P[:,k-1,j,:]; previous=P[:,k-2,j,:]`，previous仅k>=2。输出query_row/query_time_s/target_time_s；另用逐k慢速标量参考测试全部h及随机小夹具，不能复用同一向量实现证明自己。
- [ ] 增加GT当前参考方向投影测试：同一误差在固定d/n下有已知纵横向分量，零速无历史方向记invalid，不得取未来方向补齐。
- [ ] 增加Miss边界、空/少样本quantile分母测试。阈值使用严格`>2m`，同值不算miss；P95/P99始终附带窗口/log/更新数量。
- [ ] M2正负0.05恰等值均neutral，三类同分母；用float64阈值夹具区分语义与存储精度。M4低速只沿用过去最后有效方向，无方向仅该查询M4无效，不删整窗口。
- [ ] 验证误差`[[0,0,10],[4,4,4]]`的`p95_query_error=8.5`与`p95_window_mean_error≈3.9666667`分别命名，quantile method=linear；计数覆盖零事件、单log和三noise重复。
- [ ] GREEN后保持原ADE/FDE/jitter函数调用不变，提交纯指标模块。

## S4c：首个离群事件的配对与额外位置误差恢复

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
- [ ] 构造A的上层逻辑只使用固定1s提前量；主事件step4至7，第8步不加入主窗口。v1预期完整预测/配对缺失即终止；不能用不可评价跳过。仅显式声明的未来短窗协议允许insufficient_followup。
- [ ] 恢复必须在首次受损后，至少两次连续合格且窗内不再复发；未受损事件单列。确认延迟不是首次低误差时刻，更不是预测剩余时长。
- [ ] 输出每事件A序列、配对键、AUC、恢复状态和确认step，原始证据可独立重算；GREEN后提交。
- [ ] 计数断言eligible=affected+unaffected、affected=recovered+not_recovered。AUC包含未受损事件；恢复/未恢复率分母affected，未受损率分母eligible，延迟仅recovered并给计数；分母零输出null。比较策略使用共同eligible键，但affected允许依模型不同。
- [ ] 回归GT零、参考(3,0)、受扰(0,3)夹具：A=0且两预测相距sqrt(18)，报告应称额外位置误差恢复并联合绝对E，不声称状态恢复。确认延迟从step4起计，仅1.0/1.5s或未确认。

## S4d：离线窗口标签与抽查清单

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

- [ ] 实现窗口事件表：window_key、相对onset/end、instance/scene/log、事件类型、规则版本、status/reason及原始/平滑速度和方向证据。旧净航向标签和旧报告保持不变；不在3A输出canonical episode ID。
- [ ] 按设计§6实现速度区间、中值边界、加速度时间对齐和完整两区间持续条件；prior四区间不足记不可评价。方向wrap且低速/缺失断段，不能跨停车拼接S弯；0.3至1之间creeping、不算重启。先RED测试初始静止、窗尾不足、短停、阈值等值及±pi跨界，再GREEN。
- [ ] 3B再做完整track canonical事件及窗口关联；本阶段禁止onset<=1s传递闭包合并。新标签不可加入predict_motion输入。
- [ ] 只对已有窗口打标签，不改变3A采样；生成每log/tag最多20例哈希抽查清单，人工完成字段为空或pending时报告未复核，不自动声称标签正确。
- [ ] 图中标注“GT离线标签”，图像/未来标签不传给模型；GREEN后提交。
- [ ] 保存rule_label与人工label/reviewer/reason，review_status仅pending/confirmed/rejected/corrected；人工更改不能覆盖原规则，pending不计已完成。

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

- [ ] 写`test_scenario_report.py`小型端到端：文件/哈希齐全、拒绝覆盖、篡改拒绝、物理候选train_seed为空、缺预期配对抛错、无方向/空组等合法不可评价保留原因、重复噪声不膨胀独立窗口数。
- [ ] 报告至少含逐更新M1/M2、逐事件M3、新旧机动组、M4/M5、逐log、逐训练seed及有效/不可评价计数。禁止把三类合成/真实数据混为单个总分。
- [ ] 严格执行设计§5.1聚合表：均值query→noise→window，分位数query池化且linear，M2/M5比例明确查询/更新分母，M3状态守恒。window主均值与log_macro_mean分名，给有效log数；模型seed先逐模型评价再ddof=1，不混池。4个select log只给描述统计。
- [ ] 内部NaN+mask只能表示声明的不可评价，导出null/status/reason；JSON allow_nan=False。坏预测/GT错配/缺配对终止，不能转成null掩盖完整性错误。
- [ ] 预先选取按标签哈希排序的案例及各模型共同的top-error案例，固定策略/时刻集合；图含GT、预测、观测到达时刻、异常位置、同一坐标轴和评价时间点，不只画获胜案例。
- [ ] verify-only第一层从不可变P/GT重算全部指标/汇总和计数；第二层固定哈希样本模型重放对照P。旧指标调用旧函数，新指标用慢速标量参考校验，不能仅复读summary或重复调用同一向量函数。小夹具全部通过后提交验收代码。

## S7：冻结后正式全量运行与交付

- [ ] 运行新测试，确认RED→GREEN证据；执行全量`pytest`、`ruff check src scripts tests`、`scripts/check_public_repo.py`及`git diff --check`。测试临时目录先创建在outputs下。
- [ ] 审阅是否存在GT标签进入输入、缺包补GT、不同目标时间相减、缩短horizon冒充恢复、模型seed混为noise seed等问题。
- [ ] 冻结源码/配置再运行以下拟实现接口；此处是接口约定，文档阶段不执行：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/run_scenario_evaluation.py `
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

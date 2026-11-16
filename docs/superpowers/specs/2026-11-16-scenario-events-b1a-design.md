# B1a：真实时钟停车／起步候选事件协议设计

设计日期：2026-11-16；验收日期：2026-11-17。**状态：用户已批准并完成实现，真实运行与独立verify-only均通过。** 结果见[B1a验收报告](../../SCENARIO_EVENTS_B1A_RESULTS.md)。本文把首个实现单元收敛为运动学区间、停车、起步及窗口关联。前置数据见[B0b验收报告](../../SCENARIO_TRACKS_B0B_RESULTS.md)。

## 1. 为什么先做这一部分

B0b已经提供1967条完整目标轨迹、60948条标注、22231个缓存窗口和真实微秒时间。25处结构缺口形成1992个annotation连续片段；它们不是驾驶episode。下一步必须建立跨窗口一致的事件对象，才能评价停止时间、停止位置超调／不足、漏停、误停与起步延迟。

B1拆成有依赖的三个单元：B1a先建立停车／起步与删失规则；B1b另定义实际时间下的减速／急刹；B1c另定义方向段及转向事件。不能把旧窗口中的最大方向段直接改名为完整驾驶事件。本文不实现B1b/B1c，不计算预测分数，不训练，不使用holdout选规则。

## 2. 方案选择

| 方案 | 适用性与代价 |
|---|---|
| **推荐：完整连续片段上的确定性状态机** | 明确onset、确认、结束与缺口；可逐区间追踪；标签仍有阈值和标注噪声偏差，需人工复核 |
| 窗口候选按时间接近聚类 | 开发短，但不能恢复被窗口截断的事件，传递闭包会串联多个停走过程，不作为canonical定义 |
| 学习式事件分类 | 可能适应复杂行为，但当前没有已复核episode训练标签，先用它会混淆模型选择与评测对象构造，暂不采用 |

## 3. 输入、隔离与版本

仅接收正式B0b包`outputs/scenario-tracks/b0b-v1`及显式owner。父身份必须为已完成且可原生verify的full_cache范围，不将pilot自动视作完整范围。目标、类别和fit/select log角色继承父包，不能扩对象或重分split。

实现时创建新worktree与分支，旧B0b、B0a、3A源码及产物只读。父验证在原B0b owner的cwd/PYTHONPATH执行一次，保存stdout证明与输入哈希；后续独立重建核对证明和全部来源。不能在新源码环境直接import旧verify并假定原冻结成立。

新配置候选路径为`configs/analysis/scenario_events_b1a.yaml`，规则版本`stop_restart_realclock_v1`。配置完整规范化哈希进入运行身份与事件ID。未知配置键、bool冒充数值、非有限数、阈值关系错误必须拒绝。没有命令行临时阈值覆盖；修改规则必须产生新版本输出和身份。

## 4. 运动学区间：先明确能够观察到什么

对同一B0 segment内相邻点构造区间`[t_i,t_(i+1))`，记录端点annotation/sample token、整数时间、世界XY和`v_i=norm(x_(i+1)-x_i)/(dt_us/1e6)`。先做int64时间差，再转float64秒。这里只能估计整段平均速度，不声称区间内每一时刻的真实车速。

首版**不平滑、不重采样、不插值、不重复端点**。这与3A名义时钟、三点中位数规则不同，必须使用新版本，不改写旧标签。原始速度可能受annotation抖动影响；首版以持续时间和双阈值抑制短暂变化，误检／漏检在后续复核中量化，不能预先承诺效果。

例如两段各499883微秒仅累计999766微秒，首版不会把它当作完整1秒，需等待下一完整区间；这会改变确认延迟与候选数量，必须与旧两步规则分开报告。onset与confirmation都落在真实已有端点上，不声称亚采样间隔的停车时刻精度。所有候选是离线GT规则标签，不能将使用完整轨迹的确认／结束判定称作车端因果检测器。

配置如下，物理阈值取自既有3A契约，时长改为真实时间并明确完整区间数：

| 字段 | 首版值 | 比较方式 |
|---|---:|---|
| stop_speed_mps | 0.3 | `v <= 0.3` |
| restart_speed_mps | 1.0 | `v >= 1.0`，也作为先前运动证据 |
| prior_history_us | 2000000 | onset前必须有完整、未中断的2秒可观察历史 |
| stop_dwell_us | 1000000 | 连续低速完整区间累计时长至少1秒 |
| restart_dwell_us | 1000000 | 连续高速完整区间累计时长至少1秒 |
| minimum_dwell_intervals | 2 | 停／起步均至少两个有效完整区间 |
| max_interval_us | 1000000 | 大于此值属于B1不可评价区间；等值允许 |

max_interval_us是显式数据分辨率保护，不是B0结构缺口规则：即使sample相邻，超过1秒的平均速度也不用于确认停车。该区间单列`not_evaluable: temporal_resolution_gap`，两侧状态重置，不能跨过累计dwell。原始正时间和结构信息保留，不修改父分段。前置只读检查显示fit的38176个有效区间dt为394416–700621微秒；未抽标签或使用select指标调整此阈值。完整数字只是来源描述，不把1秒声称为论文或安全标准。

结构缺口、过长区间把B0 segment进一步切成B1分析block。只有一个点的片段输出`not_evaluable: no_velocity_interval`。所有区间包括不可评价区间均保留，计数守恒，不通过删坏数据改善覆盖率。

## 5. 停车候选与左删失

每个block从未知上下文开始。找最大连续低速run；其起点是第一低速区间左端annotation，dwell是完整区间真实dt之和。首次达到1秒且至少两段时确认，确认token取该完整区间右端点，不能把确认时间插成精确onset＋1秒。

停车onset前的prior窗口为`[onset-2秒,onset)`。可观察历史要求block起点不晚于该窗口左端；运动证据只取端点都在prior窗口内的完整区间，并要求至少一段`v>=1`。不从部分重叠区间伪造新测量。记录实际参与判断的区间索引，等值边界包含，浮点速度比较不增加隐式容差。

低速run满足dwell后分为：

- 有完整prior且有运动证据：产生`stop_candidate`，`status=confirmed_by_rule`。
- prior不足：记录`not_evaluable: insufficient_prior_history`；block开始即低速另记`initially_low_speed`，不能当作新停车或确定负例。
- prior完整但无运动证据：记录`no_recent_motion_evidence`，作为可观察规则未触发诊断；不能解释成现实中肯定没有停车。
- 尚未达到dwell就在block末尾结束：记录`not_evaluable: insufficient_followup`。
- 未达dwell且被正常非低速区间打断：记录`short_low_speed_run`；这是未确认候选，不计作停车事件。

同一block内，确认停车后进入已有停车上下文，直到**确认restart**才允许产生下一次停车事件。爬行、短暂加速或低速run多次出现，都不能重复生成新stop。

## 6. 起步、爬行与事件结束

有已确认stop后，`0.3<v<1`记creeping；首次连续`v>=1`开始restart_pending。高速run达到至少1秒且至少两段才确认`restart_candidate`，onset取首段左端，confirmation取达到条件段的右端。

restart事件表示一次已确认起步转换，其`end_token/end_time`等于confirmation，不把之后整段行驶归为同一事件。确认后清除原停车上下文；持续高速不会重复产生restart，必须先有新的已确认stop。未确认的低速run仍作诊断，但已有停车上下文期间不产生第二个stop。

pending期间出现`v<1`立即取消该次pending并保留诊断，仍属于原停车上下文，不能将两次短加速跨低速段相加。高速run尾部不足确认时长记`insufficient_followup`；不能当作已经起步或负例。

仅有可观察低速上下文而缺少已确认stop时，按下面的确定规则记录`restart_without_confirmed_stop`诊断，不生成stop_go配对事件。initially_low_speed后起步不能冒充本block完整观察到的停车再起步。

上述诊断必须按以下确定规则执行，不是可选输出：无已确认stop的低速run只有达到相同stop dwell及最小区间数，才建立`unpaired_low_context`。记录其来源为initially_low_speed／insufficient_prior_history／no_recent_motion_evidence；未达低速dwell时不建立上下文。上下文在本block内保留至首次已确认高速run或block边界，不跨gap继承，也不凭爬行过期。

上下文来源按以下优先级唯一确定：onset在block首端则为initially_low_speed；否则prior不足则为insufficient_prior_history；否则为no_recent_motion_evidence。原始低速run的主诊断仍按第5节输出；initially_low_speed是附加标记及上下文来源，不额外创建第二个低速run计数。

有unpaired_low_context时，高速run仍要求1秒且至少两段；满足后恰好输出一条`restart_without_confirmed_stop`诊断，附来源原因并清除上下文，持续高速不重复输出。中途取消记`short_restart_run_without_confirmed_stop`，尾部不足记`insufficient_followup_restart_without_confirmed_stop`；这两种情况保留上下文供下一高速run判断。不产生restart事件或stop_go配对。没有任何低速dwell上下文的普通高速run不产生该诊断。

每个最大低速run都必须有可追溯结果：首次合格run关联stop候选，其他run各有一条诊断。有confirmed stop上下文时，后续低速run记`low_run_with_existing_stop_context`；有unpaired_low_context时记`low_run_with_existing_low_context`。两者均记录实际dwell和是否达到时长，抑制重复发stop，直到各自的确认起步／起步诊断消费上下文。短暂加速不能重置这个屏障。

事件时间字段区分两种结束，避免把爬行也称作持续静止：

- `low_dwell_end`：第一次已确认低速run的首个非低速区间左端；若一直低速到边界，端点取block末尾，`low_end_censored=true`。
- `episode_end`：已确认restart的onset；若没有已确认restart，则取block最后可观察端点，`right_censored=true`。该字段是停车上下文结束，不表示整段车速始终<=0.3。

stop可以已确认而结束右删失；这不取消它的onset，但不能据此算出完整停车上下文时长。stop_go配对仅在同block有一个已确认stop和其后首个已确认restart时生成，保留二者ID；不跨缺口、block或新stop屏障配对。

统一事件字段映射如下；通用`end_token/end_time`不得由实现自行选择：

| event_type | onset | confirmation | 通用end | 删失与辅助字段 |
|---|---|---|---|---|
| stop_candidate | 第一合格低速run左端 | 首次满足低速dwell的右端 | 已确认restart的onset，否则block末端 | left_censored=false；无已确认restart时right_censored=true；episode_end是通用end别名，另保留low_dwell_end及low_end_censored |
| restart_candidate | 高速run左端 | 首次满足高速dwell的右端 | 等于confirmation | left/right_censored=false；low_dwell_end/low_end_censored/episode_end均为null |
| stop_go_candidate | 配对stop的onset | 配对restart的confirmation | 等于配对restart的confirmation | left/right_censored=false；保留stop_id/restart_id，其他stop专属辅助字段为null，从stop行查询 |

事件ID包含表中全部通用边界及各自非空辅助边界／删失标志。未确认或左删失停车仅在诊断表，不偷偷创建status不明的候选事件。stop_go是复合关系，统计必须按类型分别报告，不能把stop、restart和stop_go之和解释为独立驾驶过程数。

## 7. 稳定ID、排序与窗口关系

分析block ID绑定B0 segment ID、规则版本和block首尾token。事件ID规范化哈希包含schema、规则配置摘要、父dataset release、track语义摘要、role/log/scene/instance、block ID、event_type、onset/confirmation/end token及删失标志。stop_go ID绑定其stop与restart ID；不得按四舍五入时间或窗口行号生成。

规范排序为role/log/scene/instance/block首时间/event onset/event type/ID；相同输入和配置重建结果一致。改变规则、语义数据或边界应改变相关ID；表布局和源路径不进入事件语义ID，原文件哈希另作运行来源。

窗口关联按真实token时间独立生成多对多表，不合并窗口P。窗口观察区间严格定义为`W=[第一历史点时间,last_future_time)`，包括4段历史；未来onset范围为`F=[anchor_time,last_future_time)`。事件观察区间为`E=[onset_time,end_time)`，其中end按第6节逐类型表取值；右删失stop也仅延伸至block末端，不假定之后状态：

- `onset_in_future`：`anchor_time <= onset_time < last_future_time`。
- `confirmation_in_future`：onset_in_future且`confirmation_time <= last_future_time`。
- `observed_interval_overlap`：W与E有正长度交集；端点相接不算重叠。
- `episode_end_covered`：事件未右删失且`W左端 <= end_time <= last_future_time`；终点端点已被观察，允许等于last_future_time。
- `entire_event_observed`：事件未右删失、`W左端 <= onset_time`且`end_time <= last_future_time`。
- onset早于anchor但区间重叠的，仅作context，不算新未来事件；不能因历史覆盖就称作可预测事件。

分别输出原22231个缓存窗口覆盖与其中608个3A pilot覆盖。新事件覆盖并不代表已有模型预测可用于评分。多个窗口关联同一事件不增加episode数，旧窗口规则标签不自动替换或升级为新事件人工真值。

每个event_type、role、窗口全集分别报告：`events_total`；至少一个窗口满足overlap的`context_overlap_events`；至少一个满足onset_in_future的`future_onset_events`；至少同一个窗口满足onset与confirmation条件的`future_confirmation_events`；至少一个满足entire_event_observed的`entire_event_observed_events`。各自的未覆盖数严格为events_total减相应覆盖数，不输出含义混杂的单一uncovered_events。右删失stop可有confirmation覆盖，但不计完整事件覆盖。

## 8. 输出、可评价分母与验证

建议模块边界：`analysis/realclock_motion.py`仅负责区间和block；`analysis/stop_restart_events.py`仅负责纯状态机、事件与诊断；`evaluation/scenario_events.py`负责父来源、窗口关联及发布／验证；入口`scripts/prepare_scenario_events.py`显式接收`--parent/--parent-owner/--config/--output-root`，独立验收使用`--verify-only`。预测输入构建器不得导入这些GT事件模块。

候选输出为`protocol.json/run.json/parent_proof.json`、逐角色kinematics NPZ、`blocks.json/events.json/diagnostics.json/state_trace.json/window_event_links.json/summary.json`和最后写入的completed标记。state_trace按区间保存状态前后、pending累计微秒／段数、当前stop或unpaired上下文、触发的事件／诊断键；不可评价区间保存重置原因。每条诊断保留role/instance/block/边界token、status/reason、真实观察时长；JSON不允许NaN，空分母为null及原因。

summary按fit/select及类别列出输入track/segment/block/interval、不可评价区间、所有诊断reason、stop/restart/stop_go候选、左右删失、窗口覆盖／未覆盖数量。zero events是合法结果，不能调整阈值凑例；首轮不输出检测准确率、召回率或预测改善。

区间守恒为`sum(segment点数-1)=有效分析区间数+temporal_resolution_gap区间数`；本B0b快照的左侧为60948−1992=58956，不包含25处结构跨sample缺口。block首点不冒充有效区间，单点block另列诊断；未确认run和已确认事件分表计数，不能相加当作episode总量。

prepare先验证父包及来源，执行事件算法，写出后读回比较，再核对来源／源码及文件集合，最后签发completed；已有目录拒绝覆盖，失败／中断不得有完成标记。verify-only从已验证父数组及配置重新计算全部运动学、状态、ID、诊断、关联及摘要，检查dtype、计数和每个字段；不能只验证哈希或summary。

来源绑定包含父B0b原生证明、父run/completed及所有数据引用、配置、源码、Python/NumPy版本；验证前后核对来源与文件集合。旧owner全部冻结源码和产物不得变化。

## 9. 必测边界与实施顺序

1. 运动学夹具：非均匀dt、零位移、单点、超长区间、无效坐标／时间、片段首尾；不跨gap计算速度，区间计数守恒。
2. 停车夹具：恰好1秒与999999微秒、两段不足真实1秒、速度恰好0.3／1、prior恰好2秒与不足、初始静止、短停、末端未确认、低速run多次出现不重复发stop。
3. 起步夹具：stop→creep→restart、短高速取消、取消后重新累计、缺少confirmed stop、末端restart未确认、gap后不得继承停车上下文、多次完整stop/restart分别成事件。
4. ID与关联：输入布局变化不改语义ID；配置／边界改变则改ID；窗口多对多、端点相接、onset恰好anchor／最后future、confirmation越界、无窗口可覆盖。

布局变化的测试只针对纯算法输入的等价重排；修改冻结父数组，即使语义等价，仍必须因父来源哈希改变而拒绝，不能借ID等价绕过旧冻结。
5. 包级夹具：错误owner／pilot范围、来源或源码改变、篡改且重算哈希、未知配置、文件缺失／多余、中断、已有输出、独立重建失败不得冒充通过。
6. 测试先行后先本地提交源码，再冻结配置与真实运行；先报告fit，select按同一配置一次运行，不根据任何预测排名改规则。真实结果记录零样本、全部异常分母与复现身份。
7. 真实产物单独verify-only，核对B0b/B0a/3A旧冻结不变；新建episode级人审材料另作下一任务，不能将pending计作完成。

运动学之后，先测试第6节字段表及逐区间状态trace，再实现窗口关联，避免用不同end口径计算覆盖。以下所有区间均为0.5秒，数值只是合成夹具，不是新增真实实验：

| 例子 | 速度／时段 | 必须得到的边界和计数 |
|---|---|---|
| 初始低速后起步 | 0–1秒v=0；1–2秒v=1.2 | 低速诊断因prior不足；2秒确认一次restart_without_confirmed_stop；stop/restart/stop_go候选均0 |
| 停车→爬行→短加速取消→起步 | 0–2秒v=2；2–3秒v=0；3–3.5秒v=0.5；3.5–4秒v=1.2；4–4.5秒v=0.5；4.5–5.5秒v=1.2 | stop onset=2、confirmation=3、low_dwell_end=3、end=4.5；一次短restart取消诊断；restart onset=4.5、confirmation=end=5.5；stop_go onset=2、confirmation=end=5.5，各类型候选1个 |
| 右删失停车 | 0–2秒v=2；2–3.5秒v=0，block在3.5秒结束 | stop onset=2、confirmation=3、end=3.5；right_censored及low_end_censored为true；无restart／stop_go |

对最后一例，若只有一个窗口且anchor=2、最后future=3，则stop的context、未来onset、未来confirmation覆盖数均1，完整事件覆盖数0。若改为anchor=0、最后future=2且观察从-2秒开始，onset恰好落在F右端：context、未来onset、未来confirmation和完整覆盖数均0，不能因端点相接计覆盖。

## 10. 后续依赖与本轮交付状态

依赖顺序：B0b完成 → 用户确认B1a具体协议 → 写实现计划／新worktree → 状态机与完整性测试 → 规则冻结及真实候选抽取 → episode级阳性／阴性／不可评价复核 → 停车预测指标及因果anchor／噪声协议 → 算法实验。B1b减速和B1c转向共享运动学基础，但需各自定义、审查和规则版本。

本文记录已批准并验收的实现契约。运动学、状态机、窗口关联、发布及语义重建已实现，真实结果见验收报告。尚未完成的人工标签、预测指标、新训练及传感器输入不属于本轮成果。只本地提交，不推送GitHub。

设计验收记录：完成自查及两轮只读复审，已消除事件边界／ID、覆盖分母和未配对起步诊断的阻断歧义。用户已确认本协议并要求实现；执行见[实现计划](../plans/2026-11-16-scenario-events-b1a.md)。

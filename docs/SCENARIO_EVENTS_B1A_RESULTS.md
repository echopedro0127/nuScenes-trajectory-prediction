# B1a：真实时钟停车／起步候选事件验收

报告日期：2026-11-17（Asia/Shanghai）。**真实抽取与单独`--verify-only`均通过，`events_semantically_replayed=true`。** 本轮交付离线GT规则候选、诊断、状态trace和窗口覆盖统计，没有新增预测精度结果。

## 解决了什么问题

B0b完成了完整轨迹与真实时间装配，但1992个annotation连续片段不是驾驶事件。旧3A在每个窗口内生成候选，同一停走过程可能重复出现，窗口首尾也可能截断历史、停车持续时间或起步过程。没有稳定事件对象，就无法定义停车时间误差、停止位置超调／不足、漏停和误停的分母。

B1a在完整连续轨迹上建立确定性状态机，先生成跨窗口一致的stop/restart候选及稳定ID，再独立关联缓存窗口。它解决事件定义和评测对象构造，不把规则候选当人工真值，也不把窗口GT覆盖当已有预测覆盖。

## 用什么方法实现

**真实时间运动学。** 对相邻有效点计算原始位移速度，时间用整数微秒相减后转秒；不平滑、不插值、不重采样、不重复端点。停车阈值<=0.3m/s、起步阈值>=1m/s，两者均要求连续完整区间累计至少真实1秒且至少两段。两段各499883微秒只有999766微秒，需要再等完整区间确认，不能沿用旧“两步等于1秒”的说法。

**上下文与取消机制。** 停车onset前要求同block内完整2秒历史，并在该窗口中找到至少一个完整速度区间>=1m/s。确认停车后，爬行或短暂加速不生成新stop；高速run不足确认时长就下降时取消pending，下一次高速run重新累计。只有首次已确认restart消费该停车上下文并产生stop_go配对。

**删失及证据。** 初始低速、prior不足、无近期运动、短停和尾部不足分别保留原因。没有已确认stop的低速上下文后出现持续高速，只生成restart_without_confirmed_stop诊断，不冒充完整停走配对。首次低速run结束与停车上下文结束分开保存；没有观察到已确认起步时，stop保留右删失。所有参与prior判断的完整区间ID可追溯，部分重叠区间不充当运动证据。

**稳定ID与独立覆盖。** block／事件ID绑定轨迹语义、配置摘要、角色及真实边界token；stop、restart、stop_go按类型分别计数，不能把三者之和叫作独立驾驶过程数。窗口关联区分历史context、未来onset、同窗confirmation和完整事件可观察性，并分别给出全部22231缓存窗口与608 pilot窗口的覆盖及补集。没有拼接旧预测P。

**完整性与复现。** 显式在原B0b owner的cwd/PYTHONPATH执行父验证一次，保存原生证明，绑定来源与源码。产品写出后读回比对，来源和文件集合复查通过才写completed。独立verify从冻结父数据与同一配置重建全部区间、事件、诊断、ID、trace、关联和摘要；不只验文件哈希。配置未知键、重复YAML键、非法数值及运行策略篡改均被拒绝。

主要模块：`analysis/realclock_motion.py`、`analysis/stop_restart_events.py`、`evaluation/event_windows.py`、`evaluation/event_sources.py`及`evaluation/scenario_events.py`。批准契约见[B1a设计](superpowers/specs/2026-11-16-scenario-events-b1a-design.md)。

## 数据与候选数量

输入为1967目标／60948标注／1992个结构连续片段，保持B0b的fit/select与类别范围不变。共58956个区间，全部满足首版<=1秒的分辨率要求；超长区间为0。11个单点片段没有速度区间，显式输出no_velocity_interval，而非生成零速度。原25处结构缺口不跨越计算速度。

| 角色 | 停车候选 | 已配对起步候选 | 停走配对 | 右删失停车 | 低速run结束删失 |
|---|---:|---:|---:|---:|---:|
| independent_fit | 147 | 46 | 46 | 101 | 76 |
| independent_select | 80 | 14 | 14 | 66 | 56 |
| 合计 | **227** | **60** | **60** | **167** | **132** |

这里的60个stop_go是上述停车和起步的复合配对，不是额外60次独立停走。167个停车未观察到已确认restart，不能按完整停车上下文时长评分；132个连首次低速run的结束都被边界截断，二者含义不同。

按车辆类别的停车候选为car 173、truck 38、rigid bus 15、construction 1；配对起步分别43、12、4、1。bendy bus与police在该规则下为0，照实保留零样本，不修改阈值凑例；小类别不支持性能优劣结论。

## 异常与删失分母

共1077条诊断，58956条逐区间state trace。诊断不是负例真值，也不与已确认候选相加作episode总数。

| reason | 数量 | 含义 |
|---|---:|---|
| insufficient_prior_history | 358 | 低速dwell达标，但缺少完整先前2秒，不能确认新停车 |
| restart_without_confirmed_stop | 371 | 有低速上下文后确认高速，但缺少可配对的已确认stop |
| no_recent_motion_evidence | 84 | prior可观察，但没有满足阈值的完整运动区间 |
| low_run_with_existing_low_context | 82 | 未配对低速上下文中的后续低速run，抑制重复 |
| low_run_with_existing_stop_context | 30 | 已确认停车上下文中的后续低速run，抑制重复stop |
| short_low_speed_run | 98 | 低速run被打断且持续时长不足 |
| insufficient_followup | 21 | 尾部不足以确认候选 |
| short_restart_run | 1 | 已确认stop后的短加速被取消 |
| short_restart_run_without_confirmed_stop | 20 | 未配对上下文中的短加速被取消 |
| insufficient_followup_restart_without_confirmed_stop | 1 | 未配对起步在尾部无法确认 |
| no_velocity_interval | 11 | 单点片段，无速度区间 |

初始低速标记为351条，prior不足为358条；它们可能重叠，**不能相加当作左删失总数**。summary单列这些上下文计数及其原因。按类别保存segment/block/interval/不可评价区间和删失计数，并检查类别加总与角色总数守恒。

## 窗口覆盖：新增GT不等于新增模型评测

下表聚合fit/select，但各类事件分开计数。context包括历史，未来onset不包括最后future端点；同窗确认要求onset与confirmation都在同一窗口的可用未来内。完整可观察允许onset在历史中，所以它不必小于未来确认覆盖数。

| 窗口全集 | 事件类型 | 总数 | context覆盖 | 未来onset覆盖 | 同窗未来确认 | 完整事件可观察 |
|---|---|---:|---:|---:|---:|---:|
| 全缓存22231窗口 | stop | 227 | 225 | 216 | 209 | 41 |
| 全缓存22231窗口 | restart | 60 | 54 | 54 | 54 | 54 |
| 全缓存22231窗口 | stop_go | 60 | 60 | 51 | 27 | 39 |
| 旧pilot 608窗口 | stop | 227 | 31 | 26 | 21 | 4 |
| 旧pilot 608窗口 | restart | 60 | 9 | 7 | 6 | 8 |
| 旧pilot 608窗口 | stop_go | 60 | 14 | 9 | 3 | 3 |

共2352条窗口—事件关联。关联数不是事件数，多个重叠窗口不会增加事件分母。尤其旧pilot仅覆盖3个停走配对的同窗未来确认，不能据此开展具有充分样本量的停走改善宣称。下一阶段需要先冻结事件级anchor、可观察范围和噪声命名空间，不能把原窗口预测拼成一次连续驾驶。

## 测试、问题修复与旧结果保护

新增55项测试，经过RED→GREEN，包括真实时长／区间阈值、首尾与缺口、状态取消、重复低速、prior完整区间、ID、窗口端点、零事件、来源／owner、配置重复键、篡改后重算哈希、dtype、文件覆盖和中断。

复审补齐了未确认停车诊断的prior区间索引，以及类别级分母／删失统计；加入类别—角色守恒。还修复了Python中bool与0/1相等可能绕过普通字典比较的问题，改为类型敏感的规范化摘要重建比较，并有回归测试。全套收集时发现新增测试与旧analysis测试重名，改名为`test_scenario_event_pipeline.py`后正常执行。

**全套980 passed，228.93秒；Ruff、公开仓库检查及diff检查通过。** 最终只读复审无Critical/Important问题。真实运行后独立verify通过，另核对事件／诊断／trace外键和计数。

| 被保护的旧阶段 | 冻结源码不变 | 绑定产物不变 |
|---|---:|---:|
| B0b | 233 | 14 |
| B0a | 233 | 14 |
| 3A | 228 | 217 |

## 复现身份与路径

- 分支：`feature/scenario-stop-events`，worktree：`.worktrees/scenario-stop-events`。
- 源码提交：`7fac74f`；前序运动学`c7322eb`、状态机`1c8ea38`。
- 运行身份：`37fa7ff81dd82b9c17eb64dbf9ea7c558ef96a24a2327dd78259d176686e3a72`。
- 规则配置语义摘要：`347a88a48c62057c33801890d2a9cc29ce6ac327b6fe01c1683fe8435c74fa93`。
- 正式包：`outputs/scenario-events/b1a-v1`，12个绑定文件共50136419字节，不含completed标记；绑定240个源码文件及1325个来源引用。
- 额外验收记录：`outputs/scenario-events/b1a-acceptance-v1.json`，位于正式包外；生成数据不进入Git。

在该worktree使用既有Python环境和`PYTHONPATH=src`执行，装配拒绝覆盖已存在目录：

```powershell
python scripts/prepare_scenario_events.py --parent ../scenario-tracks-full/outputs/scenario-tracks/b0b-v1 --parent-owner ../scenario-tracks-full --config configs/analysis/scenario_events_b1a.yaml --output-root outputs/scenario-events/b1a-v1
python scripts/prepare_scenario_events.py --output-root outputs/scenario-events/b1a-v1 --verify-only
```

## 当前限制与下一步

这批结果是原始annotation位移与固定规则构造的离线候选；未平滑速度可能受标注噪声影响，端点时刻也不具有亚采样精度。尚无人工确认事件，manual_completed=0；旧75条window人工任务不自动转为episode标签。没有新增ADE/FDE、停车预测误差或恢复收益，更不能当作实车或闭环安全成绩。

下一步优先准备episode级阳性、规则未触发／不可评价复核队列，以及预测anchor覆盖协议；人工复核后再落实停车时间误差、停止位置超调／不足、miss与误停等指标。B1b减速／急刹和B1c转向需分别冻结规则；不凭本轮候选数调整阈值或使用历史holdout调参。没有新增训练、传感器、Cosmos或数据下载。全部本地提交，未推送GitHub。

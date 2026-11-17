# B2b1：候选级人工判决的不可变导入协议

日期：2026-11-17。**状态：用户已确认并完成实现；真实模板和全pending空版本均通过独立verify。** 结果与使用说明见[B2b1验收报告](../../CANDIDATE_VERDICTS_B2B1_RESULTS.md)。先完成候选判决这一单元；结构化边界修正和预测anchor／噪声协议分别后续设计。

前置证据：[B2a资料包](../../SCENARIO_EVENT_REVIEW_B2A_RESULTS.md)，静态身份`93687c1ebcf2e26031c43fc08656f2fbaf0135ade7e33a7ec8ee8a48f1d451e6`；规则事件来源为B1a。旧源码和产物保持只读，仅本地提交。

## 1. 要解决的问题

B2a的pending／in_progress／reviewed是阅读进度，不是事件真值。直接把reviewed或自由备注导入评分，会把“看过”“不确定”“需要修边界”混为已确认标签；复合stop_go再单独判一次，也可能与其stop/restart组成记录互相矛盾。

B2b1新增独立判决表和不可变overlay，保留原规则标签，明确哪些候选已被人工声明确认、驳回或仍无法使用。它提供后续评价的可追溯前置材料，不直接计算检测准确率、预测误差或闭环指标。

## 2. 方案选择与阶段顺序

| 方案 | 取舍 |
|---|---|
| **推荐：独立候选判决表＋不可变完整快照** | 不改变B1a/B2a，来源与未决分母清楚，单次导入可独立重建；首版不执行边界修正 |
| 扩展B2a阅读CSV，把reviewed当确认 | 省文件，但混淆阅读与标签，破坏现有可变工作表契约，拒绝采用 |
| 首版同时支持新增事件、边界重画、复合配对及模型评测 | 工作量和冲突面过大，尚无统一边界修正／预测协议，拆成后续单元 |

实施顺序为B2b1候选判决→B2b2结构化边界修正／新事件→B3预测anchor及噪声→停车指标。B1b减速和B1c转向另行冻结，不混进本任务。

## 3. 对象与来源范围

仅对B1a中的stop_candidate和restart_candidate建立人工判决行，本快照共227＋60＝287行。每行必须对应唯一B2a event_check任务。60条stop_go只作组成关系，不能另填写独立判决；1992个block_scan及诊断不自动变成人工阴性。

只读核对结果：287个原子候选和event_check一一对应；60个stop_go的stop_id/restart_id均存在，类型、block及restart.stop_id一致。B2a现有2339条阅读记录仍全pending，没有从备注或模型输出自动生成任何判决。

287／60／2339是当前已绑定快照的实测规模；实现中的守恒与覆盖校验应从源集合和冻结摘要推导，不能把这些数量硬编码进算法或夹具。

实现时新建分支／worktree保护旧B2a及所有上游。源B2a必须通过显式owner原生verify；其静态schema、父身份、源码、11个静态文件及来源全部绑定。不得换角色、对象、事件类型、block、规则版本或来源任务。

## 4. 判决CSV契约

首版命名`candidate_verdicts.csv`，与review_working.csv完全分离。UTF-8 BOM、严格CSV引号、允许排序；发布时必须恰好287个唯一来源行，未处理行保留pending，不能通过删除未决行改变分母。

锁定列按此顺序：`source_review_identity, source_task_id, source_event_id, source_event_type, role, instance, block_id`。全部从已验证静态证据生成，精确匹配，不去空白、不忽略大小写、不猜测ID。

可编辑列：`decision, reviewer, reason`。每单元格最多8192个Unicode码点，复用B2a明确的空白集合及码点计数；不把空白复核人或空理由当成真实决定。没有行级自由时间戳、类型改写、坐标或边界token输入。

| decision | 输入要求 | 导入含义 |
|---|---|---|
| pending | reviewer和reason均为零长度字符串，不接受仅含空白的字符串 | 尚无候选判决；不能用于评价 |
| confirmed | reviewer和reason非空 | 人工声明原候选类型、原边界及删失语义可接受；输出原记录，不修边界 |
| rejected | reviewer和reason非空 | 原候选被驳回；仅是该候选判断，不是全block阴性 |
| needs_correction | reviewer和reason非空 | 认为需要修正边界，首版只保存疑点；不输出为可用确认标签 |
| uncertain | reviewer和reason非空 | 当前证据不足或无法判断；不是确认，也不是阴性 |

decision必须精确为上述小写值；未知、空值、重复列、重复ID、缺行、额外行、异源行、锁定列变化及不合法CSV整次拒绝，不逐行跳过或最后一条覆盖。pending的草稿备注继续放在B2a阅读工作表；导入表是发布快照，未合规草稿不得签发完成标记。

reviewer是用户提供的本地稳定标识，不要求实名；系统只验证声明字段，不声称认证了复核人身份。代理不得把规则自动转confirmed或代填真实人工结论。只有用户实际提供判决文件才可能产生非pending记录。

## 5. 规则与人工层分离

原B1a event_id及完整原记录不修改。每条导入记录保留source_event_id、来源task、decision、reviewer、reason和本次输入文件摘要。confirmed导出为`confirmed_candidates`，附原删失标志；右删失确认不变成完整停车，不能据此计算完整停车时长。

rejected单列，不能推广为未检测区域的阴性；pending／needs_correction／uncertain统一属于未解决，分别报告原因。needs_correction的自由理由不解析成新token，也不自动裁剪、重算事件或改变其时钟。

对全部原子候选以及每个role／event_type分组，必须满足`confirmed＋rejected＋pending＋needs_correction＋uncertain＝源原子候选数`；本快照总分母287。non_pending_verdict_rows等于总数减pending，不能当作confirmed数量。缺少某状态的分组保留0计数，不从分母剔除。

“人工声明确认的候选”不等于官方nuScenes事件GT、视频感知验证或穷尽标注。首版支持候选级审阅及后续来源绑定，不输出召回率或声称发现了所有真实停车。人工规则一致性与模型预测质量分开。

## 6. 复合关系与冲突优先级

对每条源stop_go，引用其唯一stop/restart两行的decision，保留二者原状态，不新增人工判决。派生关系状态按以下唯一优先级确定：

1. 任一rejected→rejected_components。
2. 否则任一needs_correction→needs_correction_components。
3. 否则任一uncertain→uncertain_components。
4. 否则任一pending→pending_components。
5. 否则两者均confirmed→confirmed_components。

两行的role／instance／block、stop→restart外键和原始时间先后必须仍与B1a一致；任何完整性错误终止导入。confirmed_components是派生组成一致性，不是第三次独立人工标注。计数按原子候选类型与复合关系分开，不相加作独立事件。

例如stop confirmed＋restart pending：确认停车候选可单列，但配对仍pending_components；stop rejected＋restart confirmed：起步声明仍保留，配对为rejected_components；stop needs_correction＋restart uncertain：配对为needs_correction_components，两行原状态均保留。

## 7. 版本与冲突处理

每次导入生成新目录和新身份，绑定源B2a／B1a身份、来源表基线、实际CSV路径及SHA256、规范化判决摘要、schema、源码和环境。输入行重排不改变规范化判决内容摘要，但原文件SHA和本次运行身份会变化；reason、reviewer或decision变化也必须形成新身份。

首版不合并多份判决文件，不按文件修改时间选择“最新”，不自动覆盖旧overlay，也不声称建立了已认证的版本继承链。后续指标必须显式选定overlay身份；两个快照结论不同可并存，不能静默挑对模型有利的一版。缺失或相互冲突来源行一律拒绝，不做投票或last-write-wins。

模板生成与导入分开。模板恰好287行全部pending；导入全部pending也是合法空版本，状态为no_human_verdicts，不冒充人工完成。部分审阅以其余行pending表示；所有确认／驳回／未决分母完整保留。

## 8. 可变父工作表与证明

B2a的review_working.csv是唯一可变文件；它的阅读状态不得作为判决输入，不绑定其备注字节到overlay标签语义。源静态数据、review_baseline.csv和queue才是任务来源。独立判决CSV在导入前后做字节快照，之后作为不可变输入绑定，不允许边写边导入产生半版本。

初次原生父验证必须在显式旧B2a owner的cwd/PYTHONPATH执行，验证review_semantically_replayed=true和源身份。父stdout包含working_status_counts这一实时描述字段：保存完整原生证明，但验证其静态投影时必须与固定summary区分，不能把reviewed数量当GT或要求它永远等于初次读取值。

后续overlay verify校验父静态文件／源码／全部来源，复核父工作表仍满足B2a结构与唯一可变路径契约，允许合法阅读进度变化；已发布判决只依赖独立判决CSV，不随阅读备注变化。父当前工作表结构损坏应显式报错，不能绕过原父包的可变文件契约。

不在新worktree直接冒用旧verify。需要调用父CLI时必须保留其显式owner；若复用父证明做重建，当前可变表校验器必须绑定并核对原B2a版本，不能以新逻辑猜测旧格式。

父证明的静态投影**仅排除working_status_counts**：payload键集合必须恰好等于冻结summary全部键加identity／review_semantically_replayed／working_status_counts，未知字段拒绝。semantic标记严格为bool true，其他字段与冻结摘要作类型敏感比较。被排除字段仍要求恰好pending／in_progress／reviewed三个键、值为严格非负整数（不能bool），总和等于源summary.task_rows（本快照2339）；它作为当时的阅读状态快照，不要求与后来合法编辑后的当前计数相等。当前工作表仍需按源B2a契约单独校验。

## 9. 产物与独立验收

建议模块：`evaluation/candidate_verdicts.py`负责纯表校验与派生关系；`evaluation/verdict_overlay.py`负责来源、输入快照、发布及独立重建。入口区分`make_verdict_template.py`和`import_candidate_verdicts.py`，均显式parent／parent-owner；导入另给annotations文件和新output-root，不提供临时规则阈值。

导入包包括protocol/run/parent_proof、规范化atomic_verdicts、confirmed/rejected/unresolved候选及派生关系、summary、输入CSV副本及最后完成凭据。原文件和副本都绑定哈希；所有包内文件不可变，没有阅读表式mutable例外。源判决文件后续改变会被原版本verify发现；建议用户每次另存新的判决文件，不能覆写已发布版本输入。

summary分别给出227 stop／60 restart／60配对分母、所有decision计数、未决原因、confirmed的原删失分布及各角色／类别。non_pending_verdict_rows是人工声明数量，不与B2a manual_completed字段混用；所有pending时为0。没有新预测分数、训练、holdout评价或官方事件检测成绩。

独立verify从固定来源表与同一CSV重新校验所有锁定字段、状态、ID、派生关系和摘要，不只验checksum。读取前后核对判决CSV、父静态来源与源码，运行中变化整次失败。输入变更、未知策略、重新计算哈希后的类型／内容篡改、文件缺失或多余、中断均不得冒充完成；已有输出拒绝覆盖。

## 10. 边界与验证顺序

1. 先测试全pending／部分confirmed／全类型状态及严格CSV；引号、BOM、多行reason、Unicode空白／码点8192和8193边界。
2. 测试287行守恒、排序、重复／缺行／异源ID／角色类型锁定；阅读reviewed不得自动转confirmed；block无候选不得造阴性。
3. 测试复合优先级所有组合、右删失保留、未决不进入confirmed、关系外键损坏终止。
4. 测试原判决文件与副本哈希、空版本、新目录不覆盖、多快照不自动选择、重哈希篡改及中断无完成凭据。
5. 测试父合法阅读变化可通过、父可变结构坏必须报错、父静态变化拒绝、旧owner证明和环境绑定。
6. 测试通过并先提交源码后，真实生成287行pending模板并导入空版本，独立重建验收；除非用户提供真实判决，确认数量必须保持0。
7. 核对旧B2a/B1a/B0b/B0a/3A冻结均不变，报告空标签限制及新身份，不推送GitHub。

## 11. 后续契约

B2b2再定义同block原始token上的完整边界修正、新事件、复合关系重建及冲突规则；不能用本阶段needs_correction理由绕过它。B3先冻结实际查询时刻、GT目标时间匹配／容差、事件anchor覆盖及因果噪声命名空间，独立于标签确认，禁止拼接旧窗口P。停车时间误差、位置超调、miss和误停在这些前置协议之后实现。

本契约已按用户确认实施。真实验收仅导入全pending空版本，未产生人工确认或预测分数；287个原子候选和60个关系的结构核对不构成人工审阅。执行见[实现计划](../plans/2026-11-17-candidate-verdicts-b2b1.md)。

设计复审记录：未发现Critical／Important或阻断歧义；已明确父证明唯一可变投影、strict整数计数、pending零长度字段和原子分母守恒。产品另经测试先行、代码复审和真实独立重建验收，不能将空版本称作人工完成。

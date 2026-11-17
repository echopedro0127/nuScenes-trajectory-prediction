# B3a：真实查询时钟、GT匹配与覆盖审计

日期：2026-11-18。状态：设计待审阅，尚未实现B3a发布程序；已做现有B0b窗口时钟的只读统计。无新预测分数。

## 设计动机与证据

旧Phase 3A明确使用固定0.5秒相对时钟。`scenario_replay.py`冻结dt=.5，`streaming_metrics.py`使用查询step与目标列的整数偏移。B0b则保存每个窗口17个真实微秒时间戳。两种时钟不能混名，旧P不能通过补timestamp就成为真实时钟模型输出。

对已发布B0b两角色windows.npz只读核对：

| 角色 | 窗口 | 最小／最大实际间隔(us) | 最大网格偏差(us) | 非精确名义网格窗口 | 存在>50ms网格偏差 |
|---|---:|---:|---:|---:|---:|
| independent_fit | 14445 | 394416／650298 | 251324 | 14445 | 10710 |
| independent_select | 7786 | 397689／649744 | 249792 | 7786 | 5955 |

网格偏差定义为`abs(actual_offset_us - nominal_step*500000)`，覆盖历史和未来17点。共22231窗口、16665窗口存在>50ms偏差；这是时钟诊断，不是GT匹配失败数或模型误差。50ms是本设计拟冻结的工程容差，不是官方规则或行业标准。

## 路线选择与范围

1. **推荐：先做B3a元数据审计，再做B3b真实dt物理回放。** 明确查询／目标映射及模型可用范围，避免误用旧神经接口；不等待人工事件判决。
2. 把旧P直接绑定真实时间戳：实现快，但改变了旧模型原来的时间语义，不采用。
3. 将GT插值回规则网格并立即重训：引入新的观测重采样与训练协议，范围扩大，留待单独设计。

B3a只发布可独立重建的查询、GT映射及覆盖表，不生成新观测噪声、不运行预测、不训练。B2b2b新增事件／关系编辑不阻塞这一元数据阶段；人工停车指标仍依赖真实复核。

## 输入、角色和身份

- 必需：显式B0b full_cache包及其原owner，固定B1a事件包及owner。两包的B0b身份必须一致；不通过目录名猜测版本。
- B0b窗口键使用role/instance/sample，核对window_id、row、scene/log、anchor_annotation_token及17点token和真实时钟。row只用于查找，不代替身份。
- 检查原角色隔离、原连续片段、history完整性及源GT坐标；所有原缓存窗口必须出现且恰好一次。只使用independent_fit/select，不打开holdout。
- 可选显式B2b1/B2b2a声明包只用于标注来源层级；首版不接可选包，以B1a规则候选完成覆盖审计，明确label_source=rule_candidate。无人工确认不得改称人工GT。
- 原owner原生验证、静态引用、源码／环境、协议哈希按既有发布模式保留；B2a可变阅读表不是本阶段必需输入。

## 查询与目标时间协议

窗口原序列索引0..16，anchor索引4。第k次更新k=1..11，查询时刻`q_us=T[4+k]`，对应新到达的真实源annotation；同一source token只同化一次。clean阶段定义arrival_us=source_us，不模拟延迟。

固定评价提前量集合为500000／1000000／2000000微秒。对每个窗口、每次查询、每个提前量h，保存期望目标时刻`desired_us=q_us+h`。

仅在该窗口原未来12个GT点（索引5..16）中寻找**严格晚于q_us**的目标；取与desired_us绝对差最小者，且差值必须<=50000us。规则如下：

- desired_us>T[16]直接为not_evaluable: beyond_window_horizon，不用容差把已越界目标拉回窗尾。
- 最近目标存在等距离并列，记not_evaluable: ambiguous_nearest_target，不按先后顺序挑选。
- 在时域内但最小差>50000us，记not_evaluable: target_outside_tolerance。
- 无晚于查询的目标，记not_evaluable: no_future_target；先判窗尾越界，再判该条件，再判并列与容差。
- 成功保存target_token、target_us、signed_offset_us=target_us-desired_us及actual_lead_us=target_us-q_us；不得仍把实际提前量显示为精确h。
- 不插值、不跨window扩大GT范围、不跨segment、不读未来GT构造模型输入。未来GT只服务离线匹配。

容差在看模型成绩之前冻结，不为了提高覆盖而事后放宽。未来若需要不同容差须新协议／新身份，不能混入本版本。

首版固定11次查询：预期244541条query行、每个lead均244541条匹配尝试、总733623条query-lead行。这些是协议分母，不是已实测有效数。所有无效行保留status/reason及null目标，不删除它们缩小分母。合法无匹配与数据坏行严格分开。

## 窗口、事件与anchor覆盖

不新增或重选anchor，先审计现有22231个缓存窗口。模型原始anchor仍是窗口第5个点，查询时刻则随新观测推进；两者分别保存。

对每个B1a规则事件，记录关联窗口、instance/log、同一连续block，以及是否存在查询时刻严格早于onset。停车前0.5／1秒查询覆盖以`abs((onset_us-q_us)-h)<=50000`判定，并明确这是**query-to-event覆盖**，不是query-to-GT匹配；先有查询覆盖，仍需另检预测目标能否覆盖事件的确认区间。

同一事件可关联多个重叠窗口，不当作多个独立事件。覆盖表按event_id聚合任一匹配与全部匹配数，另保留逐window-query连接；同一个窗口查询可以服务不同评价视角，不重复生成观测。

右删失／低速段删失、缺少查询前历史、窗尾不足、无既有anchor、待人工复核分别记录，不能合成“未预测停车”。规则未触发block也不是穷尽负例，不报告误停车率或召回率。

## 模型兼容与噪声命名空间

B3a只生成兼容矩阵，不运行模型。旧3A模型保持nominal_dt_v1；本阶段禁止以它输出realclock_v1成绩。B3b先为物理基线显式传actual dt、实际目标时间和状态初始化；学习模型的固定步长历史、地图／邻车候选、CA保护与门控需要逐项定义适配，不能单靠KF支持dt就宣称全网络支持真实时钟。

为后续观测流冻结namespace描述：schema=realclock_source_packet_v1，source_key由dataset_release、role、scene/log、instance、annotation_token组成；窗口ID、查询序号、strategy、模型seed和事件类型都不进入基础噪声键。同一个源包跨重叠窗口／策略必须得到同一基础噪声。

noise_seed仍为2027／2028／2029，和训练seed分开。clean无随机扰动；后续Gaussian／离群方案共享显式基础噪声键，再用独立异常实例ID描述注入。不沿用旧window-seeded噪声冒充同一实验。B3a保存键的规范化结构和digest，不调用RNG或声称已生成某个分布；RNG算法、坐标系、异常时刻／幅度与缺包协议在B3b执行设计中一起冻结。

## 异常、聚合及发布

完整性错误立即失败：重复窗口／token、错角色／instance／scene/log、时钟非正或非单调、shape/dtype错、源hash变化、GT非有限、跨缺口、缺预期窗口、父身份冲突。不能把这些坏行标成正常不可评价后继续。

合法时域不足仅使对应query-lead无效，窗口及其其他lead保留。每组满足attempted=matched+各not_evaluable原因，按role/lead及log分别报告窗口、query、target、instance、event数；候选类别不混成模型排名。零分母输出null及原因，JSON禁止NaN/Inf。

输出至少包括protocol、run、completed、query表、GT匹配表、event-window-query覆盖表、compatibility、noise namespace manifest、summary、parent proofs和全部sources/references。发布不覆盖，completed最后写；verify从同一冻结父数据独立重建全部语义和文件字节。

## 实现顺序与验收

1. 新隔离分支；验证B0b/B1a原owner与公共数据身份，保留旧源码和产物。
2. 纯时间函数与独立慢速标量参考：准确目标、容差边界、1us越界、并列、窗尾、无未来点、单点block、非单调、整数严格类型。
3. 全窗口查询装配和分母守恒；小夹具核验事件覆盖与重叠窗口去重、基础噪声键跨窗口一致性。
4. 发布／verify集成测试，覆盖来源、策略、表内容和重哈希篡改；全套回归和公开检查。
5. 真实B3a全缓存审计并独立verify，报告实际匹配数、无效原因和规则候选覆盖；无模型成绩。
6. 根据审计确定B3b物理回放的可评价目标集合及输入适配；先比较真实dt的CV／KF，再单独设计学习模型迁移。

完成标准是时钟、索引、分母和可复现性正确，不要求覆盖达到预设比例，也不承诺模型获胜。当前只读统计不能冒充本设计已经实现或真实验收完成。

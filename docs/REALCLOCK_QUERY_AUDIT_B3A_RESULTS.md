# B3a：真实查询时钟、GT匹配与事件覆盖审计

日期：2026-11-18。分支`feature/realclock-query-audit`，工作区`.worktrees/realclock-query-audit`。本阶段处理真实时钟与评价范围，不进行预测、训练或holdout评价。

## 用什么方法解决什么问题

旧3A以规则0.5秒网格构造查询与未来目标索引，新B0b则保留17个实际时间戳。22231个窗口全部不是精确名义网格，16665个窗口存在超过50ms的历史或未来网格偏差；不能把旧预测张量补上绝对timestamp后称为真实时钟预测。

B3a将查询时刻绑定到实际释放的新位置token，再以期望物理提前量匹配GT：`desired_us=query_us+lead_us`，只选择本窗口未来12个原始点中严格晚于查询的唯一最近目标。匹配容差固定50ms，保存实际提前量及有符号时间偏移，禁止插值与跨窗口扩展。

| 难点 | 方法 | 避免的错误 |
|---|---|---|
| 固定step与实际时间混淆 | k=1..11查询使用17点序列的索引5..15，anchor始终为索引4 | 把“第2点之后”冒充精确未来1秒 |
| 窗尾目标被容差拉回 | 先判desired_us是否超过原最后GT时间，再进行最近点匹配 | 用最后一点给越界查询凑有效样本 |
| 匹配选择存在歧义 | 最近目标并列时不可评价，不按数组先后选点 | 结果依赖文件排序 |
| 通过删除失败行改善覆盖 | 保存每个window-query-lead尝试及原因码 | 分母被静默缩小 |
| 重叠窗口重复同一源包 | 以dataset/role/scene/log/instance/annotation token建立来源键 | 各窗口把同一观测随机扰动成不同值 |
| 事件覆盖数冒充召回率 | 规则候选来源、人工未提供状态和各删失标志单独保留 | 把规则候选当人工行为GT或完整事件清单 |
| 共享旧模型却改变时间语义 | 输出nominal-only兼容矩阵，不调用模型 | 假称旧门控已支持actual dt |
| 仅验hash无法发现重哈希篡改 | 从冻结父证据重建JSON和NPZ并逐字节比较 | 摘要或目标索引被修改后仍被验收 |

## 时间、索引与来源契约

主提前量为0.5／1／2秒，全部用int64微秒计算。有效目标保存source index、target_us、desired_us、actual_lead_us、signed_offset_us和valid mask。无效target_index／target_us／actual_lead_us为-1，offset占位0必须结合valid读取，不能当零偏差命中。

状态分为matched、beyond_window_horizon、no_future_target、ambiguous_nearest_target、target_outside_tolerance。边界包含50ms等值；超过1us即不满足容差。负数／非单调／浮点或bool时钟、int64溢出、错误shape不是合法无匹配，而是完整性错误。

窗口先按role/window_id稳定排序，每一行绑定原role_window_row和缓存row；核对role/instance/sample、scene/log、anchor token、17点真实token／时钟／offset、连续block、有限float64 GT坐标和角色log隔离。所有源窗口必须恰好出现一次。

原B0b与B1a均在其显式原owner中验证；B1a的parent路径、owner、身份及dataset release必须与本次B0b一致。新代码不冒充旧模型／数据生成源码身份，原生证明和全部来源引用均被绑定。

## 事件覆盖与噪声键

保留全部347个规则候选记录（227 stop、60 restart、60 stop_go关系），这不是347个独立人工episode。每个事件记录同block关联窗口、提前0.5／1秒是否存在真实查询、全部query-link数量、去重源查询数、不同窗口数，以及确认token是否落在该查询的剩余GT范围内。

查询覆盖与GT固定lead匹配不同：前者比较event.onset-query，后者比较目标annotation与query+lead。某事件有提前查询，不代表已有模型预测覆盖它的确认区间，也不代表预测正确。

逐事件保留no_existing_anchor、no_pre_lead_query、insufficient_query_history、insufficient_prediction_horizon和human_review_not_supplied原因；这些原因可同时出现，不可相加当互斥分母。右删失／低速段删失仍单独显示。无候选block不当作完整负例，不计算误停率或召回率。

source_packets.json记录跨重叠窗口去重的源键。noise_namespace.json仅冻结键字段、版本、排除字段及2027／2028／2029噪声seed；没有调用随机数、生成观测噪声或注入异常。后续B3b仍须冻结RNG、坐标系、噪声幅度和异常协议。

## 工程实现与测试

生产模块：query_clock.py（整数匹配）、query_audit_data.py（来源装配与覆盖）、query_audit_sources.py（冻结父证据）、query_audit.py（不可变发布与独立重建）；CLI为scripts/run_realclock_query_audit.py。新增3份测试文件，包含独立穷举标量参考，而非重复调用向量函数自证。

审查与测试补齐了浮点event/block时间可能通过整数外键比较的问题、空release拒绝、独立源query／GT目标数，以及逐event/lead连接计数。完整性错误拒绝，不通过四舍五入修复输入。

最终全套回归为**1168 passed，505.99秒**，日志`outputs/pytest-b3a-final.log`。本阶段新增51项：22项微秒时钟、20项来源／覆盖装配、9项发布集成。Ruff、公开文件扫描和diff检查通过；只读复审确认已关闭来源类型与计数问题，未发现剩余Critical／Important。源码冻结提交为`6ccc8b0`，此后仅更新文档及输出。

## 正式全缓存验收结果

真实运行与独立verify-only均成功，`query_audit_semantically_replayed=true`。运行身份为`e52bac9e3312f2902d529c6c609a696b18c6e5aa62c37957e0a02892de501fa7`。绑定260个源码文件、1577个来源引用；13个静态文件共89493322字节，不含completed。

| 项目 | 数量 |
|---|---:|
| 原缓存窗口 | 22231 |
| window-query行 | 244541 |
| query-lead匹配尝试 | 733623 |
| matched | 511854 |
| 超出窗口时域 | 126515 |
| 时域内但超50ms容差 | 95254 |
| 并列最近点／无未来目标 | 0／0 |
| 去重源查询包 | 42687 |
| 匹配涉及的不同GT标注 | 39460 |
| 窗口17点中去重源包 | 54534 |

54534包含历史与全部未来源token的键描述，42687只统计更新查询位置；两者不能混作同一个分母。源包清单没有将未来位置提供给模型，本阶段没有运行模型。

每角色、每lead尝试数固定，失败行全部保留：

| 角色 | lead(s) | 尝试 | matched | 窗尾越界 | 超容差 |
|---|---:|---:|---:|---:|---:|
| fit | 0.5 | 158895 | 137275 | 9339 | 12281 |
| fit | 1 | 158895 | 114079 | 22268 | 22548 |
| fit | 2 | 158895 | 81855 | 50400 | 26640 |
| select | 0.5 | 85646 | 73678 | 5179 | 6789 |
| select | 1 | 85646 | 61278 | 12167 | 12201 |
| select | 2 | 85646 | 43689 | 27162 | 14795 |

每行matched＋窗尾越界＋超容差等于尝试数。2秒lead剩余可用查询减少包含原窗口窗尾效应，不代表预测精度下降；这里根本没有模型误差。

事件覆盖如下，“任一”不能与两列直接相加，同一事件可能两种提前量都有覆盖：

| 规则候选 | 总数 | 任一提前量有查询 | 提前0.5s | 提前1s |
|---|---:|---:|---:|---:|
| stop | 227 | 204 | 203 | 175 |
| restart | 60 | 49 | 48 | 41 |
| stop_go组成关系 | 60 | 49 | 48 | 44 |

共2896条event-window-query-lead连接；302条规则记录存在任一提前查询，其中包含stop_go关系，不能称为302个独立episode或人工真值。仍未计算停车时间／位置误差、miss或误停率。

额外固定哈希抽取fit/select各32个窗口，用独立Python标量循环验证所有11次查询×3个lead，共2112次匹配，状态及索引全部一致。全量源查询、GT目标、状态和role/lead/log分母另行守恒核对。

旧七阶段各自冻结源码／静态文件核对通过：B2b2a 255／11、B2b1 250／13、B2a 245／11、B1a 240／12、B0b 233／14、B0a 233／14、3A 228／217。B3a的直接父为B0b/B1a，其他旁支使用各自冻结清单额外保全，不冒称本包直接依赖所有旁支；B2a阅读表保留原可变契约。

实际本地路径：

- 不可变审计包：`outputs/realclock-query-audit/b3a-v1/`。
- 包外验收记录：`outputs/realclock-query-audit/b3a-acceptance-v1.json`。
- 正式构建／独立重建日志：`outputs/b3a-build.log`、`outputs/b3a-verify.log`。
- 源预检记录：`outputs/b3a-parent-prevalidation.json`，只是预检，不冒充正式completed。

```powershell
# 本worktree，既有环境；PYTHONPATH=src
python scripts/run_realclock_query_audit.py --tracks ../scenario-tracks-full/outputs/scenario-tracks/b0b-v1 --tracks-owner ../scenario-tracks-full --events ../scenario-stop-events/outputs/scenario-events/b1a-v1 --events-owner ../scenario-stop-events --output-root outputs/realclock-query-audit/b3a-v1
python scripts/run_realclock_query_audit.py --output-root outputs/realclock-query-audit/b3a-v1 --verify-only
```

生成命令仅在新目录使用，现有b3a-v1不要覆盖。只本地提交；输出数据不进入Git，未推送GitHub。

## 产物读取方式

- windows.json：NPZ第一维对应的稳定窗口顺序，含来源row、block、17点time_us和tokens。
- window_clocks.npz：原17点实际时钟、annotation和sample token表。
- query_matches.npz：query_us形状[N,11]；desired_us／target_index／status_code／valid／target_us／signed_offset_us／actual_lead_us形状[N,11,3]，lead顺序见protocol。
- target_index指向同窗口原17点表，不是压缩后12点表的列号；-1为无效，不能用NumPy负索引直接读GT。
- event_coverage.json／event_query_links.json：完整事件分母及逐窗口、查询、提前量连接。
- protocol／run／completed／parent_proofs／summary／compatibility／noise_namespace／source_packets：协议、身份、状态与来源。

不可变目录不覆盖，completed最后写。verify-only重新读取同一冻结父数据、重建全部数组／表／分组／协议并比较文件字节，最后再次核对源码和来源hash。

## 后续实验边界

B3b先接入真正接受实际dt和实际目标时刻的CV／KF基线，状态初始化使用已到观测；随后单独适配学习模型的固定步长历史、邻车／地图候选、CA与门控。不能因为KF支持dt就宣称整条融合网络已适配。

人工停车指标还需真实标签及确认区间覆盖；本阶段无需新下载数据，不调用Cosmos，不查看或调优旧holdout。这里的覆盖结果不改写历史ADE／FDE提升。

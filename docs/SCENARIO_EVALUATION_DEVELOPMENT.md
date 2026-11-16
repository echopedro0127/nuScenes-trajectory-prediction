# 场景驱动在线轨迹预测评测：下一阶段开发设计

日期：2026-11-10；审查修订：2026-11-11。文档分支：`feature/scenario-evaluation`，worktree：`.worktrees/scenario-evaluation`。

**状态：S0–S7正式评测已完成，单独verify-only通过。** 608窗口/5472流/16策略/41案例，完整结果及负结果见[Phase 3A报告](SCENARIO_EVALUATION_PHASE3A_RESULTS.md)，协议见[S7](SCENARIO_EVALUATION_S7_PROTOCOL.md)。S4d人工复核仍pending；规则标签不是人工真值，本轮不训练、不评价holdout，不包装为全面模型提升。

2026-11-16：完成[3B准备审计](SCENARIO_EVALUATION_3B_PREPARATION.md)及独立可编辑复核资料；随后在新worktree完成[B0a真实时钟与track装配](SCENARIO_TRACKS_B0A_RESULTS.md)。461目标/16010标注/608窗口的真实运行与独立重建通过，920项测试通过。旧3A源码及产物保持不变。尚未抽取canonical episode或产生新预测分数；下一步先扩B0b目标范围，再冻结B1事件协议。

随后[B0b全缓存装配](SCENARIO_TRACKS_B0B_RESULTS.md)也已通过真实运行与独立重建：1967目标/60948标注/22231窗口，25处缺口形成1992连续片段（非episode），窗口内缺口为0。全套925项测试通过，B0a子集和旧冻结均不变。当前下一步为B1事件协议设计，仍未产生新预测分数或人工事件真值。

用户已批准[B1a停车／起步协议](superpowers/specs/2026-11-16-scenario-events-b1a-design.md)，并已完成实现、真实运行及独立重建验收：227个停车规则候选、60个起步及60个停走配对，167个停车右删失；58956区间守恒，全套980项测试通过。详细覆盖和限制见[B1a结果报告](SCENARIO_EVENTS_B1A_RESULTS.md)。旧3A中位数／名义两步标签不被覆盖，manual_completed仍为0，没有新预测分数。下一步优先episode级复核与预测anchor覆盖协议，再进入停车指标；减速和转向分别设计。

本次依据[审查报告](SCENARIO_EVALUATION_REVIEW.md)R1–R9修订协议。以下条款记录3A开发时的协议与验收要求，实际完成状态以顶部状态及各阶段报告为准；3B的新事件协议另行冻结。

## 1. 为什么改变下一阶段的重点

[Phase 2D](NOISE_AWARE_PHASE2D_RESULTS.md)已完成3个门控训练种子验证：CA在clean/普通噪声改善方向一致，保护规则缓解高曲率但损失噪声条件总体精度。当前缺口是解释修正何时有效、何时有害、异常后多久恢复，而非继续根据同一小开发集叠加组件。

下一阶段目标：建立可复核的“真实场景/受控运动—因果观测—冻结模型—事件级指标”评测。首轮不新增模型训练，不下载新数据或Cosmos权重，不修改已有holdout链，不根据新指标反向调当前模型。

## 2. 已有资源及适用边界

已确认本地有nuScenes trainval的sample/sample_annotation/ego_pose/category元数据、地图扩展，以及已有map/neighbors缓存。Phase 2D提供十候选、18维特征、目标、保护trace和9个冻结门控。

父实验位置：`../noise-aware-stability/outputs/noise-aware/phase2d`；身份：

`ca0792a001be3fd361239d41c631080e25d3d30ed490762b5d83a66fb9c2568f`

父文档提交`5d022ca`，实现冻结`7bc11d9`。父源码在原worktree验证；新分支增加源码后不能在新源码目录冒充父执行环境。

| 可用资源 | 现在可以支持 | 尚不能由此证明 |
| --- | --- | --- |
| 标注位置和固定0.5s回放 | 机动/修正/退化观测评测 | 真实检测跟踪误差、亚0.5s响应精度 |
| 静态地图、邻车元数据 | 场景挖掘、道路约束及交互上下文准备 | 当前门控已具备完整交互推理能力 |
| 原物理滤波器 | 丢包/延迟的源时钟处理 | 当前GRU/地图/CA门控已支持缺失观测 |
| samples/sweeps目录 | 后续按scene核查传感器文件覆盖 | 所有trainval原始传感器都完整可用 |

当前`build_candidates`、`build_motion_experts`、`build_guard`要求完整规则包；Phase 3A不让它们接收丢包/延迟，不用干净GT补齐历史。预测jitter是跨更新的修订量，不等同轨迹jerk或乘坐舒适性。

## 3. 分阶段交付

| 阶段 | 工作及输出 | 模型/数据边界 |
| --- | --- | --- |
| **3A：优先实施** | 冻结模型重放；固定提前量、同时间点修正增益、异常恢复、方向分解和尾部误差；新增GT事件标签 | 复用Phase 2D窗口/9门控；不训练、不换划分 |
| 3B：场景库 | 从现有fit/select挖掘急刹/走停/S弯；受控二维运动；物理基线遮挡恢复 | 真实与合成结果分开；缺失观测首轮只比较支持它的物理策略 |
| 3C：接口与扩展 | 改造学习候选的缺失/延迟接口；更大开发窗口；按覆盖审计后接入交互 | 独立设计、训练和消融；不能直接把旧完整观测门控当已适配 |
| 可选视觉扩展 | 小规模视频标签复核，之后再考虑Cosmos视觉压力测试 | 先接通感知/跟踪；生成视频不当轨迹真值 |

第一轮交付以3A为准；后续阶段的具体场景数量由资源和标签覆盖审计决定，允许报告零样本，不能为凑案例改变模型或冒充新盲测。

## 4. Phase 3A固定协议

### 4.1 对照和样本

使用父实验原352 fit窗口/22log、256 select窗口/4log，各三观测条件clean/noise_05/noise_outlier、三个噪声seed。9个门控为3控制×3训练seed，全部保留，不选最佳seed。

主对照为9个冻结门控和7个物理候选：raw_cv、alpha_beta、kf_cv、kf_nis、bounded_ca、brake_stop、hold_position。物理候选无训练seed，只报告一次噪声重复统计，不复制成三个模型。原GRU/original/map仍在门控候选中，首轮不扩大独立模型列表。

fit只作诊断，select单独汇总；指标定义冻结后不据结果调模型/阈值。不读取旧holdout结果进行选择。窗口可重叠；噪声seed、门控训练seed和采集log是不同维度，不能合并当独立样本。

### 4.2 时间及数组契约

`dt=0.5s`，GT未来点j（零基）时间为`(j+1)*dt`；查询step k为1至11，时间`k*dt`。预测`P[N,11,12,2]`仍指向原十二个绝对未来时间点，标签`Y[N,12,2]`与模型输入分离。

固定提前量h以整数步m=h/dt表示。目标列`j=k+m-1`，要求`k+m<=12`。h=0.5/1/2s分别有11/10/8个有效查询；不足提前量的更新排除并报告数量，禁止补零、重复终点或外推GT。

已有ADE/FDE/jitter调用原`forecast_metrics`、保留原输入dtype/口径；新指标不得静默替换旧定义。父缓存`per_update.npz`只有已有指标，不能直接相减当作新指标；须重建预测张量。

**唯一数组索引公式（NumPy零基）：**

```python
j = k + m - 1
current = P[:, k - 1, j, :]
previous = P[:, k - 2, j, :]  # 仅k >= 2
error = current - Y[:, j, :]
```

导出`query_step=k/query_row=k-1/target_column=j/query_time_s=k*dt/target_time_s=(j+1)*dt`。3A时间均相对窗口anchor；不要求绝对时间或独立episode ID。

### 4.3 数据装配与来源契约

父NPZ只有`features/candidates/target/ca_allowed/guard_yaw/guard_min_speed`；records只有condition/instance/log/role/row/sample/scene/seed，不能据此假定存在GT历史或anchor timestamp。

- 读取父run绑定的`source_cache_root`及原角色缓存；以`role/instance/sample`为窗口键，核对scene/log、GT未来和坐标系。row只作查找加速，发现身份不一致立即失败。
- 装配GT历史位置、history mask、anchor与未来12点，保留原缓存字段/文件引用及哈希，沿用父坐标变换与米制单位。GT历史不足以计算某标签时仅标记该标签不可评价；父协议要求完整的模型历史/未来意外缺失则终止。
- 3A使用经验证的规则2Hz相对时钟，当前GT速度为截至查询的相邻GT位移除dt。标签历史与方向评价来自GT缓存，禁止用门控特征速度替代GT。
- 模型只接父`features/candidates/ca_allowed`；GT历史、未来、标签和人工意见仅进入评价层。保存独立来源引用，不向观测流补入GT。
- 绝对timestamp、完整track和canonical episode在3B通过sample元数据装配并核对scene/log；不依靠假造绝对时间满足3A接口。

必需资源：父completed/run/records、两角色NPZ、9个checkpoint及归一化统计、原GT缓存、父源码环境和本阶段环境。重建候选抽查另需原地图/上游权重。传感器文件与Cosmos不是3A依赖。缺资源时列明路径、身份及再生入口：原`run_noise_aware_baselines.py`→`run_noise_aware_gate.py`→`run_noise_aware_diagnostics.py`→`run_noise_aware_motion.py`→`run_noise_aware_stability.py`，具体参数及上游缓存再生遵循父run引用和各阶段报告；不得下载替代权重后沿用旧身份。显式传入每层owner project root，在其原环境验证；不按目录深度猜测源码所属。

## 5. 新指标的v1定义

本节阈值是拟实施的实验协议，不是行业安全标准。执行前冻结配置和哈希；如需更改须新版本、说明原因，不能混入旧结果。

### M1：固定提前量误差

`E_h(k)=||P[:,k-1,k+m-1,:]-Y[:,k+m-1,:]||₂`。报告h=0.5/1/2s的均值、逐查询P50/P95及有效查询/窗口/log数；h=1s为首轮主要比较，P99仅作描述性附录。

同一窗口先对有效查询取均值，再在窗口/噪声重复内汇总；另外保留按k的结果。三训练seed先逐模型统计，再给配对差、均值和样本标准差，不能把物理候选伪造成多seed训练。

### M2：同一目标时刻的修正增益

只比较k>=2、且固定提前量仍有效的更新：

`G_h(k)=||P[:,k-2,j,:]-Y[:,j,:]||₂-||P[:,k-1,j,:]-Y[:,j,:]||₂`，其中`j=k+m-1`。

前后预测比较的是同一个j，不能比较各自不同的“未来1秒”。h=1s有9个有效更新对。定义`G>0.05m`为有益修正，`G<-0.05m`为有害，其余中性；报告均值、中位数、三类比例及逐窗口值。

它衡量已部署算法收到新观测后的净变化，不要求每次观测必然改善。不得用“剩余预测长度减少”解释成修正收益。

### M3：首个离群事件的额外位置误差恢复

noise_outlier与noise_05按角色、窗口键、噪声seed及同一冻结模型配对，GT和基础高斯噪声一致。主事件固定第4步t=2s；第8步的第二次注入不混入主恢复指标。

使用固定1s提前量，观察step4至7（0/0.5/1/1.5s）；第8步尚未到达，不会污染这段因果输出。定义：

`A(k)=max(0,E_outlier,1s(k)-E_noise05,1s(k))`。

- 主数值为1.5s内A的梯形积分，单位m·s（额外误差AUC），包含全部共同可评价事件，包括未受损事件。所有事件使用相同时间窗。
- 阈值delta=0.5m。整窗A都不超过delta，记“未明显受损”，不伪造0秒恢复并混入恢复时间。
- 否则从首次超阈值后寻找至少两个连续低于或等于delta的更新，且此后到窗末不再次超阈值；第二个更新为确认恢复时刻。
- 恢复比例=`recovered/affected`、未恢复比例=`not_recovered/affected`，未明显受损比例=`unaffected/eligible`。确认延迟从注入step4起算，均值仅含recovered，同时报告各状态数量；“1.5s内未恢复”不代表永久失败。
- v1完整11次预测中缺少step4至7或缺配对是完整性错误，终止运行。仅未来显式声明的短事件协议可将后续不足记不可评价；不得静默删除v1坏窗口。

先受损、再两次确认使当前确认延迟仅可能为1.0/1.5s或未确认，AUC是主指标。联合报告参考/受扰绝对E_1s；A=0不表示预测重合或物理状态恢复，例如GT=(0,0)、两预测(3,0)/(0,3)误差均3但相距约4.243m。可附预测间距离D作诊断，不能代替准确性。更长孤立异常放3B，不插值伪造精度。

### M4：纵向/横向误差

固定1s提前量，将误差向量投影到GT当前速度方向d_k及其法向。GT方向仅供评价，所有模型使用同一参考。速度<0.5m/s时沿用截至当前最后一个有效方向；若从未有效，方向指标记无定义并报告数目。

报告纵向/横向绝对误差、纵向有符号偏差。禁止用未来方向补齐静止方向，或用各模型自己的方向让比较坐标系变化。

### M5：失败尾部与Miss Rate

固定1s提前量报告P95及`E_1s>2m`比例。另按查询k报告“剩余轨迹中任一点误差>2m”的`online_MR2_remaining`；不能将不同剩余长度混成官方6s挑战分数。

当前输出单条融合轨迹；十个专家不等于十个预测模态，不报告minADE@10或把门控权重当已校准模式概率。NLL、预测置信区间覆盖率留待显式概率模型建立后再设计。

### 5.1 聚合层级、状态与分母

所有指标按role/condition/strategy/train_seed分别计算；不同观测条件不混成总分。noise_seed是同窗口重复，不是新窗口；物理策略train_seed为空。

| 指标 | 冻结口径 |
| --- | --- |
| M1均值、M2增益均值、M4误差均值 | 有效查询先窗口内均值，再对该窗口noise_seed均权，再窗口均权；逐log使用同样规则 |
| M1/M5尾部 | `p50_query_error/p95_query_error/p99_query_error`池化本组有效查询，NumPy quantile `method=linear`；若附窗口均值分位数须另命名`p95_window_mean_error` |
| M2三类比例 | 有效更新对池化同一分母，三比例和为1；另列窗口内比例再均权的`window_macro_*`，不能混名 |
| M3 AUC | 所有eligible配对事件，先noise重复均权再窗口均权；共同eligible集合比较模型 |
| M3状态及延迟 | `eligible=affected+unaffected`、`affected=recovered+not_recovered`；延迟只在recovered内汇总，并给计数；本模型affected集合可能不同，不能单靠条件延迟排名 |
| M4有效性 | 无历史有效方向只排除M4该查询，保留其M1/M2/M3/M5；报告有效方向数 |
| M5 miss | 固定lead有效查询中`E>2`计数/查询数；remaining MR按k单独计算有效窗口重复的比例 |
| log与训练seed | 主分数窗口均权；另列`log_macro_mean`及有效log数；逐模型结果完成后计算三训练seed均值、样本std(ddof=1)与配对差，不混池seed |

零分母输出`null`及`status=not_evaluable/reason`，不输出0成功率；同时保存query/pair/event/window/log/noise_repeat数量。示例`[[0,0,10],[4,4,4]]`逐查询P95=8.5，窗口均值P95约3.9667，必须分名。

完整性错误（重复键、错角色/GT/seed、缺checkpoint/配对、shape/坐标错误、预测NaN/Inf、未声明时间缺口、角色log交集）终止，不通过丢弃坏预测改善分数。合法不可评价（lead越界、无GT方向、标签历史/持续观察不足、空标签组）保留`valid_mask/status/reason`。内部NaN必须配mask；JSON转null并使用`allow_nan=False`，CSV保留原因。

## 6. 事件标签与真实场景挖掘

保留旧净航向/平均加速度标签，新增事件表，与旧标签共存，不静默重写历史分组。标签使用GT进行离线评价；模型加载/回放不接收事件名称或未来时刻。

速度区间i定义为`[t_i,t_(i+1))`，原始速度`v_i=norm(x_(i+1)-x_i)/dt`；平滑速度`s_i=median(v_(i-1),v_i,v_(i+1))`，仅序列最外端重复最近值。加速度`a_i=(s_i-s_(i-1))/dt`归于区间i起点t_i。连续两段条件表示两个完整0.5s区间，onset取第一段起点、end取最后段终点；平滑可读未来，只用于离线标签，不宣称因果事件检测或亚0.5s时刻精度。固定规则如下：

| 标签 | 初始定义 | 主要检查 |
| --- | --- | --- |
| hard_brake_candidate | 先前2s内速度曾>=2m/s，平滑加速度连续两段<=-2m/s² | 不能把单次标注抖动当急刹 |
| stop_event | 之前2s内速度曾>=1m/s，再进入速度<=0.3m/s并持续两个完整0.5s速度区间；时间戳取首个满足持续条件的低速区间起点 | 与一开始就静止分开 |
| stop_go_candidate | stop_event后重新速度>=1m/s持续至少1s | 停止后起步延迟/误停车 |
| s_turn_candidate | 可靠速度>=0.5m/s的相邻方向差，累计绝对转角>=45度、净转角绝对值<=15度，正负各累计>=20度 | 修复左右转向抵消后被当直行的问题 |
| u_turn_candidate | 可靠方向净变化绝对值>=135度 | 与低速方向噪声区分 |

方向来自未平滑的位移方向，仅当相邻两段原始速度都>=0.5m/s时累积，角差wrap到[-pi,pi)。低速或缺失方向立即断段；S弯/掉头在同一连续可靠段独立判定，不能把停车前后反向转弯拼接。所有表中>=/<=包含等值；M2/M5严格不等式不加隐式容差。

急刹/停车的prior 2s为onset前四个完整区间，不足则该事件判定`not_evaluable: insufficient_prior_history`，不当阴性。窗口开始已静止记`initially_stationary`，不生成新stop；短停不足两区间不触发stop；尾部尚无法确认持续时长记`insufficient_followup`。确认stop后0.3<s<1的爬行记`creeping`且不算restart，直到连续两段>=1确认起步。stop_go必须有本窗口可确认的先前stop，否则记左删失。保留原始/平滑速度及判定区间，禁止边界填充制造不存在的持续区间。

标签只是候选场景，不是人工行为真值。按log/类别固定哈希抽样，每组最多20个窗口人工复核，不足则全查；保存rule_label、rule_version、review_status=pending/confirmed/rejected/corrected、reviewer、修订原因及人工标签。pending不等于完成，人工修订不覆盖规则标签，也不查看模型排名后挑选案例。

首轮3A只给已有窗口增加标签，不改变采样；保存窗口键及相对onset/end，报告窗口/instance/log数，不宣称独立episode数。3B才在原fit/select完整track上抽取canonical事件后关联重叠窗口，训练/选择角色不跨log。禁止单纯按onset相距<=1s的传递闭包合并：0/0.8/1.6s可能串联多个事件。3B独立冻结绝对时钟、事件区间、stop/restart分隔和ID算法，再报告episode数。

S4d v1实施口径：阳性onset在[0,6)，历史[-2,0)仅作上下文；跨左边界急刹保留删失证据。初始状态取anchor前最后平滑速度区间；观察到低速dwell但无符合prior条件的stop，后续持续起步标缺少已确认stop，不当阴性。S/U形状在未来最大连续可靠方向段上计算，区间边界不是人工行为起止。图表及候选抽查均为离线GT辅助，人工完成状态单独记录。

## 7. 停车及道路指标的后续契约

停车时间/位置指标在标签复核之后落地：从预测未来相邻点推导速度，不用GT当前位置偷偷修正预测锚点；与GT使用相同停速/持续时间规则。只在有足够剩余点验证持续停止时评分。

按离线GT事件选择停车前0.5s/1s查询作分组，仍只向模型提供当时已到观测。未预测到停车记miss，不用最后预测点冒充停止点。停止时间绝对误差、沿最后有效行驶方向的停止位置超调/不足在命中样本上报告，同时必须报告miss数与分母。

另设未停车且非已静止的负例队列，报告误报停车率，防止始终预测停止的模型“获胜”。车辆越界率须先重建类别与地图可行驶区域；行人不能直接套用机动车道路约束。TTC/碰撞只能作为明确参考运动假设下的风险代理，不当作闭环避撞成绩。

## 8. 3B受控场景及遮挡边界

真实轨迹回放、在真实轨迹上降质观测、完全合成运动三类结果分别保存，不能平均为一个“真实驾驶总分”。首轮合成种子、轨迹生成参数和观测噪声种子分开记录。

建议先实现三个解析可核验场景：

1. 直行4m/s，t=2s开始-2m/s²减速，t=4s停止并保持；历史段保持直行，位置/速度连续。
2. 2m/s直行，t=1s以-2m/s²刹停至t=2s，保持至t=3s，再以1m/s²起步至t=5s；之后匀速。
3. 4m/s的S弯，t=0至2s以+15度/s转向，2至4s以-15度/s转回，之后直行；历史直行。模型仍只每0.5s收到位置，不声称从nuScenes获得了高频GT。

每场景先跑clean，再共享基础随机数加入0.5m噪声与单次离群。真实/合成、运动事件/观测异常的时刻分别记录，避免只测两者恰好重叠的案例。先验证无速度/位置不连续及无刹停倒退，再比较模型。

遮挡恢复先用物理基线：删除第4/5/6源时刻包，第7步重现；保持源时钟/到达时钟和不重复同化规则。已有学习候选接口仍会拒绝缺包，这属于合理失败，不能通过补GT或把重复位置当新测量绕过。学习候选接入缺失mask/观测年龄并重训属于3C的独立工作。

邻车切入/行人横穿先用现有邻车/地图标注挖掘事件，再单独设计因果交互特征；不能把当前单目标运动修正称完整交互预测。

## 9. 实现、产物与验收

首轮实现详见[Phase 3A执行计划](superpowers/plans/2026-11-10-scenario-evaluation-phase3a.md)。新实现放在新分支；旧worktree只读。预测张量由冻结checkpoint重放，不重新训练或改门控阈值。基于GT的标签/指标模块不参与模型输入构建。

产物至少包含protocol/run/completed、父文件和源码哈希、样本键、查询/目标索引、逐窗口/逐更新/逐事件CSV或NPZ、分组/逐log/逐训练seed摘要、不可评价原因、标签人工抽查清单、案例图。另必须按role/strategy保存不可变原始`P[N,11,12,2]`、行键、dtype/shape/hash和GT来源引用，GT可共享保存，不能只留分数。16策略5472条流的float32 P共92,454,912字节，不含其他产物，不能当峰值内存估计。

运行时父链原环境验证只做一次并绑定证明；明确owner project roots。沿用float32、model.eval、CPU两线程与原4096平坦批次路径，先复现旧数值再优化；同环境要求逐项精确一致，变更后端必须运行前冻结新容差协议，不在失败后放宽。按role/strategy分片，不堆叠所有模型到GPU；每角色缓存读取一次，避免内循环重复解压。

实现顺序：S0协议→S1来源/GT历史装配→S2真实单窗口重放及旧指标→S3固定lead→S4a增益/尾部、S4b方向、S4c恢复、S4d标签→S5分片重放/原始P发布→S6汇总报告→S7冻结后正式全量运行。S4a依赖S3，S4b/c依赖S1+S3，S4d依赖S0+S1；标签不阻塞纯指标数值测试。S5/S6先小夹具验收，不在S7前计算正式全量成绩。

S0–S2现有入口为`scripts/run_scenario_preflight.py`，产物标记`preflight_completed.json`，不冒充完整3A的completed。直接父owner显式传入；上游parent_path链通过Git查询实际worktree owner并将身份→root映射固化进run，不按输出目录深度猜测。现有冻结父验证器仍沿用原工作区布局；迁移父链前需先通过原入口预检。

验收以时间索引正确、因果性、数值与摘要可重算为准，**不要求新模型获胜**。既有ADE/FDE/jitter必须复现；新指标独立从预测/GT重算。恢复必须正确处理延后受损、恢复后再次超阈值、未受损、窗尾不足及配对不完整。

审计两层：从已保存P/GT重算全部指标及计数，再固定哈希抽样重放模型对照P。测试需有独立慢速标量参考，不用同一向量函数算两遍作为索引证据。预测目标晚于step8不等于已使用step8观测，因果性以输入到达时间判断。

未来GT只用于离线评价；场景阈值/采样/统计口径在计算模型成绩前冻结。所有seed保留，按log报告，不因仅4log输出显著性断言。已有holdout不再当新盲测；扩大开发窗口不等于新独立数据。

## 10. 数据、Cosmos与投入顺序

当前3A/3B优先利用已有nuScenes元数据与地图，不需要新数据集。真实视频标签复核先审计对应camera文件覆盖，不因目录存在就宣称传感器完整。

Cosmos 3只列为后续可选辅助：小样本场景解释或接入视觉感知链后的外观压力测试。生成视频不能当米制轨迹/碰撞真值，模型文字评分不替代数值评测。本机RTX3060 Laptop 6GB不优先部署完整生成/驾驶仿真；本阶段不安装、不下载、不调用收费服务。

需要更高频轨迹和跨数据集验证时再评估Argoverse 2 Motion Forecasting及地图；需要自车策略闭环时才讨论CARLA/专门驾驶仿真。新数据、算力、模型和协议独立评审，不与本阶段混合成已有成果。

协议参考（2026-11-10已核对；本项目指标仍为独立在线协议）：

- [nuScenes官方预测评测](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/eval/prediction/README.md)：6s、2Hz、多模态与Miss定义。
- [nuPlan规划指标](https://github.com/motional/nuplan-devkit/blob/master/docs/metrics_description.md)：碰撞/TTC/舒适性属于规划和自车执行评价，不能直接挪作预测安全结论。
- [Argoverse 2 Motion Forecasting](https://argoverse.github.io/user-guide/tasks/motion_forecasting.html)：轨迹场景与数据字段。
- [Cosmos模型说明](https://docs.nvidia.com/cosmos/latest/cosmos3/model_reference.html)：推理/生成接口，不能替代本项目的GT和指标。

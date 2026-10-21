# P6.3 设计提案：因果反馈与逐时域门控

日期2026-10-21，基线387460f，沿用feature/causal-context-online。
状态：八组结构消融、冻结选择及条件推进的历史val回归已完成并通过生产重放验收。准确度固定RH达标；稳定性保留旧基线。第二轮一致性实验未启动。详见[结果报告](../../CAUSAL_CONTEXT_PHASE63_RESULTS.md)。

## 1. 目标与依据

在保留冻结P4/GRU/CV/map候选、真实包时间线与精确CV回退的前提下，回答三问：
方向信息是否有效；刚刚兑现的历史预测误差能否改善可靠性判断；共享一个标量g是否限制不同未来时刻的融合。

本轮推荐路线是冻结候选的小模型改进。完整SHARP重训和终点引导新地图查询作为后续独立路线，避免同时改变候选、数据协议与门控。
论文与代码来源及迁移限制见[调研](../../research/2026-10-21-streaming-motion-forecasting-frontier.md)。

SHARP/HPNet启发历史信息参与当前决策；DeMo/DeMo++启发区分方向与时变状态；这些是迁移启发。本提案的误差EWMA、特征组合和门控结构是我们待验证的设计，不是这些论文的等价复现或已被证明的新颖贡献。

## 2. 现有瓶颈与不改变的协议

P6 gate当前为8→16→1，同一个g融合当前所有剩余未来点，8维中多项只保留范数。
P6.2运动组依赖较强且map标签使用型混合参考仍有空间；P6.1简单跨更新惩罚降低jitter时FDE可能变差。

完整train25560轨迹/468场景、train_val6621/176、val7201/136；三seed2027/2028/2029。
保留六地图包条件和每0.5秒更新、终点t0+6秒。位置仍是已释放的干净标注观测，历史val仍为固定回归集。
训练/选模只用train/train_val，val不做特征选择或超参数搜索。不引入传感器、不下载外部数据或权重。

## 3. 统一模型：337个可训练参数

对更新k及未来绝对时刻T_j，τ=T_j−t_k>0：

    x[k,j] = concat(q8[k], D4[k], R4[k], H3[k,j])
    g[k,j] = valid[k] * sigmoid(MLP_19_16_1(x[k,j]))
    prediction[k,j] = CV[k,j] + g[k,j] * (map[k,j] − CV[k,j])

两层MLP中间GELU，可训练参数19×16+16+16+1=337，是替换原161参数gate后的总量，增加176；不包括冻结候选编码器。
共享MLP在各未来点重复使用，不为每个未来点创建独立网络，也不新增多模态hypothesis。
已释放点的g强制0，指标与损失仍只读尚未释放mask；invalid当前所有点精确CV，不混入上次输出。

初始化复制同seed P6第一层前8列、bias与最后层，其余11列权重全部0。因此各变体初始预测在尚未释放的未来点与P6一致，需实际验证ADE/FDE/jitter一致及无效包逐位相等；已释放点按新协议g=0，不要求其与旧P6的无评估意义回溯值相同。
所有变体用完整19维张量/相同head计算，关闭的因素组填0；参数量、隐藏宽度、初始化和主损失一致。零填充使有效输入自由度不同，不能声称所有有效容量相等。

## 4. D：保留有符号方向信息的4维特征

当前速度方向e_parallel=v/||v||，垂直方向e_perp=(-e_y,e_x)。已观测加速度a=(v−v_previous)/.5。
D由a在两方向的投影，以及map−CV在剩余未来点上的均值修正在两方向的投影组成。
加速度投影除10m/s²，修正投影除20m，新增四维均clip到[-5,5]；旧q8原样保留。

||v||<.5m/s时沿用截至当前最近一次>=.5m/s的已观测速度方向；若整个已见前缀均不足阈值，方向投影全部0。不得访问未来速度选择方向。此规则及单位在运行前冻结，不因val表现修改。

## 5. R：只用刚刚兑现误差的4维可靠性反馈

在k>=2，新位置p_k已经到达，可比较上次发出的CV/map预测在绝对t_k处与p_k的偏差：

    error_cv[k]  = norm(CV[k−1](t_k) − p_k)
    error_map[k] = norm(map[k−1](t_k) − p_k)  # 仅上次map有效时
    E[k] = .8*E[k−1] + .2*error[k]

首次合法误差直接赋E=error，避免零初始化偏差。R为log1p(E_cv/1m)、log1p(E_map/1m)、has_cv_feedback、has_map_feedback；log项clip[0,5]。
无历史时E与flag为0。map反馈仅由上次valid=true发出的map候选更新；当前invalid时清空map反馈状态。下一次重获有效包时，不将invalid期间original回退的误差当作map反馈。
CV反馈可以继续累积。所有状态以condition/sample-key独立维护，不能跨轨迹、场景或split。

时间顺序：先接收当前观测→核对上次预测的当前落点→更新反馈→读取当前已交付包→产生新预测。只允许用t<=t_k的观测；不能用当前预测的尚未到达真实未来计算R。
反馈取冻结专家预测而非训练中gate输出，所以给定输入前缀后可确定性重建，不需要teacher forcing或训练过程中在线更新权重。
E只是经验可靠性分数，不将其称为已校准概率或严格估计的预测方差。

## 6. H：逐未来时刻的3维特征

H为τ/6、该未来点map−CV修正的纵向/20m与横向/20m投影；投影clip[-5,5]，方向规则与D一致。
启用H后，同一更新可以对近处和远处给予不同g；禁用H时输入不依赖j，退化为统一标量gate。
这些未来坐标是已经生成的候选预测，不是真实未来标签。P6.2的scalar oracle不足以证明H收益，必须消融验证。

## 7. 结构消融：完整2×2×2因素设计

固定八变体B、D、R、H、DR、DH、RH、DRH，各337参数；B仅q8+11零填充。
每个seed都从同一P6初始化；全部10epoch/patience3/batch4096，AdamW lr.001、weight_decay.0001。
训练主损失完整复用P6.1的masked SmoothL1+.05预测内部correction平滑，结构消融lambda_revision=0。
按train_val六条件平均ADE选各模型最佳checkpoint，epoch和优化顺序绑定metadata；早停导致实际训练量不同需明报，不只列最大epoch。

B是再训练预算对照；另冻结评估CV、P6 learned、P6.1 finetune_0、time_age_gate、准确度consistent_short、稳定性consistent_010。
不能只击败P6就宣称超过当前最佳，也不能将B到P6的额外训练收益归于D/R/H。
八变体名义前向和优化预算相同；R前缀预处理、H点级推理带来的实际耗时另测，不将相同参数量当作相同原生延迟。

## 8. 条件推进的一致性实验

第一轮只做八结构变体。只有DRH或其子集通过下述推进标准，才另立冻结配置比较一致性损失，不与第一轮混报。
候选第二轮对同一选定结构，从同P6初始化重训四组：无revision、固定lambda=.02、创新自适应lambda=.02、固定lambda=.02×train_mean_weight。
创新量使用当前已经到达的位置与上次CV当前点的偏差：

    innovation[k] = norm(p_k − CV[k−1](t_k))
    consistency_weight[k] = exp(−(innovation[k]/.5m)^2)
    L = L_base + .02 * weighted_overlap_revision

首步无配对；比较相同绝对未来时间，双方可导。weighted_overlap_revision按有前序的每个update先算未来点平均L2、乘w_k，再对这些updates等权平均，不除sum(w)。
train_mean_weight只在完整train的有前序updates上计算，用于固定平均系数对照，排除仅减弱正则总强度的解释；不保证各模型梯度范数相同。监督的真实未来只能进入L_base，创新权重不读取未释放标签。
目的是让直行小创新时保留连续性、转向等大创新时允许改预测；这是假设，仍可能损害FDE或稳定性。第二轮重新冻结选择，不将事后探索伪装成预注册结果。

## 9. 选模与验收

以下为本提案的预设阈值；若评审修改，先更新设计与配置再开始训练。以冻结P6.1准确度方案为reference：
- accuracy候选：六条件ADE最小；要求全条件FDE<=1.005×reference、clean逐update FDE p95<=1.01×reference、jitter<=1.01×reference。
- stable候选：全条件及clean ADE<=1.005×reference、全条件FDE<=1.005×reference、clean FDE p95<=1.01×reference，再选jitter最小。
- 同时保留reference和旧stable作可选基线；若基线获选，不称新算法成功。tie先目标metric，再ADE，再方法名字。
- 超过预算对照B的效果需报告三seed方向和配对scene区间，不能只选最好seed。

推进到历史val的标准：至少一个新变体在train_val满足相应约束，且ADE相对reference下降>=0.5%或jitter相对旧stable下降>=2%；还要优于B的对应主指标。否则本轮仅报告train_val消融，不启动val调试。

有价值的目标是val ADE比现有准确度方案再降>=1%，且FDE无恶化；稳定性路线目标jitter比旧stable再降>=3%，同时守住上述ADE/FDE约束。这些是研究目标，不是预测收益。
使用val后的全部结果按固定选择报告，不重新选择架构、阈值或特征。历史val不称新盲测。

## 10. 实现边界和验证

新模块独立于P6/P6.1/P6.2绑定源码；只读生产cache并核对原父来源。R预处理保存前缀依赖、实际数组/scene/key/hash，不能只信预计算特征自洽。
模型、优化设置、八变体mask、P6初始化hash、完成标记与全部源码绑定；输出独立outputs/context/p63。
逐点gate数组为[condition,N,11,12]，与旧标量格式区分，全部指标和权重需重预测核对。

必要测试：
1. 修改未到达观测/labels不影响当前及此前D/R/H；输入接口不向模型提供真实future。
2. R当前创新严格取上次预测的当前绝对点，首步无反馈、invalid前后不借original伪造map可靠性。
3. 重排batch/样本不串状态、切scene/split不延续状态、低速与静止无NaN。
4. 初始化全变体在未释放未来点及评估指标与P6一致；禁用H时未释放未来点g相等；invalid精确CV；时间mask与jitter对齐。
5. 篡改特征/NPZ并重算hash仍被生产来源/完整重放拒绝；支持失败stage确定性重训。

报告train_val所有八变体/三seed及固定基线，若推进则完整val六条件；ADE/FDE/jitter、逐update和maneuver、p90/p95、失效回退和固定3seed平均后的2000次配对scene bootstrap。
延迟固定256key/16warmup/batch1/CUDA同步：独立报反馈状态更新+特征+head+blend，地图查询/P4 forward另列；缓存读入和批量吞吐不能当完整在线延迟。
不因文档改动重复全量训练，也不修改旧报告来迁就新结果。

## 11. 后续强模型路线

若八因素设计仍只带来极小改善，优先转向候选：在独立环境按官方nuScenes设置训练SHARP，并用相同观测窗口、单/多模态指标分别对照。新holdout需重训任何见过该场景的上游。
终点引导地图包必须另建缓存契约，只查询当前可用的几何；不能从旧包覆盖外隐式补入地图信息。
本提案尚未启动这些工作，也不声称本模型达到论文基准的SOTA。

# P6.1：门控归因、短时收缩与跨更新一致性

日期2026-10-21，基线af9e45c，沿用feature/causal-context-online和现有worktree。
用户已批准上一轮提出的P6.1方向，并要求设计施工方案和开始实施。本阶段在该范围内执行。

## 目标与范围

回答三问：收益是否来自地图而非简单CV混合；末期短horizon负收益能否减少；能否在
保留准确度的同时减少跨更新jitter。固定P4/GRU、P6缓存、六条件和三seed，不下载数据。
新增模型不修改任何P6已绑定源码，也不覆盖P4/P5/P6原始产物。

## 固定实验组

1. sliding、ungated_map、p6_fixed、p6_learned：原始四对照，严格重现P6。
2. constant_gate：g取0:.05:1，乘相同hard-valid，按完整train_val六条件平均ADE选择。
3. time_age_gate：11个update×5个age bin(0,.5,1,1.5,2秒)，每格g在同一21点网格，
   用train标签拟合ADE最优值，未覆盖格设0；不读取val拟合。无效行hard0。
4. original_gate：161参数、与P6相同设置，从头在train训练20epoch/patience5；
   候选替换为original，quality最后一项重算为original/CV预测分歧，其他字段与hardmask
   不变。与P6地图gate比较候选几何价值，明确候选编码器容量差异仍存在。
5. short_gate：冻结p6_learned后施加cap(h,tau)=clip((h-.5)/tau,0,1)，tau=0时cap1，
   h=(12-update_step)*.5秒。tau网格0,.5,1,1.5，只按train_val选择；先三seed平均选同tau。
6. finetune_0、consistent_002、consistent_010：均从同seed P6权重初始化，10epoch/
   patience3，同批次/seed/优化器；一致性系数lambda分别0,.02,.1。lambda0提供额外训练
   预算匹配对照，不将再训练带来的收益归因于一致性项。
7. consistent_short：对上述两个非零lambda模型各做同一tau网格，只按三seed平均
   train_val ADE选择同一lambda/tau组合；保存完整选择表。

合计12个策略（4+1+1+1+1+3+1），不新增逐horizon向量gate或传感器候选，避免本轮
同时引入过多结构。constant与short校准使用train_val；lookup只拟合train。所有选择
包含missing/expired，保持所有条件同样本同更新。

## 跨更新训练

损失=原P6 masked SmoothL1+.05*smoothness(correction)+lambda*overlap_revision。
对每个抽样update k>1，同条件/同轨迹取k-1的过去输入，网络分别前向，比较二者在
当前仍未释放的同一绝对未来时间上的预测L2范数均值；当前/上次分支均可导。
第1次不配对，不跨样本/条件边界；pair损失只用两次预测，不读future labels。
训练监督labels仍单独获取。整轮从train随机shuffle所有6*N*11 updates，batch4096。
选择每模型最佳train_val ADE checkpoint，earlystop以ADE；逐epoch记录ADE/FDE/jitter
可单独审计，不能用单batch无序输出计算跨更新jitter。

## 验收和选择

旧P6对照clean及六条件指标应在数值容差内一致。选模时对完整预测序列计算jitter，
首步0，重叠时间严格对齐；soft cap不得破坏missing/expired精确CV回退。
最终accuracy方案按三seed train_val全条件平均ADE最小；stable方案在全条件平均ADE
<=1.01*P6、clean ADE<=1.01*P6时，选全条件平均jitter最小。若无非基线可行改进，
明确报告没有实现稳定性目标；不改变容忍阈值。tie按ADE后策略名字排序。
冻结所有策略/超参数/最终方案选择后才运行val，val不调参。历史val不是新盲测。
任何未来新留出若源自P4已见train，必须同时重训上游才称新留出。

报告六条件ADE/FDE/jitter、seed mean/std、update/horizon/maneuver、逐update ADE/FDE
的p90/p95（注明统计对象）、fixed3seed平均后的2000次配对scene bootstrap(seed0)。
对比各新策略−P6和最终两方案−CV；不把多个同scene更新当独立样本。
不新增离路率/MR定义，避免这一轮改变验收口径。

## 数据、产物与来源

完整train25560/468scene、train_val6621/176、val7201/136；seed2027/2028/2029。
所有split通过P6 load_verified_gate_cache，校验全部父数据与冻结模型候选重建。
新RefinementDataset只读包装GateDataset，不更改原数组；原candidate的quality重算
只在inputs中进行，labels独立接口。实际输入fingerprint、缓存manifest与P6 checkpoint
hash、执行源码、设置和scene绑定run identity，已完成训练可核对后复用。

输出outputs/context/p61/{training,selection,evaluation,report}。逐update指标保存为有shape/
dtype/hash的压缩NPZ，[condition,sample,update,metric]，权重同key另存；逐行ID、时间和
包诊断通过hash绑定已验证P6缓存，避免为12策略重复写千万行字符串CSV。report重新
预测并比较全部NPZ元素，再核对scene/group/quantile/recovery；禁止只相信summary。

延迟固定256keys/16warmup/batch1/CUDA同步，包含候选与quality读取、必要original质量
重算、gate/cap及混合返回，不包含P4 forward/地图查询；保存原始样本和scope，不将
批量吞吐作在线延迟。配对训练前向开销与推理开销分别报告。

## 模块边界

新增online/gate_refinement.py（数据适配、权重、配对损失）、online/refinement_training.py
（训练/校准/冻结选择）、online/refinement_evaluation.py（全序列度量/产物/延迟）、
analysis/refinement_report.py（来源验证/重放/统计）；scripts/{train,evaluate,report}_gate_refinement.py；
configs/models/gate_refinement.yaml；对应tests。既有P6模块全部只读。

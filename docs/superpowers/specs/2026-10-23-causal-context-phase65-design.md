# P6.5 一致性强度校准与平均系数匹配设计

日期：2026-10-23。分支feature/causal-context-online，基线a3af54a。
状态：用户已批准；实现与训练前验收通过，正式训练尚未开始，未访问新的val结果。
用户已要求继续下一阶段并确认本设计；本文件冻结系数、预算和晋级规则。

## 1. 问题与路线选择

P6.4固定lambda=.02的train_val jitter下降0.452950%，未达0.5%推进门槛；
adaptive下降0.348705%，mean下降0.282000%。三者未进入val。
训练权重均值mu=0.7076048737280716，因此原adaptive平均系数只有0.0141520974745614。
这尚不能区分正则过弱、权重分配不当和当前RH结构的稳定性上限。

候选路线：
1. 推荐：固定RH，有限强度对照与平均系数匹配；改动集中，能检验损失的强度/分配作用。
2. H的lead与逐点候选偏差拆分：利于归因，但不能直接回答当前一致性损失是否足够强。
3. 新候选或SHARP等外部模型：可能扩大精度上限，但引入新的训练/依赖与比较口径。
本轮只执行路线1，不叠加结构、候选、观测噪声、数据来源或选模规则的变化。

## 2. 训练矩阵与对照

固定RH共享head：337名义参数、15有效输入维度/273有效参数；不新增部署参数或状态。
InnovationDataset、创新尺度0.5m、weighted_revision及基础监督损失沿用冻结P6.4。
创新仅取已到达位置与上次CV在当前绝对时刻的预测之差，不读未到达观测或future标签。
权重w=exp(-(innovation/.5m)^2)，首步无配对。每update剩余点平均L2后乘w，再按有前序update平均。
双方预测均可导；不按权重和归一化，不只挑地图有效样本。

新增四组，每组三seed2027/2028/2029，共12个新模型：

| arm | 目标平均系数alpha | 原始lambda | update权重 |
| --- | ---: | --- | --- |
| fixed_005 | 0.05 | 0.05 | 1 |
| adaptive_matched_005 | 0.05 | 0.05/mu | w |
| fixed_010 | 0.10 | 0.10 | 1 |
| adaptive_matched_010 | 0.10 | 0.10/mu | w |

mu只由验证通过的完整train缓存中k>0的权重重建；每seed独立验证相同，而非从checkpoint声明读取。
mu必须有限且0<mu<=1，否则中止。lambda=alpha/mu在训练开始前固定，不能按batch、epoch或train_val重算。
保留w在[0,1]内，通过lambda缩放实现平均系数匹配；不裁剪归一化后的权重，也不改旧损失函数的范围检查。
“匹配”仅指train上lambda*w的平均系数，绝不声称梯度范数或实际计算量相等。

旧RH及P6.4的zero、fixed(.02)、adaptive(.02)、mean(.02)是冻结对照，严格加载并核对已保存产物，
不重新计作新训练模型。新的constant-mean组若取lambda=alpha/mu、weight=mu，代数上等于fixed(alpha)，
因此以损失/梯度等价测试作为均值控制，不重复训练数学相同的组。
旧adaptive/mean保留为不同平均强度的诊断，不用于证明新matched组优于所有强度匹配方案。

全部新增模型从各seed的P6生产初始化训练，不接着P6.4最佳checkpoint微调。
同一seed/初始化/shuffle；max10epoch、patience3、batch4096、AdamW lr.001/wd.0001、clip_grad_norm5。
L_base=masked SmoothL1+.05 correction smoothness；best epoch仍按完整train_val ADE。
最多120新epoch；实际早停、训练和逐轮选模耗时如实报告，不保证取得收益或固定耗时。
CUDA初始化前冻结CUBLAS_WORKSPACE_CONFIG=:4096:8。禁止事后改epoch、系数或选模依据。

## 3. 数据、评估与冻结选择

仅复用train25560轨迹/468场景、train_val6621/176；条件推进历史val7201/136。
六地图包条件：clean、delay500ms、delay1000ms、dropout、missing、stale；11次更新，固定6秒终点。
目标位置是干净的已释放标注，地图查询包退化不等于真实定位/跟踪噪声。
缺包、无覆盖或age>2秒精确回退CV；新增训练正则不改变推理路径。

旧18策略加新增4组，共22策略；必须完整报告，旧18的train_val与P6.4冻结表核对atol1e-9/rtol0。
旧zero/RH模型状态指纹须一致，完整metrics和weights核对同容差。
选择池仅RH、旧revision_fixed和四新组；不重新选择D/H/DRH或旧稳定性模型。
其余旧组仅作诊断，不能依据本轮结果扩大选择池。

固定参考RH；以三seed、六条件均值计算：
- overall ADE <= 1.002*RH；clean ADE <= 1.002*RH。
- overall FDE <= 1.005*RH；clean FDE p95 <= 1.01*RH。
- 合格池中jitter最小，tie依次ADE、策略名；没有合格新增组时保留可行的冻结基线。

无论晋级与否，都在读取val前冻结选择、全部阈值、输入/模型/源码身份。
仅获选者是四个新组之一、jitter比RH下降至少0.5%、且严格低于旧revision_fixed和旧zero，才运行val。
旧fixed或RH获选不算新方法成功，不推进val；不因接近门槛而改阈值。
研究目标：固定获选方案val jitter比RH降低至少1%，同时满足上述误差/尾部约束。
相对旧consistent_short/consistent_010/CV另报，不把“比RH更稳”说成稳定性冠军。
历史val已有多轮使用，仅作为固定回归，不能称盲测。

若晋级，则在本轮P6.5目录评估完整22策略的历史val，包含此前未进入val的四个P6.4组；
这是新冻结协议中的统一对照，不能回写成P6.4通过推进或事后选用P6.4方案。
val不选结构、强度或checkpoint；不达研究目标也完整报告。
若未晋级，本轮程序不得加载val缓存或创建val评估目录。

## 4. 身份、安全复用与模块边界

全部P6至P6.4源码、配置、测试及旧outputs保持冻结；只新增独立模块与outputs/context/p65。
训练身份绑定arm、alpha、lambda计算规则/数值、实际train mu/指纹、split/scene/key、候选契约、
父P6初始化、RH角色/mask、当前与全部父源码/config、seed/预算、CuBLAS设置与run manifest。
生产加载必须由实际验证通过的train缓存独立重建mu与指纹；同时篡改metadata/checksum也必须拒绝。
绑定P6.3和P6.4父选择与正式报告身份、父checkpoint及完整输入契约；不信任checkpoint自证。
阶段完成marker最后原子发布；中断未完成stage从同一初始化重启，完成复用必须全面核验。

实现边界：独立strength_consistency定义四arm及系数；strength_training负责训练/加载身份；
strength_artifacts/strength_evaluation负责22策略产物/选择/条件val；analysis/strength_report负责全量重放与统计。
新增三个CLI和一份配置；调用只读旧模块，不通过修改全局常量、monkeypatch或覆盖旧输出实现复用。
新增文件的精确路径、接口和TDD步骤在批准后的施工计划中确定。

## 5. 统计、成本与验收

所有22策略保存[6,N,11,3]指标及[6,N,11,12]逐点权重。
报告重读生产缓存和12模型，逐元素重算全部预测度量/权重/分组与冻结选择。
每split15对比、7范围、3指标，共315条区间；2000次scene配对bootstrap，先固定三seed平均。
15对比分别：四新组-RH、四新组-旧revision_fixed、两种强度各自adaptive-fixed、
fixed_010-fixed_005、adaptive_matched_010-adaptive_matched_005、固定选择-旧准确度/旧稳定性/CV。
即使固定选择是旧组，也保留预定义标签和15对比；不按结果删掉不利比较。
总体按scene合并六条件再采样，不能把seed/condition/update当独立样本；另报seed样本标准差。
区间未做多重比较校正，不替代预设阈值，不证明对所有训练随机性的普遍收益。

报告按condition/update/remaining horizon/maneuver、p90/p95、gate、恢复与训练loss分组，
明确恢复指标相对clean预测定义而非感知恢复。平均系数、每epoch监督/正则loss和真实训练耗时单列。
在线计时沿用同256key/16warmup/batch1/CUDA同步，计入真实当前状态更新、特征、head、blend和返回；
排除前缀重建、地图查询、P4 forward、冷缓存加载。保存raw测量和逐seed分位数。
新组与RH同推理结构，不把微基准噪声或跨轮时间变化写成加速收益。

TDD覆盖：alpha/mu固定系数、constant-mean与fixed损失及双方梯度等价、前缀因果/标签隔离、
zero/首步/非有限或非法mu、角色/系数/输入和完成复用篡改、父契约、22策略矩阵、
精度/尾部约束、tie/旧组获选/未达门槛不读val、逐点产物重算、315区间和scene/seed缺失拒绝。
旧损失原语已有测试继续回归，不为本轮改旧源或放宽身份规则。
完整pytest/Ruff/公开扫描/diff，独立只读实现和结果审查；正式训练前冻结源码，结果前后核对不变。
最终更新P6.5报告/路线图/spec/plan，本地提交；不推送、不合并、不下载、不清理旧实验或更改简历。

## 6. 失败解释与后续边界

如果fixed增强有效而matched adaptive仍落后，应归结为当前创新权重分配在本任务下未获优势，不能包装自适应创新成功。
如果更强正则违反精度/尾部约束，保留RH并报告权衡，不能事后放宽约束。
若四新组均无可推进收益，则结束本轮，建议另立协议做H归因或新候选；不追加系数扫描直至获得正结果。
本轮不启动P7，不自动把train_val提升写入简历。

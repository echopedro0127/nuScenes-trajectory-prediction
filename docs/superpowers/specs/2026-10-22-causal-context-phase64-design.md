# P6.4 创新自适应时序一致性

日期2026-10-22，基线58f17ea，分支feature/causal-context-online。
状态：实验与完整重放完成；revision_fixed的train_val jitter下降0.453%，未达0.5%门槛，未读val、未晋级。
验收详见[完整报告](../../CAUSAL_CONTEXT_PHASE64_RESULTS.md)。
授权：用户“继续下一轮”，执行已批准P6.3设计第8节的条件一致性实验；RH已满足推进条件。
本轮不重新选择结构，不拆分H，不改上游候选，不新增数据/依赖，不启动SHARP或P7。

## 方案与对照

固定337参数RH、P6生产初始化、三seed2027/2028/2029。比较四组：
- revision_0：无一致性，预算对照；须与P6.3 RH重训结果完整核对。
- revision_fixed：lambda=.02，所有有前序updates权重1。
- revision_adaptive：lambda=.02，w=exp(-(innovation/.5m)^2)。
- revision_mean：lambda=.02，固定w=train_mean_weight，等价lambda=.02×train_mean_weight。

创新使用新到位置p[k]与上次CV在当前绝对时刻的预测cv[k-1,k]之L2差，k=0创新和权重记0。
权重不使用未来标签，不做EWMA，不使用当前CV误差（当前CV在已到达点可平凡为0）。
均值仅由完整train的k>0样本计算，六条件重复不改变均值，不能在train_val/val重算系数。
固定系数方案简单但可能压制转向；自适应方案允许大创新时修订；均值对照排除仅减弱总强度的解释。

双方预测均可导；按当前未释放future mask比较相同绝对未来点。每个有前序update先平均剩余点L2，
乘w后对有前序updates等权平均，不除sum(w)，不按有效地图重新筛样本。
首步不配对。零权重或全首步返回连接双方计算图的0。标签仅进入监督损失。
L_base=masked SmoothL1+.05 correction smoothness；不变。

全部从同一P6初始化重训，10epoch/patience3/batch4096，AdamW lr.001/wd.0001。
每epoch按完整train_val ADE选best；实际早停和双前向成本单列。CuBLAS=:4096:8在CUDA初始化前冻结。
四组同seed/shuffle/初始化/名义参数；零组不执行不需要的前序前向，不能称实际计算量相同。

## 选择与推进（本轮训练前冻结）

冻结参考为P6.3 RH，不依据上轮val重新选模型。评估完整旧14策略+四新组=18策略。
选择池只含旧RH与四新组；其他旧策略仅作固定参考，不能通过本轮重选H或DRH。
以三seed×六条件均值选择jitter最小方案；要求总体及clean ADE≤1.002×RH、总体FDE≤1.005×RH、
clean FDE p95≤1.01×RH。tie按jitter、ADE、策略名。
精度容忍0.2%用于尽量保留RH相对旧准确度约1.23%的收益；不是不损失精度的承诺。

仅当获选正则组jitter比旧RH下降≥0.5%，且比新revision_0更低，才读取历史val。
若零组或旧RH获选，不称一致性算法成功，不运行val。val仅固定选择加全部消融回归，不调超参。
研究目标：固定新方案val jitter比RH下降≥1%，同时满足上述约束；另报告相对P6.1准确度/稳定性，
不能把比RH更稳说成优于旧稳定性。历史val是已使用回归集，不是新盲测。

## 数据与身份

train25560/468场景，train_val6621/176，条件推进val7201/136；六实际包条件、11更新、固定6秒终点。
位置是已释放干净标注，缺包/无覆盖/age>2秒精确CV。创新只训练使用，不增加部署推理输入或状态。
父P6/P6.1/P6.2/P6.3源码、配置、产物不改；本轮独立模块与outputs/context/p64。
训练绑定新源/config/父源、实际输入与创新数组/均值、初始权重、arm、seed、scene、cache manifest和完成标记。
completed复用及生产加载都检查arm/lambda/mean/mask，篡改并重算文件hash也不能绕过角色检查。

## 审计与报告

全部18策略保存[6,N,11,3]指标和[6,N,11,12]权重。完整重构train创新/均值并核对12模型，
重加载父缓存及全部模型，逐元素重算train_val和条件val指标/权重/分组/选择。
旧14策略与P6.3冻结表核对；revision_0与RH全部metrics/weights核对，训练状态指纹另查。
按scene固定三seed平均后2000次paired bootstrap：四组−RH、三正则组−revision_0、
adaptive−fixed、adaptive−mean、固定选择−旧准确度/旧稳定性/CV；六条件及overall，均报ADE/FDE/jitter。
按条件、update、horizon、maneuver、p90/p95、gate、恢复统计及seed mean/std完整保留。
延迟沿用256key/16warmup/batch1/CUDA同步；训练创新无需部署执行，线上仍为同RH特征/head路径。
训练双前向及逐轮选模时间与在线延迟分开。保存原始计时、模型/cache/source/选择哈希。

## 必须验证

前缀与标签扰动、当前绝对时刻索引、首步/全零/极大创新、权重分母、两侧梯度、重排不串样本；
四arm身份/均值/初始化及零组重现、完整矩阵/约束/tie/基线获选不读val；更新hash后的NPZ篡改仍拒绝。
先失败测试再实现，独立只读代码与报告审查，全量pytest/Ruff/公开扫描/diff；只本地提交。

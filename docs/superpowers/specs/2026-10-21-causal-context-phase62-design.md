# P6.2：运动候选与冻结质量特征诊断

日期2026-10-21；基线38e346c，feature/causal-context-online。用户已批准继续诊断运动候选与质量特征的增量价值。

## 目标与方案选择

直接扩大gate无法解释瓶颈；直接新增运动候选会混合候选与选模影响。本阶段采用先诊断的第三条路线：冻结所有模型，只在完整train_val（6621轨迹/176场景、seeds2027/2028/2029）估计候选与门控空间。历史val不读取，不训练新模型；不下载、不推送。

## 候选诊断

读取生产load_verified_gate_cache与P6.1 load_suite/load_selection，核对缓存、模型、选择身份及scene。CV、map、original、filtered端点统一在原地图包valid mask内可用，无效包均退回CV；这人为约束original/filter仅用于同条件比较，不意味着它们物理上依赖地图包。保留P6 learned与original_gate两个实际门控参照。

对每个轨迹更新，使用仍未释放的真实未来标签，在g=0:.05:1及该候选参照gate的实际g中选ADE最小值；map参照P6，original参照original_gate，filtered参照P6权重。端点包含0/1，严格小于时更新，平手取网格中较小g。无效包强制g=0。保存三个oracle的ADE/g；该有限网格+实际gate是可行混合中的诊断参考，不是连续最优解，也不保证可部署模型可达到。标签仅进入oracle计算与指标，不进入gate。

报告端点ADE/FDE/jitter、理想混合ADE、相对实际gate的差距、map优于CV比例、oracle零权重比例及实际gate与oracle权重差。oracle不报告部署稳定性或推理延迟，不称新方法收益。

## 冻结特征依赖

只测试P6 frozen gate：八个单特征、timing_coverage=[0,1,6]、motion=[2,3,4,5]、all八特征，共11组。重复seed0/1/2；每条件、更新步、相同实际age、valid=true内置换轨迹donor，invalid不变。同repeat同donor用于各组，组内特征整体拷贝。预测从未修改的CV/map混合，只有quality选定列替换；valid/因果时间/候选不变，完整序列后计算jitter。

置换可能因相关特征冗余、分布改变或组内常数而低估/高估重要性，不等同删除特征重训后的增量价值。每组输出实际quality变化比例、权重变化及ADE/FDE/jitter差，常数置换不得误称无用。

## 产物与统计

独立analysis/gate_diagnostics.py、scripts/diagnose_gate_candidates.py，旧P6/P6.1绑定源码不改。outputs/context/p62存每seed完整数值NPZ（端点/实际/置换指标、oracleADE/g）、scene与condition表、donor哈希、父来源与源码快照；最后原子manifest。CLI固定train_val和三seed，--verify重新加载生产缓存/模型、重算所有数值及表，检查hash/shape/key/source/完整字段集合，篡改NPZ并更新hash也拒绝。

汇总三seed均值/样本标准差；置换先平均三repeat、再固定三seed均值，以176scene为配对单位bootstrap2000(seed0)，分别clean与六条件等权平均，11组×3指标×2范围=66条区间。重复记录不当独立样本。文档如实报告小增益、负面结果与oracle限制，不据诊断宣称新的val提升。

## 验收与下一步

先失败测试验证mask、oracle只看未释放标签/端点保证、置换不跨分组、无效包回退、NPZ更新hash仍拒绝；再实现和完整运行/重放。全量pytest、Ruff、公开扫描、diff检查；旧绑定源码不变。本地提交文档、源码和测试，数据/模型/outputs不入git。

只有诊断完成才提出下一项训练：map混合仍有明显空间且motion/disagreement有依赖时，优先考虑因果方向/曲率质量特征的预算匹配消融；若候选空间不足，优先改进候选。重训消融另立冻结协议，本阶段不自动启动。

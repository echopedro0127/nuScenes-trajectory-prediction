# Independent Holdout Protocol v1

日期2026-11-09，固定在第一次新holdout预测/评价之前。依据既有select结果选定RH为主模型；这不是对整个研究事前注册，也不是新采集盲测。

## 冻结内容

数据为现有固定log互斥划分，holdout=independent_holdout（3,329轨迹、73scene、6log）；不得重划分、筛选容易轨迹或更改场景。
history2s/5步、future6s/12步、2Hz，每0.5s释放新增标注位置，11次更新，保留原始预测终点。
固定六条件clean/delay_500ms/delay_1000ms/dropout/missing/stale，固定既有地图包和fallback规则，max_age2s/fixed_scale0.5/decay1s。
输入限已有标注运动、静态HD map及时间可用信息，无Camera/LiDAR/Radar感知特征，无新训练。

主模型RH，主对照因果CV；次要模型original、map、quality_gate、revision_adaptive。revision_0与RH权重相同，不重复当作独立模型。
不因holdout上其他模型更好而更换主模型。一次性评价后不继续在同holdout上调参或选checkpoint。

## 指标与统计

主指标：六条件等权的在线ADE（每条轨迹每次更新等权，剩余时间点距离先取均值）。每条件同批轨迹同样11次更新，因此等于全部条件更新的池化均值。
主比较：RH-CV的ADE绝对差。使用log级配对cluster bootstrap，抽取6个log、有放回，重复5000次，NumPy PCG64 seed20271109；每次对所抽log的误差总和/对应样本更新数求池化均值，重复抽中的log按次数计入。
95%区间用0.025/0.975分位数，线性插值。若差值区间上界<0，仅表述该holdout上的改善证据；否则主结论为未证实优于CV，不转用次要指标改写主结论。
次要指标：FDE（原始固定终点）、jitter（相邻更新共同剩余时刻预测修订距离，首次记0）；全部模型和条件完整报告。次要比较仅描述，不做显著性筛选或新增主检验。
报告六个log的分项、场景/轨迹计数和全部点估计。log仅6组，bootstrap区间可能不稳定，不将大量相关更新当成独立样本，不称作跨seed结论。
missing条件RH/gate/adaptive的权重全0、输出精确等于CV属于实现验收，不是统计收益。

## 执行与不可变性

协议JSON绑定划分/配置、源数据cache manifests、学习模型及训练/审计记录、协议和当前执行源码SHA256。
生成前重验独立split、模型完成marker/来源/初始化，确认holdout的predictions/context/candidates/evaluation尚未生成。
先冻结JSON，再实现只读推理/评价入口；任何新的执行源码须在不读取holdout误差的情况下以独立实现清单绑定到此协议身份。
第一次推理前创建evaluation_started记录，完成后发布包含协议/实现/输入/输出哈希的completion记录。崩溃可在同一协议与权重下恢复，保留运行日志；不得根据已见误差改变模型。
若发现评价bug，保留旧运行证据，解释修复并重跑全部固定模型，不能只重算有利项。原始数据、旧结果不覆盖。

模型选择已参考新select；研究者已知历史结果，此划分只是上游未见场景的重训后诊断。冻结后尚无holdout收益结论。历史公开RH/consistent_010推荐不自动替换，不推送GitHub、不改简历。

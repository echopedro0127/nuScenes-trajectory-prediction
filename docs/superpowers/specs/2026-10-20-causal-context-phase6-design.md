# P6：地图查询包质量门控设计

基线f3025c0。P4地图相对original及参数对照有全量trainval增益，满足P6前提；
P5 Radar未优于强CV/滤波，本阶段不扩展Radar。沿用原worktree/feature分支、本地提交。

## 问题与方案

比较固定门控与学习门控是否在上下文缺失/延迟时可靠回退，并相对强基线体现价值。
冻结P4各seed的original/map及固定GRU；混合预测=CV+g*(map预测-CV)。这是新CV锚点
混合器，不是原Corrector的offset缩放，故同时报告original、ungated_map、CV、filtered、
frozen、fixed_gate、learned_gate共七策略。严格无效时g=0，输出精确等于CV。

固定g=.5*exp(-age_s)，要求包存在、地图有点且age<=2秒；学习g=sigmoid(MLP8→16→1)
乘同一有效mask，初始最终bias=-2。gate只读因果质量；不使用真实未来或oracle最佳策略。
学习网络161参数，候选完全冻结。用train训练、train_val选checkpoint；val不调参数。

## 六种固定条件

位置观测始终干净，沿用每0.5s释放一点及原t0+6s终点。只扰动地图查询结果包：
clean；delay_500ms；delay_1000ms；dropout（包4/5/6丢弃）；missing（全部不交付）；
stale（仅交付第1包）。包source=查询时刻t0+k*.5，available=source+delay。
每query只选已交付最新包，tie规则稳定；第1包之前缺失，不凭空提供anchor地图包。
超过2秒包保留age/source诊断，但不输入地图模型；ungated_map继承原original fallback，
两种gated策略回退CV。缺失行、过旧行不删除。

地图几何本身是静态数据；age指基于过去目标位置选取的局部查询包时龄，并非地图
几何产生时间。缓存坐标固定anchor，旧包不做目标运动补偿。查询包在source时刻以
当时已释放位置生成；从P4已验缓存选过去packet，绝不用当前clean位置重新选lane。
本实验只测packet退化，不把它称为目标位置退化或真实地图服务网络性能。

## 质量特征与缓存

8维：有效map点比例；clip(age/2)；当前位置与固定base当前点偏差norm/20m；
当前速度与base差norm/20mps；速度norm/20；过去差分加速度norm/10；剩余horizon比例；
剩余horizon内map/CV预测分歧norm均值/20m。最后项是候选预测、不是未来标签。
值均finite；缺包coverage=0；age/packet存在与硬有效性单独存储。

先验证P4 map mmap/NPZ、轨迹和GRU来源；验证P4 suite checkpoint与对应train/selection/
val场景和来源绑定，再批量生成各seed各split候选。只加载original/map权重。
独立NPY缓存按严格schema保存targets（仅labels方法读取）、CV/filtered/original/map候选、
quality、valid、packet source/availability/step/age、IDs/timeline；每文件shape/dtype/hash，
manifest绑定配置、P4模型/父缓存/预测及源码身份。加载重建packet时序/硬mask及质量，
拒绝错配置、漏文件、错key、修改输入、场景重叠或错P4 checkpoint。旧缓存不改写。
有界batch和mmap，不把全量条件更新展开成Python对象。

## 训练与验收

完整train25560条/468场景、train_val6621条/176场景、val7201条/136场景；固定P4
seed2027/2028/2029，每seed单独gate。hidden16、batch4096、20epoch上限、patience5，
AdamW .001/.0001；SmoothL1(masked forecast,target)+.05smoothness(g*(map-CV))。
固定shuffle seed，逐epoch日志、best checkpoint、实际数据/源码hash与场景隔离。

工程测试先失败后实现：前缀因果/未来包不影响旧输入；0.5/1s到达边界；stale第6次
age2.5s已失效；完全缺失精确CV；g范围0..1；模型调用不含label；P4 clean数值parity；
来源/状态/质量hash拒绝及场景/seed匹配；metrics/jitter/恢复/统计与旧路径一致。
先合成短训，再完整三seed。gate算子单次延迟另固定256key、16warmup、CUDA同步，
包含mmap候选读取/质量与gate/混合返回，不包含地图查询或P4候选模型forward；单独报
候选离线准备耗时和批量吞吐，不能称完整在线系统端到端延迟。

报告六条件七策略全部ADE/FDE/jitter、seed mean/std、update/horizon/maneuver、packet
覆盖与过旧/缺失硬回退、gate分布、dropout包恢复后的预测恢复。恢复定义与P5同：
相对同策略clean ADE差<=.25m连续两次，起点距第7次的秒数，未恢复2s右删失。
2000次paired scene cluster bootstrap，seed0；先固定三seed平均再按scene抽样。
报告learned相对CV/filtered/fixed/ungated_map，负面结果保留；val称历史固定回归。

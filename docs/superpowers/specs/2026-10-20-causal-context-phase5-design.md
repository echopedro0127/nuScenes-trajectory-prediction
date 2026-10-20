# P5 mini Radar 与位置退化设计

基线 c5eaffe，沿用已批准主设计第6–8节。现有worktree与feature/causal-context-online保持。
本阶段完整mini可行性实验，不下载数据、不推送，不把mini_val两场景称为泛化证明。

## 固定实验

mini_train570条/8场景内按现有scene_partition分成6/2场景训练/选模，mini_val47条/2场景
冻结后评估。固定同一mini GRU基础预测，t0+6秒终点、11次更新与既有指标。退化从t0后
开始，初始2秒历史及anchor保持，所有策略共享该初始信息，不能声称全程定位损坏。

五条件：clean；逐更新独立高斯噪声sigma=1m；位置交付延迟500000/1000000us；
固定丢弃第4/5/6次位置更新。扰动按instance/sample及固定seed2027生成，与训练seed
独立、与遍历顺序无关，不因模型不同重采样。所有策略保留相同agent/update/condition行。

延迟/丢点时只用已交付的位置。速度由最后两个不同source时刻的位置除真实时间差；
将最新位置按该速度外推到query时刻，作为所有模型、CV及Radar门控的共同因果状态。
状态包含最新source/availability、age和freshness诊断。不把重复held点算作新观测。
原Corrector复用8维residual语义，速度以该时间感知状态构造；干净时与P3数值一致。
简单滤波使用固定alpha=.6、beta=.2的alpha-beta递推，按实际交付source间隔更新，
再预测到query与原固定终点，不在mini_val调参数。

## Radar

仅读取sample/scene/sample_data/sensor/calibrated_sensor/ego_pose，关联不读取annotation。
每scene每Radar通道按原始source时间索引，query选择已可用且最多2秒旧的最近3帧。
source=availability（未模拟额外sensor延迟），固定5通道；相同source按token稳定排序。
pose必须与该帧对应且timestamp<=source，禁止未来pose插值。读取PCD时显式固定
invalid_states=[0]、dynprop_states=0..6、ambig_states=[3]，另保留quality_valid及pdh0。

位置sensor→ego→global→固定anchor，速度vx_comp/vy_comp仅由3D旋转及anchor yaw旋转。
门控半径4m，以共同退化状态位置为圆心；无annotation bbox、未来位置或clean中心。
不对历史点做目标运动补偿，保留age；ego-motion对齐仍可能拖影/混入邻近物体。

13维统计：log1p(count)/log(129)、相对XY中位数/4m、补偿VX/VY中位数/10mps、
VX/VY MAD/10mps、RCS中位数/20、quality_valid比例、age min/median/max除2秒、
有效通道数/5。无点时features全0、present=False；count/source token/frame age独立记录。
官方依据：https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py

## 训练、缓存与报告

三个learned模型original、radar、matched_radar，加frozen/sliding/filtered共六策略。
每seed在五条件等权训练与选模，seed2027/2028/2029，hidden64、batch1024、最多20epoch、
patience5，AdamW .001/.0001，SmoothL1+.05smoothness。先训练original再冻结fallback；
radar零初始化condition head，完全无回波时严格回退；matched匹配total参数且全部可训练。
保留继承预训练与trainable参数/计算不匹配限制。原Corrector指同架构P5重新训练对照。

独立P5 NPZ+manifest保存因果状态、Radar统计、mask、frame tokens与时间；不修改原缓存。
来源绑定metadata与实际已读PCD文件hash、配置、轨迹/预测身份及key；加载语义验证，
checkpoint绑定split、场景、数据/模型源码hash。label仅由loss/evaluator取得。

报告每条件/策略三seed均值/std、ADE/FDE/jitter、update/horizon/maneuver、coverage/fallback。
mini_val仅两场景，配对scene bootstrap2000次seed0仅作描述，不据CI宣称泛化。
丢点恢复：第7次交付恢复后，预测ADE相比同策略clean条件差值<=.25m且连续两次，
第一次满足的起点距第7次的秒数；没恢复者右删失，单报数量，不能当0秒。

计时分缓存路径与真实Radar路径；固定256个condition/sample/update key、16次warmup，
CUDA同步。真实路径清空解码缓存，包含PCD读取/解码、变换、聚合、特征/搬运及预测；
metadata索引启动不含在单次时间，操作系统文件缓存可能已热，不称物理冷盘性能。

工程验收包括future/clean中心扰动不变性、延迟与丢点真实dt、恢复删失、三维速度旋转、
显式过滤/缺帧/空回波、来源hash与场景重叠拒绝、fallback一致及全策略key一致。

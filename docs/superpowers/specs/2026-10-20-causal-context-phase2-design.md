# P2 Map-only 细化设计

日期：2026-10-20。继承 causal-context-online-design.md 的实验合同，P1 基线 8fa93b8。
本阶段实现地图数据准备、独立缓存、训练及生产 evaluator 通路，仅运行 mini 可行性验证。

## 决策与边界

使用离散 lane/lane_connector 折线，默认半径 50 m、最多 8 条、每条最多 20 点、2 m
离散间隔。配置先固定用于 mini 通路，不用 mini_val 调参；正式参数比较仅在 train_val。
相较 raster CNN，这条路径无需新依赖、便于核验方向；相较只取最近车道，保留交叉路口
多候选。按当前可见位置与离散车道点的最小距离、token 排序，保留距当前最近点附近的
连续折线片段。坐标使用原 agent anchor（前向 +Y），方向只旋转、不平移。每个点为
[x,y,dir_x,dir_y]，另有 bool mask、lane token、coverage 和有效点数量。

元数据只读取 sample、scene、log、sample_annotation；核验 instance/sample、场景、
anchor translation/yaw，使用 anchor 的 timestamp，随后严格 +k*500000。地图是静态
上下文，来源时间/可用时间为 0，frame age 不适用，不能伪造为动态观测。位置来自因果
annotation replay，结果明确标注 oracle target positions；候选选择绝不读取剩余标签。

上下文 NPZ 独立于原 trajectory cache。数组为 N×11×K×P×4；cache contract 包含配置、
split、原 trajectory manifest/archive 哈希、metadata/map 文件哈希、时间/坐标/oracle
策略、完整有序 instance/sample/scene key。content-addressed NPZ 校验 hash；加载时
核验 contract、完整 key 顺序、时间戳和形状。所有行保留，无地图行不删除。

旧 prediction NPZ 没有 token：强制绑定同目录 per_sample.csv 的完整有序 key，并核验
NPZ target 与 trajectory future 一致；metrics.json 必须记录固定 GRU 的 checkpoint、
model config 与 data config 身份，checkpoint 记录三份文件哈希并核验跨 split 模型一致。未来新格式可携带
内嵌 key，本阶段不改旧 writer。跨 split 评估只比较预处理配置和来源哈希，不能要求评估
缓存 hash 等于训练缓存；checkpoint 分别记录训练/选择上下文 hash 与场景清单哈希。

模型：原 8 维 residual 特征严格由 ObservationView 与固定 base 生成。共享点 MLP +
masked mean 编码地图，与 residual 拼接后预测附加 offset；无有效地图时附加 offset
严格为 0，最终完全等于原 ResidualCorrector。past horizon mask 保持为 0。

mini_train 的 8 个场景按名称排序，最后 2 个作 selection，前 6 个 train；mini_val
两个场景用于冻结后评估。发现旧 mini Corrector 在整个 mini_train 上训练/选择，不能
将其用于该内部 selection 的干净对照；故本阶段先在同一 train 子集重新训练原架构
Corrector 并用 selection 选模，然后冻结它，训练地图附加 head。固定 GRU 是已存在的
baseline（含其历史训练来源），不重新训练。报告明确该限制，不宣称盲测或性能提升。
trainval 通路使用独立 train/train_val cache，严格拒绝场景重叠。任何 evaluation 场景
不得出现在 checkpoint train/selection 场景中。

Evaluator 对每行每次 update 运行 frozen/sliding/corrected/map_corrected，predictor
仅接收 ObservationView、固定 base 和 map view；label 只交 metrics。记录每种策略
ADE/FDE/jitter、按 update/剩余 horizon/maneuver 分组、coverage、fallback 计数；计时
分 residual+map 准备、head、缓存路径完整 update，GPU 同步。准备阶段另报 metadata/
地图读取与编码耗时。缓存路径 latency 不等于在线地图读取端到端 latency。

## 验收

合成测试覆盖坐标/方向、候选排序、mask/全缺失 fallback、未来扰动、缓存/配置/split/
来源/行序拒绝、场景隔离、checkpoint 溯源、真实模型边界与四策略同行集合。
全量 pytest、ruff、公开仓库扫描和 diff check。真实 mini 跑 prepare/train/evaluate，
保存场景、seed、hash、命令、覆盖率和失败；输出全部位于 outputs/context/p2，数据、
模型和指标不提交。P2 工程完成不代表多 seed/trainval 效果验收，后者属于 P4。

地图 API 核验来源：[nuScenes 官方 map expansion 教程](https://www.nuscenes.org/tutorials/map_expansion_tutorial.html)
与 [devkit map_api.py](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/map_expansion/map_api.py)，运行使用本机固定 1.2.0。

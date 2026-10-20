# 因果在线轨迹预测：上下文扩展设计

日期：2026-10-20。根据已讨论的地图/邻车优先、Radar 次之的方案制定。
范围：阶段化工程设计；本轮交付为开发分支和文档，未产生新增实验结果。

## 1. 目标与现状

将现有每 0.5 s 释放位置观测的 replay 扩展为可接入地图、邻车历史和异步 sensor 的
在线修正系统。基础 GRU 及既有 checkpoint 固定；逐步扩展 Corrector 条件输入。

现有 `online/replay.py` 的 ReplayUpdate 同时携带观测与 remaining_target；
`online/dataset.py` 从已揭示的位置构造 8 维 residual 特征；`online/evaluator.py`
比较 frozen/sliding/corrected。新增模型入口需要显式排除 remaining_target。

现有 cache.py 严格校验旧 trajectory NPZ 字段，因此上下文使用独立 cache 和 manifest。
继续保留旧 replay、Corrector 和 CLI，作为兼容对照与回归基线。

## 2. 实验合同

时间单位统一为整数微秒。anchor 为 t0，更新 tk=t0+k*500000，k 为 1 至 11。
所有策略始终预测原始 t0+6 s 终点之前的剩余轨迹。当前最多 2 秒历史，初版 sensor
聚合窗口也最多 2 秒。每次重启完整 6 秒 horizon 属于另一个实验，不在本合同内。

每个动态输入记录 source_timestamp_us 和 available_timestamp_us，满足
source_timestamp_us <= available_timestamp_us <= tk。初版延迟非负，可由后续实验
显式模拟。不同模态帧不要求严格同步；按当前可用时间选取并记录 frame age。

固定 anchor frame 用于轨迹、地图、邻车和 sensor 对齐。sensor -> ego -> global ->
anchor 的变换使用该观测已经可用的标定和 pose；不读取未来 pose 进行插值。速度向量
只旋转。邻车速度仅用过去差分，候选车道仅按静态几何和当前/历史位置选择。

## 3. 数据与模型接口

```text
metadata / sensor records
  -> timestamp index -> availability filter -> causal context encoders
historical target observations ----------------------+
fixed base forecast --------------------------------+-> Corrector -> forecast
evaluation-only remaining target -------------------------------> metrics
```

P1 定义以下稳定边界：

- `FrameRef`：token、channel、source timestamp、availability timestamp；只携带引用，
  不主动加载未来文件。
- `latest_available`：在同通道、已到达且未过旧的帧中按 source timestamp 选择最新帧。
  相同时间按 token 稳定打破平局；缺失返回 None。
- `ObservationView`：update step、当前绝对时间、只读历史位置、当前位置和 FrameRef
  元组。无 future/target 字段。
- `EvaluationLabel`：只读 remaining_target，仅由训练 loss 和 evaluator 持有。
- `replay_observations`：适配旧 replay，产出 `(ObservationView, EvaluationLabel)`。

P1 先建立并测试独立接口；P2 接入新 Corrector 时让 predictor 只接受 ObservationView
及由它派生的 context。该约束用于减少误接线，不代表在同一 Python 进程中建立安全
隔离。测试必须验证实际模型调用参数不含 label。

P2 后每项 context 包含 features、validity mask、来源时间、frame age 与质量统计。
缺失值用 mask 明确表示，不把“无 sensor 输入”伪装成“真实值为零”。

## 4. 模块文件职责

以下新增路径是规划；阶段开始时为对应子项目补充独立细化计划。

| 文件/目录 | 所属阶段 | 职责 |
| --- | --- | --- |
| context/schema.py、context/timeline.py | P1 | FrameRef 与可用时间筛选 |
| online/observations.py | P1 | 观测/标签接口与旧 replay adapter |
| context/nuscenes_index.py | P2 | token、场景、sample/annotation 时间查询 |
| context/maps.py、context/cache.py | P2 | 地图特征与独立缓存，避免改旧 NPZ schema |
| online/context_features.py | P2 | 只从观测构造 residual/context 输入 |
| online/context_dataset.py、context_corrector.py | P2 | 对齐训练样本、轻量编码与 offset 预测 |
| online/context_evaluator.py | P2 | 新模型调用边界、误差/覆盖率/分段计时 |
| scripts/prepare_context.py | P2 | 可指定数据根的独立 context cache CLI |
| scripts/train_context_corrector.py | P2 | split 隔离与 checkpoint selection |
| scripts/evaluate_context_online.py | P2 | 统一策略与相同更新集合的在线评估 |
| context/neighbors.py | P3 | 邻车因果历史与共享编码输入 |
| configs/context/、configs/models/context_corrector.yaml | P2–P3 | 版本化上下文及模型配置 |
| context/radar.py、online/observation_noise.py | P5 | 雷达关联/统计与可重复观测退化 |
| online/quality_gate.py | P6 | 基于质量/帧龄的修正强度 |
| tests/context/、tests/online/ | 各阶段 | 时间边界、几何、缺失及集成测试 |

训练、索引、encoder 与可视化分开，禁止在单个 evaluator 中加入全部模态处理逻辑。

## 5. Map 与 Neighbors 主线

Map 首版使用局部 lane polylines 的位置与方向；配套 map coverage/mask。候选数量、
半径、离散间隔在 train_val 上开发确定，不用真实未来轨迹选 lane。可行驶区域用于
单独诊断，适用 agent 类别与栅格配置写入结果。

Neighbors 使用当前范围内最多 8 个交通参与者的 2 秒历史相对位置和历史差分速度；
先按当前距离、再按 token 排序。排除目标自身，历史不足与消失用 mask；类别映射
只由固定配置决定。首版用共享 MLP + masked pooling，必要时才加入 attention。
邻车来自 annotation，属于 oracle interaction context。

运行 Map-only、Neighbors-only、Map+Neighbors 与原 Corrector；加入参数量近似匹配
的原 Corrector 对照。完全缺失 context 时回退到原 Corrector；所有策略保留相同
agent/update 行，不因模态缺失删除样本。

## 6. Radar 与观测退化

Radar 首版读取过去可用帧，提取位置、补偿速度、RCS 和质量；统一 frame 后在目标
因果位置附近做门控，聚合点数、稳健速度统计及离散度。速度旋转单独验证。

位置门控不依赖未来标注；若另做标注框关联，只能作为显式 oracle 对照。多 sweep
ego-motion 对齐不等于目标运动补偿，须保留时间差并记录拖影限制。

分别在干净位置、固定噪声、0.5/1.0 s 延迟、预先固定丢点扰动下比较原 Corrector、
Radar Corrector 与简单滤波基线。模型特征、目标关联和地图查询都应使用同一退化
观测状态，避免一边损坏位置、一边从 clean 标注位置悄悄恢复信息。

## 7. 训练与复现

train 训练、train_val 选择 checkpoint。val 已发布历史结果，后续称固定验证集回归
对比；不称全新盲测。mini 在 mini_train 内做场景级开发划分，mini_val 用于冻结后
可行性检查，记录实际场景清单；不以随机 agent split 替代场景隔离。

context cache key 包含 instance/sample、update step、模态及其配置哈希。
manifest 记录源数据版本、时间策略、来源文件/metadata 哈希、oracle 标记与 split。
checkpoint 记录基础预测文件哈希、context hash、训练/选择场景集合哈希。
加载时拒绝配置、split、token 顺序或上下文来源不一致，不能只按数组行号拼接。

输出写入 `outputs/context/<phase>/<run-id>`；缓存和 data/output 均不提交。
固定 GRU baseline，所有策略在匹配的样本/update 集合上重跑，不能把旧完整数据指标
直接拿来与新的可见 sensor 子集比较。

## 8. 验收与结果报告

工程验收：时间筛选、未来扰动不变性、坐标/速度变换、mask、fallback、split/哈希
拒绝、相同评估行集合、旧基线回归均通过。

效果验收：ADE/FDE、jitter、按 update/horizon/maneuver 分组、输入覆盖与 fallback
计数、多 seed；置信区间以场景为单位。无收益和退化结果必须记录。

延迟分别记录特征提取、encoder/head 和完整 update 处理；完整处理包括数据读取、
解码、变换、搬运和预测，另报模拟输入延迟及 frame age。预缓存 embedding 只能
衡量缓存路径，不代表 sensor encoder 端到端耗时。原 Corrector 2.430 ms p95 只作为
原路径历史记录。

## 9. 推进范围

本轮只完成 P0 文档与 Git 隔离。下一开发批次实施 P1，不加载真实 sensor、不训练。
P1 验收后依次执行 Map、Neighbors、trainval 消融和 mini Radar。Camera/LiDAR 在
主线效果明晰后再选一种；当前无需追加下载或改写原有 README 结果。

详细阶段清单见 [开发路线图](../../DEVELOPMENT_ROADMAP.md)，P1 任务见
[P1 实施计划](../plans/2026-10-20-causal-context-phase1.md)。

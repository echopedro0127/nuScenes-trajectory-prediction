# P4：完整 trainval 三种子消融

状态：正式训练进行中，尚未声明 P4 完成。训练代码锁定于 `a975c14`；下述协议在查看完整 val 结果前确定。

## 数据与协议

| split | 轨迹 | 场景 | 每策略更新 |
| --- | ---: | ---: | ---: |
| train | 25,560 | 468 | 281,160 |
| train_val | 6,621 | 176 | 72,831 |
| val | 7,201 | 136 | 79,211 |

三组场景两两不相交。固定 GRU、原始轨迹缓存和基础预测均复用已有产物；加载绑定
NPZ/CSV/metrics、token 顺序、target、模型配置及预处理。val 是已有历史固定验证集回归，
不称新盲测。target 与 neighbor 位置来自 annotation oracle，未读取或下载 sensor。

流式解析 annotation 建 SQLite 场景索引，逐场景 query；NumPy NPY 只读映射配合原格式
NPZ。加载分块验证全部语义，并流式核对 NPZ 成员与独立 NPY 完全相同。mini 的全部
缓存数组与 P2/P3 精确一致；向量化邻车 query 测试覆盖旋转、缺失与延迟。

固定 seeds=2027/2028/2029、hidden=64、batch=4096、最多20 epoch、patience=5；
AdamW lr=0.001、weight_decay=0.0001，SmoothL1 + 0.05 smoothness。
只用 train_val ADE 选 checkpoint；每个 seed 训练 original、map、neighbors、fusion
及三种 matched 宽度对照，评估再加 frozen/sliding，共九策略。

上下文模型继承并冻结同 seed 的 original fallback；宽度对照匹配 total 参数、全部参数
可训练。因此不是 trainable 参数或总训练计算量匹配，比较不能单独证明上下文的因果价值。
训练输入与标签由不同方法取得，预测函数只接收可见输入。缓存/配置/实际数组内容及模型
源码共同绑定 stage 复用身份；已完成 stage 可复用，未完成 stage 从头运行。
复用要求相同源码与训练记录中的 git revision，变更 revision 时使用新输出目录。

指标沿用 P3：只评估每次更新后的剩余轨迹，终点固定为 t0+6s；首步 jitter=0，后续只
比较相邻预测重叠的绝对未来时刻。批量九策略数值与 P3 逐条真实预测路径测试一致。

## 统计与计时

每策略报告三 seed 均值与样本标准差（ddof=1）。先对固定三 seed 的场景 sum 取平均，
以场景为配对 cluster 做2000次 bootstrap（统计随机种子0），用抽中场景的 sum/count
计算样本加权均值，报告策略自身及相对 original、对应 matched 对照的95%区间。
区间只反映固定 seed 集合下的场景不确定性；训练 seed 波动单独列示，不把相邻 update
或 seed×scene 当作独立样本。

报告检查实际压缩逐行 CSV、scene sum/count、summary、分组文件的哈希与一致性，
三 seed 九策略必须有完全相同 key 顺序；frozen/sliding 跨 seed 指标相同，漏行不丢弃。
输出按 update、remaining horizon、maneuver 分组；maneuver 仅为标签侧事后诊断。

批量 forward 耗时仅表示吞吐。单次延迟另固定随机抽256个 sample/update key（三 seed
同一集合，抽样种子2027），每策略16次 warmup 后走 P3 实际 predictor，CUDA 前后同步。
总时间包含预缓存上下文取值、view 构造、特征/传输、head 和输出，单列 feature、head
及总计 p50/p95。它不包含 metadata/sensor 冷读，也不表示传感器端到端延迟。

## 复现

在开发 worktree 执行；所有路径参数支持绝对路径，以下共享路径为本仓库布局。

```powershell
$python = (Resolve-Path ../trajectory-mvp/.venv/Scripts/python.exe).Path
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& $python scripts/prepare_scaled_context.py --trajectory-root ../../data/cache/trainval --dataroot ../../data/nuscenes --version v1.0-trainval --output-root outputs/context/p4/trainval --splits train train_val val
& $python scripts/train_scaled_interactions.py --trajectory-root ../../data/cache/trainval --context-root outputs/context/p4/trainval --prediction-root ../../outputs/trainval --dataroot ../../data/nuscenes --output-root outputs/context/p4/training --device cuda
& $python scripts/evaluate_scaled_interactions.py --trajectory-root ../../data/cache/trainval --context-root outputs/context/p4/trainval --prediction-root ../../outputs/trainval --dataroot ../../data/nuscenes --training-root outputs/context/p4/training --output-root outputs/context/p4/evaluation --device cuda
& $python scripts/report_context_ablation.py --runs outputs/context/p4/evaluation/seed-2027 outputs/context/p4/evaluation/seed-2028 outputs/context/p4/evaluation/seed-2029 --output outputs/context/p4/report
```

`--mini --epochs 1 --device cpu` 的三 seed 训练，加 mini_val 完整评估及报告已通过 smoke；
正式实验使用完整 train/train_val/val 与上述固定训练设置。原始数据、NPY/NPZ、模型和
逐行结果全部位于被忽略的 data/outputs，不进入代码提交，不推送远程。

## 实际结果

待正式三 seed 训练、完整 val 评估、延迟测量和最终验收完成后填写。

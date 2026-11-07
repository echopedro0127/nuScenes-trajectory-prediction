# P6.6C/2 新 GRU 预测与因果上下文准备

日期：2026-11-07。接续[独立划分与上游重训](CAUSAL_CONTEXT_INDEPENDENT_UPSTREAM_RESULTS.md)。

本阶段只准备 independent_fit / independent_select 的新预测和确定性上下文，不训练或选择新修正器，不生成 holdout 预测。
沿用新划分、seed2027 的新 GRU best.pt、原地图/邻车配置及既有准备实现；不复用旧学习模型的预测或权重。

## 已完成的新预测

| 角色 | 轨迹数 | 静态 ADE / m | 静态 FDE / m |
| --- | ---: | ---: | ---: |
| independent_fit | 14,445 | 3.832407971 | 9.150227848 |
| independent_select | 7,786 | 4.208823220 | 10.029098137 |

这些是静态 GRU 指标，不是在线修正收益或独立测试成果。导出器用 double 计算逐样本距离，训练验证用 float32 分批聚合；因此 select 与前阶段日志有约 1.7e-7 m 的 ADE 差异。

实际复用 scripts/evaluate.py，history_seconds=2，history_steps=5。两个角色均传入新 data.yaml、绝对 model-config/checkpoint/output-dir。
load_predictions 已验证样本顺序、target 与轨迹缓存逐项一致、形状/有限性、角色及配置身份；另核验 checkpoint 文件哈希。

- checkpoint SHA256：0804c98a4b3db907ab4b4b7f8debe744d47c7c9e599fd4e2806b97464cb94757。
- data_config SHA256：d81d38f229bf5d9679db773d1b4fdfe3ed1b32721ed08776a09fe3bca38539ce。
- fit predictions.npz：a35e75852037d7e0a6494b7ea08fa3a43880e86dea9a4671890b390ac8d93b32。
- select predictions.npz：af2cf11ab1caae2958b8cdeac30c92191f7a8122d4530e80243d938865ad7985。

产物：outputs/context/p66-independent/predictions/{independent_fit,independent_select}/gru_eval，含 predictions.npz、per_sample.csv、metrics.json、official_predictions.json。
predictions/verification.json 记录核验身份及 holdout_evaluated=false。

## 上下文准备协议

```powershell
python scripts/prepare_scaled_context.py --trajectory-root outputs/context/p66-independent/cache --dataroot ../../data/nuscenes --output-root outputs/context/p66-independent/context --version v1.0-trainval --splits independent_fit independent_select
```

地图：50 m 半径、最多 8 条车道、每条 20 点、2 m 离散间距。邻车：50 m 半径、最多 8 个邻居、5 步历史、最大陈旧 0.5 s、延迟 0。
每条轨迹生成 11 次更新，时间为原锚点后 0.5 至 5.5 s；查询仅使用当次已释放位置和当时可用的邻车注释。
仍属于标注位置因果回放，未使用 Camera / LiDAR / Radar 感知特征。

准备器按场景查询元数据，生成只读 NPY sidecar 与绑定哈希的 NPZ；发布前校验键、schema、更新时间、邻车自身排除及可用时间。
上下文合同绑定新角色、新轨迹缓存、新配置、地图与元数据文件。原始元数据索引可含全部场景，但训练角色由轨迹缓存决定；读取索引不等同于用 holdout 标签拟合或评价。

## 当前状态及验收

新预测已完成并核验。fit/select 上下文生成及发布前校验完成，分别耗时 466.013 / 174.024 秒（不含元数据索引构建）；两个角色均已通过 load_scaled_bundle 独立重载验收。
相关现有测试 **53 passed in 2.63s**（evaluation、scaled_context、independent_split）。

| 角色 | 更新数 | 地图 NPZ SHA256 | 邻车 NPZ SHA256 |
| --- | ---: | --- | --- |
| independent_fit | 158,895 | 17fc678ee7094fe77fc919b413a4ef68ef6764c4e4b77ac39f6cf1d192097a5c | b7357d62bc6ab8540f86d6737cd10fb07d4dd1d61bba0aebe85f01634a3bc553 |
| independent_select | 85,646 | 8e172ea43496410e627ceeca5bf39e999ed53e5ce9ba1f34263dda4941e8ed01 | ead06cffcbf00fe3b441a18bf6ffd6aacad7752e2040df0798e13c2d36eab1e0 |

重载检查归档及 sidecar 哈希、数组 schema、逐更新因果约束、预测配置/checkpoint 身份；另逐项比较轨迹/地图/邻车标识顺序并确认上下文数组不可写。
fit 地图覆盖 158,811/158,895 次更新，邻车覆盖 145,853/158,895；select 分别为 85,646/85,646 和 79,351/85,646。覆盖表示至少一个有效地图点/邻车观测，不表示预测精度。
核验结果保存在 context/verification.json，SHA256 为 89ae6c5c46662a8afbb84c700bd673991e19c061aba4483b48118a20cd112444。
holdout 的 context/predictions 子目录均不存在；未执行 holdout 预测与误差评价。
本阶段没有修改模型或数据准备源代码。输出不入 Git，不推送 GitHub，不改简历。完整独立训练链仍未完成。

下一步冻结地图修正器训练预算，并为自定义 fit/select 角色增加独立训练入口（旧 CLI 固定为 train/train_val，不直接套用）。后续 quality gate、RH、一致性 head 均需沿新 fit/select 链重训；配置冻结后才可开展 holdout 一次性评价。

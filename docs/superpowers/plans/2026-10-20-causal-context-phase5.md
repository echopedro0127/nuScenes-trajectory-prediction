# P5 Radar and degraded observations Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline with task-level checkpoints.

**Goal:** 完成mini因果Radar、同位置退化对照、三seed可复现实验及真实输入路径计时。
**Architecture:** 可重复measurement stream→CausalState→Radar统计/原residual→三模型六策略。
独立cache绑定原始PCD及metadata；预测函数不接收label；旧P1–P4接口保持。
**Tech Stack:** Python3.12、NumPy、nuScenes1.2.0、PyQuaternion、PyTorch、pandas、pytest。

设计见 ../specs/2026-10-20-causal-context-phase5-design.md。
执行环境：../trajectory-mvp/.venv/Scripts/python.exe，PYTHONPATH=当前src；pytest basetemp
为outputs/pytest-GUID。所有写入在已指定开发worktree，沿用本地提交，不下载、不推送。

## Task 1：可重复位置退化与简单滤波

文件src/trajectory_prediction/online/observation_noise.py；tests/online/test_observation_noise.py。
- [ ] 写失败测试：noise同key/前缀一致；1s延迟首两步只用initial history；drop4/5/6
  不会把重复点当零速度；source时间差1.5s差分正确；未来位置改变不影响早期状态。
  示例：history=[[-2,0],[-1.5,0],[-1,0],[-.5,0],[0,0]]，future=[k*.5,0]，
  delay1s在query=.5时position=[.5,0]、velocity=[1,0]、source=0。
- [ ] 聚焦pytest确认失败，实施MeasurementConfig/CausalState/replay_states与alpha-beta
  状态递推；residual由position与position-.5*velocity调用现有residual_batch，不传label。
- [ ] 回归clean与P3 residual_features、无观测/非法输入和alpha-beta匀速精确性；提交。

## Task 2：因果Radar索引、变换与统计

文件context/radar.py、context/radar_index.py；tests/context/test_radar.py。
- [ ] 写失败测试：带平移的两级旋转只改变位置，不改变速度平移项；未来pose拒绝；
  同scene五通道过去3sweep窗口；future文件替换不影响query；空点mask与半径4m边界。
  示例：sensor yaw90deg、translation=[10,0,0]，point=[1,0,0]变[10,1,0]，
  velocity=[2,0,0]变[0,2,0]，anchor_yaw=pi/2时保持global XY方向。
- [ ] 聚焦pytest确认失败；实现RadarConfig、RadarView、transform_returns、aggregate_returns，
  RadarIndex查询只接受CausalState、scene、anchor_xy/yaw；不接收annotations/labels。
- [ ] 固定PCD过滤参数，记录frame/point source/availability与阶段耗时；文件缺失报错，
  时间窗口无帧是mask缺失；测试点投影及translation不变性后提交。

## Task 3：独立缓存与准备入口

文件context/radar_preparation.py；scripts/prepare_radar_context.py；configs/context/radar.yaml。
测试tests/context/test_radar_preparation.py。
- [ ] 写失败测试：prepare/load roundtrip；错key/config/base/hash/PCD/metadata拒绝；
  Radar查询中心与保存的退化CausalState一致；labels不进入query签名。
- [ ] 生成NPZ的condition×sample×11状态/统计/frames，manifest记录所有来源与schema。
  CLI：--trajectory-root --prediction-root --dataroot --output-root，处理mini_train/mini_val。
- [ ] 完整mini实际prepare，核对570/47轨迹、5条件、11更新、无丢行；提交。

## Task 4：模型、训练与checkpoint

文件online/radar_corrector.py、online/radar_training.py；scripts/train_radar_corrector.py；
configs/models/radar_corrector.yaml；tests/online/test_radar_corrector.py、test_radar_training.py。
- [ ] 失败测试：空Radar精确fallback；模型调用仅features/mask/context/present；
  train/selection/eval场景重叠与数据身份拒绝；三模型共享同状态数据；best epoch回载。
- [ ] 实现原Corrector→冻结fallback的Radar head→matched宽度对照，五条件均匀训练；
  checkpoint包含model/metadata、实际数据hash、训练源码hash、seed、完整场景与进度。
- [ ] 合成短训练通过后，完整mini_train6/2场景运行三seed20epoch/patience5；提交。

## Task 5：评估、恢复时间、真实路径计时

文件online/radar_evaluation.py、analysis/radar_report.py；scripts/evaluate_radar_online.py；
tests/online/test_radar_evaluation.py、tests/analysis/test_radar_report.py。
- [ ] 失败测试：六策略同condition/agent/update；clean数值parity；未来label不入predictor；
  首步jitter0；drop恢复连续两步阈值与右删失；两场景cluster配对、三seed key/source检查。
- [ ] 评估使用独立label构造forecast_metrics；输出压缩逐行CSV、condition/update/horizon/
  maneuver、coverage/fallback、recovery、scene CI与三seed std。不删除无回波样本。
- [ ] 固定256key+16warmup，真实RadarIndex无解码缓存计时含PCD读/解码/变换/特征/
  head与返回，CUDA同步；另测缓存路径，不混称sensor端到端。
- [ ] 完整mini_val三seed六策略五条件运行并汇总，保留无收益及低覆盖结果；提交。

## Task 6：验收交付

- [ ] 请求独立代码审查；修复实证问题并聚焦回归；全量pytest、Ruff、公开扫描、diff检查。
- [ ] docs/CAUSAL_CONTEXT_PHASE5_RESULTS.md写真实指标、恢复删失、覆盖、延迟及来源。
- [ ] 仅实际实验完成后将路线图P5标完成；P6按收益条件评估；本地提交且工作区干净。

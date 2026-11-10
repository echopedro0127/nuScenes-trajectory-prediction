# Noise-Aware Phase 2A Implementation Plan

**Goal:** 重建共享带噪观测上的七候选，执行三种可靠性特征消融及常数门控对照。
**Architecture:** packets→候选/诊断特征→fit-only训练→select评价。新文件隔离于Phase 1源码；模型输入与评价标签分离。
**Tech Stack:** NumPy、PyTorch、nuScenes静态地图、pytest。

在现有隔离worktree内按[设计](../specs/2026-11-10-noise-aware-phase2a-design.md)顺序执行，不另行启动子代理。

- [ ] Task 1：先写tests/robustness/test_noisy_candidates.py，运行确认缺少noisy_candidates模块的RED。用真正LaneMap和小型确定模型覆盖未来包变更不影响已发布前缀、带噪历史重新推理、map查询随测量改变、共同平移等变、空图回退及拒绝延迟。
- [ ] Task 2：实现robustness/noisy_candidates.py的build_candidates(packets, gru, original, map_model, lane_map, anchor_xy, anchor_yaw, timestamp_us)，返回candidate[11,7,12,2]及features[11,18]。只读已到包，所有torch前向inference_mode，输入不含GT。运行Task 1测试为GREEN。
- [ ] Task 3：先写tests/robustness/test_reliability_learning.py，RED后实现reliability_learning.py。测试feature mask/相同名义参数量/初始化、fit-only标准化、凸融合范围、真正拟合可分专家与缺失目标拒绝。固定20epoch，fit才可反传。
- [ ] Task 4：实现phase2_experiment.py及scripts/run_noise_aware_gate.py。来源绑定Phase 1采样与模型缓存，重建608×3×3条流。单候选及三门控保存逐轨迹指标、训练日志与权重；verify-only拒绝篡改并重算摘要/预测抽查。新增小型集成测试，先RED后GREEN。
- [ ] Task 5：全量pytest、Ruff、公开扫描通过，Git提交冻结实现/设计，再运行outputs/noise-aware/phase2a真实pilot。命令：PYTHONPATH=src，既有trajectory-mvp/.venv环境运行scripts/run_noise_aware_gate.py。
- [ ] Task 6：verify-only独立验证、核对报告数值，写docs/NOISE_AWARE_PHASE2A_RESULTS.md及更新路线。保留负面结果、单训练seed/旧冻结模型/准确航向假设等限制；本地提交，不推送。

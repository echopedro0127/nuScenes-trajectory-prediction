# P4 Trainval Ablation Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline with task-level review checkpoints.

**Goal:** 完成全样本trainval三seed七模型消融和场景级不确定性报告。
**Architecture:** bounded metadata SQLite + mmap sidecars；P3模型与来源合同复用，新增batch执行和统计报告。
**Tech Stack:** Python3.12、sqlite3/json、NumPy、PyTorch、pandas、pytest。

设计见 ../specs/2026-10-20-causal-context-phase4-design.md。
python沿用../trajectory-mvp/.venv/Scripts/python.exe，PYTHONPATH=当前src，pytest使用独占outputs/pytest-GUID。

## Task 1：流式metadata与mmap准备

文件context/streaming_index.py、context/mmap_cache.py、context/scaled_preparation.py、scripts/prepare_scaled_context.py。
测试tests/context/test_scaled_context.py。
- [x] 先测试JSON跨chunk/非法输入、SQLite source绑定/scene隔离、mmap roundtrip/错hash/shape/key拒绝。
- [x] 运行失败后实现流式array decoder、按scene annotation索引，复用P3NeighborIndex与map query。
- [x] memmap保存已有schema并生成NPZ/manifest、NPY sidecar；加载只读mmap并校验所有hash/contract。
- [x] CLI指定--trajectory-root --dataroot --version --output-root --splits（预测由训练/评估入口绑定），绝对路径可用。
- [x] mini完整prepare核对P3数组，再准备train/train_val/val，保存进度与失败计数，提交。

## Task 2：批量因果数据与指标

文件online/batched_context.py、online/batched_evaluation.py；tests/online/test_batched_context.py。
- [x] 每step对照residual_features与P3真实predictor，验证未来扰动、标签边界、history/velocity/age一致。
- [x] 批量计算ADE/FDE/jitter，逐行与P3九策略核对；首步jitter=0，不延长horizon。
- [x] 实现只接受模型输入tensor的调用入口；label独立获取；输出逐seed压缩CSV与分组，提交。

## Task 3：有界内存多seed训练与checkpoint

文件online/scaled_training.py、scripts/train_scaled_interactions.py、configs/models/trainval_interaction.yaml。
测试tests/online/test_scaled_training.py。
- [x] 短合成/mini运行，验证场景重叠拒绝、source/base/key不匹配拒绝、checkpoint恢复与每stage记录。
- [x] 复用P3模型/parameter控制和metadata，批量切片训练，独立shuffle，20epoch/patience5上限。
- [x] 每stage写日志和原子checkpoint，支持已完成stage的严格身份校验续跑，不丢失长实验进度。
- [x] 完整train/train_val三seeds2027/2028/2029训练，不读取val选模；记录配置hash，提交。

## Task 4：完整评估、延迟与场景统计

文件analysis/context_ablation.py、scripts/evaluate_scaled_interactions.py、scripts/report_context_ablation.py；
测试tests/analysis/test_context_ablation.py。
- [x] 配对bootstrap测试常量delta、场景聚类、不匹配key/seed/漏行拒绝，seedstd口径明确。
- [x] 全val九策略批量评估；固定256key逐条warmup后分段计时，保存对应key和设备信息。
- [x] 完整三seed报告、分组、覆盖/fallback、95%场景CI、相对original与matched对照差值，提交。

## Task 5：验收与交付

- [x] 独立审查，修复后聚焦回归；全量pytest/Ruff/公开扫描/git diff --check。
- [x] docs记录完整样本数、场景、seeds、epoch、命令、hash、正负结果、CI及延迟范围。
- [x] P4满足实际运行条件后才标完成，保留工作分支，不提交data/outputs/checkpoint，不推送。

实际结果、复现命令及第二次独立复核的服务限制见 ../../CAUSAL_CONTEXT_PHASE4_RESULTS.md。

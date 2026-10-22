# P6.4 Innovation Consistency Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans inline with independent read-only review checkpoints.

**Goal:** 执行已批准条件一致性四组、三seed训练、冻结选择与条件val完整重放。
**Architecture:** 只读HorizonDataset→独立创新权重与pair loss→RH共享head→身份绑定训练→18策略评估→生产重放报告。
**Tech Stack:** 现有Python3.12/PyTorch/CUDA/NumPy/pandas/pytest；不新增依赖。

协议见[设计](../specs/2026-10-22-causal-context-phase64-design.md)。用户已批准上一轮第8节并要求继续本轮。

## Task 1：创新数据与损失

文件src/trajectory_prediction/online/innovation_consistency.py，tests/online/test_innovation_consistency.py。
- [x] 先写失败测试，断言k=1只用cv[:,0,1]，未到达positions/cv及全部labels扰动不改此前权重。
- [x] 测试：now−past距离[2,4]且权重[1,0]时损失=1而非2；无前序/零权重为0，两侧梯度存在且相反。
- [x] 实现InnovationDataset(HorizonDataset)：innovation[:,1:]=norm(position[:,1:]-cv[:,arange(10),arange(1,11)],axis=-1)，w=exp(-(innovation/.5)^2)，w[:,0]=0；train_mean只含[:,1:]。
- [x] 实现weighted_revision：mask&has_previous；每行剩余点mean L2×w，再有前序行mean；返回可导0并检查shape/device/finite/range。
- [x] 聚焦RED→GREEN、Ruff，本地提交。

## Task 2：四arm身份绑定训练

新增online/consistency_training.py、configs/models/innovation_consistency.yaml、scripts/train_innovation_consistency.py、tests/online/test_consistency_training.py。
- [x] 失败测试：完成复用/输入与权重变化拒绝、arm/mask/lambda/mean篡改拒绝、scene重叠、lambda0与P6.3 RH同初始化同顺序重现。
- [x] 以独立文件复用P6.3的严格完成标记/父契约/初始化与源码身份；固定variant=RH，新增arm及train_mean绑定，保持旧文件不变。
- [x] run_epoch用当前与previous_indices预测，零arm不取前序；正则三arm分别1、data.weights、train_mean，lambda=.02；监督部分逐句保持P6.3。
- [x] production main只读train/train_val，要求P6.3冻结选择accuracy=RH；四arm同P6初始化、seed/shuffle与默认预算。
- [x] 初始全量指标核验，零arm训练后完整metrics/weights及state指纹对比旧RH；独立只读审查通过再冻结源码跑12模型。

## Task 3：18策略评估与冻结选择

新增online/consistency_artifacts.py、online/consistency_evaluation.py、scripts/evaluate_innovation_consistency.py、tests/online/test_consistency_evaluation.py。
- [x] 失败测试完整SEEDS×CONDITIONS×18、ADE/FDE/p95约束、tie、旧RH/零arm获选不访问val、NPZ重算hash仍拒绝。
- [x] 父14策略直接调用冻结P6.3 context/evaluate，不重选结构；新四arm使用同RH measure_horizon，保存全部逐点权重和指标。
- [x] 独立严格artifact发布与重放；derive_tables覆盖18策略并保留场景/剩余时域/分位数/恢复等。
- [x] train_val按设计choose_consistency冻结selection；与旧P6.3全部14基线核对，随后才决定是否val。
- [x] 同keys计时：新arm走冻结RH streaming currentstate更新，旧六基线使用RefinementDataset；不包含训练创新计算。

## Task 4：完整重放与区间

新增analysis/consistency_report.py、scripts/report_innovation_consistency.py、tests/analysis/test_consistency_report.py。
- [x] 失败测试缺scene/seed/count拒绝，固定三seed平均后配对区间，选模与推进重算；完整零arm重现核对。
- [x] 从生产train重建创新/均值/指纹并核对12训练记录；train_val/条件val重算全部18策略、weights和分组。
- [x] 12对比×7范围×3指标=252区间/split，2000次scene bootstrap；报告seed std、损失/权重分布与负面结果。
- [x] 验证真实单步状态更新、raw latency、key/源/模型哈希及旧14策略完整基线一致。

## Task 5：运行、验收与收尾

- [x] 全量pytest/Ruff/公开扫描/diff及独立只读审查，旧绑定源码diff为空。
- [x] 12模型后运行完整train_val→冻结选择→条件val→生产重放；失败不放宽校验，不据val重选。
- [x] 写docs/CAUSAL_CONTEXT_PHASE64_RESULTS.md，更新roadmap/spec/本计划；本地提交，保留worktree、不推送。

命令统一现有venv：../trajectory-mvp/.venv/Scripts/python.exe；PYTHONPATH=src；pytest独立outputs/pytest-GUID。
训练参数：--cache-root outputs/context/p6/cache --p6-training-root outputs/context/p6/training --horizon-training-root outputs/context/p63/training-deterministic --horizon-selection-root outputs/context/p63/selection --output-root outputs/context/p64/training --device cuda。
评估/报告额外指定--training-root outputs/context/p64/training、--reference-training-root outputs/context/p61/training、--reference-selection-root outputs/context/p61/selection、--horizon-evaluation-root outputs/context/p63/evaluation、--selection-root outputs/context/p64/selection、--output-root outputs/context/p64/evaluation。

## 实际验收

12模型/120epoch训练与完整train_val生产重放完成，23,597,244条度量及权重、252条区间核验。
冻结revision_fixed/advance=false；jitter下降0.453%未达0.5%门槛，条件val按协议不执行。
493项pytest、Ruff、公开扫描及独立只读审查通过。保留RH；负面结果和下一项建议见[实验报告](../../CAUSAL_CONTEXT_PHASE64_RESULTS.md)。

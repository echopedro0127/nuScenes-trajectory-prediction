# 因果在线轨迹预测：分阶段开发路线图

更新日期：2026-10-20。当前状态：P5 mini Radar 五条件三 seed 实验已完成；相对小型Corrector有平均改善，但强CV/滤波更优，P6仍按证据条件推进。

## 开发入口

- 基线：公开 `master` 的 `54fb25b`。
- 开发分支：`feature/causal-context-online`。
- 工作目录：仓库根目录下 `.worktrees/causal-context-online`。
- 根目录仍使用 `local/resume-bullets`；简历文档和之前的本地实验提案保留原处。
- P0–P5 的文档、代码与验收记录保留在该开发分支；P2/P3/P5使用mini，P4使用完整trainval，不推送远程。

进入开发目录：

```powershell
Set-Location .worktrees/causal-context-online
git branch --show-current
```

预期分支名为 `feature/causal-context-online`。不要在根目录直接切换到这个已被 worktree
检出的分支。

设计见 [技术设计](superpowers/specs/2026-10-20-causal-context-online-design.md)；
已执行 [P1 实施计划](superpowers/plans/2026-10-20-causal-context-phase1.md) 与
[P2 实施计划](superpowers/plans/2026-10-20-causal-context-phase2.md) 与
[P3 实施计划](superpowers/plans/2026-10-20-causal-context-phase3.md)。
P4 见 [实施计划](superpowers/plans/2026-10-20-causal-context-phase4.md) 与
[实验协议和结果](CAUSAL_CONTEXT_PHASE4_RESULTS.md)。
P5见 [实施计划](superpowers/plans/2026-10-20-causal-context-phase5.md) 与
[完整可行性报告](CAUSAL_CONTEXT_PHASE5_RESULTS.md)。

## 阶段及验收

| 阶段 | 工作范围 | 验收依据 | 数据 | 状态 |
| --- | --- | --- | --- | --- |
| P0 | 分支、worktree、设计、计划、基线验证 | 开发文档可执行，原有测试通过 | 不需要新数据 | 已完成 |
| P1 | 时间戳输入选择、观测/标签边界 | 未来帧扰动不影响当前输入，旧 replay 回归通过 | 合成数据 | 已完成 |
| P2 | 地图上下文与 Map-only Corrector | mini 通路、因果特征测试、无地图 fallback | 现有地图和 metadata | 已完成（工程） |
| P3 | 邻车历史与 Map+Neighbors 融合 | 历史差分、邻车缺失/排序测试、四组消融通路 | 现有 metadata | 已完成（工程） |
| P4 | trainval 消融与复现报告 | 相同样本/更新集合、多 seed、分组与延迟报告 | 现有 trainval metadata | 已完成 |
| P5 | mini Radar 与位置退化实验 | 同扰动对照、速度变换、回波覆盖和恢复时间 | 现有 mini sensor | 已完成（mini实验） |
| P6 | 质量门控与延迟/缺失实验 | 固定/学习门控对照，输入过旧时可解释 fallback | 前阶段数据 | 条件推进 |
| P7 | Camera 或 LiDAR 小规模扩展 | 单模态消融、真实 encoder 耗时、覆盖率 | 现有 mini sensor | 可选 |

P1–P5 是主线；P6 以至少一种上下文在对照中表现出价值为前提。P7 根据 P4/P5 的
结果选择一种模态开展，届时再评估完整 trainval sensor 下载需求。

每次“继续”默认推进下一个尚未完成的阶段；不直接跳到所有模态同时训练。

## 每阶段开发节奏

1. 阅读该阶段设计和文件清单，检查工作区已有变更。
2. 为该阶段编写独立实施计划；P1 已提供，其余阶段在前一阶段验收后细化。
3. 先写有行为意义的测试，再补最小实现；新数据通路先用合成样本验证。
4. 跑聚焦测试、原有相关测试、Ruff 和公开仓库扫描。
5. 需要真实数据时，先完成 mini smoke，保留失败计数与输入覆盖率。
6. 提交该阶段代码与实验说明；记录 seed、split、哈希、命令和实际结果。
7. 在阶段总结中明确已完成、未完成、限制与下一阶段入口。

工程验收与效果验收分开：功能正确但未改善指标也应保留结果，不能把“训练成功”写成
“性能提升”。所有新增准确率/延迟数字均在运行后填写实际记录。

## 本机数据与环境

2026-10-20 已核验：mini 共 10 个场景，31,206 个 sensor 文件齐全；trainval 有
850 个场景 metadata；本地 sensor 文件集合全部属于 mini；4 个地区地图 JSON 齐全。
GPU 为 RTX 3060 Laptop，6 GiB。P1–P5 无需下载新数据。

worktree 不复制 data、outputs 或虚拟环境。P2 起新增命令须支持绝对 `dataroot`、缓存和
基础预测文件路径；从共享数据根读取，实验输出写到 worktree 自己的 `outputs/context`。
现有缓存拒绝额外数组，因此新增上下文必须放在独立 cache，不修改旧 NPZ schema。

本机根目录 `.venv` 缺少 torch/nuscenes；已验证的完整环境在旧 worktree 内。
从当前开发目录执行：

```powershell
$python = (Resolve-Path ../trajectory-mvp/.venv/Scripts/python.exe).Path
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& $python -c "import torch, nuscenes; print(torch.__version__, torch.cuda.is_available())"
```

2026-10-20 验证输出：PyTorch `2.6.0+cu124`，CUDA 可用。
该环境仅复用依赖；`PYTHONPATH` 显式指向当前 worktree 的代码。
新电脑按现有 [复现文档](REPRODUCIBILITY.md) 创建独立 Python 3.12 环境。

## 基线与验收命令

默认系统 pytest 临时目录存在权限冲突，因此为每次运行生成独占的临时路径：

```powershell
New-Item -ItemType Directory -Path outputs -Force | Out-Null
$testRunPath = Join-Path (Join-Path (Get-Location) 'outputs') ('pytest-' + [guid]::NewGuid().ToString('N'))
& $python -m pytest -q --tb=short --basetemp $testRunPath
& $python -m ruff check src scripts tests
& $python scripts/check_public_repo.py
git diff --check
git status --short
```

`--basetemp` 必须使用本次生成的独占子目录，不指向原始数据、整个 outputs 或项目根。
2026-10-20 开发基线：**246 passed**。该记录指原有代码，不能写成新增模态已通过验收。

## 实验记录要求

每次实验输出到 `outputs/context/<phase>/<run-id>`，保存：

- git commit、模型与数据配置哈希、context cache 哈希；
- train/selection/evaluation 场景集合及其哈希，seed 与 device；
- oracle 输入说明、时间戳/延迟策略、覆盖率与 fallback 计数；
- 每 agent/update 的误差与抖动、各环节 latency、p50/p95；
- 所有策略相同的有效样本 key 集合与结果表；
- 对应的运行命令、失败案例和结论。

P4 的开发/选模使用 train_val；val 曾经报告过结果，后续在相同 val 上的比较称为固定
验证集回归评估。mini 只做可行性验证。置信区间按场景聚合，不把相邻 update 当成
独立样本。训练阶段允许未来标签参与 loss，模型输入始终受可用时间约束。

## 当前检查点

- [x] 新开发分支与隔离工作目录已创建。
- [x] 公开主线与本地简历分支的差异已确认只涉及文档。
- [x] 用完整依赖环境与独占临时目录通过 246 项基线测试。
- [x] 技术设计、路线图与 P1 实施计划已编写。
- [x] P1：实现并验收时间戳选择和观测/标签接口。
- [x] P2：完成 Map-only cache、训练、evaluator 和 mini smoke 工程验收。
- [x] P3：完成邻车历史、融合/参数量对照、九策略 evaluator 与 mini smoke 工程验收。

下一次开发从细化 P4 trainval 消融实施计划开始。

## P1 验收记录（2026-10-20）

观测 adapter 已可用，生产 context evaluator 接入在 P2。

- `FrameRef` 校验非空 token/channel、非负整数微秒，以及 availability 不早于 source。
- `latest_available` 按通道筛选已可用且不超过 2 秒的帧，按 source timestamp、token
  选择最新记录；边界包含等号，缺失返回 `None`。
- `ObservationView` 与 `EvaluationLabel` 分离，数组复制为 float32 只读快照；旧 replay
  保持原样。adapter 限定 12 点，每次更新 0.5 秒，共 11 次更新，终点保持为 t0+6 秒。
- 新增 **65 项测试**：timeline 27 项，observations 38 项；旧 replay 7 项连同新测试
  的聚焦回归共 **72 passed**。本轮先重跑原有基线得到 **246 passed**，全量验收得到
  **311 passed**。未来位置扰动覆盖全部 11 次更新，另覆盖未来帧删除、延迟释放、过旧、
  输入修改不影响快照、只读标签、非法输入及 dummy predictor 的观测字段边界。
- Ruff、公开仓库检查和 `git diff --check` 均通过。
- 本地代码提交：`1220d1d`（帧选择）、`07c9e4c`（观测/标签 adapter）；验收文档另作
  `docs: record causal context phase one verification` 提交。

实际执行命令（开发 worktree 内，所有 pytest 均生成新的独占 basetemp）：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$python = (Resolve-Path ../trajectory-mvp/.venv/Scripts/python.exe).Path
# 新测试先分别确认 context / observations 模块缺失导致失败，再实现并验证。
& $python -m pytest tests/context/test_timeline.py -q --basetemp (Join-Path (Get-Location) ('outputs/pytest-' + [guid]::NewGuid().ToString('N')))
& $python -m pytest tests/online/test_observations.py -q --basetemp (Join-Path (Get-Location) ('outputs/pytest-' + [guid]::NewGuid().ToString('N')))
& $python -m pytest tests/context/test_timeline.py tests/online/test_observations.py tests/online/test_replay.py -q --basetemp (Join-Path (Get-Location) ('outputs/pytest-' + [guid]::NewGuid().ToString('N')))
& $python -m pytest -q --tb=short --basetemp (Join-Path (Get-Location) ('outputs/pytest-' + [guid]::NewGuid().ToString('N')))
& $python -m ruff check src scripts tests
& $python scripts/check_public_repo.py
git diff --check
```

限制：本阶段仅用合成数据与 sensor token 引用，无真实 sensor I/O、训练、缓存扩展或
新增性能结论；每通道仅取最新一帧，多 sweeps 留到 P5。观测/标签分离用于减少误接线，
不是同一 Python 进程内的安全隔离；只读数组沿用 NumPy writeable 标志约定。
P2 将接口接入地图特征、新 Corrector 与实际 evaluator，届时验证真实模型调用边界。

## P2 验收记录（2026-10-20）

Map-only 工程通路完成。细节见 [P2 设计](superpowers/specs/2026-10-20-causal-context-phase2-design.md)
与 [P2 计划](superpowers/plans/2026-10-20-causal-context-phase2.md)。旧 replay、模型、训练 CLI、
trajectory cache schema 和原有 README 结果均未修改。

实现：metadata anchor 校验、车道折线/方向与稳定候选排序、独立 content-addressed context
cache、固定 GRU 的 token/target/模型身份绑定、ObservationView-only residual 特征与地图
head、场景隔离训练和 checkpoint、四策略同更新行 evaluator、覆盖/fallback 与分组计时。
原 prediction NPZ 必须同时提供 per_sample.csv 和带 GRU provenance 的 metrics.json，
加载时拒绝错 split、错配置、错 token 顺序、错来源及不同 base checkpoint/model config。

本轮新增 **28 项行为测试**，P1 的 311 项加上新增后全量 **339 passed**。覆盖未来扰动、
方向旋转、padding、非零地图 head 的 mask 不变性、无地图严格 fallback、完整缓存准备、
场景隔离、checkpoint provenance 拒绝和实际 evaluator 预测边界。Ruff、公开仓库扫描与
`git diff --check` 通过。独立审查发现的完整来源校验、padding 测试和统计口径问题已修复，
复审无遗留重要问题。

### mini 可行性运行

- 数据为已有 mini metadata/地图，未读取 sensor 文件或下载新数据。
- 地图配置固定为 radius 50 m、8 条车道、20 点/条、2 m 离散间隔；未用 mini_val 调参。
- mini_train：570 条轨迹、6270 次更新，地图覆盖 6270 次；metadata/地图准备约 15.36 s。
- mini_val：47 条轨迹、517 次更新，地图覆盖 517 次；准备约 4.08 s。
- mini_train 按场景名排序，6 个场景训练（4499 updates）、2 个场景选模（1771 updates）。
  原 mini Corrector 曾在全 mini_train 上训练/选模，因此本轮重新在同一 6 场景上训练
  原 ResidualCorrector，以同一 selection 选模后冻结，再训练地图附加 head。
- seed=2027，CUDA，原 Corrector 与 map head 各 2 epoch；固定历史 GRU 不重新训练。
  mini_val 仅在 P2 模型冻结后运行，不称全新盲测，固定 GRU 的历史训练/选模来源仍被继承。
- 四种策略各 517 行、共 2068 行，场景为 scene-0103、scene-0916；无失败、fallback=0。
  此真实子集未触发缺地图路径，该路径由合成模型/evaluator 测试验证。

以下均是本次匹配 update 行上的 smoke 记录，不与旧完整 horizon 指标直接比较：

| 策略 | ADE (m) | FDE (m) | jitter (m) |
| --- | ---: | ---: | ---: |
| frozen | 7.239052 | 8.946203 | 0.000000 |
| sliding | 2.203285 | 5.005146 | 1.018701 |
| corrected | 7.728077 | 8.935945 | 2.313310 |
| map_corrected | 8.478229 | 9.472696 | 3.057674 |

本次 Map Corrector 退化，尚无收益证据；两轮 smoke 用于验证工程通路，不能推出正式效果
或继续调 mini_val 选参数。参数量匹配对照、多 seed、场景级置信区间及 trainval 消融仍属
P3/P4 后续工作。mask/coverage 目前表示存在可用车道折线，不等同检测道路几何适用性。

计时分 cached replay 准备、特征/数据搬运、encoder/head 与返回搬运；完整 update 计时
仅覆盖缓存路径且包含 replay 的 label snapshot 开销，不包含地图磁盘读取和离散化。
CUDA 已同步，但未做 warmup，因此只作为可行性运行记录，不发布实时性能结论。
地图静态来源/可用时间为 0，age 标记 static_not_applicable，动态 target 位置为 annotation
oracle；不读取未来 pose，预测终点始终保持 t0+6 秒。

### 复现与本地文件

在开发 worktree 内运行，下列相对数据路径指向共享仓库；所有输出都保留在本 worktree：

```powershell
$python = (Resolve-Path ../trajectory-mvp/.venv/Scripts/python.exe).Path
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& $python scripts/prepare_context.py --trajectory-cache ../../data/cache/mini/mini_train --split mini_train --dataroot ../../data/nuscenes --output-dir outputs/context/p2/mini-train
& $python scripts/prepare_context.py --trajectory-cache ../../data/cache/mini/mini_val --split mini_val --dataroot ../../data/nuscenes --output-dir outputs/context/p2/mini-val
& $python scripts/train_context_corrector.py --trajectory-cache ../../data/cache/mini/mini_train --context-cache outputs/context/p2/mini-train --split mini_train --base-predictions ../../outputs/mini/gru_train_eval/predictions.npz --dataroot ../../data/nuscenes --output-dir outputs/context/p2/smoke-seed2027 --epochs 2 --device auto
& $python scripts/evaluate_context_online.py --trajectory-cache ../../data/cache/mini/mini_val --context-cache outputs/context/p2/mini-val --split mini_val --base-predictions ../../outputs/mini/gru_eval/predictions.npz --checkpoint outputs/context/p2/smoke-seed2027/best.pt --dataroot ../../data/nuscenes --output-dir outputs/context/p2/mini-val-evaluation --device auto
$testRunPath = Join-Path (Get-Location) ('outputs/pytest-' + [guid]::NewGuid().ToString('N'))
& $python -m pytest -q --tb=short --basetemp $testRunPath
& $python -m ruff check src scripts tests
& $python scripts/check_public_repo.py
git diff --check
```

场景清单及哈希、源 metadata/地图 hash、输入模型身份与全部 artifact hash 保存在 context
manifest、training.json、checkpoint metadata 及 evaluation/summary.json；各 update 行与
update/horizon/maneuver 分组 CSV 保存在 mini-val-evaluation 目录。数据、checkpoint 和
运行指标文件全部被 gitignore 排除，不提交。

- 本次评估代码提交：`0aa319070b9bfeff77bb7fcc1d17fea5e5ae48e9`。
- 训练 context archive SHA256：`1215cf6378edf820b66435eb74315ff4be08dbac5e9957b6f5f306ae568b9c2e`。
- 评估 context archive SHA256：`1783c5587c26d6f6db879db4be554116b5dbd4b8038920b96d3f4d95cd25b9fb`。
- 最终 smoke checkpoint SHA256：`4bb4eb297b1c366b642c8f39f02518caf78d5e1742aea201b64acb1e44a20595`。
- 固定 GRU checkpoint SHA256：`d0af99801477de295b11fe6eeae1e8dada4ad74eb2cec57e6dfb8cb521da5ddf`。

本地提交：8779f32（设计/计划）、e658b8c（地图）、0548a6d（缓存）、a6f8438（模型）、
63421fe（训练）、117cb90（评估）、0aa3190（来源校验修复）；本文验收记录另作 docs 提交。
下一阶段为 P3 Neighbors，不直接跳到全模态或 trainval sensor 训练。

## P3 验收记录（2026-10-20）

Neighbors 与 Map+Neighbors 的工程通路完成，见 [P3 设计](superpowers/specs/2026-10-20-causal-context-phase3-design.md)
与 [P3 计划](superpowers/plans/2026-10-20-causal-context-phase3.md)。本阶段新增独立模块/CLI，
P1/P2 代码、旧 trajectory cache schema 与历史 README 指标保持原样。

- 邻车来自当前已可用的场景 annotation，类别固定为 vehicle.*（含 cycle）与
  human.pedestrian.*；排除目标，按当前距离/token 选最多 8 个，不沿用已消失对象。
- 5 个历史槽覆盖过去 2 秒，历史 source 不晚于槽时间、available 不晚于当前时间，
  每槽最大 age 0.5 秒，重复 sample 只保留最新槽；不读取未来 pose 或插值。
- rel_xy 相对目标当前观测、速度仅为邻车 anchor frame 后向差分，使用真实时间差。
  position/velocity masks、类别 onehot、source/available 时间分开记录。MLP 另编码
  velocity validity 与 age；缺失模态/全缺失 fallback 经过非零 head 测试。
- 独立 neighbor archive 绑定配置、类别/instance/annotation/sample/scene/log/map 来源、
  P2 map archive/contract、原 trajectory 与 token 顺序；加载拒绝错来源、错 split/key、
  checksum、历史槽过旧及不符真实差分的 velocity。
- 共用原架构 Corrector 冻结对照，独立训练 Map-only、Neighbors-only、融合 head；
  各自增加原 ResidualCorrector 宽网络参数量对照，记录 total/trainable 参数。
- checkpoint 校验固定 GRU 身份、map/neighbor 合同、场景哈希及训练/选模场景隔离；
  记录每个 stage 的 best_epoch/completed_epochs。此 suite 是冻结评估产物，不支持 resume。
- 九策略有完全相同的 instance/sample/update 集合，输出 per-update 与 update/horizon/
  maneuver 分组、coverage/fallback、历史/速度有效率、frame age 和分段缓存路径计时。

本轮新增 **36 项行为测试**（neighbors 16、neighbor cache 8、model 8、training 1、
evaluator 3），全量由 P2 的 339 项增至 **375 passed**。覆盖未来 annotation 新增/删除/
扰动不变性、真实时间差、方向、target 排除、类别、radius/排序、history/velocity masks、
延迟/过旧、缓存/来源拒绝、非零梯度、实际 evaluator 单模态缺失与全缺失、九策略同 key。
Ruff、公开仓库扫描、`git diff --check` 通过。独立审查发现历史槽 age 校验与 evaluator
非空邻车测试覆盖不足，均已修复并复审；无遗留重要问题。

### mini smoke 与结果

固定 seed=2027，CUDA，7 个 learned stage 各 2 epoch，场景划分延续 P2：6 个训练场景
（4499 updates），scene-1094/scene-1100 两个选模场景（1771 updates）。本次每个 stage
均在第 2 epoch 选中 checkpoint。不同 stage 重置 seed，并使用独立固定 DataLoader
生成器，使 batch 顺序一致。固定历史 GRU 不重新训练，其历史训练/选模来源被继承，
mini_val 不参与 P3 选模，不能把这次运行称为新的盲测。

- mini_train：570 条轨迹、6270 次更新，5768 次有邻车；准备约 12.36 s。
- mini_val：47 条轨迹、517 次更新，500 次有邻车（96.7118%）；准备约 0.89 s。
- mini_val 地图覆盖 100%，融合至少一模态覆盖 100%；Neighbors-only 的 17 次无邻车
  update 严格回退原 Corrector，融合 fallback=0，无失败。9 策略各 517 行，共 4653 行。
- 共选择 4000 个邻车实例/update，历史有效点 15252/20000（76.26%），速度有效点
  10846/15252（71.11%）；当前邻车 frame age p50=99798 us，p95=499582.1 us。
- 原 Corrector 与各个模型均在本次同一场景/更新集合重跑，不直接套用 P2 历史数值。

| 策略 | ADE (m) | FDE (m) | jitter (m) |
| --- | ---: | ---: | ---: |
| frozen | 7.239052 | 8.946203 | 0.000000 |
| sliding | 2.203285 | 5.005146 | 1.018701 |
| original | 7.513973 | 8.736928 | 2.392542 |
| map | 8.191405 | 9.166367 | 3.084374 |
| neighbors | 7.684330 | 8.793473 | 2.917715 |
| fusion | 7.638667 | 8.679637 | 2.903470 |
| matched_map | 6.602393 | 7.813867 | 4.153845 |
| matched_neighbors | 6.573249 | 7.791869 | 4.403310 |
| matched_fusion | 5.975780 | 6.921620 | 4.639333 |

本次上下文未显示稳定收益：融合 ADE 高于原 Corrector，虽 FDE 略低，但 jitter 更高；
宽网络对照 ADE 更低而 jitter 较高。2 epoch、单 seed、mini 场景结果只验证通路，不能
用于声称上下文有效或据此调 mini_val。正式 train/train_val 开发、多 seed、场景级置信
区间、充分训练及相同样本消融仍属 P4。

| 模型 | total 参数 | trainable 参数 | 与对应 context total 差异 |
| --- | ---: | ---: | ---: |
| original | 4866 | 4866 | — |
| map | 14212 | 9346 | — |
| matched_map | 14127 | 14127 | 0.5981% |
| neighbors | 14468 | 9602 | — |
| matched_neighbors | 14366 | 14366 | 0.7050% |
| fusion | 23108 | 18242 | — |
| matched_fusion | 23070 | 23070 | 0.1644% |

宽度对照匹配 total 参数，context 模型的 common fallback 冻结、宽网络全部可训练。
因此这不是 trainable 参数或总训练计算量匹配：context 还继承原 Corrector 的预训练。
两项差异明确保留，不把宽网络结果误写成纯上下文增益的因果归因。

计时是预缓存输入路径，包含 replay label snapshot、feature/transfer、encoder/head 与
返回搬运；CUDA 同步但无 warmup。不含 metadata 磁盘读取/索引构建，不代表 sensor
端到端性能。annotation neighbor 与 target 位置均为 oracle，vehicle/cycle 共用一类；
缺失 mask 只表达已定义窗口内的可用性，不代表检测准确率。

### 复现与来源

```powershell
$python = (Resolve-Path ../trajectory-mvp/.venv/Scripts/python.exe).Path
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& $python scripts/prepare_neighbors.py --trajectory-cache ../../data/cache/mini/mini_train --map-cache outputs/context/p2/mini-train --split mini_train --dataroot ../../data/nuscenes --output-dir outputs/context/p3/mini-train
& $python scripts/prepare_neighbors.py --trajectory-cache ../../data/cache/mini/mini_val --map-cache outputs/context/p2/mini-val --split mini_val --dataroot ../../data/nuscenes --output-dir outputs/context/p3/mini-val
& $python scripts/train_interaction_corrector.py --trajectory-cache ../../data/cache/mini/mini_train --map-cache outputs/context/p2/mini-train --neighbor-cache outputs/context/p3/mini-train --base-predictions ../../outputs/mini/gru_train_eval/predictions.npz --split mini_train --dataroot ../../data/nuscenes --output-dir outputs/context/p3/smoke-seed2027 --epochs 2 --device auto
& $python scripts/evaluate_interaction_online.py --trajectory-cache ../../data/cache/mini/mini_val --map-cache outputs/context/p2/mini-val --neighbor-cache outputs/context/p3/mini-val --base-predictions ../../outputs/mini/gru_eval/predictions.npz --checkpoint outputs/context/p3/smoke-seed2027/best.pt --split mini_val --dataroot ../../data/nuscenes --output-dir outputs/context/p3/mini-val-evaluation --device auto
$testRunPath = Join-Path (Get-Location) ('outputs/pytest-' + [guid]::NewGuid().ToString('N'))
& $python -m pytest -q --tb=short --basetemp $testRunPath
& $python -m ruff check src scripts tests
& $python scripts/check_public_repo.py
git diff --check
```

manifest、training.json、checkpoint metadata 与 summary.json 记录完整场景、hash、参数
与独立选模进度；每行结果及分组 CSV 位于 mini-val-evaluation。所有 outputs/cache/model
均被 gitignore 排除，未读取 sensor 文件、未下载新数据、未推送远程。

- 最终 smoke 代码提交：`a22c3d6ede7226da063b04bdcf416173deda66f1`。
- train neighbor archive SHA256：`bbd05c03b21ddd98cf77e982d41a903552d145c7cd59da836b625eb587da61e4`。
- eval neighbor archive SHA256：`377309c12e1bfff277b0778e5d4c693773c5e89b16ceaf885438b69bb3d08e37`。
- suite checkpoint SHA256：`45db6aa32f795a11c1f3a77e7e2578cada07447a4953d173af08433e96d18969`。
- 固定 GRU checkpoint SHA256：`d0af99801477de295b11fe6eeae1e8dada4ad74eb2cec57e6dfb8cb521da5ddf`。

本地提交：ec1795a（设计/计划）、97d61fb（邻车）、720bfe3（缓存）、b15a520（融合模型）、
f5346fc（训练）、a22c3d6（评估与历史校验修复）；验收记录与最后补充测试另作提交。
下一阶段细化 P4 trainval 消融计划，先核验缓存/预测与资源预算，再进行多 seed 训练。


## P4 完整 trainval 消融验收（2026-10-20）

完整 train 25,560条/468场景、train_val 6,621条/176场景，seeds2027/2028/2029，
七模型/seed；完整 val 7,201条/136场景、九策略，每策略79,211更新/seed，合计
2,138,697条评估记录，失败0。已完成2000次配对场景bootstrap、分update/horizon/
maneuver报告及相同256key、16warmup的同步GPU单次延迟测量。

| 策略 | 三seed平均ADE/m | FDE/m | jitter/m |
| --- | ---: | ---: | ---: |
| sliding | 1.937731 | 4.235810 | 1.108377 |
| original | 2.328250 | 4.240271 | 1.555624 |
| map | 2.170413 | 3.983093 | 1.493390 |
| neighbors | 2.334388 | 4.225095 | 1.562015 |
| fusion | 2.184295 | 3.970296 | 1.495179 |

map-original ADE差值=-0.157837m，场景95% CI=[-0.177403,-0.137334]m；
map也优于对应total参数宽度对照。neighbors ADE与jitter退化，融合未显示优于map的
ADE；全部learned模型ADE和jitter仍高于sliding。oracle与参数/预训练不匹配限制保留，
val仍称历史固定验证集回归，不作为新盲测。

402项pytest通过；Ruff、公开仓库扫描、diff检查通过。独立审查原缓存/训练身份问题
已修复并回归；后续评估/统计的再次独立复核遇服务使用上限，具体范围已记录。
完整配置、来源哈希、选模epoch、参数量、三seed标准差、置信区间、延迟、分组与命令
见 [P4报告](CAUSAL_CONTEXT_PHASE4_RESULTS.md)。原始产物位于outputs/context/p4且未提交。

本阶段代码提交：acff8bf（计划）、908c68e（全量准备）、a975c14（批量训练评估）、
213cec6（场景统计与延迟）、ecdc794（源码变化回归）、67fc46e（延迟一致性）。
下一阶段为P5 mini Radar与位置退化实验，继续时先细化独立实施计划。


## P5 mini Radar 与位置退化验收（2026-10-20）

完整mini_train570条/8场景，内部409条/6场景训练、161条/2场景选模；seeds2027/2028/2029。
五条件为clean、1m高斯噪声、0.5/1s交付延迟、固定丢弃第4/5/6次位置更新。所有策略
共享同一时间感知退化状态和固定GRU，Radar半径门控也用该状态，无标注框关联。
完整mini_val47条/2场景，六策略×五条件×11更新，三seed共46,530条记录，失败0。

| 条件 | original ADE/m | Radar ADE/m | matched ADE/m | sliding ADE/m | filtered ADE/m |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean | 6.112666 | 5.958220 | 6.141214 | 2.203285 | 3.387421 |
| noise | 7.109338 | 6.924692 | 7.236954 | 7.190922 | 3.940704 |
| delay 0.5s | 6.252575 | 6.023069 | 6.267482 | 3.026487 | 4.263374 |
| delay 1s | 6.614336 | 6.327768 | 6.644946 | 3.879909 | 5.119225 |
| dropout | 6.285151 | 6.108900 | 6.320326 | 2.785395 | 4.228504 |

上述为三seed均值。Radar相对小型original/matched平均ADE更低，但若干两场景描述性
区间跨0，训练seed方向不一致；强CV/滤波基线仍更优，不能声称稳定泛化或整体方案优越。
原Corrector是P5混合退化条件重训，同P4权重和数据不同；初始干净oracle历史/anchor保留。

Radar覆盖88.39%–90.91%，缺回波行均fallback，覆盖不表示目标关联准确。丢点恢复要求
相对clean ADE差<=.25m且连续两次，Radar与original的平均恢复时间均0.092199s，
未观察加速；滤波6/141行在2s右删失。141为47轨迹×3固定seed，不当作独立样本。

真实PCD路径禁用解码缓存，CUDA同步、固定256key/16warmup，p95为41.56–43.37ms；
缓存统计路径p95为2.95–4.09ms，二者分开记录，OS文件缓存可能已热。
完整阶段计时、seed标准差、场景区间、组别结果和来源见 [P5报告](CAUSAL_CONTEXT_PHASE5_RESULTS.md)。

420项pytest通过，Ruff/公开扫描/diff检查通过；独立审查发现的问题均先复现后修复再复核。
代码提交：9d7a8c1（设计计划）、2d639b5（因果Radar与退化）、a746e2b（缓存审计/训练评估）、
c93b800（子集恢复ID）、0202e1e（来源一致性与完整统计报告）。全部模型/数据产物位于
outputs/context/p5且未提交；保留开发worktree，不推送。

P6仍为条件推进：先以P4已有证据的地图和强CV/滤波对照细化门控实验，不因P5弱Corrector
上的平均改善自动增加Radar编码复杂度或下载更多sensor。P7仍可选。

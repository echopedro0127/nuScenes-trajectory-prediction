# 因果在线轨迹预测：分阶段开发路线图

更新日期：2026-10-20。当前状态：P0 分支、设计和开发计划已建立；新增功能尚未实现。

## 开发入口

- 基线：公开 `master` 的 `54fb25b`。
- 开发分支：`feature/causal-context-online`。
- 工作目录：仓库根目录下 `.worktrees/causal-context-online`。
- 根目录仍使用 `local/resume-bullets`；简历文档和之前的本地实验提案保留原处。
- 本轮只向开发分支提交文档，不推送远程、不训练新模型。

进入开发目录：

```powershell
Set-Location .worktrees/causal-context-online
git branch --show-current
```

预期分支名为 `feature/causal-context-online`。不要在根目录直接切换到这个已被 worktree
检出的分支。

设计见 [技术设计](superpowers/specs/2026-10-20-causal-context-online-design.md)；
下一步执行 [P1 实施计划](superpowers/plans/2026-10-20-causal-context-phase1.md)。

## 阶段及验收

| 阶段 | 工作范围 | 验收依据 | 数据 | 状态 |
| --- | --- | --- | --- | --- |
| P0 | 分支、worktree、设计、计划、基线验证 | 开发文档可执行，原有测试通过 | 不需要新数据 | 已完成 |
| P1 | 时间戳输入选择、观测/标签边界 | 未来帧扰动不影响当前输入，旧 replay 回归通过 | 合成数据 | 待开始 |
| P2 | 地图上下文与 Map-only Corrector | mini 通路、因果特征测试、无地图 fallback | 现有地图和 metadata | 待开始 |
| P3 | 邻车历史与 Map+Neighbors 融合 | 历史差分、邻车缺失/排序测试、四组消融通路 | 现有 metadata | 待开始 |
| P4 | trainval 消融与复现报告 | 相同样本/更新集合、多 seed、分组与延迟报告 | 现有 trainval metadata | 待开始 |
| P5 | mini Radar 与位置退化实验 | 同扰动对照、速度变换、回波覆盖和恢复时间 | 现有 mini sensor | 待开始 |
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
- [ ] P1：实现并验收时间戳选择和观测/标签接口。

下一次开发从 P1 开始；本轮不包含 P1 生产代码。

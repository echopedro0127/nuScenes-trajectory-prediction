# Phase 3B / B0a：真实时钟与目标轨迹装配验收

日期：2026-11-16。**真实装配与独立 `--verify-only` 均通过，`track_and_window_replayed=true`。** 本阶段交付后续场景抽取所需的数据底座，不产生新预测分数。

后续进展：[B0b全缓存装配已验收](SCENARIO_TRACKS_B0B_RESULTS.md)，1967目标/60948标注/22231窗口；本报告以下保留B0a交付时的范围与后续安排。

## 解决了什么问题

原3A在名义0.5秒网格上评测，但nuScenes原始相邻sample不严格间隔500000微秒。例如名义两步lead实际跨度为0.799584–1.150706秒。直接用步数解释物理时间，会影响后续减速、停车持续时间、事件边界及恢复时长的定义。

B0a不修改既有3A时钟或成绩，而是从原始sample恢复整数微秒，并把608个旧窗口逐项映射回真实annotation/sample token。后续算法因此可以明确区分名义步数与实际时间，而不需要插值伪造GT。

原窗口还截断了目标的运动历史，无法可靠判定跨窗口是否同一个驾驶事件。B0a收集既有目标instance的完整annotation链，提供连续性和缺口证据，为下一阶段事件状态机与去重建立输入；本轮不把连续片段自动当成驾驶episode。

## 方法与实现

1. **有界流式读取。** 原annotation文件583,417,244字节，逐块扫描1,166,187条记录，只保留pilot目标。读取器拒绝重复字段、非有限数、非法JSON空白、截断及超长记录，不新增解析依赖，不把annotation整表载入。
2. **链与角色边界验证。** 检查首尾token、记录数、prev/next互反、时间严格递增、instance/sample唯一性、scene/log/category来源，以及fit/select目标与log隔离。场景sample中存在但目标annotation跳过的帧形成显式缺口；不根据时间差除0.5猜缺帧数。
3. **可追溯分段。** 世界坐标保留float64，时间保留int64微秒；track语义摘要和segment ID由规范化链及角色/边界生成，不依赖JSON行排列。缺口两侧断开，不补点；片段首行时间增量标为无效。
4. **精确窗口重建。** 对每个旧anchor沿链取4个历史点、12个未来点，复用原几何变换与原float32转换顺序。anchor/history/future要求逐元素精确一致，旧window ID与原events/windows.json一致。真实绝对时间和相对偏移另存，不覆盖名义步数。
5. **冻结隔离与独立重建。** 在新worktree开发；原3A验证器只在显式旧owner的cwd/PYTHONPATH运行一次，保存其语义重放证明。新包绑定源码、元数据、缓存、父凭据与产物哈希，最后写完成标记。独立verify-only重新扫描原始输入并重建数组、窗口、ID、协议和摘要；复用已保存父证明并复核其来源，不再次运行父模型。

主要入口：`scripts/prepare_scenario_tracks.py`。核心模块为`data/metadata_stream.py`、`analysis/track_timeline.py`、`evaluation/track_sources.py`及`evaluation/scenario_tracks.py`。

## 实测规模与验收

| 项目 | 结果 |
|---|---:|
| 目标instance | 461（fit 268 / select 193） |
| 完整目标annotation | 16,010 |
| 旧窗口 | 608（fit 352 / select 256） |
| 与缓存精确重建的窗口 | 608 / 608 |
| 窗口内annotation缺口 | 0 |
| 完整链上的跨sample缺口 | 4 |
| annotation连续片段 | 465，非episode数 |
| 插值或填充GT点 | 0 |
| 类别 | car 362 / truck 66 / rigid bus 32 / construction 1 |
| 新包冻结源码文件 | 233 |
| 绑定来源引用 | 1,076 |
| 包内绑定文件 | 14，合计2,520,918字节，不含completed标记 |

新增测试共58项，覆盖流式边界、链与分段、窗口装配、来源变更、显式owner、篡改后重算哈希、协议/运行标志/运行环境、额外文件和中断无完成标记。全套 **920 passed，234.02秒**；Ruff、公开仓库检查及diff检查通过。最终只读代码复审未发现Critical/Important问题。

真实运行后另核对旧3A的 **228个冻结源码与217个产物均未改变**。其运行身份仍为`5cd60defeb086e0216d3885b4cb6a337eda7e5f8899639d60c30084d02dee019`。

## 复现身份与命令

- 分支：`feature/scenario-episodes`，工作区：`.worktrees/scenario-episodes`。
- 源码提交：`09cbe1c`；前序流式读取及链实现提交：`75ae6cd`、`7602554`。
- 输出：`outputs/scenario-tracks/b0a-v1`，原始数据和生成包均不进入Git。
- 运行身份：`d8a904ed958ece5bf797fdf5a21aba6d755680d3f79b8e9809baf03088664102`。

以下命令在新worktree、既有Python环境和`PYTHONPATH=src`下执行。目录不可覆盖；重跑装配需新版本目录，现有包用verify-only验收。

```powershell
python scripts/prepare_scenario_tracks.py --parent ../scenario-evaluation/outputs/scenario-evaluation/phase3a-v1 --parent-owner ../scenario-evaluation --output-root outputs/scenario-tracks/b0a-v1
python scripts/prepare_scenario_tracks.py --output-root outputs/scenario-tracks/b0a-v1 --verify-only
```

包内包含两角色tracks/windows NPZ、tracks/segments/gaps/windows索引、audit/summary/protocol/run、parent_proof及完成凭据。绑定含本机来源绝对路径和环境身份，当前支持原环境复查；搬迁目录或升级环境不能沿用旧身份声称通过。

## 边界与下一步顺序

本轮未训练、未评价holdout、未重新计算预测分数，不能声称精度新增提升。75条人工复核仍pending、manual_completed=0；没有canonical episode、人审GT导入器或新传感器输入。

下一步先设计并实现 **B0b范围扩展**：仅扩到原缓存1265 fit＋702 select目标，预计60,948条annotation，须实际装配验收后才确认为成果。不得扩成允许log内所有对象，也不得把608 pilot窗口成果套到全量。

随后才是 **B1事件协议冻结与实现**：使用真实时间定义减速、停车、重新起步、转向边界和stop/restart屏障，明确缺口、首尾删失和不可评价规则，生成稳定episode ID并关联旧窗口；禁止以近邻onset传递闭包替代事件定义。再独立推进episode级人工复核/阴性队列与指标，最后据已冻结假设考虑训练。仍不使用历史holdout调参。

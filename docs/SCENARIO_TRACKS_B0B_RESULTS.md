# Phase 3B / B0b：全缓存目标轨迹与真实时钟验收

日期：2026-11-16。**真实装配及单独verify-only均通过**，`track_and_window_replayed=true`。本轮将B0a的pilot数据装配扩展到原fit/select缓存的全部目标；没有新增预测分数、模型训练或holdout评测。

## 解决了什么问题

B0a只核验了3A已有461个目标、608个窗口，不能据此认为全缓存已具备完整轨迹与真实时间。后续识别刹车、停车、重新起步事件，需要看到目标的完整annotation链，不能把窗口起止位置误当作事件边界，也不能把名义0.5秒当作准确物理间隔。

B0b从原缓存每一行推导目标与窗口，形成全部1,967个目标的完整GT链、整数微秒时间及22,231个窗口的精确映射。这为后续定义事件边界和评测分母提供数据基础；本轮仍未执行事件抽取。

## 实现方法与边界

复用已有流式读取、链完整性校验、真实时间和几何重建模块，为装配器增加显式范围选择：默认`pilot`保留旧行为，`full_cache`使用新schema与scope。两种范围进入运行身份、协议、摘要及独立重建流程，未知范围和篡改后重算哈希的范围变更均被拒绝。

全量目标仅来自原`independent_fit`和`independent_select`缓存。逐行保留role/instance/sample/scene/log/row，拒绝重复窗口键；旧pilot的键及原缓存行号必须是全量的匹配子集。不能因某个log允许使用，就把其中非缓存目标对象纳入。测试特意加入同scene的额外对象，确认它们不被选中。

完整链仍通过instance首尾token、annotation prev/next互反、计数、sample唯一性、scene/log与角色边界校验。583MB annotation文件分块读取，扫描1,166,187条记录，仅保留所选目标，不安装新依赖或加载整表对象。

每个缓存窗口沿链取4个历史点、anchor和12个未来点；复用原浮点转换与坐标变换顺序，对anchor/history/future逐元素精确比较，不能用误差容限掩盖来源差异。保留真实微秒时间、名义步数与原window ID，并增加`phase3a_pilot_member`，防止把新增数据覆盖误称已有模型预测覆盖。

GT仅用于离线装配、标签与评测；本轮未向模型输入构建器接入未来GT。旧3A预测也没有按新窗口拼接或扩展。

## 真实结果

| 角色 | 目标instance | 完整annotation | 原缓存窗口 | scene | log |
|---|---:|---:|---:|---:|---:|
| independent_fit | 1,265 | 39,466 | 14,445 | 291 | 22 |
| independent_select | 702 | 21,482 | 7,786 | 104 | 4 |
| 合计 | **1,967** | **60,948** | **22,231** | **395** | **26** |

- 22,231/22,231窗口的anchor/history/future精确重建通过；历史mask均为完整5点。
- 完整目标链发现25处跨scene sample缺口，形成1,992个annotation连续片段。**这不是1,992个驾驶事件。**
- 缓存窗口内跨标注缺口数量为0，不可评价窗口（source_annotation_gap）为0；这不代表未来的事件抽取没有首尾删失或覆盖不足。
- 未插值、补GT或重采样；`nominal_time_is_exact=false`。旧3A指标仍按原名义时间解释，没有改写成严格物理1秒lead。
- 类别为car 1,540、truck 289、rigid bus 114、bendy bus 12、construction 9、police 3；不包含行人预测结论。

与B0a相比，已验证目标增加1,506个、完整标注增加44,938条、窗口映射增加21,623个。**这三个数字是数据装配工作量，不是新增模型训练样本或已评测预测数量。** 3A已有预测覆盖仍为608个唯一窗口。

## 独立验收与旧结果保护

`--verify-only`从相同原始元数据和缓存重新构建完整链、数组、时间映射、ID、协议及摘要，并核对源码/来源/输出哈希，独立进程执行成功。父3A验证器在其原owner/cwd/PYTHONPATH运行一次，保存原生语义重放证明；本次verify-only核对该证明及不变来源，不重复训练或重新打分。

另对B0a子集作投影比较，按role/instance或role/instance/sample重新对齐，而非假定新旧扁平行号相同：

| 比较项 | 结果 |
|---|---|
| 461条track元数据和语义摘要 | 完全一致（排除布局行偏移） |
| 16,010行track数组 | 所有字段逐元素一致，包括坐标、rotation、时间、token和segment ID |
| 608个旧窗口 | 原ID、来源字段及全部token/真实时间数组一致（新旧行偏移另作索引） |
| 原465个连续片段 | 片段ID及元数据一致 |
| 旧B0a | 233个冻结源码、14个绑定产物均不变 |
| 旧3A | 228个冻结源码、217个绑定产物均不变 |

新增5项测试覆盖全量范围、新增目标、排除无关对象、重复键、跨角色目标、未知scope和重算哈希篡改。先运行RED，观察到接口缺失导致失败，再实现并运行GREEN；相关定向测试共63项通过。**全套925 passed，210.95秒**；Ruff、公开仓库检查、diff检查均通过。最终只读复审未发现Critical/Important问题。

## 复现身份、路径与命令

- 本地分支：`feature/scenario-tracks-full`，源码提交`350d892`。
- worktree：`.worktrees/scenario-tracks-full`；旧`.worktrees/scenario-episodes`及`.worktrees/scenario-evaluation`保持只读。
- 运行身份：`9c9324f34fdea1f1069d8e6dbb8e19626d8d1407978d7dbb8644bd15717b21d8`。
- 正式目录：`outputs/scenario-tracks/b0b-v1`，14个绑定文件共22,893,302字节（不含completed标记）。
- 绑定233个源码文件和1,076个来源引用；含原环境和本机绝对路径，不能搬迁或升级环境后沿用旧身份。
- 子集/旧冻结检查记录：`outputs/scenario-tracks/b0b-acceptance-v1.json`，放在正式包外，避免向已完成包加入额外文件。

在本worktree使用既有Python环境和`PYTHONPATH=src`执行：

```powershell
python scripts/prepare_scenario_tracks.py --scope full_cache --parent ../scenario-evaluation/outputs/scenario-evaluation/phase3a-v1 --parent-owner ../scenario-evaluation --output-root outputs/scenario-tracks/b0b-v1
python scripts/prepare_scenario_tracks.py --output-root outputs/scenario-tracks/b0b-v1 --verify-only
```

装配拒绝覆盖现有目录。verify-only读取保存的scope重新装配；不传`--scope`的新装配默认仍为pilot，不能据默认命令宣称已跑全量。数据与输出保持本地，代码和文档仅作本地提交，未推送GitHub。

## 下一步及尚未完成的事项

进入B1前先冻结独立事件协议：基于真实dt定义减速、停车、重新起步和转向状态，明确确认时刻、结束边界、stop/restart屏障、缺口、首尾删失和不可评价分母，随后生成稳定episode ID并关联窗口。不能把近邻onset的传递闭包当作事件定义。

再准备episode级人工复核及阴性/不可评价队列；目前75条旧window任务仍pending、manual_completed=0，旧window规则标签不是人工事件真值。后续episode预测评测还需冻结anchor覆盖和噪声命名空间，不能拼接具有不同初始化/噪声历史的旧窗口P。

新增预测指标、受控运动、遮挡恢复或新训练均在这些协议之后分阶段推进。本轮不新增传感器、Cosmos或数据集下载，不使用历史holdout选择规则。

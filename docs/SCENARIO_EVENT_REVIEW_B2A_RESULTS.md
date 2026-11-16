# B2a：事件复核资料与既有窗口覆盖索引验收

日期：2026-11-17。**真实生成及独立verify-only通过，review_semantically_replayed=true。** 交付离线GT查看器、完整阅读队列和工作表，不导入人工事件标签，不选择新预测anchor，不计算新预测分数。

## 解决的问题与方法

B1a规则候选不能直接当作人工真值；只看阳性抽查也无法检查未触发上下文。B2a把完整GT、真实速度、规则边界、诊断和原覆盖索引组织成可浏览资料，保留所有分析块，便于记录标注噪声、删失、规则漏触发和覆盖不足等疑点。没有模型预测曲线或排名，也没有新增抽样阈值。

轨迹从父run明确指向的已绑定B0b数组提取，按annotation端点token核对instance、segment、时间及区间守恒，不猜owner层级。图中仅平移首点，米制X/Y等比例；速度取原B1a真实dt区间值。单点显示无有效速度区间，不画伪零速度。事件、诊断、窗口关联及coverage与B1a完全一致，不重新抽标签。

## 真实规模

| 项目 | 数量／状态 |
|---|---:|
| 完整分析块扫描任务 | 1,992 |
| 候选核对任务 | 347，含60个复合配对 |
| 无block诊断单独任务 | 0 |
| 总阅读任务 | **2,339** |
| fit／select任务 | 1,529／810 |
| 无规则候选分析块 | **1,767**，不是人工阴性 |
| 原始诊断 | 1,077 |
| 原始GT点／速度区间 | 60,948／58,956 |
| 初始pending／in_progress／reviewed | 2,339／0／0 |
| 已导入人工事件标签 | 0 |

347条记录为227 stop＋60 restart＋60 stop_go，不能叫作347个独立驾驶过程；任务也不等于独立事件。所有诊断均关联对应block并保留。全量扫描包含未触发、单点和删失上下文，但没有人工复核的“无候选”不能作为负例真值。

## 离线使用与工作表

入口：本worktree下`outputs/event-review/b2a-v1/review.html`，自包含HTML/SVG/JavaScript及数据，无需服务、网络或新依赖。支持角色、任务类型与来源ID筛选，查看GT、原始速度、onset／confirmation／end、删失及原窗口关联。无匹配时清空旧任务并禁用编辑。

`review_working.csv`是唯一可编辑文件，只允许review_status／reviewer／notes变化。可排序，不可删除、增加任务或修改来源字段。允许pending／in_progress／reviewed，reviewed须填复核人和备注，**只表示阅读记录，不是confirmed事件GT**。

打开网页后先选择候选核对或完整块扫描，记录疑点并下载更新CSV；继续阅读时导入同源CSV。浏览器不会自动覆盖本机文件。若要用CLI核验，将下载文件另存为资料目录中的review_working.csv，不覆盖review_baseline.csv。

Python与浏览器共享空白规则及每单元格8192个Unicode码点上限；非BMP字符不按UTF-16长度误计。支持UTF-8 BOM、引号、逗号和多行备注；拒绝错列、重复／遗漏任务、非法状态、未闭合引号与异源工作表。导入失败整次拒绝，不部分覆盖已有记录。

## 冻结及独立验收

原生B1a验证只在显式旧owner的cwd/PYTHONPATH执行一次，严格解析两条角色stage及semantic结果，绑定旧源码、父产物及全部来源。静态包最后写completed；11个静态文件校验哈希并从原始来源重建HTML/JSON/CSV逐字节比较。已有输出不覆盖，中断不签发完成标记。

完成凭据显式声明review_working.csv这一可变例外：不绑定其字节，但每次verify都从重建队列核对列、任务、锁定字段、状态、长度和必填项。实时阅读数量与固定manual_completed=0分开输出，阅读记录不写回B1a、不进入模型输入或评分。

真实审计确认：事件／诊断／窗口关联／coverage完全继承，60948个GT token无遗漏和重复，58956个区间速度保留，初始working与baseline字节相同。这里仅实现**既有覆盖索引**，尚未实现新anchor／噪声协议或扩展预测覆盖。

## 验证与问题修复

新增31项定向测试，覆盖GT／队列守恒、来源和外键、CSV排序／多行／身份、Unicode空白及码点长度、原生proof、重算哈希篡改、唯一可变文件边界和中断。**全套1011 passed，250.01秒**；Ruff、公开检查、diff及最终只读复审通过。

Node实际执行CSV往返及锁定字段校验；Edge无头在合成夹具上验证空筛选／恢复、图表、单点、File导入、Blob导出和异源拒绝的原子性。再对真实2339任务页按事件ID加载并截图，确认显示正常且仍pending。夹具中的reviewed记录未写入正式资料，不是人工复核成果。

复审修复了Python strip与JS trim的Unicode空白差异、默认CSV长度限制不一致、零筛选仍显示旧任务和图表独立缩放失真。浏览器异步导入QA等待实际状态变化，不用固定延迟推定文件读取完成。

| 旧阶段 | 冻结源码不变 | 绑定产物不变 |
|---|---:|---:|
| B1a | 240 | 12 |
| B0b | 233 | 14 |
| B0a | 233 | 14 |
| 3A | 228 | 217 |

## 复现身份与命令

- 分支：`feature/scenario-event-review`；worktree：`.worktrees/scenario-event-review`。
- 源码提交：`36924ae`，初始装配／查看器提交`ff34ac0`。
- 身份：`93687c1ebcf2e26031c43fc08656f2fbaf0135ade7e33a7ec8ee8a48f1d451e6`。
- 正式目录：`outputs/event-review/b2a-v1`。
- 11个静态文件32,849,101字节；初始working 499,899字节，完成标记另计。
- 绑定245个源码文件、1577个来源引用；原本机路径和环境被绑定，搬迁不能沿用旧身份声称通过。
- 额外验收记录：`outputs/event-review/b2a-acceptance-v1.json`，位于正式包外。
- 真实页面截图：`outputs/real-browser-qa/review.png`；合成交互QA位于`outputs/ui-smoke/`。

在本worktree使用既有Python环境及PYTHONPATH=src：

```powershell
python scripts/prepare_event_review.py --parent ../scenario-stop-events/outputs/scenario-events/b1a-v1 --parent-owner ../scenario-stop-events --output-root outputs/event-review/b2a-v1
python scripts/prepare_event_review.py --output-root outputs/event-review/b2a-v1 --verify-only
```

## 限制与下一步

全量队列不要求一次读完；未复核仍为pending，部分阅读不支持召回率或检测精度结论。阅读记录尚无标签overlay导入器，也未产生人工事件GT。

下一步分别冻结人工标签导入规范（确认／驳回／边界修正、复核人、来源及冲突处理）与预测anchor／噪声协议。复核和覆盖要求满足后才评停车时间、超调、miss与误停。本轮无新训练、holdout调参、传感器或数据下载。原始数据和生成包保持本地，仅提交代码／文档，未推送GitHub。

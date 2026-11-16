# B2a 复核资料与既有覆盖索引实现计划

> **For agentic workers:** 使用superpowers:executing-plans顺序执行、测试先行；只读审查可委托agent。用户已要求继续实现B1a之后的复核和覆盖工作，沿用既有规则，不新增标签／预测协议。

**Goal:** 将冻结B1a转换为可打开的离线GT查看器、完整复核队列和可编辑工作表，保留原覆盖分母，供人工检查。

**Architecture:** 显式旧owner验证→全量block与事件索引／GT装配→自包含HTML及CSV→不可变静态包与单独可编辑表→语义重建。无新采样、规则、模型预测或GT导入。

**Tech Stack:** Python、NumPy、标准库CSV/JSON；原生HTML/SVG/JavaScript，不下载依赖。

## 范围与设计约束

对B1a所有block建立block_scan任务，并对所有候选建立event_check任务；规则未触发block也必须出现，不从既有阳性抽查推断召回率。复合stop_go不是额外独立驾驶过程，任务数与独立对象／block／事件类型计数分开。全部诊断保留，关联对应block；无block诊断另建diagnostic_check任务，不静默丢弃。

每个任务保留原role/instance/block/event/diagnostic ID，GT与速度只来自已绑定B0b/B1a数组。查看器包含轨迹、真实时间速度、规则边界、诊断和原window覆盖；没有模型名称、排名或P。所有任务初始pending。

静态包含review_data.json、queue.json、coverage.csv、review_baseline.csv、review.html、README、run/protocol/parent_proof/summary/progress及最后完成标记。review_working.csv是唯一可编辑例外，静态清单不绑定其字节，但每次验证其列、任务集合、不可编辑字段和允许状态。允许pending/in_progress/reviewed；reviewed须有reviewer与notes，只表示阅读记录，不是确认事件GT。工作表可排序，不可删除或添加任务。没有人工标签overlay导入器，不将阅读状态写回B1a或用于评分。

HTML可编辑本次会话工作表并下载CSV；不会自动写本机文件。打开时显示初始baseline状态，若要继续已有复核，可手工导入同源working CSV。导入必须验证任务身份和全部锁定字段，错误整次拒绝，不能部分覆盖。CSV解析须支持引号、逗号、换行和UTF-8 BOM。

两种方案取舍：全量队列避免引入新抽样协议，按需浏览不用渲染1992张图片；只导出抽样图较小但无法覆盖未触发上下文；新事件GT导入／anchor选择会引入未冻结语义，留下一阶段。此处仅实现第一种。

## T1 新worktree与基线

- [x] 从880292c新建feature/scenario-event-review，工作区.worktrees/scenario-event-review，旧B1a只读。
- [x] 创建outputs并运行既有55项B1a定向基线；复用原环境。

## T2 索引与工作表

Files: `evaluation/event_review_data.py`, `tests/evaluation/test_event_review_data.py`。

- [x] 先写RED：完整任务守恒、事件／诊断外键、无候选block、单点、错误GT来源、原coverage不变。
- [x] 实现`assemble_review(parent, references)`：通过父run显式B0b路径读取绑定数组，不猜owner层级；按block端点token提取原始XY／时间，关联已有事件／诊断／窗口索引。
- [x] 实现CSV生成及`validate_working(text, baseline)`，只允许三列编辑，拒绝错列／重复／漏行／锁定字段篡改，允许排序、带引号多行notes。
- [x] GREEN后提交。

## T3 离线查看器

Files: `evaluation/event_review_html.py`, `tests/evaluation/test_event_review_html.py`。

- [x] RED：自包含、安全JSON嵌入、角色／任务筛选、source ID定位、GT与规则边界、原覆盖、CSV导出和导入验证。
- [x] 原生SVG实现按需渲染，XY用首点平移显示但保留world数据；速度使用原始真实dt；通过文本节点显示源字段与人工notes。
- [x] 不引入网络资源，不自动打开可见浏览器；本地可用的Node／Edge用于语法和无头渲染核验，无则报告验证限制。

## T4 父证明、发布与独立验证

Files: `evaluation/review_sources.py`, `evaluation/event_review.py`, `scripts/prepare_event_review.py`, `tests/evaluation/test_event_review_pipeline.py`。

- [x] 先RED：显式owner、B1a原生证明（两条stage JSON＋结果JSON）、构建→独立重建、静态篡改重哈希、工作表允许修改／拒绝任务篡改、中断不完成、输出不覆盖。
- [x] 父CLI只在旧owner中调用一次；原生stdout严格解析，检查stage和semantic true及身份，绑定旧源码、父12产物和全部来源；不重新抽取不同规则或评分。
- [x] 发布时核对全部静态文件集合与哈希，唯一mutable路径明确写入协议；语义verify重建静态HTML/JSON/CSV，并另验工作表结构。初始manual_completed=0，live工作状态另作descriptive计数。
- [x] 只读代码审查，Ruff／全套pytest／公开检查／diff通过后先提交源码。

## T5 真实运行和交付

验收记录：源码36924ae；31项新增测试及全套1011测试通过（250.01秒），Ruff/公开检查/只读复审通过。真实生成及独立verify-only通过；2339任务全pending，1992块及347候选、1077诊断完整保留。Node/Edge交互与真实页面显示核验通过；旧四owner冻结不变。运行身份93687c1ebcf2e26031c43fc08656f2fbaf0135ade7e33a7ec8ee8a48f1d451e6，详见[结果报告](../../SCENARIO_EVENT_REVIEW_B2A_RESULTS.md)。

```powershell
python scripts/prepare_event_review.py --parent ../scenario-stop-events/outputs/scenario-events/b1a-v1 --parent-owner ../scenario-stop-events --output-root outputs/event-review/b2a-v1
python scripts/prepare_event_review.py --output-root outputs/event-review/b2a-v1 --verify-only
```

- [x] 固定源完整生成，独立verify-only通过；核对1992个block、347候选及1077诊断保留，未触发block数量照实报告。
- [x] 无头浏览器核验首个事件显示和工作表导入导出；记录实际支持的交互检查。
- [x] 旧B1a/B0b/B0a/3A来源及产物均不变；文档记录资料包路径、pending状态、真实测试与限制。本地提交、不推送。

下一步：实际人工复核及其版本化导入协议、新预测anchor与噪声协议，之后才是停车指标；本轮不创建人工GT或新预测分数。

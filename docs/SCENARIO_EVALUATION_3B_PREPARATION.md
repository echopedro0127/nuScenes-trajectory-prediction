# Phase 3B准备：真实时钟/track审计与人工复核工作区

日期：2026-11-16。完成的是只读数据审计和现有复核资料复制，**尚未实现B0产品代码、抽取canonical episode或产生新预测评分**。3A冻结源码和产物不修改。

## 已完成的工作与发现

完整sample、sample_annotation、instance、scene、log、category元数据均在本地，无需新增数据下载。sample_annotation.json为583,417,244字节；当时可用内存约1.5GB，因此使用标准库分块扫描，没有把annotation整表载入或安装新依赖。

扫描34,149条sample、64,386条instance及1,166,187条annotation。按原fit/select缓存的log范围仅作可用性计数，不对holdout打标签或评分；未保留非目标对象的轨迹坐标。

| 范围 | fit | select | 说明 |
|---|---:|---:|---|
| 原缓存窗口 | 14,445 | 7,786 | 原有可用数据，不改变3A pilot |
| 原缓存目标instance | 1,265 | 702 | 后续可扩展目标范围，总计1967 |
| 目标instance在instance表的预期annotation数 | 39,466 | 21,482 | 60,948条，尚未逐track装配全量范围 |
| 允许log内所有对象annotation | 445,391 | 176,218 | 包含非目标类别，不能当作本项目目标集合 |
| 允许log内所有对象instance | 25,806 | 8,147 | 不自动扩进3B |

首轮审计聚焦3A已有的461个目标instance：完整收集16,010条annotation，并检查nbr_annotations、首尾token、prev/next互反、instance/sample唯一性、scene/log和时间严格递增。缺失实例、链错误、重复instance/sample和非正时间差均为0。目标类别为362 car、66 truck、32 rigid bus、1 construction；不能将这些轨迹泛称为行人或全部仅为小汽车。

发现4处annotation跨过中间scene sample，形成465个“annotation连续片段”。**465是结构性连续片段数，不是驾驶事件数。** 原608窗口内部未发现跨帧缺口，raw annotation重建的anchor、history、future与缓存逐项float32精确一致（608/608）。

## 名义时钟与真实时钟不同

461条track的15,545对相邻scene sample时间差范围为397,689–650,284微秒，中位499,883微秒；没有一对恰好500,000微秒。含annotation缺口的最大时间差为1,651,696微秒。608窗口中，原始时间相对anchor的偏移与名义步数×0.5秒最大相差249,763微秒。

| 3A名义lead | 唯一GT查询数 | 实际跨度最小s | 中位s | 最大s |
|---|---:|---:|---:|---:|
| 0.5秒 / 1步 | 6688 | 0.399167 | 0.499884 | 0.650284 |
| 1秒 / 2步 | 6080 | 0.799584 | 0.999771 | 1.150706 |
| 2秒 / 4步 | 4864 | 1.749580 | 2.000034 | 2.152681 |

这些计数按608个唯一窗口，不乘噪声或模型重复。3A仍遵循冻结的名义0.5秒网格，旧分数不改；“固定1秒lead”应解释为名义两步lead，不是物理时钟严格1秒。已向3A报告追加该限制。3B必须保留原始整数微秒和真实token映射，不能伪造500000微秒时间戳或插值补GT。

## 已准备的人工复核资料

工作区：`outputs/scenario-evaluation/phase3b-preparation-v1/review/`。

- `review.html`与`figures/`：62张GT三联图，按原固定哈希选例，无模型排名。
- `review_working.csv`：75条可编辑任务，初始全部pending。
- `baseline/`：原始CSV、review JSON及608窗口证据副本。
- `README.md`：可编辑字段、允许标签、复核人与修订原因要求。

复制前逐项校验S4d及3A已绑定文件，并比较608窗口的规则、事件、trace和旧标签，全部一致；75个窗口×标签复核键也一致。只允许编辑working CSV，其余文件由review_pack_manifest.json记录静态哈希。没有复制旧completed标记来假装新工作区已完成标签验收。

75条都是规则阳性抽查，不支持召回率或假阴性结论；需要另冻结阴性/不可评价复核队列。窗口级人工确认不自动成为新episode级确认。当前manual_completed仍为0，尚无新导入器或人工标签覆盖3A结果。

## 下一步设计

具体B0设计见[真实时钟与目标track装配设计](superpowers/specs/2026-11-16-scenario-episodes-b0-design.md)。先在新worktree实现pilot目标的时钟/track/窗口映射，再扩到原缓存1967目标；新事件算法和停车评分分开设计，不把近邻窗口连通分量称为canonical episode。

本地完整审计JSON：`outputs/scenario-evaluation/phase3b-preparation-v1/data_audit.json`；包含输入哈希、范围、链检查、时间统计和窗口对齐结果。准备包清单为同目录review_pack_manifest.json。此阶段不训练、不评价holdout、不计算新预测分数、不推送GitHub。

准备验收：69个静态文件哈希通过，75条工作表与baseline初始逐行一致且全pending；12个新审计来源引用复查通过。3A的217个绑定产物、228个冻结源码字节均未改变。本轮没有新增产品代码或测试，不重复声称运行了全套模型评分。

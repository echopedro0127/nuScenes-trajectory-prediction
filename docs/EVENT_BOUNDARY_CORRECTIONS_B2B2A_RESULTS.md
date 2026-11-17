# B2b2a：原始 token 边界修正与独立重建

日期：2026-11-17。分支：`feature/event-boundary-corrections`，工作区：`.worktrees/event-boundary-corrections`。本阶段只有本地开发，不推送GitHub。

## 解决的问题

B2b1能记录“候选需要修正”，却不能用自由文本安全地决定新的停车／起步边界。若直接把修订理由解析为时间，容易产生跨连续片段、单位混淆、首尾删失被错误解除、停车与起步关系错配等问题。

B2b2a增加独立结构化修正层：先校验父判决overlay，只对needs_correction候选建立完整模板，再用同block原始annotation token映射出真实微秒时间。原规则事件、人工判决、修正声明分开保存；没有人工输入时不代填标签。

本阶段验证数据／标签处理链路，不增加模型、训练或预测精度成绩。旧ADE/FDE提升不归因于本阶段。

## 方法与实现

| 问题 | 实现方法 | 实际作用 |
|---|---|---|
| 任意候选被越权修改 | 仅接纳显式B2b1 overlay中的needs_correction；7个来源字段锁定 | pending／confirmed／rejected／uncertain不能借修正表改边界 |
| 时间单位或区间错配 | token→timestamp_us映射限定同role／instance／block；完整校验原token与时间 | 不接手填秒数，不跨缺口插值 |
| 停车时长与事件终点混用 | 分别保留onset、confirmation、end、low_dwell_end | 保留低速持续区间和完整走停区间的不同含义 |
| 观察结束冒充真实结束 | 原right_censored／low_end_censored保持不变，删失端点约束在block终点 | 不把右删失事件用于完整停车时长结论 |
| 单边修正破坏配对 | 检查stop.end=restart.onset及restart.confirmation=end | 冲突整次拒绝，需提供一致修正；不隐式重新配对 |
| 重复或重叠事件 | 有效边界视图按block检查正时长重叠，端点相接允许 | 不按输入顺序选优，也不自动合并 |
| 修改后的边界仍标为规则确认 | 修正视图标corrected_declared，原规则状态保存在source_event | 人工声明不冒充原规则重新验证或官方GT |
| 已驳回关系被混入未决 | rejected_components单列；关系按role/category统计 | 保持已驳回／未决／可用组成的分母语义 |
| 只有checksum，无法确认语义 | 从父证据和同一CSV重新生成全部文件并逐字节比较 | 即便重新计算篡改文件的hash，仍会被独立重建拒绝 |

### 记录与状态

锁定列为source_overlay_identity、source_review_identity、source_event_id、source_event_type、role、instance、block_id。

可编辑列为action、onset_token、confirmation_token、end_token、low_dwell_end_token、reviewer、reason。pending行的token及声明必须全空；propose必须提供完整适用边界及非空复核人／理由。restart的low_dwell_end_token为空。

CSV保持所有eligible行，允许排序，不允许删未决行、重复、外来行、改锁定字段或多列。沿用严格UTF-8 BOM／引号／多行解析、Unicode空白判定和8192码点单元格上限。公开纯函数入口同样重新校验，不能绕过CSV检查传入伪造字典。

原子事件输出source_event、effective_event、status、correction。effective_event中的event_id仍指向原事件，不是重新抽取的canonical事件ID；视图由修正包身份和来源ID共同限定，不能替换原B1a事件表。

复合关系不单独接受人工票：两端均为confirmed或corrected_declared且边界一致才派生derived_declared_components；含rejected则为rejected_components，其余保留unresolved_components。各子事件声明完整保留。

### 来源与发布

父版本必须是完成的B2b1 overlay，模板不能冒充输入。使用显式旧owner的原生CLI、cwd与PYTHONPATH进行验证；新代码副本不能冒充旧源码身份。绑定父静态产物、completed、全部冻结源码和引用。

每次独立verify重新运行原owner验证器；只有父阅读状态计数允许变化，仍要求已知状态、严格非负整数及总任务数守恒。B2a阅读工作表不能因合法阅读修改而使后代静态证据失效，但结构损坏会失败。

模板包、独立可编辑草稿、永久输入快照和不可变修正输出分离。输出不覆盖，completed最后写，失败不产生成功标记。输入原CSV路径／字节／包内副本与规范化修正摘要同时绑定。

## 测试与审查发现

纯规则测试覆盖完整eligible分母、删失、跨block token、逆序／相等边界、同块端点相接／重叠、配对单边冲突／双边一致、非法CSV和Unicode、分组守恒等。

真实B2a包含只有一个GT点的block，它没有速度区间，但仍是合法证据。初版错误要求至少两个点；新增先行失败测试后修正为至少一个点，事件边界仍必须满足严格时间次序。

另一个先行测试发现，修改后的effective_event会继承confirmed_by_rule状态；现已显式标为corrected_declared。只读审查提出的驳回关系分类和关系分组计数也已补充并通过回归测试。

发布层测试使用真实冻结owner程序及合成小数据，覆盖模板／pending导入／人工声明夹具／空eligible版本、原CSV或owner变更、包内副本篡改、重哈希篡改、额外文件、策略变更、父模板拒绝、合法阅读变化／损坏及中断。合成声明不进入真实数据。

最终全套回归：**1117 passed，502.05秒**，日志`outputs/pytest-b2b2a-final.log`。本阶段新增48项（35项纯规则＋13项发布集成）。全仓Ruff、公开仓库扫描、diff检查与只读代码复审通过。首次基线的15项测试因系统Temp访问权限未能执行，改用工作区内basetemp后原58项全部通过；这不是算法失败。一次中途全套运行因落实审查修复主动终止，未计为验收通过，最终计数只来自上述完整回归。

工程交付包括3个生产模块、2个CLI入口、2份测试文件及设计／执行／结果文档。源码冻结提交为`632c3b2`，纯规则首版提交为`014d74a`。冻结后只更新文档与输出，不再修改源码。

## 真实验收结果

真实模板及空修正overlay均独立verify-only通过，`corrections_semantically_replayed=true`。没有实际人工判决或边界修正，不能报告新预测收益。

| 项目 | 结果 |
|---|---:|
| 继承原子事件／复合关系 | 287／60 |
| eligible修正候选 | 0 |
| pending／proposed修正行 | 0／0 |
| 可用声明组成／已驳回／未决关系 | 0／0／60 |
| 人工边界声明 | 0 |
| 父阅读任务pending | 2339 |
| 修正比例 | null，no_eligible_corrections |

两个包各绑定255个源码文件。静态文件数量与字节数不含completed：

| 包 | 身份 | 文件／字节 | 来源引用 |
|---|---|---:|---:|
| template-v1 | `38c1e18b48a971427dfe072fd91811ed9ba7373902f59062ef4ce0ef20fbd22a` | 8／537914 | 2099 |
| empty-v1 | `80a9c6ab5a820cedd022f485f2a88457b82e76677483dec4c336aa0bc3aac1d6` | 11／1734992 | 2100 |

规范化修正摘要均为`4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945`，对应空列表。192字节CSV只有表头，SHA256为`9797ac9af8b11f5dfac89e3d8ebdc8a1da516dd798df2b2adecb206ef6a245d0`。模板CSV、永久输入、包内副本及初始草稿字节一致；草稿路径未进入已发布引用，后续修改草稿不影响该空版本。

保全审计按每个旧阶段自身冻结清单检查了当前源码和静态产物：

| 阶段 | 源码文件 | 静态文件 |
|---|---:|---:|
| B2b1 | 250 | 13 |
| B2a | 245 | 11 |
| B1a | 240 | 12 |
| B0b | 233 | 14 |
| B0a | 233 | 14 |
| 3A | 228 | 217 |

B0a是独立旁支，源码不在本包主链扁平引用中；它使用自身冻结清单额外保全，不能声称本包直接绑定了B0a源码。B2a阅读工作表保留原可变契约，其余列出的静态产物均校验通过。

本worktree的实际产物：

- 验收审计：`outputs/boundary-corrections/b2b2a-acceptance-v1.json`。
- 不可变模板：`outputs/boundary-corrections/template-v1/`。
- 不可变永久输入：`outputs/boundary-corrections/input-snapshots/pending-empty-v1.csv`。
- 不可变空版本：`outputs/boundary-corrections/empty-v1/`。
- 独立草稿：`outputs/boundary-corrections/drafts/boundary_corrections.csv`，当前仅表头。
- 独立重建日志：`outputs/correction-template-verify.log`、`outputs/correction-empty-verify.log`。

## 使用入口与依赖顺序

```powershell
# 在本worktree及既有环境中，PYTHONPATH=src
python scripts/make_correction_template.py --parent ../candidate-verdicts/outputs/candidate-verdicts/empty-v1 --parent-owner ../candidate-verdicts --output-root outputs/boundary-corrections/template-v1
python scripts/import_boundary_corrections.py --parent ../candidate-verdicts/outputs/candidate-verdicts/empty-v1 --parent-owner ../candidate-verdicts --annotations outputs/boundary-corrections/input-snapshots/pending-empty-v1.csv --output-root outputs/boundary-corrections/empty-v1
python scripts/make_correction_template.py --output-root outputs/boundary-corrections/template-v1 --verify-only
python scripts/import_boundary_corrections.py --output-root outputs/boundary-corrections/empty-v1 --verify-only
```

当前B2b1真实父版本全部pending，因此修正草稿只有表头，没有可填写事件行；不能手工向它新增候选。实际使用需要先在B2b1判决草稿中提供真实needs_correction声明并另行发布新overlay，再以新身份重新生成修正模板。

后续顺序：真实候选判决／边界声明；B2b2b新增事件及关系编辑独立设计；B3固定实际查询时刻、GT时域匹配、anchor覆盖与噪声命名空间；最后才是停车时间／位置误差、miss及误停评价。B3的协议准备可以不依赖人工完成，但人工事件指标不能用全pending空版本充数。

本阶段不支持新增事件、重新配对或解除删失，也不提供穷尽负例标签。不下载数据、不调用Cosmos、不修改既有holdout、不根据新结果调模型。

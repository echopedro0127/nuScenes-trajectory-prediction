# B2b1：版本化候选判决导入验收

日期：2026-11-17。**模板与全pending空overlay均完成真实生成及独立verify-only，verdicts_semantically_replayed=true。** 本轮验证导入链路，没有实际人工确认标签，没有新的预测分数。

## 解决的问题与方法

B2a的reviewed表示阅读，不代表确认事件。B2b1新增独立candidate_verdicts CSV，严格区分pending／confirmed／rejected／needs_correction／uncertain，并将每次选定输入发布为不可变快照。规则记录、阅读记录和人工声明三层分离，不改写B1a或B2a。

模板只对227个stop和60个restart建立287行；60个stop_go保留为组成关系，不重复判决。每行锁定源review身份、task/event/type、role、instance和block，仅decision/reviewer/reason可编辑。必须保留全部源行，可排序，但不能删未决行缩小分母。

CSV采用严格解析、共享Unicode空白规则和8192码点上限。pending要求复核人／理由均为零长度字符串；非pending必须有非空复核人与理由。未知状态、重复／缺失／异源行、改锁定列、额外列、非法引号或长度整次拒绝，不跳坏行、不合并多表、不按文件修改时间选最新。

复合关系按rejected→needs_correction→uncertain→pending→confirmed的优先级确定组成状态，保留两行原decision。confirmed_components只是组成一致性，不是第三次人工标注。needs_correction／uncertain／pending不进入确认输出；rejected也不等于全block阴性。原候选边界和删失标志深拷贝保留，右删失确认不能被用于完整停车时长。

## 可变父证明与不可变发布

父B2a只在显式旧owner/cwd/PYTHONPATH做原生verify。父payload仅working_status_counts允许随阅读变化，仍严格检查三个状态键、非负整数和task总和；其余固定字段类型敏感比较，未知字段拒绝。

每次验收还检查当前父工作表结构。所复用的校验器必须与冻结B2a原版相同，仅允许Python通用换行差异；真实新worktree与旧文件确实存在CRLF/LF差异，已按该契约核对。阅读备注字节不作为判决依赖，也不转成标签。

导入前后核对原CSV字节，绑定源路径、SHA256、规范化判决摘要、来源基线、源码和环境；包内保存原CSV副本。输出全部不可变，没有mutable例外。verify从原静态证据及同一CSV重新构造锁定行、分组、派生关系及所有文件，不只验checksum。重排输入行不改规范化内容摘要，但源CSV字节／路径变化会产生不同运行身份。

## 真实验收结果

| 项目 | 模板／空版本 |
|---|---:|
| 原子候选 | 287（stop 227、restart 60） |
| pending | 287 |
| confirmed／rejected／needs_correction／uncertain | 均为0 |
| 非pending人工声明 | 0 |
| 复合关系 | 60，均pending_components |
| confirmed／rejected输出 | 空列表 |
| unresolved输出 | 287 |
| 父B2a阅读状态 | pending 2339，其他0 |

两个包的规范化判决摘要一致：`451eda2c95afb94c42f5bac68f21a389a3df6bc806d5812c5a246204bfe4b00d`。输入快照、包内CSV副本、模板CSV及初始可编辑草稿字节一致；可编辑草稿路径不在已发布overlay引用中，后续编辑草稿不会使空版本失效。

| 包 | 身份 | 静态文件／字节（不含completed） | 来源引用 |
|---|---|---:|---:|
| template-v1 | `ea8b252fca5b5dca3ebfc209a7f13f39a3d23b35e098b4581b05f66f79b7b9bc` | 8／724096 | 1834 |
| empty-v1 | `692ee1506abf23ad1a06d4d11af127a8128e9d28624f7e26ad7e6408168eec30` | 13／1659100 | 1835 |

源码提交`3bb6e96`，分支`feature/candidate-verdicts`，worktree为`.worktrees/candidate-verdicts`；前序纯校验提交`80f657f`。两个包均绑定250个源码文件。验收记录在`outputs/candidate-verdicts/b2b1-acceptance-v1.json`，位于不可变包外。

## 测试与修复

本版本新增58项测试，覆盖25种配对组合、五状态、原删失、未决隔离、CSV完整性和Unicode、来源外键、父可变阅读、strict证明计数、原文件／副本变更、重哈希篡改、输出不覆盖及中断无完成标记。**全套1069 passed，273.74秒**；Ruff、公开扫描和最终只读复审通过。续跑未修改源码，补完了真实模板／空版本独立验收。

复审发现公开derive入口可能绕过CSV校验，已通过RED→GREEN修复：`derive_overlay(data, rows, review_identity)`要求显式可信身份并重建baseline，复用统一记录校验，直接传字典也不能伪造锁定列或空人工声明。角色／类型分组与全局判决分母另做守恒。

合成测试中的确认仅用于代码测试，未进入真实数据。代理没有把规则或阅读记录自动转confirmed，也未代填复核人。

| 旧阶段 | 冻结源码不变 | 静态产物不变 |
|---|---:|---:|
| B2a | 245 | 11 |
| B1a | 240 | 12 |
| B0b | 233 | 14 |
| B0a | 233 | 14 |
| 3A | 228 | 217 |

B2a工作表仍按其可变契约验证，不将它的字节声明为永远不变。其余旧源码和静态证据均核验通过。

## 文件与使用步骤

以下路径相对本worktree：

- **可填写草稿：** `outputs/candidate-verdicts/drafts/candidate_verdicts.csv`。
- 不可变模板：`outputs/candidate-verdicts/template-v1/`，其中CSV不要直接编辑。
- 已发布空输入：`outputs/candidate-verdicts/input-snapshots/pending-empty-v1.csv`，不要编辑。
- 已发布空版本：`outputs/candidate-verdicts/empty-v1/`，全部文件不可编辑。
- GT复核页面位于兄弟worktree：`../scenario-event-review/outputs/event-review/b2a-v1/review.html`。

填写草稿时可按source_event_id搜索网页，或用source_task_id定位对应任务。只填decision/reviewer/reason；不改来源ID，不用B2a的reviewed替代decision。needs_correction仅记录需修正的理由，首版不接受token重画或新增事件。

填写后另存新的永久输入快照，使用新的output-root发布；不要让后续草稿编辑改变已发布版本所绑定的输入。示例命令尚未对真实人工判决执行：

```powershell
# 在本worktree，既有Python环境，PYTHONPATH=src
if ((Test-Path outputs/candidate-verdicts/input-snapshots/reviewed-v1.csv) -or (Test-Path outputs/candidate-verdicts/reviewed-v1)) { throw 'Choose a new version name' }
Copy-Item -LiteralPath outputs/candidate-verdicts/drafts/candidate_verdicts.csv -Destination outputs/candidate-verdicts/input-snapshots/reviewed-v1.csv
python scripts/import_candidate_verdicts.py --parent ../scenario-event-review/outputs/event-review/b2a-v1 --parent-owner ../scenario-event-review --annotations outputs/candidate-verdicts/input-snapshots/reviewed-v1.csv --output-root outputs/candidate-verdicts/reviewed-v1
python scripts/import_candidate_verdicts.py --output-root outputs/candidate-verdicts/reviewed-v1 --verify-only
```

已完成空版本可直接复查：

```powershell
python scripts/make_verdict_template.py --output-root outputs/candidate-verdicts/template-v1 --verify-only
python scripts/import_candidate_verdicts.py --output-root outputs/candidate-verdicts/empty-v1 --verify-only
```

## 限制与下一步

当前只有全pending空版本，没有实际人工确认标签、官方事件GT或穷尽标注；不能报告检测召回率或新增停车预测改善。读取正确的reviewer/reason也不等于认证了人的身份。真实判决仍需用户审阅后提供。

后续B2b2定义结构化边界修正／新增事件与冲突规则；B3独立冻结实际查询时刻、GT时域匹配、anchor覆盖及噪声命名空间，再实现停车指标。本轮无新训练、holdout调参、传感器或数据下载。代码／文档只本地提交，生成数据不进入Git，未推送GitHub。

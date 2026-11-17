# B2b2a：已有事件的结构化边界修正设计

日期：2026-11-17。状态：用户已通过“继续”确认，正在实现与验收；尚无真实人工修正结果。

## 问题与阶段选择

B2b1已经验收287条原子候选的五状态导入，但所有真实记录仍为pending。needs_correction只能说明需要修改，不能用自由文本可靠表达新的事件区间。直接进入停车时间误差评测，会把未确认的规则边界当成行为真值。

可选路线：

1. **推荐：B2b2a先修正已有事件边界**，沿用原始token、完整来源和人工声明；新增事件及关系编辑单独进入B2b2b。范围较小，也能独立验证时间与删失语义。
2. 同时实现修正、新增、删除与关系编辑：功能完整，但同时引入事件身份和冲突消解，验收面过大。
3. 先做B3预测anchor协议：不依赖人工完成，但不能解除当前未决标签，停车指标仍需等待。

本设计选择路线1。它不要求现在填写人工标签；没有真实输入时只验收空修正版本。

## 输入与可信边界

- 显式指定一个经原owner验证的B2b1 overlay及其B2a静态来源；禁止自动找“最新版本”。模板包不能冒充判决overlay。
- 仅接纳该overlay中decision=needs_correction的已有stop_candidate/restart_candidate；pending、confirmed、rejected、uncertain均不可被修正表越权改写。
- 修正必须附reviewer/reason。人工声明不是身份认证，不把模型生成意见登记为人工结果。
- 以同一role、instance、block的真实annotation_token选择边界，由冻结GT映射出timestamp_us；不接收手填浮点时间或跨缺口插值。
- 不改变B1a/B2a/B2b1，不覆盖原始事件；输出追加来源与修正层。

## 两层数据与状态

每个eligible候选生成一条固定身份行：source_overlay_identity、source_review_identity、source_event_id、source_event_type、role、instance、block_id；均不可修改。

可编辑字段为action、onset_token、confirmation_token、end_token、low_dwell_end_token、reviewer、reason。

- action=pending：所有token、reviewer、reason均为空。
- action=propose：必须填完整适用边界及非空reviewer/reason。它表示人工提出的修正声明，输出状态为corrected_declared，不提升为官方GT或穷尽标注。
- restart不适用low_dwell_end_token，要求为空。
- 停车低速段边界与完整事件end分开保存，不用一个end字段冒充停车时长。

必须包含所有eligible候选，允许重排行；不允许缺行、重复、外来行、额外列、隐式合并或只选容易修正的子集。沿用B2b1的严格CSV、Unicode空白与单元格长度限制。

## 边界和删失语义

所有token必须属于冻结block，时间严格按实际微秒排序。要求onset < confirmation <= end；停车另要求confirmation <= low_dwell_end <= end。

原right_censored和low_end_censored不可由修正表直接改成false。删失边界必须仍指向原block观察终点；禁止把观察终点解释为真实运动终点。解除删失或扩展观察范围属于后续独立协议。

修正可以不满足原自动候选的速度阈值，因为这是人工边界声明；仍保存原速度证据与规则结果，不将人工声明伪装成规则重算结果。声明的时间精度受真实token采样限制。

## 关系和冲突

不独立编辑stop_go，不新增配对，不自动把restart改绑其他stop。

涉及一条既有复合关系时，从原事件及修正声明构造“有效边界视图”，并检查同block、stop.end=restart.onset、restart.confirmation=restart.end等原关系契约。单边修正不能破坏另一端；有冲突整次导入失败，要求同时给出一致的修正，或转入后续关系编辑协议。

未决子事件保留未决；corrected_declared与B2b1的confirmed分开计数。只有两个子事件均有可用声明且时间关系一致时，才输出derived_declared_components，不冒充第三次独立人工确认。

审查澄清：只要子事件包含rejected，关系单列rejected_components，不能混入unresolved_components；关系总分母和各状态另按role/category报告守恒计数。

同一block内同类有效事件不允许重复区间或正时长重叠；端点相接允许。stop/restart之间仅允许原配对规定的边界连接。检测到关系歧义或不同候选修正为同一事件，不按文件顺序择优、不自动合并，整次拒绝。

## 发布与重建

沿用模板、可编辑draft、独立永久输入快照、不可变输出四层分离。输出至少保存protocol、run、原CSV副本、规范化修正记录、原事件与有效边界视图、关系状态、分组计数、source references、completed。

同时绑定原CSV路径/字节哈希和规范化内容摘要；源码、配置、环境及父证明纳入身份。输出目录不得覆盖，completed最后写。失败不留下可被误识别为成功的完成标记。

verify-only必须从原父数据和同一输入重建token映射、边界、关系、分组和全部产物；仅检查checksum不算通过。父B2a的阅读工作表仍可变，继续只绑定静态证据并验证当前工作表结构。

## 验收与异常

- 标量小夹具：stop/restart正常修正、同token等值、逆序、删失边界、同block端点相接与重叠、复合关系单边冲突和双边一致。
- 身份攻击：错role/instance/block、外来token、跨缺口、修改locked字段、不是needs_correction的候选、源表漏行/重复/多列。
- 发布完整性：源CSV中途变更、原文件/包内副本篡改、重哈希篡改、输出已存在、中断无completed、父阅读状态合法变化。
- 零eligible输入必须生成可验证空版本，状态no_eligible_corrections，计数为0；不能据此宣称人工修正已完成。
- 摘要按role/category/event_type报告eligible、pending、proposed、censored及关系状态；零分母输出null及原因，不输出虚假成功率。
- 真实验收保护旧五阶段及B2b1冻结包；不重训、不调holdout、不下载数据、不计算新的预测成绩。

## 实现顺序与退出条件

1. 新隔离worktree；固化父overlay身份与基线，旧owner只读。
2. 纯函数：eligible集合、严格记录校验、token映射、边界/删失/关系冲突检查，先测异常。
3. 发布入口：生成模板、导入修正、独立verify；复用冻结来源契约，保持修正语义独立。
4. 小夹具与全套回归、静态检查、公开文件扫描；空输入真实发布及独立verify，记录实际计数。
5. 用户提供真实修正后另发布新版本；新增事件/关系编辑在B2b2b定义，之后再冻结B3实际查询时钟、GT目标匹配、anchor覆盖和噪声命名空间。

本设计不承诺修正数量或精度提升。边界：无新增事件、无删除/重新配对、无解除删失、无穷尽负例标签、无停车指标。设计通过后再写逐项实现计划。

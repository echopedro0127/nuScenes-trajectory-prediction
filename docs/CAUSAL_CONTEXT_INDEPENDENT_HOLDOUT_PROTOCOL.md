# P6.6C/8 一次性holdout协议与模型冻结

日期：2026-11-09。接续[独立一致性重训](CAUSAL_CONTEXT_INDEPENDENT_CONSISTENCY_RESULTS.md)。
本阶段已冻结模型、划分、指标与统计规则，**没有运行holdout推理或评分**。这份锁是后续执行的输入，不能视作holdout验证成果。

## 固定模型与主结论

- 主模型：本轮独立RH，best epoch10，checkpoint ee957f2f…f5d3c05。
- 主对照：每0.5秒释放标注位置后的因果CV。
- 次要对照：original、map、quality_gate、revision_adaptive。全部固定报告，不据holdout改换主模型。
- revision_0与RH的权重和指标完全一致，只作训练控制，不重复视为独立模型。

选择主模型参考了已见select，理由为保留RH且不因一致性臂0.274%的微小jitter收益增加正则；这不是对整个研究的事前注册，也不构成新采集盲测。
原始数据为现有固定重划分：holdout3,329轨迹、73scene、6log。历史结果曾被研究者看到，须保留这一限制。

## 锁定的评价规则

完整不可变规则见[设计协议](superpowers/specs/2026-11-09-independent-holdout-design.md)。
history2s、future6s、2Hz、11次更新、原始固定终点，沿用六种地图包条件、既有特征和回退规则。没有传感器特征，也没有新训练。

主指标为六条件等权的在线ADE，主比较为RH-CV的ADE绝对差。
log级配对cluster bootstrap：抽6个log有放回、5000次、NumPy PCG64 seed20271109；按所抽log的总误差/更新数池化均值，保留重复抽样次数。
95%区间使用线性分位数0.025/0.975。仅当差值区间上界<0时表述该holdout上的改善证据，否则报告“未证实优于CV”。
FDE、jitter及所有次要模型/条件均完整描述，不做显著性筛选或事后改主检验。
报告逐log结果；只有6组，bootstrap区间可能不稳定。不能以大量相关回放更新替代独立log数或跨seed证据。

冻结后不得根据holdout误差重新调参、选checkpoint、筛场景或更改主模型。运行崩溃可按相同锁恢复并保留日志；评价bug修复须解释并重算全部固定模型，不能只改有利项。

## 机器可验证锁

本地输出：outputs/context/p66-independent/holdout-protocol/protocol.json及completed.json。

**协议identity：adbc5d7c948a57c7e76173a54cb481d551842b8b80416309d8a823327ccb1231**。

protocol.json文件SHA256：038f157fb42cadb4c5abcd399b25e72064622a1e73ead7a1d47de447e58d8c47。锁定33个实验文件及130个源码/配置/设计文件。

协议正文包含数据/配置/划分身份、所有模型相对路径及SHA256、完成和审计记录、执行源码/配置与设计协议哈希。
所有已选模型会自动加入文件校验集合，并验证模型清单与文件集合一致，不能只锁名字或摘要。
验证同时检查内容identity、protocol.json文件哈希、父文件和源码哈希；拒绝覆盖及实验引用路径逃逸。
source相对路径允许指向实验根之外的仓库代码，artifact引用则必须位于实验根内。

冻结前已重验split机械来源、GRU训练角色、map/gate/RH/一致性模型完成marker、初始化与权重依赖，zero精确复现RH。
确认真实holdout尚无predictions/context/candidates/evaluation及started产物。锁中false字段表达“冻结时尚未评价”，不代表今后每次verify时都未评价。

```powershell
python scripts/freeze_independent_holdout.py
python scripts/freeze_independent_holdout.py --verify-only
```

生成会拒绝已有目录；日常只运行verify-only，不覆盖已有协议。运行时从实验根解析模型引用，不依赖个人绝对目录。

## 测试与实现

初版be217d1；额外反例发现并修复基础模型仅列于清单但可能未进入文件验证的问题，最终实现91e22c8。
新增7项测试覆盖锁发布/重验、协议内容及父模型/源码篡改、路径逃逸、覆盖拒绝、已有holdout产物拒绝及所有已选模型强制校验。
最终全量 **573 passed in 187.84s**，Ruff通过；此前公开扫描通过，交付前再次扫描文档。
真实freeze与独立verify均成功，后续再次重验所有模型均包含在artifact集合，且未产生holdout推理产物。

## 下一阶段

实现只读holdout输入准备和评价入口，并在第一次推理前将新执行源码清单绑定本协议identity。
既有独立候选loader故意拒绝holdout，不能通过重命名为select或放宽旧检查绕过；需要单独验证与fit/select的scene/log互斥及固定新上游。
执行时先发布started记录，最终发布完整模型/条件/指标、log统计和completion。任何结论都在完成这一步后才产生。
本轮不推送GitHub、不改简历，不替换历史公开推荐，P7未启动。

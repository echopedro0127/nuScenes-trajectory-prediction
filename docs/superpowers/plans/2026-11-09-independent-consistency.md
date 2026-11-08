# Independent Innovation Consistency Plan

**Goal:** 隔离链上重训revision_0/revision_adaptive，核验正则效果与RH零正则复现。
**Architecture:** 新独立wrapper复用InnovationDataset和train_consistency，不改旧训练/损失源码；独立模型格式、输入/初始化来源及完成marker。
**Tech Stack:** PyTorch、NumPy、pytest。

用户已授权继续。保留原P6.4训练方法：从本轮新标量gate同起点训练RH结构；revision_0 λ0、revision_adaptive λ0.02。不是从已训练RH继续微调，不冒用历史consistent_010名字。
两臂seed2027、最多10epoch、patience3、batch4096、lr0.001、AdamW wd0.0001、smoothness0.05、337参数；按select平均ADE选模。自适应权重exp(-(innovation/0.5)^2)，仅用已释放观测与此前CV预测。训练均值只从fit计算。
要求revision_0全部权重与本轮RH一致，排除额外轮数/随机初始化差异；adaptive相对该对照分析。未跑固定/均值匹配正则臂，不主张创新权重优于其他正则。
不追加预算，不改历史推荐，不运行holdout。

- [ ] 写端到端测试并确认RED：新gate/RH来源、短双臂训练、zero精确复现RH、错角色/身份与覆盖拒绝。
- [ ] 独立训练/重载入口；绑定新gate与RH、候选合同和源码，重载指标/缺包回退并发布marker。
- [ ] 全量pytest、Ruff、公开扫描通过后冻结源码，执行两个真实训练臂。
- [ ] 独立重载与实际数据身份/完整指标重算，报告ADE/FDE/jitter及条件差异，保留单seed选择集边界。

输出outputs/context/p66-independent/consistency-training/seed-2027/{revision_0,revision_adaptive}；完整配置冻结后再考虑holdout。

# Independent RH Training Plan

**Goal:** 从新quality gate初始化RH，沿新fit/select重训并验收。
**Architecture:** 新independent_horizon_training模块复用既有HorizonDataset、HorizonGate、train_horizon/measure_horizon；不修改历史绑定源码。独立模型格式与loader绑定新候选和新gate，拒绝已有输出。
**Tech Stack:** PyTorch、NumPy、pytest。

用户已授权继续。预算固定seed2027，RH单臂，337参数/19输入/hidden16，最多10epoch、patience3、batch4096、AdamW lr0.001、wd0.0001、smoothness0.05、revision_weight0、按select平均ADE选模。
初始化使用本轮新gate（69f49457…87bd8），新特征权重初始为0；训练前核对初始RH与标量gate的剩余预测权重/指标一致。R只使用此前预测在当前已释放位置上的已实现误差，H只描述剩余预测时长与候选差异，不读未来标签。
只重训RH，不重做八结构消融/容量匹配，不增加一致性正则；不计算holdout指标，不改推荐。

- [ ] 写测试并确认RED：实际新gate初始化、短RH训练/重载、初始权重一致、错身份/角色和覆盖拒绝。
- [ ] 新独立入口/loader，验证新gate来源、候选与场景/log隔离、初始化指纹；封装模型、完整重载指标与marker。
- [ ] 全量pytest、Ruff、公开扫描通过后冻结源码，运行真实RH训练。
- [ ] 独立重载复算选择集ADE/FDE/jitter及六条件，核对缺包CV回退，报告限制。

输出outputs/context/p66-independent/horizon-training/seed-2027/RH；后续一致性仍需重训，holdout封存。

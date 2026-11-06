# P6.6B 保守精度约束 checkpoint 选择实验

日期2026-11-06；父提交1112c7e。用户已要求继续实验。

## 问题和固定预算

P6.6A发现train_val上的保守选模空间，但只复用了旧best checkpoint。本轮直接在训练
过程中保存每个epoch，比较相同训练轨迹的最低ADE与约束下最低jitter两种选择规则。
冻结一个arm adaptive_matched_005，seed2027/2028/2029，固定每seed10epoch，共3模型/
30epoch。冻结batch4096、AdamW lr.001/wd.0001、clip5、SmoothL1+.05 smoothness、
创新权重及alpha=.05/mu；mu仅从完整train重建。无早停、无额外强度或结构搜索。

初始化、shuffle与P6.5同seed同arm严格一致；从P6初始化开始训练，不在旧best上微调。
固定RH架构与候选缓存。CUDA初始化前设置CuBLAS=:4096:8。

## 数据与选择规则

仅train训练、train_val逐epoch评估，不读取val数据或指标。位置仍为干净已释放标注，
六条件是地图包可用性变化，不代表新的真实传感器或跟踪噪声实验。

每seed单独以该seed的冻结P6.3 RH作参考：overall与clean ADE <=1.0005*RH，
overall FDE<=1.0025*RH，clean FDE p95<=1.005*RH。该逐seed规则比P6.6A跨seed均值
规则更保守，事先固定而不根据运行结果改变。

ADE规则：在10个新epoch中按overall ADE、epoch升序选择。
约束规则：在全部满足四项约束的epoch和RH参考(epoch0)中，按jitter、ADE、epoch升序
选择。无合格新epoch就选RH，不把强制选择训练checkpoint伪装为可行解。

保存每epoch五项统计及state fingerprint。对RH、ADE-best、constrained-best重新进行
完整train_val的measure_horizon批量推理，精确核对统计、硬CV回退、checkpoint身份。
ADE-best还须与P6.5相同arm的旧best模型state fingerprint及五项统计核对，检查本轮
训练是否确实仅改变选择规则。若不匹配，实验归因验收失败，不更改容差掩盖差异。

## 来源与产物

加载冻结P6.5 selection固定SHA256，复用P6.6A父表重建；真实train/train_val缓存由
load_verified_gate_cache校验，加载旧strength模型及RH走原生产loader，校验父契约、
初始化、来源、场景、hash。不得修改旧P1至P6.6A源码、配置、测试或输出。

新增online/precision_checkpoint.py、scripts/train_precision_checkpoint.py和独立测试。
复用旧run_consistency_epoch、HorizonGate、measure_horizon；不修改全局变量或旧选模。
run身份绑定协议/当前及父源码、seed/超参、父checkpoint/selection、实际train和
train_val指纹、候选契约。保存所有epoch.pt、log、选择结果、三策略metrics及完成marker。
输出为全新outputs/context/p66-checkpoint；拒绝覆盖已有目录。

完成后独立重载三个被选模型，完整重算逐点metrics并核对保存值；检查所有epoch文件
哈希、state fingerprint和来源。检查可揭示结果的日志后仍不能改规则或追加epoch。

## 结论边界

这轮有真实新训练，但train_val已用于开发，选择出的指标存在选择偏差。结果只能
证明当前数据上的选模行为和训练轨迹复现，不能作独立泛化或新val收益声明。
若两规则选出同一epoch，明确报告没有新增选模收益。若约束规则退回RH，也如实记录。
不因运行结果自动更换当前推荐、读取val、推送远程、改简历或开启P7。

测试覆盖精度/clean/尾部约束、RH回退、稳定tie、非法记录、checkpoint哈希/模型状态
篡改、场景重叠、完整保存/重载、实际训练loss路径与旧训练轨迹一致性。
全量pytest、Ruff、公开扫描通过后冻结源码，再启动30epoch；最后记录预算与结果。

# B3b2b：源键离群观测与真实时钟 NIS 降权实验

状态：用户已批准并实现；28组合正式实验、独立重放/重评分、外部审计及全套1289项测试通过；最终结果见[阶段报告](../../REALCLOCK_PHYSICS_OUTLIER_B3B2B_RESULTS.md)。

## 目的与已有证据

B3b2a 在相同源包、真实时间间隔、相同无噪声 GT 下证明：select、1 s、三噪声 seed 均值，0.5 m Gaussian 时 actual KF 比 actual CV 误差低 38.83%，1 m 时低 50.91%。但 clean 中 KF 比 CV 更差，1 m 时 actual CV 比 fixed-dt CV 略差。因此不能预设某种滤波策略始终更优。

本阶段回答：在少量源位置出现突跳时，使用当前新息计算的固定 NIS 降权能否减少预测污染？它会否对没有注入异常的真实运动也持续降权，付出预测误差代价？这不是新学习门控，不代表真实感知误差评测。

父报告：[B3b2a 实验报告](../../REALCLOCK_PHYSICS_GAUSSIAN_B3B2A_RESULTS.md)。父源码提交 `73070d2`，文档提交 `91251b8`。新实现必须在新的分支和 worktree 内，保持 Gaussian、clean 与之前各 owner 的源码、输出不变。

## 备选方案与取舍

1. **采用：固定 NIS 自适应测量方差。** 沿用历史 Phase 1 的 9.21 / 25 参数，但单独实现并验证 actual-dt 版本；不搜索阈值，控制变量清晰。
2. 硬拒绝观测或切换 CV：拒绝真实机动可能恶化，增加状态恢复和切换边界；留作后续消融。
3. 学习可靠性门控：需要训练、候选和数据开发隔离；本阶段先确定检测信号是否具有增量价值。

## 实验矩阵与共同输入

沿用 B3a / B3b2a 的 22,231 个开发窗口、fit/select 角色、真实时间、有效 GT 配对和 0.5 / 1 / 2 s 提前量。不得读取 holdout、重新抽样窗口或按离群标记删除预测。

每策略每条件每 seed 应保持 244,541 个 query 和 511,854 个有效位置比较。若不符，验收失败，不通过重新选择窗口修补。

四策略：`cv_actual_dt`、`kf_actual_dt`、`kf_nis_actual_dt`、`kf_nis_fixed_dt_state_control`。前两策略必须精确复用既有物理实现；NIS 的 fixed-dt 控制只更改状态初始化和同化 dt=0.5 s，预测仍使用真实 query 至 target 时间差。

三个条件：clean、Gaussian 0.5 m、Gaussian 0.5 m + 稀疏 5 m 离群位置。三个噪声 seed：2027、2028、2029。

两个带噪条件 × 三 seed × 四策略 = **24 个组合**；clean × 四策略只跑一次，共 **28 个唯一组合**。clean 不复制成三份统计数据。1 m Gaussian 不纳入主矩阵，不根据结果追加有利条件。

噪声 seed 是观测扰动重复，不是训练 seed。所有策略接收完全相同的世界坐标观测；sigma 和离群标记不能输入估计器。

## 源键离群生成

Gaussian Z 完全复用 B3b2a `normal_vector(source_key_digest, seed)`，不得改变 schema 或 RNG。仅为离群另建独立通道。

离群 key 为 canonical JSON：`schema=realclock_outlier_packet_b3b2b_v1`、`source_key_digest`、`noise_seed`、`channel`；按键排序、紧凑分隔符、UTF-8、禁止 NaN。channel 分别为 `occurrence`、`direction`。使用 SHA256，不依赖 Python hash、全局 RNG、window_id、query、策略、sigma、GT 数值或机动类别。

对 occurrence 哈希首 8 字节按无符号大端解释为整数 u，且仅当 `u < floor(2**64 / 20)` 时注入离群，名义概率约 5%。direction 哈希首 8 字节为 v，角度 `theta = 2*pi*(v / 2**64)`；离群向量为 `5*[cos(theta), sin(theta)]` m。事件和方向通道独立，不重抽、不强制每窗口恰有异常。

每个唯一源键、每 seed 的离群标记和向量冻结一次。同一源包在重叠窗口中必须同标记、同方向；不得沿用窗口内第 4/8 步注入方案。角色属于源身份，不跨角色拼接。

世界观测为 `z_i = GT_i + 0.5*Z_i + outlier_i`。clean 不含任何扰动。Gaussian-only 与 outlier 条件的 Z 完全相同，因此两者差异只来自额外离群向量。

索引 0..15 都允许注入，包括两点初始化和暖启动；索引 16 仍不得同化。初始化异常不能被偷偷排除，需要作为边界单独统计。未来包的异常信息只有评价器可以离线查看，估计器只逐时刻接收已到 XY 与时间。

局部原点为本条件第一个带噪观测。保存全量扰动 D 时，局部观测为 `g_i + D_i - D_0`，评价 GT 为 `g_i - D_0`。GT 永远保持原始世界坐标真值，不能把 outlier 同时加到评分真值。

## NIS 状态核与回退含义

状态 `[x,y,vx,vy]`、q=1、基础 R0=0.25 I、两点初始化及 P 与 clean 物理核一致。真实时间间隔从 int64 微秒差得到，不四舍五入成 0.5 s。

在每个 j=2..15 的已到观测上先预测 x-、P-，然后：

```
innovation = z_j - H*x_minus
S0 = H*P_minus*H.T + R0
nis = innovation.T * solve(S0, innovation)
scale = min(25, max(1, nis/9.21))
R_effective = scale * R0
K = solve(H*P_minus*H.T + R_effective, H*P_minus).T
x = x_minus + K*innovation
P = (I-KH)*P_minus*(I-KH).T + K*R_effective*K.T
P = (P+P.T)/2
```

仅用基础 R0 的一次 pre-update NIS；不得用更新后残差、未来 GT 或重新计算的 NIS 迭代调参。触发定义严格为 `nis > 9.21`，恰等于阈值不触发；不截断观测，不重置状态，不硬切 CV。

这里的回退是降低当前测量权重、更依赖运动学预测；不是断言已实现学习门控自动切换或完全排除异常。初始化两点没有 pre-update NIS，保存为不可检测，不用 0 冒充有效 NIS。

阈值 9.21 只沿用历史参考值；本阶段不把报警率解释为理论 1% 误报率。q/R 失配和真实机动都可能使残差升高。

算术失败、奇异 S、非有限状态、非有限 NIS 或明显负 NIS 均终止该次生成，不跳窗口或静默输出 CV。对舍入级负值的容忍范围须在实现前测试中明确为 -1e-12，范围内置 0，范围外报错。正时间、严格递增与输入 dtype 验证沿用父协议。

## 评价与新增诊断

主结论预先指定为 select、1 s、Gaussian+outlier 条件，`kf_nis_actual_dt` 对 `kf_actual_dt` 的窗口宏平均误差配对差值；其他 lead、fit 和 fixed 控制完整报告，不只选最好值。

保留原有每 seed、逐窗口、逐 log、log 宏平均，P50/P95 与 E>2 m 比例；跨 seed 先同窗口均值，再窗口均值。尾部分位数保留每 seed 值及其均值/样本标准差，不将池化分位数冒充种子均值。

必须报告三个控制差值：NIS-KF−KF（鲁棒更新增量）、outlier−Gaussian（额外离群代价）、actual−fixed（时间适配差异）。正差为退化，不裁为零。clean、Gaussian-only 下 NIS 的代价必须与收益并列。

新增诊断全部只用于评价，不作为模型输入：

- 每个 j=2..15 保存 pre-update NIS、R scale、报警、时钟与可用性；query 输出仍为 j=5..15。
- 源包级注入率：唯一 source/seed 去重后计数，另报窗口观测暴露次数，避免重叠窗口重复计数被解释为独立异常数量。
- NIS 诊断以“窗口内一次滤波更新”为单位，因为相同源包在不同窗口有不同历史状态；明确不能与唯一源包分母混用。
- 对 j>=2 的更新，按当前是否有合成离群统计 TP/FP/FN/TN、precision/recall/FPR 及有效分母；无分母返回 null+reason。初始化两点不计检测 FN。
- FP 只表示“没有合成离群时的报警”，不声称真实传感器异常误报。clean 单独叫机动/模型残差报警。
- 恢复观察：只对 query 区间 j=5..15 中离群更新建立事件；检查随后 k=0,1,2,3 个已到包，且从 j+1 到 j+k 没有新离群，才纳入 k 的有效集合。固定 1 s lead 且父 GT valid 才评分；每 k 保留数量、原始误差和 NIS-KF−KF。k 表示到达包数，同时报告真实 elapsed 秒，不伪称固定 0.5 s。
- 连续离群、末尾离群、无有效目标和截断恢复窗口分别计数。按 window/event 计数，不当作独立物理事件；先每窗口汇总，再跨窗口宏平均。
- 窗口按初始化0/1是否有离群、query前暖启动2..4是否有离群、query5..15是否有离群做可重叠暴露诊断，并报告交集数量。不用未来暴露标签筛选算法。

恢复曲线用于配对描述，不把某个阈值首次达标时间称为恢复时间；本轮不引入事后阈值。无额外人工 GT，不宣称官方 ADE/FDE 或新的盲测泛化。

## 实现依赖与产物

顺序固定：父链证明与源清单 → 源键离群表 → 含原点偏移的数据装配 → NIS 核和独立标量实现 → 检测/恢复指标 → 流式正式生成 → 独立验收 → 结果文档。

预计新增 `robustness/source_outlier.py`、`robustness/realclock_nis.py`、`evaluation/outlier_data.py`、`evaluation/outlier_metrics.py`、`evaluation/physics_outlier.py` 及对应测试、CLI。源码拆分以职责为界，不改 frozen owner。复用只读 Gaussian 生成器与物理基线。

输出使用全新 `outputs/realclock-physics-outlier/b3b2b-v1/`，记录协议、父 proof、source/ref hashes、运行环境、唯一源扰动表、窗口映射、观测/原点/GT有效掩码、各组合状态/协方差/预测/误差、NIS诊断、逐seed和聚合结果、验收证明。completed 最后写，不覆盖已有目录。数据忽略于 Git，不推送。

Gaussian 父由原 owner CLI 独立 verify，绑定 identity `6eca3e0cabfd216d00d007ac3f3673a52aedb52866344ef112d3c9595c073551`。clean 另绑定 B3b1 原 owner。复制进新分支后不得用新源码“代验”旧包。

## 验收边界

1. 源键黄金向量、跨窗口/策略/运行一致，通道与 seed 隔离；全局 RNG 状态不变。
2. 修改未来观测不能改变过去状态；修改 GT 只能改评分；状态接口不接异常标记/GT/sigma。
3. 初始噪声与离群原点变换经世界坐标独立检查；GT不受污染。
4. clean 的 CV/KF 与 B3b1、Gaussian 的 CV/KF 与 B3b2a 完整精确回归；Gaussian seed 共享不变。
5. NIS<=9.21 时与对应普通 KF 更新一致；阈值边界、R cap、极端残差、真实加减速/转弯、初始化异常与连续异常均有测试。
6. 独立标量核复核固定哈希选出的 fit/select 各16窗口，全时刻状态/NIS/scale/协方差 at most atol=rtol=1e-10；检查协方差对称和 PSD 容差 -1e-10。
7. 保存的预测从原始 GT 重评分，重新计算每 seed、聚合、配对、检测和恢复统计；故意篡改任何组统计须拒绝。
8. 所有28组合 GT mask一致，不能对报警/无报警分别改变目标配对。空组用 null+reason，不使用0冒充均值。
9. 流式写盘、单进程固定线程策略；不将批量生成耗时称在线延迟或 FPS。测试、公开仓库扫描、父包保护验证全部通过才能报告完成。

## 完成判断

完成以协议和验收通过为准，不以 NIS 必须胜出为准。若 NIS 伤害真实机动、初始化异常难以恢复或 fixed-dt 更优，完整保留；后续依据这些边界讨论更好的 q/R 建模或可靠性学习，不能据本轮 select 成绩事后改阈值再冒充预注册结果。

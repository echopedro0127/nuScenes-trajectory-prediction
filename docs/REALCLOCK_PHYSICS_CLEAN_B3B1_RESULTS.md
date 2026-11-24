# B3b1：真实dt的CV／KF clean对照实验

日期：2026-11-24；实施从2026-11-23延续。分支`feature/realclock-physics-clean`，工作区`.worktrees/realclock-physics-clean`。结果只解释理想标注位置流上的预测，不是感知／跟踪或闭环驾驶成绩。

## 从评估准备进入预测对照

B3a已经把真实查询时刻与GT目标冻结。B3b1在同一批窗口、同一批目标上运行四个物理策略，回答两个不同问题：真实时间差是否改善状态估计；真实时间差下KF是否比最近两点CV更适合clean轨迹。

| 策略 | 状态初始化／同化时间 | 评价时的目标提前量 |
|---|---|---|
| cv_actual_dt | 相邻已到位置的真实微秒差 | 同一实际GT目标 |
| kf_actual_dt | 同上 | 同上 |
| cv_fixed_dt_state_control | 每次固定0.5秒 | 同上 |
| kf_fixed_dt_state_control | 每次固定0.5秒 | 同上 |

固定dt组只用于状态同化时间的消融，不是旧3A完整输出的重命名；它们也在相同实际lead外推。因此比较不会被“各自预测到不同时间点”混淆。

## 方法与因果链路

1. 从冻结B3a加载窗口顺序、17点真实时间和GT匹配，按role／annotation token从原B0b提取float64世界坐标，再减去第一个已到观测作局部原点。
2. 状态核只接前16点。第0、1点初始化，第2..4点逐包暖启动，第5..15点每到一个包同化一次并保存查询状态；第16点只作评价目标，绝不同化。
3. CV用最近两点除以该组dt求速度；KF用[x,y,vx,vy]、连续白加速度Q、R=0.25I、q=1和Joseph covariance更新。参数沿用旧算法，未根据fit/select分数调参。
4. 先冻结各查询的p、v，再由评价器按B3a matched target_us采样连续函数p+v*lead。未来GT时间／位置和事件标签不进入状态估计器。
5. 四策略共享完全相同valid集合，逐查询保存位置误差；按窗口／log聚合，并计算同目标配对差。

clean没有注入噪声，R=.25是KF假定的测量方差，不是实际加了0.5m扰动。CV没有建模不确定度，分片中的全零covariance只作占位，协议明确不可解释为确定性置信区间。

## 指标与分母

误差名称为position_error_at_matched_gt，单位米。名义lead为0.5／1／2秒，匹配实际提前量在父容差内；它不是官方6秒ADE、FDE或多模态Miss Rate。

每组先对窗口内有效查询求均值，再窗口均权；fit/select分开。另报告逐log结果、log宏平均，以及池化query的P50/P95和E>2m比例。重叠窗口和重复源token不是独立驾驶样本，不据此做显著性断言。

配对增益为reference误差减compared误差，正值表示compared更好：cv_dt比较fixed CV→actual CV，kf_dt比较fixed KF→actual KF，actual_kf_vs_cv比较actual CV→actual KF。无效lead窗口保留mask与原因；无效数组位置以0占位，不能当作零误差命中。

## 工程验证与问题修复

- 时间核：不规则匀速解析CV、规则0.5秒下actual/fixed一致、KF独立逐包标量参考、历史暖启动、未来观测扰动不影响此前状态、非法时钟／shape／dtype／非有限输入拒绝。
- 评价层：未来GT变化只改评分，不改状态；按同一目标连续取样；窗口均权与log宏平均区别、池化分位数、零分母及paired mask一致性。
- 发布层：原B3a owner证明、源文件变更、重哈希改预测／状态／GT／策略、额外文件、中断无completed和输出不覆盖。
- 审查补强：给向量核统一加有限偏差的RED用例证明“同一核重放＋保存预测重算”不足以独立验证状态。现正式prepare和verify都按固定SHA256规则每role最多抽16个窗口，调用独立H矩阵＋inverse逐包参考，核对四策略状态与covariance，atol／rtol均为1e-10；样本身份和检查数保存在scalar_reference.json。
- 运行环境：原先仅写threads=2会误报实际线程数。现run据实绑定OMP／MKL／OpenBLAS请求；正式进程在启动前将三者设2，并用外部threadpool_info核验实际OpenBLAS，不增加生产依赖。

最终全套回归：**1213 passed，870.16秒**，日志`outputs/pytest-physics-final.log`；新增45项（19项状态核、8项GT评价、6项指标、12项发布／运行信息）。Ruff、公开扫描、diff检查与只读复审通过，两个审查问题均已关闭。源码冻结提交`e1f3f39`；冻结后不根据真实分数修改算法或参数。

## 全缓存实测结果

真实发布、独立重放、保存预测重算及正式标量参考均通过：physics_clean_semantically_replayed、saved_predictions_independently_rescored、scalar_reference_verified均为true。运行身份为`272dd11799b4a545882578cfb8866be619e09d25e1609db15fa7a08e5e6feb86`。

每策略22231窗口、244541查询、733623匹配尝试、511854个有效query-target比较，四组共用相同有效集合。fit为14445窗口，select为7786窗口；2秒lead分别只有14355／7730个窗口存在有效GT匹配，其余保留not_evaluable，不是算法失败或丢弃坏预测。

### 窗口平均位置误差（m）

先窗口内平均有效query误差，再窗口均权。下表越小越好，所有四组完整保留：

| 角色 | 策略 | 0.5s | 1s（主要比较） | 2s |
|---|---|---:|---:|---:|
| fit | CV fixed-dt state control | 0.265437 | 0.666735 | 1.914095 |
| fit | CV actual-dt | 0.233273 | 0.605452 | 1.800407 |
| fit | KF fixed-dt state control | 0.475468 | 0.958664 | 2.361503 |
| fit | KF actual-dt | 0.468491 | 0.945304 | 2.332566 |
| select | CV fixed-dt state control | 0.280048 | 0.711013 | 2.035658 |
| select | CV actual-dt | 0.240652 | 0.633016 | 1.899615 |
| select | KF fixed-dt state control | 0.503836 | 1.024706 | 2.534617 |
| select | KF actual-dt | 0.492380 | 0.998683 | 2.479966 |

### 时间适配带来的配对改善

相对下降=(fixed-control均值−actual均值)/fixed-control均值，沿用同一有效集合。以下仅为select开发验证角色，不是新盲测holdout：

| 算法 | lead | 绝对下降(m) | 相对误差下降 |
|---|---:|---:|---:|
| CV | 0.5s | 0.039396 | 14.07% |
| CV | 1s | 0.077997 | **10.97%** |
| CV | 2s | 0.136043 | 6.68% |
| KF | 0.5s | 0.011455 | 2.27% |
| KF | 1s | 0.026022 | **2.54%** |
| KF | 2s | 0.054651 | 2.16% |

1秒lead的有效query为fit 114079、select 61278。其余lead分母继承B3a，不按策略变化。select的1秒log宏平均同样从CV fixed的0.711357降到actual的0.636996，KF从1.030025降到1.006685；主结论不是只在窗口加权口径下出现。

### 尾部与负结果

select的1秒query级结果：

| 策略 | P95误差(m) | E>2m比例 |
|---|---:|---:|
| CV fixed control | 2.063388 | 5.6219% |
| CV actual | 1.906143 | 4.3311% |
| KF fixed control | 2.683541 | 12.7795% |
| KF actual | 2.678417 | 12.4857% |

**actual KF并未胜过actual CV。** select的1秒窗口均值从CV 0.633016到KF 0.998683，增加0.365667m（57.77%）；0.5／2秒也更差。当前clean数据和冻结R=.25／q=1下，不能声称滤波必然提高精度。固定R下的平滑偏差／滞后是后续分组分析的解释假设，本轮没有机制消融证明原因。

时间适配也不是尾部全项改善：select KF的2秒P95从fixed的6.403167到actual的6.426161，略有上升；完整逐log／lead结果保存在summary。没有因这些退化调整参数。

本轮支持的结论是：实际时间差能改善这套固定参数物理基线的平均误差；clean场景应保留原始CV对照，KF的抗噪价值需在B3b2检验。10.97%属于本协议的matched-GT位置误差，不是旧融合网络ADE/FDE的新提升，也不是官方nuScenes挑战成绩。

## 验收证据与工作量

包绑定267个源码文件、1851个来源引用。13个静态文件共135237893字节，不含completed。固定哈希样本为fit/select各16个窗口，共核对1408个query state、704个KF covariance矩阵及704个CV零covariance占位。

正式执行前后实际OpenBLAS均为2线程；构建总耗时104.02秒，包含原父验证、数据读写和压缩，不是单次推理延迟或实时FPS。复核另从保存预测和GT用hypot重新计算误差、分组和配对差，并对原状态全部重新生成，源代码未变化。

旧八阶段各自冻结源码／静态产物校验通过：B3a 260／13、B2b2a 255／11、B2b1 250／13、B2a 245／11、B1a 240／12、B0b 233／14、B0a 233／14、3A 228／217。除直接父链外，其他旁支按自身清单额外保全；B2a阅读工作表仍遵守可变契约。

本阶段交付6个生产模块（状态核、数据、指标、标量参考、来源、发布）、1个CLI、4份测试文件和45项新增测试。main工作区用户已有文档及图片未动。

实际本地文件：

- 不可变实验包：`outputs/realclock-physics-clean/b3b1-v1/`。
- 包外验收：`outputs/realclock-physics-clean/b3b1-acceptance-v1.json`。
- 正式构建／独立verify日志：`outputs/physics-build.log`、`outputs/physics-verify.log`。
- 同进程执行前后线程审计：`outputs/physics-thread-audit.json`。
- 父来源预检：`outputs/physics-parent-prevalidation.json`，不冒充正式completed。

## 产物与复查入口

shared.npz保存共同原点、全GT位置、真实时间、匹配索引和mask；windows.json给出同一稳定窗口顺序。每策略NPZ保存states、covariances、query_us、predictions、errors、valid以及逐窗口均值／query计数；paired_deltas.npz保存同窗口配对均值与mask。

独立verify从原观测重放四策略并逐字节重建产物，同时用保存的预测与GT、独立hypot公式重新计算误差及汇总。正式标量抽样不调用向量状态核，避免同一错误在生成和验证两端同时出现。

```powershell
# 本worktree和既有环境；进程启动前设置线程环境
$env:PYTHONPATH='src'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
python scripts/run_realclock_physics_clean.py --parent ../realclock-query-audit/outputs/realclock-query-audit/b3a-v1 --parent-owner ../realclock-query-audit --output-root outputs/realclock-physics-clean/b3b1-v1
python scripts/run_realclock_physics_clean.py --output-root outputs/realclock-physics-clean/b3b1-v1 --verify-only
```

生成命令仅用于新目录，已完成版本不可覆盖。输出数据保持本地、忽略于Git；本轮不推送GitHub。

## 适用边界与下一步

本轮没有学习模型、训练seed、观测噪声、丢包、地图输入、人工停车GT或holdout评价；物理控制组不能被包装成门控融合新成果。四组的改善和退化都应完整报告。

后续B3b2单独冻结source-key RNG、噪声／离群及共享基础扰动，检查clean结论在感知降质下是否仍成立；学习门控的actual-dt接口和训练另做设计。已有历史ADE/FDE提升与本实验分开归因。

# Phase 2D：多门控训练种子与因果转弯保护

日期2026-11-10。分支`feature/noise-aware-stability`，worktree`.worktrees/noise-aware-stability`，冻结提交`7bc11d9`。依据[Phase 2C](NOISE_AWARE_PHASE2C_RESULTS.md)继续执行[设计](superpowers/specs/2026-11-10-noise-aware-phase2d-design.md)及[计划](superpowers/plans/2026-11-10-noise-aware-phase2d.md)。

九个真实模型完成后，CA在clean/普通噪声下的改善在3个训练种子方向一致，离群条件只有2/3改善。保护规则缓解高曲率和低速误差，但普通噪声/离群条件总体ADE相对未保护CA变差。它不是统一改进，也没有解决高曲率相对原门控的退化。

## 固定控制和随机性维度

- 原七候选`baseline7`、加CA的`plus_ca`、加因果保护的`guarded_ca`三个控制，每个训练seed2027/2028/2029各训练一次，共9个门控。
- 模型seed同时控制门控初始化与批次shuffle；同seed的控制共享初始化/顺序。上游GRU/map始终冻结，不是整条神经链三次重训。
- 原608窗口、352 fit/22log、256 select/4log、三观测条件、噪声seed2027/2028/2029不变。噪声seed与模型seed虽使用相同数字，属于不同维度。
- 938名义参数、原18特征、同一十候选缓存（停止/保持头关闭）、fit-only标准化、每模型20epoch/700更新、原Adam预算。没有选最佳seed、早停、选epoch或看结果改阈值。

保护规则：截至当前的最近四个NIS-KF速度快照（跨度1.5s），任一速度<0.5m/s，或相邻航向差绝对值累计>=15度时，禁止CA；其他情况允许。mask在fit和预测的softmax之前屏蔽CA、重归一化旧七头。只读已到packet，未来机动标签只用于评价。

“禁止CA”表示该模型回到自己的七头混合，并不保证等于另一个baseline模型的预测。规则同时含转弯与低速条件，不能把它的全部作用单独归因于转弯识别；它只能响应已经观察到的变化，不能预知未来转弯。

## Select：三个模型训练种子的ADE

均值±样本标准差（ddof=1），单位m。每个模型内先对相同窗口/噪声重复取均值，再统计3个模型seed；不是预测轨迹集成，不是95%置信区间。

| 控制 | clean | noise_05 | noise_outlier |
| --- | ---: | ---: | ---: |
| baseline7 | 2.145186±0.005698 | 3.143655±0.003644 | 3.870416±0.026951 |
| plus_ca | 2.105486±0.008066 | 3.052923±0.008028 | 3.842010±0.053497 |
| guarded_ca | 2.081715±0.008555 | 3.091209±0.003286 | 3.859925±0.044548 |

逐seed配对ADE差（负为改善），所有seed都纳入：

| 条件 | 对比 | 平均差/m | 改善seed数 |
| --- | --- | ---: | ---: |
| clean | plus_ca-minus-baseline7 | -0.039699 | 3/3 |
| noise_05 | plus_ca-minus-baseline7 | -0.090733 | 3/3 |
| noise_outlier | plus_ca-minus-baseline7 | -0.028406 | 2/3 |
| clean | guarded_ca-minus-plus_ca | -0.023772 | 3/3 |
| noise_05 | guarded_ca-minus-plus_ca | +0.038287 | 0/3 |
| noise_outlier | guarded_ca-minus-plus_ca | +0.017915 | 0/3 |

三seed均值之比：CA相对baseline的ADE在clean/普通噪声约降1.85%/2.89%；离群条件平均降0.73%，但训练seed2028略差0.064%。不能说CA收益已对所有训练随机性稳定。

保护相对未保护CA，clean约再降1.13%，普通噪声/离群却分别增加1.25%/0.47%，两项退化在3个训练seed均出现。不能把保护版直接设成统一默认，也不能只报告clean改善。

FDE/jitter及逐seed原值完整保留在`seed_summary.json`。例如离群条件jitter均值：baseline7为3.398337，plus_ca为3.502264，guarded_ca为3.463937；保护虽缓解CA的jitter，仍高于原门控。当前没有显著性或独立数据泛化结论。

## Clean机动分组：缓解了什么，仍缺什么

下表为3模型seed的ADE均值，组标准差和逐seed配对见JSON。标签沿用固定未来轨迹描述，不进入模型。加减速各15窗口、低速38、高曲率28，均只覆盖4个select log；标签可重叠。

| 标签 | baseline7 | plus_ca | guarded_ca |
| --- | ---: | ---: | ---: |
| straight | 1.927993 | 1.833131 | 1.816001 |
| turn | 2.386133 | 2.433065 | 2.426511 |
| high_curvature | 3.275633 | 3.487615 | 3.395028 |
| low_speed | 1.370678 | 1.327426 | 1.232428 |
| acceleration | 3.552748 | 3.313661 | 3.366539 |
| deceleration | 4.580009 | 4.363393 | 4.339876 |

保护相对未保护CA：高曲率ADE下降2.65%、低速下降7.16%、减速下降0.54%，三组均3/3训练seed改善。但高曲率相对原baseline仍高3.64%；相对未保护CA的改善不等于解决原问题。

低速相对baseline平均改善10.09%、减速改善5.24%，这些是相同开发样本上的门控随机性检查，不能外推到未见采集log。加速组保护反而削弱了部分CA收益，完整表格保留该代价。

## 保护覆盖率及解释

CA被禁止的更新比例，与模型seed无关；只由固定观测流决定：

| 条件/组 | 禁止CA比例 |
| --- | ---: |
| clean全部 | 15.45% |
| clean高曲率 | 57.79% |
| clean直行 | 6.97% |
| clean低速 | 38.04% |
| noise_05全部 | 46.85% |
| noise_outlier全部 | 68.43% |

高曲率组的57.79%不是转弯识别召回率：组标签描述整体未来，分母是该组所有更新，并非逐时刻真实转弯事件。clean的三个噪声seed也完全重复，不能把更新次数当独立样本。

噪声越大时保护屏蔽更多CA，同时总体误差相对未保护CA变差，与规则过于保守的解释一致；尚未拆分航向/低速条件或做阈值因果归因。本轮没有因为结果把15度或0.5m/s调成其他值。

## 审计与复现

运行身份：`ca0792a001be3fd361239d41c631080e25d3d30ed490762b5d83a66fb9c2568f`。

输出`outputs/noise-aware/phase2d/`：父链/原始缓存引用、保护trace、9模型、训练日志、逐轨迹/逐更新指标、机动/权重/覆盖率及跨seed统计、完成哈希。180,576条策略记录=5,472观测流×33策略（旧24参照+9新模型），不是独立轨迹数。9模型训练加评价/汇总约44.10秒CPU，不含父链验证、保护生成和输入复制，不是在线端到端延迟。

- 636项pytest、Ruff及公开文件扫描通过。
- 全部指标/逐更新、标签/机动/权重、保护覆盖率、逐模型/跨训练seed配对和标准差重算通过；fit-only统计与log隔离通过。
- 2027的baseline7和plus_ca权重/标准化及每个5,472流指标精确复现父模型，确认seed扩展没有改变旧数值路径。
- 父输入/十候选/目标逐元素一致；64流的保护mask、航向变化和最小速度重新生成，逐元素一致。没有称为保护全量第二次重算。
- 父Phase 2C及其上游在原worktree验证；旧noise-aware-motion仍为0d7c3d7、无修改。没有新增holdout评价、下载数据或重训GRU/map。

CSV中的`seed`是噪声seed，`train_seed`是门控训练seed；旧参照行的`train_seed/variant`为空，不纳入跨模型seed统计。自定义CSV读取建议指定`dtype={'variant':'string'}`或`low_memory=False`，避免空旧行与新字符串列的分块类型推断警告；指标已逐元素重算一致。

在新worktree使用既有环境：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/run_noise_aware_stability.py --verify-only
# 复现训练须给新目录，已有目录拒绝覆盖。
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/run_noise_aware_stability.py --output-root outputs/noise-aware/phase2d-reproduction
```

## 后续判断

CA有较一致的clean/普通噪声开发收益；当前硬保护是局部折中，保留对照，不依据本轮均值替换历史推荐。下一步需要解释高曲率残余误差，并在未参与本轮开发的数据或更大开发划分上验证；继续堆叠小样本均值不能替代泛化评价。

限制仍是4个select log、3个门控训练seed、反复开发窗口、合成位置噪声、准确航向参考系和冻结上游。均值/标准差仅描述门控训练随机性，不能当全链稳定性、官方nuScenes成绩或研究SOTA。

本轮只本地提交，未推送GitHub、未修改简历；物理运动先验、KF/NIS和门控融合均不声称首创。

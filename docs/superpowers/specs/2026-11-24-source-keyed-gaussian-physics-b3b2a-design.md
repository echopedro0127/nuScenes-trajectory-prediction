# B3b2a：按源包固定的Gaussian位置噪声对照

日期：2026-11-24。状态：待审阅，未实现、未运行新噪声实验；B3b1的clean结果保持冻结。

## 动机与路线

B3b1发现actual-dt CV在select 1秒误差0.633016m，actual KF为0.998683m；相较各自fixed-dt控制，时间适配均改善均值，但KF没有在clean获胜。下一步应检验同一模型在位置观测降质后的收益，而不是根据clean分数调R或换模型。

现有`robustness/packets.py`使用窗口key和相对step生成扰动，不满足B3a源包命名空间的跨窗口一致性。新实验不能直接沿用旧噪声文件并称为同一source-keyed协议。

路线比较：

1. **推荐：B3b2a先两档Gaussian，B3b2b再离群／NIS。** 单独检验位置噪声与时间适配，不混入异常位置、到达时钟和专家变化。
2. 一次加入Gaussian、离群、丢包、延迟及NIS：覆盖广，但随机因素与模型接口同时变化，难以归因。
3. 立即重训学习门控：当前学习接口仍nominal-only，属于独立迁移工作。

本方案采用路线1。沿用B3b1四组CV/KF，不增加新算法、不根据本轮分数选参数；本设计不承诺KF一定获胜。

## 数据、条件和参数

显式已完成B3b1包及原owner为直接父，继承B3a/B0b身份、窗口顺序、GT、匹配及所有分母。clean结果从父包读取，不把同一clean输出复制成三个独立重复。

正式新增条件仅为：

| 条件 | 每个全局XY轴Gaussian标准差 |
|---|---:|
| gaussian_05 | 0.5m |
| gaussian_10 | 1.0m |

每条件保留noise_seed=2027／2028／2029，四个策略与B3b1相同，共24个condition-seed-strategy组合。每组合22231窗口、511854个有效query-target比较；重复seed是同一数据上的噪声重复，不是新增窗口或训练seed。

R=.25I、q=1保持B3b1冻结值，包括sigma=1.0条件；不把R自动改成sigma²，不引入oracle观测质量。这是固定滤波器对不同噪声强度的对照，R自适应须另立实验。

没有偏置、离群、丢包、延迟、时钟抖动、NIS、地图或学习模型。source_us=arrival_us继续成立，时间戳不加噪声。

## 源包键与确定性随机数

从原B3a的source_packets.json读取已冻结source_key_digest及其dataset_release／role／scene／log／instance／annotation_token来源，核对重复键和完整覆盖。

每个(source_key_digest,noise_seed)只产生一个二维标准正态向量Z。新键为：

`{"schema":"realclock_gaussian_packet_b3b2a_v1","source_key_digest":...,"noise_seed":...,"channel":"world_xy_standard_normal"}`。

规范化为UTF-8 JSON：sort_keys=True、separators=(',',':')、ensure_ascii=False、allow_nan=False；取SHA256摘要前8字节、big-endian解释为无符号64位种子。显式使用NumPy Generator(PCG64(seed_uint64))，调用standard_normal(size=2,dtype=float64)一次，XY分量顺序固定。绑定NumPy版本、算法、键规范及向量清单哈希，不依赖Python hash或全局RNG。

观测扰动为sigma*Z。sigma、condition、window_id、query_step、strategy、模型seed、事件类型及GT排名均不进入Z的键：两档sigma共享同一基础Z，所有策略共享同一观测，重叠窗口中同一token共享同一扰动。role保留在source_key中，不跨角色混合来源。

生成噪声清单可以离线覆盖全部源token，但只有对应观测包到达时才能向状态核释放位置。不得通过噪声大小提前选择模型、裁剪大扰动、重抽“不好看”的样本或按结果替换seed。经验均值／方差只作生成诊断，不强制每次有限样本恰好均值0、方差1。

## 观测／GT隔离和局部原点

对每窗口的索引0..15，包括两点初始化和全部历史暖启动，都使用z_i=GT_i+sigma*Z_i。第16点仍绝不同化。GT评分位置永远使用原始无噪声标注；同一token在另一个窗口被用作观测时可以带噪声，在GT通道中仍为真值。

局部原点为**第一个已到的带噪声观测z0**，不能用父clean原点当额外准确观测。若父GT局部坐标为g_i=GT_i−GT_0，则新观测局部坐标为g_i+sigma*(Z_i−Z_0)，评价GT为g_i−sigma*Z_0；保存源Z、sigma、原点偏移及变换关系，以免把GT也注入噪声或漏减原点。

状态核只接16点带噪声观测及原时间，接口仍不接受未来目标／标签。四策略逐包初始化与更新保持B3b1一致。未来观测扰动不得改变此前状态，未来GT修改只能改变评分。

sigma=0只用作测试退化条件：应精确复现父clean状态和误差；它不另算三次正式噪声实验或新统计样本。

## 评价及解释

完全继承父valid和GT目标：每组合511854有效比较，按role／sigma／strategy／lead分组。保存每seed结果，再对同一窗口的三seed均权，最后窗口均权；另保存逐log、log宏平均。零有效lead保留mask／reason，不能按噪声或策略删行。

报告窗口平均位置误差、query级P50/P95、E>2m比例；每seed的尾部／比例先独立报告，跨seed用等权平均并明确名称，不把三seed池化分位数当成均值分位数。保留各seed数量与描述性跨seed标准差，不将其称为训练稳定性或独立数据置信区间。

同条件、同seed、同query-target比较actual KF与actual CV；另保留actual/fixed时间适配配对差。相对clean的额外误差为noisy−clean，同目标配对，正值表示退化。所有组都报告，包括KF未改善和尾部恶化，不选择最优seed或强度。

核心问题是：噪声下是否出现CV/KF优势转移、时间适配收益是否保留，以及尾部是否与均值方向一致。Gaussian是合成位置降质，不等于真实检测／跟踪误差；clean与noisy结果不能合成一个“真实驾驶总分”。不计算人工停车指标或官方nuScenes ADE/FDE。

## 发布与验收

按condition／seed／strategy逐片生成，避免同时堆叠24份状态与covariance。保存共享GT／时钟／匹配、按源键排序的标准Z与key manifest、观测原点／输入摘要、各策略状态／预测／误差、逐窗口指标、分组／配对／clean退化摘要；不能只存平均分数。

绑定父B3b1原owner证明、全部来源、完整随机规范、源码／环境与线程请求；正式执行前后额外核对实际OpenBLAS2线程。输出目录拒绝覆盖，completed最后写；不修改任何父包。

独立verify从source key重建Z和观测，重放全部组合并从保存预测／GT独立重算；固定哈希窗口用逐包标量参考核对状态／covariance。新增测试包括：跨窗口／策略／强度共享Z、排序／分片／全局RNG种子不影响键结果、不同seed／token分离、固定黄金向量、sigma=0复现clean、带噪声首原点平移、GT始终无噪声、未来扰动因果性、完整分母、重哈希篡改和中断无completed。

## 实现顺序

1. 新隔离worktree与执行计划，旧B3b1和全部上游只读。
2. 源键Gaussian生成与黄金向量测试，再接带噪声历史／query和无噪声GT评价。
3. 小夹具四策略、三seed聚合、同目标配对和clean退化，独立标量／保存预测重算。
4. 完整发布／verify、只读审查、全套测试与公开扫描；冻结源码和协议。
5. 正式全缓存24组合实验，独立验收并如实报告所有正负结果；只本地提交，不推送。

之后再依据Gaussian结果设计B3b2b离群／NIS或自适应可靠性实验。没有实测前不填写噪声提升百分比，不据成绩修改本协议。

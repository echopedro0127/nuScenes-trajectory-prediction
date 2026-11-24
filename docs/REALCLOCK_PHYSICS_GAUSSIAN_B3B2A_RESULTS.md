# B3b2a：按真实源包固定的Gaussian抗噪对照

日期：2026-11-24。分支`feature/realclock-physics-gaussian`，工作区`.worktrees/realclock-physics-gaussian`。本阶段在B3b1冻结物理策略和相同GT目标上降质位置观测，评价合成噪声下的收益与退化，不评价真实感知或闭环驾驶。

## 研究问题与对照

B3b1的clean结果中，select 1秒窗口平均误差为actual CV 0.633016m、actual KF 0.998683m。当前问题是位置观测加入噪声后，CV/KF优势是否转移，而不是根据clean分数改KF参数。

新增两档各XY轴独立Gaussian标准差0.5／1.0m，保留2027／2028／2029三个噪声seed，运行CV/KF × actual-dt/fixed-dt-state-control四策略，共24组合。q=1、R=.25I完全沿用B3b1；sigma=1时也不把R偷偷改为1，没有oracle噪声质量输入。

clean结果读取冻结父包，不复制成三个“独立重复”。四策略共享同一观测、同一GT、同一有效query-target集合；seed重复不是新增驾驶样本或训练seed。

## 方法：从窗口噪声改为源包噪声

旧位置噪声按窗口key与相对step生成，同一个annotation在两个重叠窗口中可能被扰动成不同位置。B3b2a使用冻结B3a源包键：dataset_release／role／scene／log／instance／annotation_token，来源先核对digest、真实时间和完整覆盖。

每个(source_key_digest,noise_seed)生成一次二维标准Z：新版本键规范化JSON后取SHA256前8字节big-endian作为PCG64种子，显式standard_normal(size=2,dtype=float64)。sigma、窗口、策略、查询序号和事件不进入随机键，因此：

- 重叠窗口中同一源包的扰动一致。
- 四策略严格共享观测；两档sigma共享基础Z，只改变幅度。
- 遍历顺序、分片顺序和全局RNG状态不改变结果。
- 不裁剪、重抽或挑选seed；经验均值／方差只作诊断。

黄金向量锁定source_digest为64个0、seed2027时Z=[0.9757769166040525,0.36552841812889825]。这不是统计目标，而是键编码、种子端序和算法版本的回归证据。

## 因果观测与GT隔离

历史初始化及暖启动也加入噪声。每窗口只同化索引0..15；第16点仍只供GT评价。时间戳不扰动，source_us=arrival_us，未加入丢包、延迟、偏置、离群或NIS。

局部原点取第一个已到的带噪声观测。若父clean局部GT为g，标准噪声为Z，则观测局部坐标为g_i+sigma*(Z_i-Z_0)，评价GT为g_i-sigma*Z_0。这只是共同坐标平移，GT本身仍是原始无噪声位置；不能向GT加sigma*Z_i，也不能用父clean原点当额外准确观测。

噪声清单可以离线生成，但状态核只接已释放前缀。同一token在别的窗口作为观测时带噪声，在GT通道仍读无噪声真值。未来观测扰动不能改变此前状态，未来GT变化只能影响评分。

## 聚合与解释

按role／sigma／strategy／lead分别报告。先每seed逐窗口平均有效query误差，再同一窗口三seed均权，最后窗口均权；另保存逐log和log宏平均。

尾部先分别计算每seed query级P50/P95，再报告这三个分位数的均值／样本标准差；不把“分位数均值”冒称为池化分位数。E>2m比例同样保留每seed结果及seed均值。跨seed标准差是噪声敏感性的描述，不是训练方差或独立数据置信区间。

三种物理配对仍为actual/fixed CV、actual/fixed KF及actual KF versus actual CV；另保存noisy−clean有符号额外误差，正值表示退化，不裁剪为0。全部组和全部seed保留，不挑最好强度／seed。

## 验证、问题与交付

新增源键黄金向量／全局RNG隔离、源清单完整性、跨窗口／幅度共享Z、带噪声原点、sigma0父复现、未来扰动因果、三seed分母和尾部口径、24组合发布与篡改拒绝测试。

两项验收补强：保存预测重算不仅核对单seed，还核对跨seed分组、配对和clean退化；sigma0直接与冻结父包四策略的states／covariances／query_us／predictions／errors／valid精确比较，防止只用当前实现自比较。clean回归不计为新的噪声重复。

发布按condition／seed／strategy逐片写出，不把24份covariance或预测同时堆在内存。verify从源key重建全部Z、观测和预测，逐字节重建文件；再从保存预测与独立GT平移重算误差／分组／配对／跨seed汇总。全部24组合使用固定哈希窗口的独立逐包标量CV/KF参考。

最终全套回归为**1256 passed，954.45秒**，日志`outputs/pytest-gaussian-final.log`。新增43项：13项源键、8项观测／GT通道、8项seed聚合、14项发布／重算／父回归。Ruff、公开扫描、diff检查和只读代码审查通过。源码冻结提交为`73070d2`，冻结后不依据真实成绩改算法、sigma、seed或滤波参数。

## 正式全缓存结果

24组合真实生成和独立verify均通过，gaussian_semantically_replayed、saved_predictions_independently_rescored、scalar_reference_verified均为true。运行身份`6eca3e0cabfd216d00d007ac3f3673a52aedb52866344ef112d3c9595c073551`，每组合22231窗口、244541查询、511854个有效query-target比较；合计12284496次带方法／噪声重复的有效评分，不是新增独立驾驶样本。

### 主要结果：clean与Gaussian的优势转移

select开发验证角色、1秒提前量、窗口均权，Gaussian先在同窗口三seed均权。单位m，越小越好；clean读父包单次结果，不伪造重复：

| 条件 | actual CV | actual KF | KF相对CV误差下降 |
|---|---:|---:|---:|
| clean父对照 | 0.633016 | 0.998683 | -57.77%（KF更差） |
| 每轴sigma=0.5m | 2.400904 | 1.468641 | **38.83%** |
| 每轴sigma=1.0m | 4.596162 | 2.256338 | **50.91%** |

Gaussian两档的三个seed中，1秒主指标均为KF优于CV。当前证据支持的是固定参数物理滤波在合成位置噪声下的收益；不能据此声称真实感知／跟踪提升、官方ADE/FDE改善或已完成actual-dt自适应门控。

### 全部四组窗口平均误差（m）

| 角色 | sigma | 策略 | 0.5s | 1s | 2s |
|---|---:|---|---:|---:|---:|
| fit | 0.5 | CV actual | 1.445285 | 2.393175 | 4.546233 |
| fit | 0.5 | KF actual | 0.894142 | 1.423810 | 2.905788 |
| fit | 0.5 | CV fixed control | 1.447518 | 2.400510 | 4.565806 |
| fit | 0.5 | KF fixed control | 0.897069 | 1.430873 | 2.924482 |
| select | 0.5 | CV actual | 1.446382 | 2.400904 | 4.591116 |
| select | 0.5 | KF actual | 0.909118 | 1.468641 | 3.043307 |
| select | 0.5 | CV fixed control | 1.450201 | 2.415664 | 4.624707 |
| select | 0.5 | KF fixed control | 0.913710 | 1.481575 | 3.078008 |
| fit | 1.0 | CV actual | 2.829551 | 4.600637 | 8.363618 |
| fit | 1.0 | KF actual | 1.536982 | 2.224510 | 3.978396 |
| fit | 1.0 | CV fixed control | 2.821893 | 4.584065 | 8.324339 |
| fit | 1.0 | KF fixed control | 1.539540 | 2.230174 | 3.992358 |
| select | 1.0 | CV actual | 2.826196 | 4.596162 | 8.352064 |
| select | 1.0 | KF actual | 1.543431 | 2.256338 | 4.087654 |
| select | 1.0 | CV fixed control | 2.819831 | 4.584854 | 8.327701 |
| select | 1.0 | KF fixed control | 1.546747 | 2.264910 | 4.110176 |

fit/select分母不混合，仍按原B3a有效集合；不同策略不删行。select各lead每seed有效query为73678／61278／43689，三seed重复评价为221034／183834／131067，unique窗口／源token没有乘三。

### 三个seed完整保留

select 1秒窗口平均误差，单位m。列顺序固定，不选择最优seed：

| sigma | seed | CV actual | KF actual | CV fixed | KF fixed |
|---|---:|---:|---:|---:|---:|
| 0.5 | 2027 | 2.381439 | 1.463197 | 2.393566 | 1.475005 |
| 0.5 | 2028 | 2.397319 | 1.471875 | 2.414187 | 1.484000 |
| 0.5 | 2029 | 2.423954 | 1.470853 | 2.439240 | 1.485721 |
| 1.0 | 2027 | 4.557583 | 2.245040 | 4.542659 | 2.251627 |
| 1.0 | 2028 | 4.590085 | 2.258899 | 4.581380 | 2.266734 |
| 1.0 | 2029 | 4.640817 | 2.265075 | 4.630522 | 2.276369 |

actual CV／KF在0.5m条件的主指标跨seed样本标准差为0.021483／0.004743m，1.0m时为0.041948／0.010260m。它们仅描述三次噪声重复，不能当训练seed方差、标准误或独立数据置信区间。

### 尾部与相对clean的退化

select 1秒，表中P95是三个seed各自query P95的均值，不是池化P95：

| sigma | 策略 | P95的seed均值(m) | E>2m的seed平均比例 |
|---|---|---:|---:|
| 0.5 | CV actual | 4.722802 | 57.4426% |
| 0.5 | KF actual | 3.145345 | 22.6106% |
| 0.5 | CV fixed | 4.752809 | 57.8652% |
| 0.5 | KF fixed | 3.169093 | 22.9626% |
| 1.0 | CV actual | 9.014861 | 85.6321% |
| 1.0 | KF actual | 4.453628 | 53.0876% |
| 1.0 | CV fixed | 8.994182 | 85.6947% |
| 1.0 | KF fixed | 4.470562 | 53.2698% |

actual CV相对其clean父误差增加1.767888／3.963146m；actual KF增加0.469958／1.257655m（分别为0.5／1.0m噪声）。KF的噪声额外误差较小，但没有恢复到clean精度。

### 负结果：真实dt不是所有条件都改善

select 1秒，actual相对fixed state-control：

| sigma | CV误差下降 | KF误差下降 |
|---|---:|---:|
| 0.5 | 0.61% | 0.87% |
| 1.0 | **-0.25%（退化）** | 0.38% |

1.0m时actual CV为4.596162m，fixed为4.584854m；fit及其他lead的宏平均也有同方向的小幅退化。因此不能把clean的CV时间适配下降10.97%外推到所有噪声条件。两点速度差分对时间间隔和噪声的敏感性是后续诊断假设，本轮没有做机制分组消融来证明原因。

### 结果解释与下一步

clean中CV更好、Gaussian中KF更好，说明需要保留专家适用边界，并为后续使用**因果可靠性信号**选择专家提供动机。目前只比较固定专家，没有把已知sigma条件作为oracle门控，也没有验证新的actual-dt门控或NIS收益。

后续应单独冻结源键离群、NIS诊断及退化／恢复协议，继续保留clean与Gaussian回归条件；不能只按离线条件标签选择胜者后称为实时自适应系统。R=.25与q=1在本轮全部保持不变。

## 正式验收与工作量

包绑定273个源码文件、2132个来源引用，72个静态文件共722728999字节，不含completed。固定哈希fit/select各16个窗口，24组合核对8448个query state和4224个KF covariance矩阵；sigma0另精确复现冻结clean父978164个query state，不计入新噪声重复。

包外验收独立重算96个实际源键RNG向量，检查全部case观测在全局坐标等于GT+sigma*Z，并在32窗口抽查17736个有效预测的全局坐标评分。所有KF covariance与clean冻结值一致，符合固定R／q且无NIS的线性滤波设置。三seed标准Z经验标准差各轴在0.9960至1.0058之间，未强制中心化或重抽。

正式构建前后实际OpenBLAS均为2线程；总构建耗时321.21秒，含父验证、clean回归、生成与I/O，不是单次推理延迟或FPS。独立verify重建所有文件，并从保存预测独立核对每seed及跨seed全部指标。

旧九阶段源码／静态产物按自身冻结清单保全：B3b1 267／13、B3a 260／13、B2b2a 255／11、B2b1 250／13、B2a 245／11、B1a 240／12、B0b 233／14、B0a 233／14、3A 228／217。旁支额外保全不冒称直接依赖；B2a阅读表保留可变契约。

本阶段交付5个生产模块、1个CLI、4份测试文件和43项新增测试。未修改冻结clean算法或旧工作区数据。

实际本地路径：

- 不可变实验包：`outputs/realclock-physics-gaussian/b3b2a-v1/`。
- 包外验收：`outputs/realclock-physics-gaussian/b3b2a-acceptance-v1.json`。
- 构建／独立重建日志：`outputs/gaussian-build.log`、`outputs/gaussian-verify.log`。
- 同进程线程审计：`outputs/gaussian-thread-audit.json`。
- 父来源预检：`outputs/gaussian-parent-prevalidation.json`，不冒充正式completed。

## 产物读取和复查入口

- shared_clean.npz／windows.json：原GT、时钟、匹配和稳定窗口顺序。
- source_packets.json／source_point_indices.npz／normals/{seed}.npz：源键、每窗口点到源包行的映射和每seed标准Z。
- {condition}/seed_{seed}/observations.npz：16点带噪声观测、原点和原点偏移。
- 同目录四策略NPZ：状态、covariance、预测、误差、valid及逐窗口指标；CV covariance仍是明确标为不适用的零占位。
- paired_deltas.npz：物理配对及noisy−clean逐窗口差。
- aggregates/：三seed逐窗口值、均值、mask、配对和clean退化。
- clean_regression.json／scalar_reference.json／summary.json／protocol／run／completed／parent_proof：验收与来源。

每个保存预测以该case第一个带噪声观测为局部原点；读GT时必须结合origin_offset，不能直接减shared_clean坐标。无效预测／均值的0占位必须结合valid读取。

```powershell
$env:PYTHONPATH='src'
$env:OMP_NUM_THREADS='2'
$env:MKL_NUM_THREADS='2'
$env:OPENBLAS_NUM_THREADS='2'
python scripts/run_realclock_physics_gaussian.py --parent ../realclock-physics-clean/outputs/realclock-physics-clean/b3b1-v1 --parent-owner ../realclock-physics-clean --output-root outputs/realclock-physics-gaussian/b3b2a-v1
python scripts/run_realclock_physics_gaussian.py --output-root outputs/realclock-physics-gaussian/b3b2a-v1 --verify-only
```

生成只用于新目录，不覆盖父包或已完成输出。数据留本地并忽略于Git；仅代码／文档本地提交，未推送。

## 后续边界

Gaussian是合成白位置噪声，未建模真实检测误差的时间相关性或遮挡。学习门控仍未迁移actual dt。本阶段不调参、不重训、不读取holdout，也不声称人工停车GT或官方6秒ADE/FDE。

后续B3b2b再冻结离群／NIS或可靠性门控实验，用本轮clean／Gaussian结果解释优势边界；不能把尚未运行的异常恢复成绩写成已完成。

# P4 trainval 多 seed 消融设计

日期：2026-10-20，基线49ced3a。沿用P1–P3模型、因果输入与原t0+6秒终点，不下载sensor。
数据核验：train25560条/468场景，train_val6621条/176场景，val7201条/136场景；场景两两
无重叠，三份预测由同一固定GRU checkpoint ec253a44... 产生且token/target/config一致。
本机16GiB RAM，当前约1.55GiB可用，GPU3060 Laptop6GiB，E盘约273GiB可用。

## 有界内存准备

583MB annotation JSON流式逐记录解析，建立只含query必需字段的SQLite索引；不把整个
annotation表解码留在RAM。索引由metadata文件哈希绑定，原子发布，不复制sensor。
每次只加载一个scene的annotation，为已有NeighborIndex构造相同输入。地图每地区加载
一次，固定anchor坐标、时间、category和mask策略与P3完全相同。

用numpy.open_memmap写map/neighbor各数组；token用U32（对实际token长度显式校验），
按既有P2/P3 schema验证后生成content-addressed NPZ与原manifest。额外mmap_manifest
记录每个NPY的sha256/dtype/shape并绑定原manifest/NPZ身份。训练只映射需用数组，token
和metadata不搬到GPU；加载拒绝错源/配置/split/row order/sidecar checksum。旧CLI兼容。

## 批量训练与评估

新增向量化residual函数只接受当前/上一观测与固定base，不能接收remaining labels。
批量特征与P3逐条函数逐step测试一致；输入包含同样map和neighbor history/age/masks。
DataBundle按sample/update索引批量取输入，label通过独立方法仅交loss或metrics，不进入
模型调用。避免将281160个update展开成Python对象列表。

固定训练设置：seeds2027/2028/2029，hidden64，batch4096，最多20epoch，patience5，
AdamW lr0.001/weight_decay0.0001，smoothness0.05，与P3相同七个learned模型；每stage
固定同seed及shuffle生成器。original先train/train_val选模后冻结，再各自训练三context
head和三个total参数量宽度对照。记录实际stage进度、配置hash、全部来源、GPU与耗时。
三seed在同一训练/选模全集运行，未用val调参、选seed或选checkpoint。

九策略批量计算全部val更新，独立label比较得到ADE/FDE；jitter与同sample前一update
的重叠点比较，首update为0。与P3逐条输出逐行核对。为每seed保存key顺序一致的压缩
per-update CSV和group报告。所有missing context行均保留，fallback计数保持。

批处理耗时只报吞吐；另固定抽256个相同的val sample/update key，16次warmup后逐条
使用P3实际predictor路径测特征/搬运、head/返回和缓存路径总耗时，CUDA同步。报告明确
不含metadata/sensor冷读，不将batch吞吐当单次延迟。

## 统计与验收

每seed/策略记录ADE/FDE/jitter、coverage/fallback、分update/remaining horizon/maneuver。
校验三seed九策略具有完全相同的key集合/顺序，frozen/sliding跨seed一致。
按scene聚合sum/count，固定seed集合求平均后进行paired scene-cluster bootstrap（2000次，
统计seed0），计算每策略及相对original和matched宽度对照的95%区间；单独报告训练seed
均值/样本标准差。CI只表示场景不确定性，不将相邻update或seed×scene伪装独立样本。
val为已有历史固定验证集回归，不称新盲测。

先合成/mini验证流式解析、mmap checksum、batch parity、future invariance、label边界、
scene隔离、checkpoint源hash拒绝、jitter与bootstrap正确性，再完整prepare/train/evaluate。
独立审查，pytest全量、Ruff、公开扫描、diff检查；报告无收益/退化以及参数匹配限制。
所有运行产物在outputs/context/p4，代码与复现说明本地提交，不推送、不提交模型数据。

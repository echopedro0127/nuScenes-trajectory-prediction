# P3 Neighbors 与地图融合细化设计

日期：2026-10-20。基线 fd587b2，继承主设计的时间、anchor、标签隔离与 fixed endpoint 合同。
P2 mini smoke 未显示地图收益，本阶段仍只完成邻车融合工程及 mini 可行性，不调 mini_val。

## 邻车因果输入

选择场景中 source<=tk 且 available=source+固定非负延迟<=tk 的最新 sample。
当前帧中存在且半径50 m内的 vehicle.* / human.pedestrian.* 对象为候选，排除目标自身，
按当前距离、instance token 排序，最多8个。类别为固定两组（vehicle含cycle、pedestrian），
不从未来轨迹推断类别。没有当前 annotation 的对象不沿用旧位置，消失历史用 mask。

以 tk 为末端的5个0.5s历史槽（含当前，跨度2秒），每槽选不晚于槽时间且当前已可用的
最近场景sample，最大 age 0.5s；不插值、不读取未来pose，历史 source 不能早于 tk-2s。
相同 sample 不重复占槽。邻车位置转入固定 anchor 后减目标当前观测位置；速度仅由相邻
有效历史槽的真实时间差做后向差分，并只旋转，不减 target 速度。缺历史或时间不递增时
velocity_mask=false，不将无速度伪装为真实0。每点输入 rel_xy、vx/vy、类别onehot(2)，
另传 history/velocity masks、source/available timestamps；后者可验证age且用于报告。
首版共享点MLP和masked pooling，将时间槽顺序以固定age输入编码（速度valid也进MLP）。
来源明确为 oracle annotation interaction context，不宣称感知器输出。

## 独立数据与模型

P2 map cache不变。独立 neighbor NPZ/manifest 绑定 map archive/contract、原trajectory、
完整token顺序、配置和sample/annotation/instance/category/map源文件hash。加载时拒绝
错split、map来源、时间戳、padding、排序token、mask与key；输出写 outputs/context/p3。

新增 InteractionCorrector 有 map / neighbors / fusion 三种构建模式：每种只注册使用的
encoder，共享原ResidualCorrector fallback但不共享可训练head。MLP后masked pooling，
拼接residual特征、模态presence。所有所需context完全缺失时，附加offset严格为0；单模态
缺失时仍可使用另一模态，历史不足不删agent/update。past horizon mask始终为0。

重新在相同6训练场景/2选模场景上训练一次原Corrector并冻结；分别训练3个head，每个
stage固定同seed和相同batch顺序/预算。为每种context模型另训练原ResidualCorrector
宽度对照，使total参数量与对应模型尽量接近，并记录total/trainable数量与实际误差。
参数匹配按total参数量，context的fallback冻结而宽网络全训练，必须明确此限制。

训练记录map和neighbor train/selection合同、三份GRU prediction artifact hash、固定
GRU身份、场景hash、配置/seed、每stage选模与参数量。允许mini_train内部分场景或
train/train_val；mini_val不训练/选模。checkpoint加载验证所有合同与场景隔离，不把
不同split artifact hash强行比较相等。旧P2代码与模型加载行为保持兼容。

## 生产评估与验收

新增CLI运行frozen/sliding/original、map/neighbors/fusion与三个参数匹配control（共9种），
所有策略具有相同instance/sample/update行；predictor仅接收ObservationView、fixed base、
MapView和NeighborView，label只交metrics。报告ADE/FDE/jitter、update/horizon/maneuver
分组、map/neighbor覆盖、fallback、历史/速度有效率、frame-age和分段缓存路径latency。
无地图/无邻车/全缺失路径用合成测试；真实mini按既有6/2场景训练，各stage2epoch smoke，
冻结后在mini_val的47条轨迹517次更新评估。无收益和退化结果如实记录，不运行P4多seed。

测试包含未来annotation新增/删除/扰动不变性、后向速度（真实dt）、旋转、类别、目标排除、
稳定排序、消失/历史不足、过旧/延迟、cache/key/hash拒绝、真实model mask/padding/fallback
与梯度、checkpoint来源/场景拒绝以及实际evaluator同更新行和label边界。最终全量pytest、
Ruff、公开扫描、diff check与独立代码审查；本地提交，不推送，不提交数据或模型。

## 实施补充

缓存加载逐个历史槽核验 max_age，并校验有效 velocity 与真实dt后向差分一致。
checkpoint逐stage记录best_epoch/completed_epochs，suite顶层epoch表示各stage实际完成轮数
的最大值，不代替各独立选模状态；该suite不用于训练resume。报告已由有真实有效邻车的
单模态缺失/双模态用例验证，计数和age统计不只依赖全空输入测试。

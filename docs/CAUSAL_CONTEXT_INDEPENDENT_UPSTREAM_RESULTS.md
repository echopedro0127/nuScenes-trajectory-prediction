# P6.6C/1：log互斥划分、旧上游暴露审计与GRU重训

日期2026-11-07；分支feature/causal-context-online。
状态：新划分/缓存、暴露审计、一个GRU从头训练与select重载核验完成。
**当前仅GRU上游完成隔离；地图修正和门控链尚未重训，holdout未进行预测评估。**

## 为什么需要重建上游

P6.6B仅在既有train_val上观察到很小的checkpoint选择收益，不能当成独立泛化证据。
旧GRU的metadata记录原train拟合、train_val选模，其data_config_sha256与原train缓存
一致；P6缓存的base_hashes.model_identity也绑定同一GRU文件与配置。

新holdout取自原train，所以它的73个场景全部已进入旧GRU的训练来源。旧GRU以及
使用其预测的旧map/gate/RH/consistency链不能直接用于新的“上游未见”评估。
本轮暴露计数来自源数据和模型契约审计，不运行旧模型的holdout预测。

## 事前冻结的划分与实际数量

协议0078e9d，实现5f8d2a7。仅使用原train的25,560条有效轨迹、468场景、32个log。
按SHA256('p66c:20271107:'+log_token)排序，以log数量70%/15%/余量的固定取整规则
分配fit/select/holdout。相同log所有scene处于同一role；不依据标签或模型效果重抽seed。

| role | 轨迹数 | 场景数 | log数 |
| --- | ---: | ---: | ---: |
| independent_fit | 14,445 | 291 | 22 |
| independent_select | 7,786 | 104 | 4 |
| independent_holdout | 3,329 | 73 | 6 |
| 合计 | 25,560 | 468 | 32 |

scene、log和instance/sample key无交叉，源行恰好分配一次，保留原行序。比例针对log
数量，不保证轨迹/场景数恰好70/15/15；32个log取整后实际为22/4/6。

holdout标签只做缓存机械复制、哈希和逐行一致性校验，没有进入本轮模型拟合、
checkpoint选择或预测指标计算。本轮没有加载旧train_val/val用于重新划分或选模。

## 地区覆盖与统计限制

以下仅查询scene/log metadata，不据此调整划分：

| 地区 | fit场景/log | select场景/log | holdout场景/log |
| --- | --- | --- | --- |
| boston-seaport | 179 / 11 | 59 / 2 | 30 / 3 |
| singapore-hollandvillage | 35 / 3 | 27 / 1 | 0 / 0 |
| singapore-onenorth | 30 / 4 | 0 / 0 | 29 / 2 |
| singapore-queenstown | 47 / 4 | 18 / 1 | 14 / 1 |

fit覆盖四地区，select/holdout各三地区；分布不完全匹配。未来holdout只有6个log，
应按log分组估计不确定性，不把73个scene或大量相邻update当成完全独立观测。
地区描述使用log.json，SHA256为774c8256554a5569f64660e5f450394ce7e3a82168edfab530edab81f9ae7edc。

## 新GRU实际训练与重放

调用原scripts/train.py和configs/models/gru.yaml，没有更改旧训练实现或加载旧权重。
seed2027、单层GRU hidden64、输入历史5点/6维特征、未来12点；Adam lr.001、
weight_decay.0001、batch256、最多80epoch、patience12、按新select ADE选best。
运行设置CuBLAS=:4096:8，OMP/MKL线程2；PyTorch2.6.0+cu124、RTX3060 Laptop。

训练实际完成80epoch，达到预算上限；best为epoch72。日志字段val_ade/val_fde在本轮
指independent_select，不是官方val或旧train_val。

重载best.pt后，以相同batch256在7,786条select轨迹上完整evaluate_epoch：

| select指标 | 训练日志 | 重载评估 |
| --- | ---: | ---: |
| ADE/m | 4.208823391443778 | 4.208823391443778 |
| FDE/m | 10.029098075672682 | 10.029098075672682 |
| SmoothL1 loss | 2.086321888991502 | 2.086321888991502 |

最佳epoch与全日志最小ADE一致，best/last metadata均绑定新data.yaml哈希、原模型配置
哈希、independent_fit/independent_select和history_steps=5。新模型拟合/选模场景与
holdout交集为0；分组/缓存逐行回放再次通过。

原缓存的history_features为逐轨迹XY、displacement和velocity，不包含在旧全train
拟合的mean/std归一化统计；因此可在保留来源的条件下复用确定性轨迹特征。

这些select指标不是独立测试结果，也不能与旧GRU的不同数据划分或在线rolling指标
直接比较。当前仅一个seed，不声称跨seed鲁棒性、性能提升或完整预测链独立验收。

## 产物与身份

输出为outputs/context/p66-independent：

- data.yaml、partitions.json、report.json、completed.json；
- cache/independent_fit、cache/independent_select、cache/independent_holdout；
- gru-seed-2027/best.pt、last.pt、metrics.csv。

准备阶段完成marker绑定配置、分组、报告和各cache manifest，最后发布。
report.json中的independent_chain_complete保持false，避免将GRU完成误写成整个系统完成。

| 身份/文件 | SHA256 |
| --- | --- |
| partition_identity | d1d805caaa27db8c6c5d42694ce38832b7a0ffb54c95cca1b039ae39898f2d4d |
| 新data_config_sha256（规范化配置） | d81d38f229bf5d9679db773d1b4fdfe3ed1b32721ed08776a09fe3bca38539ce |
| data.yaml文件 | f41427d0f58ba40e7f29a1fccc7dcf5a6b7718e852b5a90fd712fd6a57fa83ca |
| partitions.json | 6adb391a29efa5d62427b348548533904efddd2255363312b9f48469267c1e4f |
| report.json（准备阶段） | 85cc846ef239d33382f6e834e39c70ba67cb5a5bd00450a27deda48ad434d02d |
| 新GRU best.pt | 0804c98a4b3db907ab4b4b7f8debe744d47c7c9e599fd4e2806b97464cb94757 |
| 新GRU last.pt | 98a2ec816063e2b4907f47be55654a4dde142586f1e52466636f09491917a79c |
| 新GRU metrics.csv | bda23c357b37ac6a9a30d36f61d1ba4051eaa2744ad2f22d31592676c0c49e56 |

## 验收与复现

新增7项测试，全量 **541 passed in 58.56s**，Ruff/公开扫描通过。覆盖确定性分组、
顺序不变性、缺失/冲突log、重复key、源checkpoint身份、缓存逐行来源、输出篡改与
拒绝覆盖。包括重哈希报告后伪称“旧上游未见”的情况，验证器会重算并拒绝。

准备命令见[实施计划](superpowers/plans/2026-11-07-causal-context-independent-upstream.md)。
训练实际使用新data.yaml，model-config/output-dir为绝对路径，命令模板为：

```powershell
python scripts/train.py --data-config $dataConfig --model-config $modelConfig --train-split independent_fit --val-split independent_select --history-seconds 2.0 --output-dir $gruOutput
```

dataConfig为新data.yaml，modelConfig为configs/models/gru.yaml，gruOutput为新的
gru-seed-2027目录。没有--resume。重跑使用新输出根，不能覆盖本轮目录；保留旧原始
数据、缓存和全部实验结果。数据/模型/outputs不入Git，不推送、不合并、不改简历。

## 下一阶段

新GRU的fit/select预测已生成并核验，地图/邻车上下文准备进度见
[P6.6C/2报告](CAUSAL_CONTEXT_INDEPENDENT_CONTEXT_RESULTS.md)。后续按新
key/时间/配置重建候选，重训map corrector、quality gate、RH及一致性head。
完整配置与模型选择冻结后才考虑holdout预测评估，不得直接套用旧候选缓存或旧权重。

这是现有数据的重划分泛化诊断，不是从未用于研究的新采集盲测。研究者已知旧实验
结果，这一局限不能靠重命名数据集合消除。当前完整独立验证仍未完成，P7未启动。

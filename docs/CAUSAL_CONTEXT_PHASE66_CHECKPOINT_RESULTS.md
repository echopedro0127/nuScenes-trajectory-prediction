# P6.6B 保守精度约束 checkpoint 选择：真实训练结果

完成日期：2026-11-07。分支 feature/causal-context-online。
状态：三seed/30epoch真实训练、所选模型完整train_val重放与独立产物核验完成。
本轮没有新的val评估，不替换当前推荐模型。

## 冻结问题、预算与身份

协议0a0b36f，实现c379c52；新训练启动前通过完整534项pytest、Ruff及公开扫描。
只运行adaptive_matched_005，seed2027/2028/2029，每seed固定10epoch，无早停。
初始化、训练loss、shuffle、优化器与P6.5同seed同arm一致；不从旧best继续微调。
保存全部epoch，比较同一条训练轨迹的ADE-best与constrained-best。

训练使用train（25,560轨迹/468场景），逐epoch选择使用train_val（6,621轨迹/176场景）。
六地图包条件、11次0.5秒更新、固定原始6秒终点。输入位置仍是已释放的干净标注，
地图包退化不是新增感知/跟踪噪声实验。

每seed以冻结RH为参考：overall/clean ADE上限1.0005，overall FDE上限1.0025，
clean FDE p95上限1.005。满足约束的epoch与RH参考共同按jitter、ADE、epoch升序选。
ADE-best仅按ADE、epoch升序；无合格新epoch时constrained-best可回退RH。

## 实际选择和训练预算

| seed | 完成epoch | ADE-best | constrained-best | 合格新epoch | 训练与逐轮选模秒数 |
| --- | ---: | ---: | ---: | --- | ---: |
| 2027 | 10 | 10 | 10 | 10 | 354.908 |
| 2028 | 10 | 10 | 9 | 9、10 | 306.410 |
| 2029 | 10 | 10 | 10 | 10 | 336.550 |

共3个新训练模型、30epoch，计时997.868秒（约16.63分钟）。计时覆盖forward/backward/
optimizer与逐epoch完整train_val评估，不含前置缓存/父来源核验、checkpoint写入和
最终重放；不把它当成任务总墙钟时间或训练加速结论。

全部三seed的ADE-best状态指纹与P6.5同arm原best完全一致，旧train_val指标也在
atol1e-9/rtol0下核对通过。因此这轮变化可归因于选择规则，而非训练轨迹漂移。

## 三seed汇总结果

以下是三个独立seed模型的指标均值，不是模型预测集成。总体对六条件等权；
clean FDE p95为三个seed各自clean p95的平均值，不是合并样本的p95。

| 规则/参考 | ADE/m | FDE/m | jitter/m | clean ADE/m | clean FDE p95/m |
| --- | ---: | ---: | ---: | ---: | ---: |
| RH | 1.773337 | 3.884974 | 1.107842 | 1.706452 | 12.666380 |
| ADE-best | 1.773605 | 3.888250 | 1.092107 | 1.705750 | 12.690068 |
| constrained-best | 1.773848 | 3.888764 | 1.090729 | 1.705999 | 12.689760 |

| constrained-best相对对象 | ADE变化 | FDE变化 | jitter变化 | clean ADE变化 | clean FDE p95变化 |
| --- | ---: | ---: | ---: | ---: | ---: |
| RH | +0.028793% | +0.097570% | -1.544722% | -0.026516% | +0.184589% |
| ADE-best | +0.013669% | +0.013229% | -0.126197% | +0.014635% | -0.002426% |

所选模型逐seed均满足冻结的四项精度约束。相对RH的1.5447%抖动降幅包含旧一致性
训练的作用；本轮更换checkpoint规则的额外降幅只有相对ADE-best的0.1262%，且伴随
ADE/FDE代价，不能把全部降幅说成新选模方法带来的提升。

只有seed2028换选：相对其epoch10，epoch9 ADE增加0.000727310m，jitter下降
0.004134625m。另两个seed没有额外选模收益。更早的低抖动checkpoint未通过精度约束。

## 产物和实际验收

输出为outputs/context/p66-checkpoint。每seed包含reference.pt、epoch-001.pt至
epoch-010.pt、run.json、progress.json、report.json、selected_metrics.npz和completed.json。
experiment.json在三seed训练/重放及父模型对照全部通过后发布。

训练后从磁盘重载RH、ADE-best、constrained-best，运行完整measure_horizon推理并
核对统计与硬CV回退；完成标记最后发布，绑定报告及逐update指标文件SHA256。

另外执行了一次独立核验：33个checkpoint字节哈希/状态指纹/epoch/运行身份、父模型
与缓存manifest哈希、当前源码、冻结选择重算；从保存的逐update数组重算三seed汇总。
所有保存数组形状为[6,6621,11,3]，不存在将seed/条件/相邻更新当作独立新增场景的计数。
本轮没有额外生产延迟测量、真实传感器输入或新恢复能力结论。

| 文件 | SHA256 |
| --- | --- |
| experiment.json | 097077a958a071985bdf4bb848efa644156b922a3afde8df9cdd81986dba0c81 |
| seed2027/report.json | 4014c10c2426faf30f8c542e382ee185450b548b1c1be6ff9b265fe46b7c623b |
| seed2028/report.json | 3e229a91d1d9aab846aaf7f31ee1ca3b92fd6cf2b194a6d82848443bff976cd5 |
| seed2029/report.json | 3aedc0eba8a08fd65f44f882d7f46b2a1453044bcda47ae7bc44818b08c8e3f6 |

源码/协议和父来源的细粒度SHA256保存在experiment.json及各seed/run.json中。
原P1–P6.6A源码、配置、测试和输出保持原内容。数据、模型、outputs不入Git，
仅在本地feature分支提交代码和文档，不推送、不合并、不修改简历。

## 结论与局限

工程目标已完成：真实训练、保存候选、精度约束选模和可追溯重载工作正常。
效果是一个有限的train_val精度/抖动权衡：仅1/3 seed改变选择，额外抖动改善很小。
不宣称统计显著、全面领先或新泛化收益。

train_val已参与多轮开发和本轮checkpoint选择，存在选择偏差；本轮没有独立测试。
当前推荐仍保留RH准确度方案与consistent_010稳定性方案。后续若要形成新的独立收益
证据，需要事前留出上游也未见的场景并重新训练相关上游，不能仅在已反馈的历史val
上换规则、再称盲测。本轮不自动重训上游或开启P7。

## 复现入口

在开发worktree使用原完整Python/PyTorch CUDA环境，PYTHONPATH=src：

```powershell
$env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
python scripts/train_precision_checkpoint.py --device cuda --output-root outputs/context/p66-checkpoint-rerun
```

新输出目录必须尚不存在。本轮正式运行使用默认outputs/context/p66-checkpoint。
重跑仍固定30epoch；核对父产物和源码身份，不能覆盖旧结果或将CPU/GPU差异忽略。

协议见[设计](superpowers/specs/2026-11-06-causal-context-phase66-checkpoint-design.md)，
执行见[计划](superpowers/plans/2026-11-06-causal-context-phase66-checkpoint.md)。

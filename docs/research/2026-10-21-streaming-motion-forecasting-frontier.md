# 流式轨迹预测相关方法调研与迁移判断

检索日期：2026-10-21。项目基线387460f；本文是论文与官方仓库查阅，不是外部方法复现。
优先阅读2026流式预测与2025扩展工作，并补充2024直接相关机制；只采用作者论文、会议站点、作者项目页和官方仓库作为技术来源。

## 已核实的相关工作

| 方法/状态 | 核心机制 | 对本项目的迁移判断 |
| --- | --- | --- |
| SHARP，CVPR2026 | 短观测窗口、实例对应的上下文传播，以及流式/单窗口双分支训练 | 借鉴因果状态与冷启动对照；完整替换需训练新编码器 |
| Streaming Real-Time Trajectory Prediction Using Endpoint-Aware Modeling（SEAM），WACV2026 Oral | 用上次预测终点定位目标区域上下文，并传播上下文/历史预测 | 终点引导查询值得后续候选改进；不能从旧局部包外补读地图 |
| DeMo++，2025-07 arXiv扩展稿 | 分离方向意图与时变状态，并引入跨场景交互、状态锚点细化 | 借鉴逐未来时刻建模；本项目初版采用共享小MLP，不复刻Attention/Mamba |
| DeMo，NeurIPS2024 | 意图查询与动态状态查询解耦 | 为共享标量gate的表达限制提供设计启发，并非证明我们的门控必然有效 |
| HPNet，CVPR2024 | 用历史预测注意力建模连续预测之间的动态关系 | 历史预测应提供决策信息；与一律压低修订幅度的惩罚存在区别 |
| SmartRefine，CVPR2024 | 场景自适应细化与质量评分控制细化潜力/迭代 | 借鉴可靠性评估，初版不增加迭代细化链 |

SHARP的双训练指有/无流式上下文两种预测监督，不等同于P6.1的相邻输出L2修订损失。
DeMo经论文首页和官方仓库核对为NeurIPS2024；DeMo++在已读官方README中列为arXiv预印本，本文不替其推断录用会议。
上述机制到本项目的迁移均是待验证推论，不是作者在本项目设置下已经验证的结论。

## 直接来源

- [SHARP论文正文](https://arxiv.org/html/2603.28091v1)，重点§3.2/3.5；[作者项目页](https://a-pru.github.io/sharp/)、[官方实现](https://github.com/a-pru/sharp)。
- [SEAM论文正文](https://arxiv.org/html/2603.01864v1)，重点§3.3/3.5；[论文首页及WACV2026状态](https://arxiv.org/abs/2603.01864)。
- [DeMo++正文](https://arxiv.org/html/2507.17342v1)，重点§III-C/D/E；[作者官方仓库](https://github.com/fudan-zvg/DeMo)。
- [DeMo论文首页](https://arxiv.org/abs/2410.05982)、[正文](https://arxiv.org/html/2410.05982v1)。
- [HPNet会议论文页](https://openaccess.thecvf.com/content/CVPR2024/html/Tang_HPNet_Dynamic_Trajectory_Forecasting_with_Historical_Prediction_Attention_CVPR_2024_paper.html)、[论文PDF](https://openaccess.thecvf.com/content/CVPR2024/papers/Tang_HPNet_Dynamic_Trajectory_Forecasting_with_Historical_Prediction_Attention_CVPR_2024_paper.pdf)、[官方实现](https://github.com/XiaolongTang23/HPNet)。
- [SmartRefine会议论文页](https://openaccess.thecvf.com/content/CVPR2024/html/Zhou_SmartRefine_A_Scenario-Adaptive_Refinement_Framework_for_Efficient_Motion_Prediction_CVPR_2024_paper.html)、[官方实现](https://github.com/opendilab/SmartRefine)。

## 复现与比较边界

官方SHARP README说明其提供nuScenes/AV1/AV2预处理和训练支持，公开预训练权重描述为AV2 single/multi-agent；不能把AV2权重当作现成nuScenes checkpoint。README测试环境为torch2.8/CUDA12.8，与本地既有环境不同，完整复现应使用独立环境。

DeMo官方README涉及Mamba依赖与独立环境，且2026-02记录过安装兼容更新；引入完整模型并非直接替换161参数gate。当前不安装依赖、下载权重或运行外部仓库。

论文常报告固定窗口、多模态minADE_K/minFDE_K或联合多agent指标；我们目前是固定终点、每0.5秒更新、单轨迹剩余时域ADE/FDE及跨更新jitter。不能把论文的minADE6与我们的ADE相减，不能以本项目结果声称超过这些方法。

## 与本地证据的对照

- P6.2 train_val：map有限网格+实际g的标签使用型混合参考1.615440m，实际P6为1.800475m，差0.185034m；说明存在诊断空间，不是可实现收益保证。
- 运动四特征置换ΔADE+0.018670m（场景CI[0.014096,0.023438]），全特征置换+0.019713m；提示先检查运动表达。
- P6.1一致性正则只小幅降低jitter并带来FDE代价；因此后续应允许新观测触发合理修订，而不只扩大正则权重。
- 当前一个g支配全部未来点，无法表示同一更新内短时/长时对map候选的不同信任程度。这个限制来自代码定义，并非P6.2 oracle已经量化了逐时域gate增益。

## 三种路线

1. **优先：冻结候选，改进因果反馈与逐时域门控。** 不动上游缓存，方向/可靠性/未来时长三因素消融，可控地归因，适合当前单卡实验。
2. **后续：重训流式编码器/复现SHARP。** 直接建立更强候选与标准协议对照，但需要独立预处理、环境、训练和新的数据划分审计。
3. **后续：用历史预测终点引导地图查询与候选细化。** 更接近SEAM/SmartRefine，但须重建地图包来源和候选缓存；旧包不存在的区域必须保持不可用。

不把扩散、IRL、多模态输出或增加传感器直接堆入本轮，先验证可归因的小模型机制。路线1的具体设计见[提案](../superpowers/specs/2026-10-21-causal-context-phase63-design.md)。

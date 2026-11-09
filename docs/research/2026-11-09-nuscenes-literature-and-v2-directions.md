# nuScenes轨迹预测近期论文与下一阶段建议

检索日期：2026-11-09。依据作者论文、会议/期刊页面和官方仓库；以周围交通参与者轨迹预测为主，不把自车规划、检测或地图构建成绩混作预测成绩。是近期相关工作筛选，不声称穷尽全部新论文。
本轮只调研，不运行外部代码、下载权重、修改冻结源码或启动新实验。下列迁移建议是本项目的待验证设计，不是论文已在本项目设置下证明的收益，也不是新颖性认证。

## 已核实的近期工作

| 工作与状态 | 研究问题/机制 | 数据与本项目关系 |
| --- | --- | --- |
| BEVTraj，JCDE2026；arXiv v3于2026-11-03更新，首稿2025-09 | 从原始多传感器BEV取场景上下文，以稀疏目标候选解码多模态未来，不依赖预建HD map | nuScenes、AV2 Sensor；走向真实传感器输入的直接参考，不能用我们的metadata-only结果与其数值排名 |
| CMTF，Sensors2026，出版社日期2026-10-10 | 用Bernstein曲线控制点作为显式监督，加入结构化输入/辅助头和几何一致性，研究跨域泛化 | nuScenes、AV2、WOMD；几何一致性不是相邻在线预测jitter正则。出版社页面为early access，本文未核验完整复现实验 |
| CAA-GMM，VTC2026-Fall；arXiv2026-10-06 | Attention编码上下文和交互，Gaussian mixture表示多种未来与不确定性；研究通信/感知丢失 | nuScenes和AV2预测；通信/观测丢失消融在AV2，不能描述成nuScenes上全部验证 |
| Neural Fusion of Multiple EKF-Based Trajectory Candidates，arXiv2026-09-17 | 把Trajectron++和EKF产生的匀速、加速、减速、左/右转候选做后期融合 | nuScenes；与物理候选+学习门控直接重叠。使用robot配置与future ego plan，不能照搬其提升比例到本项目 |
| SIPTraj，arXiv2026-08-01 | Camera/LiDAR的BEV特征与agent历史交互，结合物理状态引导迭代解码；无需推理时HD map | nuScenes、AV2 Sensor；完整复现涉及感知骨干，明显重于小门控模型，本文未确认会议录用 |
| UATP，arXiv2026-03-31 | 将在线地图的位置不确定性与语义不确定性分开建模，再输入轨迹预测器 | nuScenes，四种地图估计器/两种预测基线；提示地图不仅会迟到，还可能位置/类别错误。不能把双头分歧直接当成已校准概率 |
| SHARP，CVPR2026，arXiv2026-03-30 | 短窗口、实例对应的上下文传播、流式/单窗口双训练与历史预测relay，适应观测长度变化 | nuScenes/AV1/AV2；nuScenes有K=5的官方挑战结果。学习历史状态而非一律压低更新幅度，适合流式路线参考 |
| MASAR，ICRA2026，arXiv2026-02-13 | 多视角图像外观指导过去运动修正，再联合3D检测与未来预测；tracking-free/map-free | nuScenes；直接针对“历史轨迹不应假设完美”，是从标注位置回放走向感知误差的参考 |
| Online Aggregation of Trajectory Predictors，ICRA2025 | 预测器作为专家，随下一时刻真值逐步释放而在线更新混合权重，研究跨城市/跨数据分布 | nuScenes城市专家、Lyft迁移；是我们每0.5s因果释放与专家混合的关键先行工作，不能把在线加权本身宣称首创 |

arXiv仅标预印本的工作不据此推断顶会录用；BEVTraj的2026版本也不能被误写为2026首次提出。

## 直接来源

1. [BEVTraj最新摘要与版本历史](https://arxiv.org/abs/2509.10080v3)，[期刊DOI](https://doi.org/10.1093/jcde/qwag088)，[作者代码](https://github.com/Kongminsang/bevtraj)。本次最新v3正文HTML不可用，因此不将旧v2实验细节冒充v3验证结果。
2. [CMTF出版社页面](https://www.mdpi.com/1424-8220/26/20/6412)，[DOI](https://doi.org/10.3390/s26206412)。日期采用出版社2026-10-10，不采用二次索引显示的10月27日。
3. [CAA-GMM摘要/会议说明](https://arxiv.org/abs/2610.09174)，[正文§III-C](https://arxiv.org/html/2610.09174v1)。
4. [EKF融合摘要](https://arxiv.org/abs/2609.19813)，[正文§II–IV](https://arxiv.org/html/2609.19813v1)。其robot-conditioned设置包含future ego plan；物理分支只需目标历史，不代表整个系统仅需目标历史。
5. [SIPTraj摘要](https://arxiv.org/abs/2608.00779)，[正文](https://arxiv.org/html/2608.00779v1)。Sensor Encoder使用BEVFusion的Camera/LiDAR特征；map-free不等于无感知输入。
6. [UATP摘要](https://arxiv.org/abs/2603.29362)，[正文§IV–V](https://arxiv.org/html/2603.29362v1)。
7. [SHARP论文](https://arxiv.org/abs/2603.28091)，[正文§3/§4.5](https://arxiv.org/html/2603.28091v1)，[作者官方仓库](https://github.com/a-pru/sharp)。
8. [MASAR摘要/ICRA状态](https://arxiv.org/abs/2602.13003)，[正文](https://arxiv.org/html/2602.13003v1)，[作者项目页](https://iv.ee.hm.edu/projects/masar/)。
9. [Online Aggregation论文](https://arxiv.org/abs/2502.07178)，[正文](https://arxiv.org/html/2502.07178v1)，[作者机构会议页面](https://research.nvidia.com/labs/avg/publication/tong.sharma.etal.icra2025/)。

## 相邻方向与容易混淆之处

- [MapLightning，2026-10-01预印本](https://arxiv.org/abs/2610.01905)：从图像构建矢量地图、研究效率与标定扰动，含下游预测实验；主要贡献是地图构建，不应把其mAP称为轨迹预测误差。
- [SEAM，WACV2026 Oral](https://arxiv.org/abs/2603.01864)：历史预测终点引导局部上下文，主要在AV2/AV1实验；[2026-10-19边缘部署扩展](https://link.springer.com/article/10.1007/s00138-026-01939-1)也以Argoverse为主。可借鉴，不列作已核实的nuScenes实验基线。
- [UniTraj，2024作者机构页](https://www.epfl.ch/labs/vita/research/prediction/vehicle-trajectory-prediction/unitraj-a-unified-framework-for-scalable-vehicle-trajectory-prediction/)：统一数据/模型/评价与跨域研究，适合作标准比较框架，不属于本次最新2026论文。
- [nuScenes官方预测任务定义](https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/python-sdk/nuscenes/eval/prediction/README.md)：6s、2Hz的预测挑战。论文常用多模态minADE_K/minFDE_K；本项目固定原始终点的单轨迹剩余时域ADE不能直接相减排名。

SHARP官方仓库支持nuScenes预处理/训练/推理，但所列公开预训练权重为AV2 single/multi-agent；不能把它们当作nuScenes checkpoint。README测试环境为Python3.11/torch2.8/CUDA12.8，与当前项目不同，应隔离环境，先验证小样本及显存，不能承诺笔记本完整复现。

## 本地证据决定的下一步

最新固定holdout：RH对CV的ADE下降5.4067%、FDE下降6.1838%，但jitter上升5.3328%。original/map的平均ADE还差于CV；一致性正则只带来小幅jitter变化。见[完整报告](../CAUSAL_CONTEXT_INDEPENDENT_HOLDOUT_RESULTS.md)。
因此优先改进“观测可靠性与候选质量”，不再单独扩大一致性lambda，也不先堆视觉大模型。

### 优先1：带噪/缺失观测下的可靠性建模

保留每0.5s的因果释放，给观测引入预先固定的噪声、偏置、漏帧与延迟，再逐步替换为真实检测/跟踪输出。真值保留作监督/评价，不能作为在线滤波输入。
加入KF/EKF的协方差、观测年龄和归一化innovation，区分“真实机动”与“测量抖动”；已有R是已实现误差反馈，新方案必须证明协方差/可靠性校准有增量，而不是改名。
对照至少包含CV、纯KF/EKF、简单EMA、现有RH、可靠性版本，公平控制可用观测、训练步数与模型容量。检验ADE/FDE/jitter、缺失恢复和p95延迟的权衡，阈值在新开发集固定，不承诺收益。
这是UATP/CAA-GMM/MASAR提供的研究动机到本项目的迁移建议，不是原论文已经验证我们的噪声模型。

### 优先2：小规模多物理专家与因果记忆

补充CA、转弯/EKF候选，先验证是否比现有map候选更好，再做在线聚合。需要比较“增加候选”与“更复杂门控”，防止把专家质量提升误归因给门控。
加入小型因果状态记忆或有明确更新条件的relay；只用已到达观测和此前预测，不在急转弯/急刹车时强迫输出保持平滑。
来源为EKF融合、Online Aggregation与SHARP；这些思想已有先行工作。可研究的新问题是异步/噪声下的可靠性与精度/修订权衡，不能直接声称论文级新颖性。
此阶段可先用现有metadata，不需要新增传感器下载。

### 后续3：真实感知特征与多模态未来

如果目标是增加实际Camera/LiDAR输入，借鉴BEVTraj/MASAR/SIPTraj：先用预训练感知编码器缓存特征、只训练小预测头，确认标签/预测时间与坐标变换；不直接在个人电脑上联合训练整套BEV感知与预测。
同时审计预训练感知模型是否见过测试场景；“冻结编码器”不等于没有上游泄漏。先核对所选场景的图像/点云覆盖，mini只能支持流程验证，不据其少量场景声称规模泛化。
多模态输出与概率校准应作为独立实验，再补官方兼容minADE_K/minFDE_K/MR与概率指标；不能只因minADE_K低于单轨迹ADE而称改进。

## 推荐的研究问题与评价纪律

推荐问题：在不完美、异步到达的位置/地图观测下，协方差校准的因果专家聚合能否在预设精度容差内降低不必要修订，并保持对真实机动的快速响应？
这是待检验问题，不是已实现成果。现有holdout已被使用，只保留作冻结回归/探索性误差分析；新开发与最终评价需另留未参与该轮设计的数据，无法做到则明确报告探索性边界，不能换一个split seed就声称真正新盲测。
建议先做优先1，再按开发结果决定优先2或传感器路线。本轮不改代码、不下载、不启动训练、不额外推送GitHub。

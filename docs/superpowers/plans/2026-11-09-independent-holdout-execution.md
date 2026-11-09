# Independent Holdout Execution Plan

**Goal:** 按adbc5d7c…7ccb1231锁完成一次新holdout推理评价。
**Architecture:** 仅新增evaluation/holdout_statistics.py与independent_holdout.py和脚本/测试。锁内已有源码与设计不改；新实现清单绑定协议identity，started先于第一次推理，输出/完成记录可重验。
**Tech Stack:** PyTorch、NumPy、nuScenes metadata/map、pytest。

主模型RH、主对照CV、主指标六条件等权ADE，log级配对bootstrap5000次/PCG64 seed20271109；全部其他模型/指标仅描述，规则不变。
使用独立holdout缓存及已冻结新GRU/map/gate/RH/adaptive，不训练。保持角色名，验证与fit/select的scene/log互斥，不放宽旧loader。
新holdout候选入口直接复用冻结generate_candidates/audit_candidates与gate数组schema，但绑定专用holdout合同。预测阶段只保存GRU预测/身份，不提前输出静态误差。
新执行源码在首次推理前封存；中断恢复必须匹配同一协议/源码/模型，保留已有完成阶段，不覆盖其数据。只有最终完整结果写入评分报告。

- [ ] 先写统计与执行状态测试，确认RED；覆盖池化log bootstrap、确定性、无效输入、started身份/覆盖拒绝。
- [ ] 实现统计、holdout scope/源码绑定、预测/上下文/候选准备与固定模型推理、完整指标与完成验证。
- [ ] 全量pytest、Ruff、公开扫描后冻结新源码；核验原协议仍有效。
- [ ] 执行真实holdout，独立验证输入/模型/输出、bootstrap及主结论，完整报告条件/log/局限。

输出根outputs/context/p66-independent/holdout-evaluation，数据产物分别在predictions/context/candidates的independent_holdout下。不得根据holdout误差改模型或统计，不推送、不改简历。

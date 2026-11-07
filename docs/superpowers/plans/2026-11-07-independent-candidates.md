# Independent Candidate Cache Plan

**Goal:** 为新fit/select和新original/map生成六种包条件的门控候选输入。
**Architecture:** 新context/independent_candidates.py复用既有generate_candidates、audit_candidates及gate schema；旧P6入口/校验器不改。新合同绑定新划分、新GRU预测、新map pair、全部源文件，禁止holdout。
**Tech Stack:** Python、NumPy mmap、PyTorch、pytest。

直接接续用户已授权的独立重训链。本轮只生成和验收候选，不训练gate/RH/一致性，不修改旧预算、不运行holdout。
设计选择独立入口，避免旧P6只接受train/train_val/val和旧七结构checkpoint的限制；不把新角色重命名为旧角色，不修改历史绑定源码。
固定seed2027，配置沿用configs/context/gate.yaml：max_age_s2、fixed_scale0.5、decay_s1；条件clean/delay_500ms/delay_1000ms/dropout/missing/stale。
每条轨迹11次更新，每次只向模型释放当时位置与已到达的地图包。future_xy仅用于逐步释放标注观测及单独的训练/评价标签，不输入未来标签。
输出outputs/context/p66-independent/candidates；拒绝已有目录，不恢复、不覆盖。先生成全部数组并audit_candidates全量重算original/map，再发布schema/哈希校验的manifest；验收重载两组，核对来源、缓存数组和候选重算。

- [ ] 新测试验证holdout/错角色、旧或错base模型、错地图来源和错scene拒绝，运行RED。
- [ ] 实现独立入口及loader；在新测试中覆盖真实候选保存/重载和篡改拒绝，运行GREEN。
- [ ] 全量pytest、Ruff、公开扫描通过后冻结源码，生成fit/select真实缓存。
- [ ] 重载与候选重算验收，记录样本/条件/更新数、模型/缓存哈希、耗时及边界；完整独立链仍未完成。

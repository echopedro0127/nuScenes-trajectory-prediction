# Phase 3B B0b 全缓存目标装配计划

> **For agentic workers:** 使用superpowers:executing-plans在本会话顺序执行，先测试后实现；仅代码审查可委托只读agent。

**Goal:** 将已通过B0a的完整GT链、真实微秒时钟与窗口精确映射扩展到原fit/select缓存全部目标。

**Architecture:** 复用B0装配器，以显式scope区分pilot与full_cache，并将scope及schema写入运行身份。完整范围从原缓存逐行推导，不能从允许log中的全部对象推导。在新worktree执行，旧B0a与3A均只读。

**Tech Stack:** 既有Python 3.12、NumPy、pyquaternion及pytest；不新增依赖。

## 已确定设计的细化

上位设计为[原B0设计](../specs/2026-11-16-scenario-episodes-b0-design.md)，用户在B0a验收后回复“继续”，执行此前说明的B0b。预计1265 fit＋702 select目标、60948标注，14445＋7786缓存窗口；实际计数须经完整性检查后确认。

方案比较：采用共享装配器显式scope，避免复制几何与来源校验；独立复制一套B0b模块会产生验证逻辑分叉；直接将原pilot默认范围改为全量会混淆旧接口。因此保留默认pilot，仅`--scope full_cache`进入新范围，使用独立schema/scope身份。

缓存窗口按role/instance/sample稳定排序，保留原缓存row和原window ID公式。全量键必须唯一，两个角色非空；instance/log角色隔离仍由链装配器执行。原608窗口必须是全量窗口子集、对应cache row和旧ID不变。所有窗口均精确重建缓存anchor/history/future；缺口窗口显式not_evaluable而非补GT。全量窗口是数据映射，不代表已为所有窗口生成预测。

完整链上的segment ID不因pilot扩大为全量而变化；真实数据验收另比较旧B0a的461条track语义摘要和对应片段、608窗口内容。非目标对象不纳入，holdout不读取、不训练、不评分。B1事件规则、人工overlay与新评分不在本任务。

## T1 新工作树与基线

- [x] 从ae52dd1新建`.worktrees/scenario-tracks-full`，分支`feature/scenario-tracks-full`；确认.worktrees被忽略。
- [x] 复用既有环境，58项B0定向基线通过；实查原缓存22231窗口均有5点完整历史。

## T2 范围扩展测试与实现

Files: `tests/evaluation/test_scenario_tracks.py`、`src/trajectory_prediction/evaluation/scenario_tracks.py`。

- [ ] 先写夹具：每角色追加一个未进pilot的真实annotation链及缓存row；更新父绑定哈希而非绕过来源核验。
- [ ] 加入full_cache构建/独立verify断言：4目标/68标注/4窗口而pilot仍为2目标；逐项覆盖新增row，标记pilot成员，ID与原子集一致。
- [ ] 加入重复缓存键、错误scope、重算哈希篡改scope/schema被拒绝测试。执行RED并确认失败因为接口/功能未实现。
- [ ] 实现`_full_cache_windows(cache_data, scene_by_name)`，从每行cache构造role/instance/sample/scene/log/row，拒绝重复键；全量targets直接取这些窗口。
- [ ] `_load(..., scope='pilot')`校验scope，完整缓存模式验证pilot键及row为子集；`_derive`在全量模式保留旧ID核对并追加pilot成员标记。
- [ ] `prepare_tracks(..., scope='pilot')`与CLI的`--scope {pilot,full_cache}`串联；binding/protocol/summary写独立scope与schema，verify按已保存且受支持的scope重新装配，拒绝未知策略。
- [ ] 运行`pytest tests/evaluation/test_scenario_tracks.py tests/data/test_metadata_stream.py tests/analysis/test_track_timeline.py -q`并确认GREEN。

## T3 验证与真实装配

- [ ] 只读代码审查、Ruff、全套pytest与公开检查；验证通过后先本地提交源码。
- [ ] 运行以下入口（PYTHONPATH=src，既有环境），产物不得覆盖：

```powershell
python scripts/prepare_scenario_tracks.py --scope full_cache --parent ../scenario-evaluation/outputs/scenario-evaluation/phase3a-v1 --parent-owner ../scenario-evaluation --output-root outputs/scenario-tracks/b0b-v1
python scripts/prepare_scenario_tracks.py --output-root outputs/scenario-tracks/b0b-v1 --verify-only
```

- [ ] 检查实际计数、角色与类别、缺口和不可评价分母，单列全量窗口与pilot预测范围。比较旧B0a语义子集及其233源码/14产物，旧3A的228源码/217产物均不变。
- [ ] 写结果报告并更新路线图，记录身份、真实测试和负面结果；只本地提交，不推送GitHub。

## 异常与失败策略

错误scope、重复窗口、跨角色instance/log、缺元数据、链损坏、缺anchor、坐标不一致、文件/来源变更和中断均不得签发完成标记。正常annotation缺口保留并分段；不从名义0.5秒构造实际时间。范围扩大后计数可能与预审不同，必须报告实际验证结果，不硬编码预期数字或放宽精确比较。

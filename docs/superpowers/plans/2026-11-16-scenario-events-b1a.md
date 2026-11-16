# B1a Stop / Restart Implementation Plan

> **For agentic workers:** 使用superpowers:executing-plans顺序实现；测试先行，仅只读审查委托agent。

**Goal:** 按用户已批准的[B1a设计](../specs/2026-11-16-scenario-events-b1a-design.md)，从冻结B0b提取真实时钟停车／起步规则候选，并独立重建验收。

**Architecture:** 配置与运动学→纯状态机／trace→窗口关联与分母→显式owner父验证／不可变发布。新worktree保护旧三个owner；不训练、不评价holdout、不计算预测改善、不推送GitHub。

**Tech Stack:** 既有Python 3.12、NumPy、PyYAML、pytest，无新依赖。

## T1 配置、运动学和block

Files: `configs/analysis/scenario_events_b1a.yaml`, `analysis/realclock_motion.py`, `tests/analysis/test_realclock_motion.py`。

- [ ] 用不均匀时间、边界1秒、超长间隔、单点和非法坐标／时间／配置夹具先RED。
- [ ] 实现`validate_config(config)`：严格键和值类型、有限数、阈值关系；`build_motion(track, segment, arrays, config, release)`返回block、所有interval及诊断。块ID绑定segment/config/边界tokens；区间守恒，不跨gap求速度。
- [ ] GREEN后本地提交。示例接口：

```python
config = validate_config(dict(rule_version='stop_restart_realclock_v1', stop_speed_mps=.3,
    restart_speed_mps=1., prior_history_us=2000000, stop_dwell_us=1000000,
    restart_dwell_us=1000000, minimum_dwell_intervals=2, max_interval_us=1000000))
assert config['stop_dwell_us'] == 1000000
```

## T2 状态机、字段、ID与trace

Files: `analysis/stop_restart_events.py`, `tests/analysis/test_stop_restart_events.py`。

- [ ] 先写设计中三个最小例子的精确边界，以及999999微秒、短高速取消、重复low run、初始低速、prior不足、多个停走、缺口、空事件的RED测试。
- [ ] 实现`extract_events(block, intervals, config)`，返回events/diagnostics/state_trace；所有端点引用原annotation。按最大低／高速run处理但逐interval记录状态，pending累计整数微秒和段数。
- [ ] stop结束与low_dwell_end分开；restart及stop_go通用end=restart确认；unpaired上下文必须产生确定诊断；所有候选ID按配置和语义边界生成，排序固定。
- [ ] 测试ID等价重排稳定、配置／边界变化改变ID；GREEN后提交。

## T3 窗口关联与覆盖

Files: `evaluation/event_windows.py`, `tests/evaluation/test_event_windows.py`。

- [ ] 先RED：onset=anchor／最后future，confirmation=最后future，历史context、端点相接、右删失、同事件多窗口、无窗口覆盖。
- [ ] 实现`link_windows(events, windows, window_arrays)`和`coverage_summary(events, links)`；W含历史、E按通用end，按全缓存与pilot两个全集、角色及类别分别报告各覆盖及其补集；不同事件类型不得相加作独立episode。
- [ ] GREEN后提交。不得拼接P或把新增GT覆盖写成已预测覆盖。

## T4 来源、发布与独立验证

Files: `evaluation/event_sources.py`, `evaluation/scenario_events.py`, `scripts/prepare_scenario_events.py`, `tests/evaluation/test_scenario_events.py`。

- [ ] 微型B0b fixture及显式旧owner CLI证明；先RED测试build→verify、错误scope／owner、来源变化、篡改重哈希、额外文件、未知配置、已有目录、中断无completed。
- [ ] 父校验`parent_snapshot(parent, owner, proof=None)`在原owner的cwd/PYTHONPATH运行B0b CLI一次；保存并严格核验stdout/schema/full_cache/flags，绑定父全引用及源码。后续核对证明不再重跑旧模型。
- [ ] 装配全部block、运动学、事件、诊断、trace与关联；输出按role kinematics NPZ和协议/来源/run/summary；读回及来源/文件集合复查后最后写completed。
- [ ] verify-only从冻结父数组及配置重新计算所有产物和分母，严格比较dtype/数组/JSON/ID/运行策略，不只验哈希。
- [ ] GREEN后只读代码审查，修复阻断问题；Ruff、全套pytest、公开检查、diff检查通过后先提交源码。

## T5 冻结规则与真实运行

- [ ] 配置随源码提交后固定，不按结果修改；按相同配置先fit再select。

```powershell
python scripts/prepare_scenario_events.py --parent ../scenario-tracks-full/outputs/scenario-tracks/b0b-v1 --parent-owner ../scenario-tracks-full --config configs/analysis/scenario_events_b1a.yaml --output-root outputs/scenario-events/b1a-v1
python scripts/prepare_scenario_events.py --output-root outputs/scenario-events/b1a-v1 --verify-only
```

- [ ] 核对58956区间守恒、类别／角色／删失／诊断／全缓存与pilot覆盖分母；零样本照实报告。
- [ ] 独立verify-only通过；检查B0b233源码/14产物、B0a233源码/14产物、3A228源码/217产物不变。
- [ ] 写结果报告、更新路线图和设计状态，本地提交；人工复核仍未完成，不计算预测分数。

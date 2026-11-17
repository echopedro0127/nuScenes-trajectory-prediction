# B2b1 Candidate Verdict Overlay Implementation Plan

> **For agentic workers:** 使用superpowers:executing-plans顺序实现、测试先行；仅只读审查可委托agent。用户已确认[B2b1协议](../specs/2026-11-17-candidate-verdict-overlay-b2b1-design.md)，无需再次确认。

**Goal:** 生成完整pending判决模板、严格导入独立判决CSV为不可变overlay，并从源证据独立重建验收。

**Architecture:** 纯候选／表校验与派生→B2a静态父证明及单独阅读结构校验→模板和overlay发布／verify。新worktree保护B2a及上游；没有真实输入时只导入空版本，不代填确认。

**Tech Stack:** 既有Python、标准库csv/json、NumPy；无新依赖。

## T1 基线与范围

- [x] 从7bf0611创建feature/candidate-verdicts及.worktrees/candidate-verdicts，.worktrees已忽略。
- [x] 创建outputs，复用环境运行B2a新增31项基线。
- [x] 固定模板／导入两种模式；模板包不可变，另存CSV副本供填写。真实空导入使用专门empty输入，不占用用户后续可编辑草稿。

## T2 纯候选与判决

Files: `evaluation/candidate_verdicts.py`, `tests/evaluation/test_candidate_verdicts.py`。

- [x] 先RED：源stop/restart/task一一对应、配对外键／时间／删失，pending模板，CSV完整性／严格Unicode、五状态及25种配对组合。
- [x] `make_template(identity,data)`从已冻结data生成锁定7列＋decision/reviewer/reason；`parse_verdicts(text,baseline)`严格校验并按source_event_id固定排序。
- [x] `derive_overlay(data,rows,review_identity)`保留原事件记录，confirmed/rejected/unresolved分表、stop_go按既定优先级派生，角色／类型／类别分母守恒且空类保留0。
- [x] GREEN后提交；改变reason/reviewer会改变语义摘要，行重排不改变语义摘要，不能把reading reviewed转换成判决。

## T3 可变父证明

Files: `evaluation/verdict_sources.py`, `tests/evaluation/test_verdict_overlay_pipeline.py`。

- [x] 先RED：B2a唯一mutable路径、原owner native proof、静态投影仅排除working_status_counts，strict非负整数及task总和，未知字段拒绝。
- [x] 绑定父11静态文件／completed、245源码及全部引用，不能给working备注绑定不可变hash；使用冻结原B2a版本校验当前working结构。
- [x] 校验器版本检查允许Python通用换行差异，但逻辑源文件必须相同；若不同则拒绝复用，不猜测格式。原owner首次CLI验证；后续复用证明并允许合法reading状态变化。

## T4 发布与独立重建

Files: `evaluation/verdict_overlay.py`, `scripts/make_verdict_template.py`, `scripts/import_candidate_verdicts.py`, `tests/evaluation/test_verdict_overlay_pipeline.py`。

- [x] 先RED：模板→导入→verify、合法reading变化、破坏reading结构、源CSV或副本变化、重哈希策略／派生／删失篡改、已有目录、中断无completed。
- [x] 模板绑定源B2a身份和基线，全部pending；导入前后读取输入字节并绑定path/hash、副本及规范化内容摘要，不合并多表、不自动选最新。
- [x] 完成包内全部文件不可变；verify从源review_data与同一CSV重新构建所有输出和分母，严格比较策略／环境／类型／文件集合，不只哈希。
- [x] 只读复审，Ruff／全套pytest／公开扫描／diff通过，先本地提交源码。

## T5 真实空版本验收

验收记录：源码3bb6e96；58项新增及全套1069项测试通过（273.74秒），Ruff／公开扫描／只读复审通过。模板与全pending空overlay均真实构建并独立verify通过，287原子全pending、60关系pending_components、人工确认0；旧五owner冻结不变。报告与实际文件路径见[B2b1验收](../../CANDIDATE_VERDICTS_B2B1_RESULTS.md)。

```powershell
python scripts/make_verdict_template.py --parent ../scenario-event-review/outputs/event-review/b2a-v1 --parent-owner ../scenario-event-review --output-root outputs/candidate-verdicts/template-v1
python scripts/import_candidate_verdicts.py --parent ../scenario-event-review/outputs/event-review/b2a-v1 --parent-owner ../scenario-event-review --annotations outputs/candidate-verdicts/input-snapshots/pending-empty-v1.csv --output-root outputs/candidate-verdicts/empty-v1
python scripts/import_candidate_verdicts.py --output-root outputs/candidate-verdicts/empty-v1 --verify-only
```

- [x] 将生成模板复制为专用empty输入与另一个可编辑draft；不得编辑模板包或已发布empty源文件。
- [x] 真实模板及空overlay独立verify通过；287原子全pending、60关系pending_components、确认0，原始删失和来源不变。
- [x] 检查旧B2a/B1a/B0b/B0a/3A冻结不变；写验收报告和使用说明，更新设计／路线图。本地提交，不推送。

边界修正、新事件、新anchor／噪声与预测指标均不在本轮；pending或needs_correction不进入确认标签，空版本不包装成人工完成。

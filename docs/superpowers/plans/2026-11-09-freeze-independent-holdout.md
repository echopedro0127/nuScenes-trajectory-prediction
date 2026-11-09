# Freeze Independent Holdout Implementation Plan

**Goal:** 在新holdout推理之前发布可验证、拒绝覆盖的模型/数据/源码协议锁。
**Architecture:** 新evaluation/independent_protocol.py重验现有独立模型链，发布内容寻址JSON及完成marker；公开快照仅包含实验根下相对路径和哈希。原准备/训练源码不改，不运行holdout推理。
**Tech Stack:** Python、PyTorch、pytest、JSON。

协议见[设计](../specs/2026-11-09-independent-holdout-design.md)。用户已授权继续冻结。

- [x] 写测试并确认RED：锁文件发布/验证、内容或父文件篡改拒绝、路径逃逸与覆盖拒绝。
- [x] 实现源文件快照/锁验证及真实独立模型链冻结CLI；冻结时拒绝已产生holdout预测/指标。
- [x] 全量pytest、Ruff、公开扫描，通过后冻结实现源码。
- [x] 运行真实freeze，独立验证锁及模型来源；记录主模型/统计规则/哈希，确认holdout仍未推理。

最终实现91e22c8；573项测试通过；真实协议adbc5d7c…7ccb1231已生成并独立重验，holdout尚未推理。见[阶段报告](../../CAUSAL_CONTEXT_INDEPENDENT_HOLDOUT_PROTOCOL.md)。

本轮不实现或运行holdout推理/评分，后续实现清单须绑定此锁身份。输出outputs/context/p66-independent/holdout-protocol，拒绝已有目录。

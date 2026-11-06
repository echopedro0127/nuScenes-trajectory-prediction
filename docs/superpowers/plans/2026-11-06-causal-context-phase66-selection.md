# P6.6A Precision Selection Diagnostic Implementation Plan

**Goal:** 在不读取 val 的条件下核验 P6.5 train_val 汇总，执行冻结的保守精度选模与 Pareto 诊断。

**Architecture:** 独立分析模块加载冻结父选择，重建三 seed 的选择表；纯函数计算候选、
敏感性与 Pareto；薄 CLI 写入全新输出目录。保留旧模块和实验结果。

**Tech Stack:** Python、NumPy、pandas、pytest，复用原环境。

设计见 [冻结协议](../specs/2026-11-06-causal-context-phase66-selection-design.md)。

## 任务

- [ ] 新增 tests/analysis/test_precision_selection.py：合成完整 22×6×3 矩阵，覆盖总体/
  clean/尾部约束、Pareto/tie/回退、完整矩阵、输入身份和输出不可覆盖。
- [ ] 先运行上述测试确认模块缺失，再实现 analysis/precision_selection.py。
- [ ] 加入真实父产物加载：固定 selection SHA256，复用父选择校验，重建 train_val
  条件与 p95 表并核对，禁止使用 val 路径。
- [ ] 新增 scripts/audit_precision_selection.py，固定默认输出目录 p66-selection，
  输出 CSV/JSON 和最后发布的 completion marker；拒绝已有目录。
- [ ] 运行聚焦测试、全量 pytest、Ruff、公开扫描和 diff 检查，冻结实现提交。
- [ ] 实际运行 CLI 产生候选、Pareto 与敏感性结果。核对旧冻结文件未改，输出不入 Git。
- [ ] 依据实际结果编写 docs/CAUSAL_CONTEXT_PHASE66_SELECTION_RESULTS.md，更新路线图，
  提交结果文档；没有新训练或 val 自动推进。

## 命令

在开发 worktree 中设置 PYTHONPATH=src，使用 ../trajectory-mvp/.venv/Scripts/python.exe。
pytest --basetemp 使用 outputs 下每次新建的 UUID 子目录。

```powershell
& $python -m pytest tests/analysis/test_precision_selection.py -q --basetemp $testRunPath
& $python -m pytest -q --tb=short --basetemp $fullTestPath
& $python -m ruff check src scripts tests
& $python scripts/check_public_repo.py
& $python scripts/audit_precision_selection.py --selection-root outputs/context/p65/selection --output-root outputs/context/p66-selection
git diff --check
```

本计划当前覆盖选模诊断；新的 checkpoint 训练计划须根据结果另立协议。

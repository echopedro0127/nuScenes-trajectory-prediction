# P6.6B Precision-Constrained Checkpoint Implementation Plan

**Goal:** 对同一真实训练轨迹比较ADE-best与保守精度约束下的jitter-best。
**Architecture:** 独立模块复用旧epoch训练/评估/生产加载；保存全部epoch，纯函数选择，
训练后重载并重放所选模型。只使用train/train_val。
**Tech Stack:** Python3.12、PyTorch2.6、NumPy、pytest。

设计见[冻结协议](../specs/2026-11-06-causal-context-phase66-checkpoint-design.md)。

## 分步执行

- [ ] 新增tests/online/test_precision_checkpoint.py：测试精度约束、RH回退、tie、
  非有限值/重复epoch拒绝；用小合成数据训练并重载，篡改state/hash必须拒绝。
- [ ] 先运行测试确认缺少模块，再实现online/precision_checkpoint.py纯函数和训练
  runner。记录每epoch checkpoint、统计、指纹；完成marker最后写入。
- [ ] 新增scripts/train_precision_checkpoint.py，入口固定预算，父选择/缓存/模型走
  原生产验证；不提供val参数。只接受全新输出目录。
- [ ] 聚焦测试、全量pytest、Ruff、公开扫描；冻结实现提交。
- [ ] 实际执行三seed×十epoch，在训练结束后重载所选模型做完整train_val重放；
  验证ADE-best与P6.5原同arm模型及指标一致。
- [ ] 校验epoch身份、输出/源码哈希，形成阶段报告和路线图，本地提交。

## 命令

开发worktree中使用旧完整环境，PYTHONPATH=src，pytest临时目录为outputs下UUID子目录。

```powershell
& $python -m pytest tests/online/test_precision_checkpoint.py -q --basetemp $testRunPath
& $python -m pytest -q --tb=short --basetemp $fullTestPath
& $python -m ruff check src scripts tests
& $python scripts/check_public_repo.py
& $python scripts/train_precision_checkpoint.py --device cuda
```

预算固定30epoch。运行中不增加arm或重写选择规则；完成后仅报告实测结果，不推送。

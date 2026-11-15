# S7：正式运行入口、冻结与验收协议

日期：2026-11-15。本文件描述正式运行的执行约束，不代表已产生实验成绩；真实结果另见 `SCENARIO_EVALUATION_PHASE3A_RESULTS.md`。

## 先冻结，再执行

新增 `scripts/run_scenario_evaluation.py`，统一调用S5预测发布、S6a查询指标、S6b事件/机动组、S6c训练seed统计与共同案例。冻结和执行是两条独立命令：没有有效冻结凭据不能启动统一实验。

正式冻结要求src/scripts/configs及项目依赖声明无Git改动，记录初始HEAD；无推理资源预检后，再次检查这些路径干净且HEAD未变化。冻结源码覆盖当前工作区所有src/scripts Python文件、configs及pyproject/requirements；同时绑定协议、S2凭据、原始父实验引用和Python/NumPy/Torch/Matplotlib运行环境。验证时以源码字节与原始引用为准，后续仅提交结果文档不会改变已冻结的源码身份。

资源计划沿用固定父实验的全部开发观测流：fit 352窗口/3168流，select 256窗口/2304流；9个冻结学习模型、7个物理策略，共32个原始预测分片，float32预测92,454,912字节，不含指标/GT/图像/运行内存。不训练、不改变选模/阈值、不评价已使用的holdout、不下载新数据。

## 执行命令

在 `.worktrees/scenario-evaluation` 内使用既有Python环境，先通过新测试、全套回归、Ruff、公开仓库扫描及diff检查，再提交源码；随后执行：

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/run_scenario_evaluation.py `
  --preflight outputs/scenario-evaluation/preflight-v2 `
  --freeze-root outputs/scenario-evaluation/freeze-phase3a-v1 --freeze-only
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/run_scenario_evaluation.py `
  --preflight outputs/scenario-evaluation/preflight-v2 `
  --freeze-root outputs/scenario-evaluation/freeze-phase3a-v1 `
  --output-root outputs/scenario-evaluation/phase3a-v1
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/run_scenario_evaluation.py `
  --output-root outputs/scenario-evaluation/phase3a-v1 --verify-only
```

config/parent参数如有提供，必须与S2预检一致；执行时preflight还必须与实际freeze绑定路径一致，不会静默忽略参数。所有目录拒绝覆盖，包括失败的部分实验；中断记录failed进度，不写顶层completed。修复实现问题后须说明原因、重新验收并新建冻结/输出版本，不能在同一目录改容差或补造成功标记。

## 顶层完成与模块完成

四个子目录依次为predictions、queries、events、synthesis，各有模块完成凭据。这些凭据不能单独证明整条正式实验通过；模块中的formal_full_run=false表达“模块自身不签发顶层正式验收”，不以它替代顶层状态。S6c scope/HTML已调整为组件语义，避免把正式输入误称为夹具。

顶层先严格核对四阶段的路径链，再重放全部冻结预测、原始旧指标、独立标量新指标和汇总、标签及案例PNG/HTML；各阶段文件落盘后即记录快照，最后再次核对完整文件覆盖和哈希。只有全部通过，才写顶层completed.json和phase3a_verified状态。小夹具使用purpose=fixture且formal_full_run=false；正式开发评测使用formal_development，验收后才标formal_full_run=true。

顶层产物包含protocol.json、run.json、keys.json（关联共享GT/观测流文件）、event_review.csv、summary.json、audit.json、progress.json和completed.json。其余逐查询/事件NPZ、逐log/model-seed JSON及案例图位于子目录。人工状态保持pending，manual_completed=0；查看案例图不是完成标签真值复核。

## 结果解释边界

所有条件分别报告；均值query→noise→window，训练seed在模型级再做ddof=1；M3恢复率按affected，零affected为null；窗口主均值与log宏均值分开。select只有4个log，只作描述统计，不额外增加显著性主张。规则候选按窗口计数，不称独立物理episode。

研究对象是标注位置/静态地图和合成观测退化下的因果回放，不含Camera/LiDAR/Radar感知特征，也不是闭环驾驶或官方nuScenes挑战成绩。新报告评价的是冻结模型在更多指标上的表现，不自动等同模型提升；负结果、空组和人工pending必须保留。

## 开发验收关注点

新增用例覆盖冻结无推理、冻结来源变化拒绝、完整管线及重哈希统计篡改、分阶段失败无顶层完成、CLI预检与freeze错配拒绝、冻结末尾dirty/HEAD变化拒绝。审查发现并修复后两类时序/绑定问题。完整验证记录和正式冻结身份将在实际验收后写入结果报告。

入口开发验收：6项新增测试通过（248.04秒），全套862项pytest通过（892.19秒），Ruff、公开仓库扫描、CLI help及diff检查通过。独立只读复审确认两项必须修复问题已关闭。下一步提交此源码，再新建正式冻结与输出目录。

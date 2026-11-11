# Phase 3A就绪检查：S0协议、S1 GT来源、S2冻结重放

日期：2026-11-11。分支`feature/scenario-evaluation`；实现基线为`d5459bc`，本次新增源码以预检run中的文件哈希绑定。范围来自[执行计划S0–S2](superpowers/plans/2026-11-10-scenario-evaluation-phase3a.md)。

**结论：S0–S2已经实现，并在真实父缓存/冻结checkpoint上通过预检。尚未实现或计算M1–M5等新指标，没有训练新模型或新增holdout评价。** 这次取得的是数据、接口、因果边界与旧数值兼容的验收，不是新的模型提升百分比。

## 1. S0：把文字协议落实为可拒绝错误输入的配置

新建`configs/analysis/scenario_evaluation.yaml`，schema为`scenario_evaluation_v1`。`validate_protocol()`递归检查字段、类型、固定值与未知字段；不允许在同一版本中静默换阈值、跳过缺失配对或换推理后端。

固定规则包括：

- `dt=.5`，query step 1至11；数组行是step−1。
- 固定lead步数1/2/4；目标列`j=k+m-1`，只在`k+m<=12`时有效。
- 同目标前次预测读取row k−2，仅k>=2；1s lead有9个更新对。
- h=.5/1/2s的有效查询分别11/10/8；整数索引、相对查询/目标时间一起输出。
- fit/select两角色、三观测条件、三个noise seed、三控制×三个门控训练seed。
- 固定分位数linear、分母层级、缺失配对报错、JSON非有限数拒绝等后续指标契约。
- float32、CPU两线程、原4096平坦批次、精确兼容；配置不能通过容差选项绕开失败。

新指标阈值虽然已配置，指标实现仍待S3/S4，不能因YAML存在就宣称已产生新结果。

## 2. S1：先追溯GT，再装配评价输入

父Phase 2D NPZ没有历史轨迹或时间戳。新增`scenario_inputs.py`从父run的`source_cache_root`加载原角色轨迹缓存，按`role/instance/sample`建立窗口身份，再核对row、scene、log与未来GT。

返回`ScenarioInputs`，其中模型输入和GT评价输入分开：

| 字段组 | 来源及内容 | 用途 |
| --- | --- | --- |
| datasets | 父features/candidates/target/ca_allowed和保护trace | 冻结推理及旧指标兼容；模型只用前三类中的features/candidates/ca_allowed |
| records | role、原row、instance/sample、scene/log、condition、noise seed | 观测流身份、完整性与配对检查 |
| gt_windows | 原GT历史5点、mask、未来12点、anchor、坐标与来源引用 | 后续方向、事件及误差评价 |
| 相对时间 | 历史−2至0s、查询0.5至5.5s、未来0.5至6s | 不伪造绝对timestamp；canonical episode留3B |
| references/owner_roots | 文件SHA256、parent身份及实际工作区映射 | 独立重验与追溯 |

采用现有anchor-agent局部坐标（y轴向前），单位m；保留原anchor全局位置与yaw，不把门控自身估计速度当GT。完整历史mask为v1前提，缺失时拒绝，不补干净GT到观测流。

检查覆盖整个原角色缓存的scene/log/key交集，而非只检查608个pilot窗口；两角色partition及预处理身份必须一致。每个所选窗口必须完整包含3条件×3noise seed的9条观测流，缺预期配对或重复流直接报错。

### 真实装配数量

| 角色 | 不同pilot窗口 | 观测流 | 原角色log数 |
| --- | ---: | ---: | ---: |
| independent_fit | 352 | 3,168 | 22 |
| independent_select | 256 | 2,304 | 4 |
| 合计 | 608 | 5,472 | 26 |

本次绑定586个文件引用；parent_path链记录4个父实验身份→owner工作区映射。直接父root由参数显式指定，上游root通过Git查询实际worktree并固化，不依赖`parent.parents[n]`推断。本机冻结父入口原有布局通过验证；复制到新目录后仍需满足原父验证器的路径约束或重新准备来源。

## 3. S2：冻结推理并严格复现父数值

门控调用原`predict_motion`，不实现另一套前向；使用原checkpoint、fit-only标准化、`model.eval()`和CPU两线程。为保留原4096平坦批次，内部对完整role数组前向，再提取确定性选定窗口及其批次邻域；不改成batch=1后放宽容差。

7物理策略直接读取父候选列，不重估滤波参数。所有策略的旧指标仍调用原`forecast_metrics`，GT仅在预测完成后进入评价。

| 项目 | 实际核验范围 |
| --- | --- |
| 门控 | baseline7/plus_ca/guarded_ca × seed2027/2028/2029，共9模型 |
| 物理策略 | raw_cv/alpha_beta/kf_cv/kf_nis/bounded_ca/brake_stop/hold_position，共7策略 |
| fit批次邻域 | 原观测流索引2234–2606，共373条；平坦batch id 6 |
| select批次邻域 | 原观测流索引1489–1861，共373条；平坦batch id 4 |
| 神经指标 | 9模型在上述两角色的ADE/FDE/jitter逐更新数组，与父引用精确相等 |
| 物理指标 | 7策略的ADE/FDE/jitter逐窗口均值，与父CSV精确相等 |
| 磁盘重载 | 两角色×16策略，共32个P/旧指标NPZ分片，再加载模型复算并逐项对照 |

物理策略父产物并非全部保存逐更新数组，所以本阶段只主张其窗口均值对父兼容；磁盘P重放检查与该兼容口径分别记录，不夸大为7策略全部逐更新父引用比较。

本次共核验746条所选观测流×16策略，即11,936条策略级重放记录。存在大量重复窗口/噪声重复，不是11,936条独立轨迹。fit所选邻域全部为outlier条件，select包含47条noise_05、326条outlier；这是按原批次边界选定的接口预检，不代表三条件均衡全量评价。

保存的每策略P形状为`[373,11,12,2]`、float32。原始预测坐标共12,604,416字节，不含指标/键和峰值RAM；本轮32个压缩NPZ合计12,273,961字节。

## 4. 父链与独立审计

首次预检只调用一次原工作区的`run_noise_aware_stability.py --verify-only`，保存原stdout、stderr、执行命令及owner。原证明确认：全门控指标/机动/权重重算、fit-only标准化、2027两控制复现父模型均通过，未评价holdout；guard trace仍为原64流抽查口径。

`--verify-only`执行本阶段独立审计：

1. 重验预检完成标记、run身份、新源码/config、586个来源引用。
2. 重新装配GT与窗口身份，核对确定性选样、协议时间网格和键文件。
3. 完整复查父证明全部三个成功标志、identity、holdout、owner、命令与stdout最后JSON一致性；复用已记录的原生验证证明，不每策略递归调用父入口。
4. 重载冻结模型，按原批次路径重新预测所选邻域，比较保存的P及旧指标。
5. 从重放输出重新计算检查数量和报告结构，不能仅校验summary文件哈希。

测试额外验证：改变后来query特征/候选，不改变此前已发布预测；修改评价侧GT历史不进入预测；禁止创建Adam或调用backward；checkpoint被改、错误统计、错角色/身份、非有限输入、缺包/缺配对及输出覆盖均拒绝。

这证明冻结适配层的因果接口；上游观测候选如何构建仍由已冻结父链及既有前缀测试保证，不把一个适配层测试说成真实感知端到端因果验证。

## 5. 审查发现、修复与版本选择

首次真实预检`preflight-v1`已经数值通过，随后独立只读审查发现一个Important：复验遗漏父证明的fit-only和2027父控制成功标志，仅更新产物哈希就可能掩盖这些字段与首次标准不一致。

已先添加5项更新文件哈希后的证明篡改回归，确认全部失败，再统一首次与复验的`check_parent_proof()`。同时将当前`noisy_candidates.py`加入直接依赖来源绑定。复核确认原问题解决，无新增Critical/Important。

正式使用重新运行的`preflight-v2`；`preflight-v1`保留追溯，不覆盖。两版本差异是证明核验与来源绑定，不能当作模型性能改进。新增S0–S2测试共55项，包含协议20项及后续装配/重放/审计边界。

最终全量回归**691 passed in 163.47s**，Ruff通过；原基线636项加新增55项。v2独立`--verify-only`通过，32个分片内全部P、旧指标及索引数组与v1逐项相等，确认审计修复不改变预测。文档链接/编码、工作量算术、公开文件扫描与diff检查均通过；代码审查原Important已复核关闭。

## 6. 正式产物与复现

输出：`outputs/scenario-evaluation/preflight-v2/`。预检identity：

`12e2584ecf0a0a0fd0d3e847cc9279f124d084f346ea2954f221a0e7ec9b9fab`

父Phase 2D identity：

`ca0792a001be3fd361239d41c631080e25d3d30ed490762b5d83a66fb9c2568f`

产物含`run.json/protocol.json/parent_proof.json/report.json`、两个角色键文件、32个预测/指标NPZ、`preflight_completed.json`。就绪schema为`scenario_readiness_s0_s2_v1`，不使用完整3A的completed名称。

运行时HEAD仍为实现前提交，run同时绑定实际新源码哈希；不能仅凭旧HEAD推断本次实现内容。最终本地提交纳入这些源码，后续复验以绑定的文件哈希和身份为准。

```powershell
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:OMP_NUM_THREADS = '2'
$env:MKL_NUM_THREADS = '2'
# 已完成输出不能覆盖；复现请另给新目录。
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/run_scenario_preflight.py `
  --parent ../noise-aware-stability/outputs/noise-aware/phase2d `
  --parent-project-root ../noise-aware-stability `
  --config configs/analysis/scenario_evaluation.yaml `
  --output-root outputs/scenario-evaluation/preflight-reproduction
& '..\trajectory-mvp\.venv\Scripts\python.exe' scripts/run_scenario_preflight.py `
  --output-root outputs/scenario-evaluation/preflight-v2 --verify-only
```

## 7. 下一步与边界

下一步S3实现固定lead纯指标及慢速标量参考，再按依赖接入S4a增益/尾部、S4b方向、S4c恢复和S4d窗口标签。本阶段还没有这些新分数、事件标签或人工复核成绩，也没有S5/S6正式发布器及完整S7全量评测。

原Phase 2D及其父工作区源码、模型和成绩保持不变。本阶段不推送GitHub、不上传数据/模型、不修改简历；本地详细进展文档另行保留。

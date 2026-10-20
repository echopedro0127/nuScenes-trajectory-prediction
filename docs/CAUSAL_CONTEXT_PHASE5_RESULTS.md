# P5：mini Radar 与同观测退化实验

状态：P5工程与完整固定mini实验已完成。420项pytest通过；Ruff、公开仓库扫描和diff检查通过。
开发分支保留feature/causal-context-online，不推送；原始数据、缓存、模型、逐行结果未提交。

## 数据与固定协议

mini_train570条/8场景按scene_partition分为409条/6场景训练、161条/2场景选模。
五条件各11更新，共22,495/8,855个训练/选模条件更新。完整mini_val47条/2场景，
每条件每策略517更新、每seed15,510行；三seed共46,530行，失败0，缺回波样本不删行。

固定seeds2027/2028/2029、hidden64、batch1024、最多20epoch、patience5，
AdamW lr=.001/weight_decay=.0001、SmoothL1+.05smoothness。五条件等权混合训练/选模；
只按训练集内部选模场景的平均ADE选checkpoint，未用mini_val调参数。
P5 original是同8维架构重训的对照，不是P4完整trainval checkpoint；基础mini GRU固定。

退化从t0后开始，初始2秒历史和anchor干净且所有策略共享，不能称全程定位损坏。
五条件：clean、逐更新sigma=1m高斯噪声、0.5s/1s交付延迟、丢弃第4/5/6次位置更新。
扰动按instance/sample/update与固定seed2027生成，与模型训练seed及遍历顺序无关。
仅使用已交付位置，速度按不同source的真实dt计算，再外推到query；所有模型和Radar
门控均使用该同一因果状态。简单滤波固定alpha=.6、beta=.2，不在mini_val选参数。
仍只预测原t0+6秒终点之前的剩余轨迹；首步jitter=0，后续比较共享绝对未来时刻。

## Radar输入与适用限制

每scene五通道各取过去最多2秒内最近3帧，source=availability；不使用未来pose插值。
位置sensor→ego→global→固定anchor；补偿速度vx_comp/vy_comp只旋转，不加平移。
固定过滤invalid_states=[0]、dynprop_states=0..6、ambig_states=[3]；依据见
[nuScenes官方RadarPointCloud实现](https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py)。

退化位置附近4m半径门控，完全不用annotation bbox或未来位置。13维统计含点数、
相对位置、补偿速度中位数/MAD、RCS、质量、age及通道比例；缺失为全0+mask。
补偿速度是回波测量，未证明为目标真实速度；门控可能混入其他物体，高覆盖不等于
关联正确。历史点只做ego-motion对齐、不做目标运动补偿，保留age及拖影限制。

实际读取mini_train4,554个PCD、mini_val982个PCD；未读Camera/LiDAR文件、未下载。
独立NPZ/manifest绑定metadata与实际PCD哈希，加载独立重建所需来源集合、anchor、
五条件state/filter和每query应选frame；旧trajectory schema保持。

Radar继承并冻结同seed original，完全缺失回波时严格fallback。matched_radar宽度74，
匹配total参数（差约0.59%）且全部可训练，故预训练继承、trainable参数、训练计算
不匹配，不能将宽度对照解释为纯上下文收益的因果证明。多数最佳epoch靠近20轮上限，
本小数据固定训练预算不代表模型已充分优化。

## 场景与实际进度


训练场景：scene-0061, scene-0553, scene-0655, scene-0757, scene-0796, scene-1077。
选模场景：scene-1094, scene-1100。
评估场景：scene-0103, scene-0916。

| 模型 | total/trainable参数 | best/completed 2027 | 2028 | 2029 |
| --- | ---: | ---: | ---: | ---: |
| original | 4866/4866 | 18/20 | 18/20 | 20/20 |
| radar | 6404/1538 | 20/20 | 19/20 | 20/20 |
| matched_radar | 6366/6366 | 18/20 | 18/20 | 20/20 |

## 五条件六策略完整指标

单位米；均值±三个训练seed样本标准差（ddof=1）。

| 条件 | 策略 | ADE | FDE | jitter |
| --- | --- | ---: | ---: | ---: |
| clean | frozen | 7.239052 ± 0.000000 | 8.946203 ± 0.000000 | 0.000000 ± 0.000000 |
| clean | sliding | 2.203285 ± 0.000000 | 5.005146 ± 0.000000 | 1.018701 ± 0.000000 |
| clean | filtered | 3.387421 ± 0.000000 | 6.694888 ± 0.000000 | 0.937381 ± 0.000000 |
| clean | original | 6.112666 ± 0.237376 | 9.323569 ± 0.523794 | 4.522025 ± 0.325069 |
| clean | radar | 5.958220 ± 0.087878 | 9.184056 ± 0.133529 | 4.151541 ± 0.220564 |
| clean | matched_radar | 6.141214 ± 0.077422 | 9.193884 ± 0.126857 | 4.183237 ± 0.079602 |
| noise | frozen | 7.239052 ± 0.000000 | 8.946203 ± 0.000000 | 0.000000 ± 0.000000 |
| noise | sliding | 7.190922 ± 0.000000 | 12.062221 ± 0.000000 | 11.513826 ± 0.000000 |
| noise | filtered | 3.940704 ± 0.000000 | 7.225124 ± 0.000000 | 2.067565 ± 0.000000 |
| noise | original | 7.109338 ± 0.196387 | 10.057051 ± 0.408121 | 6.329342 ± 0.139591 |
| noise | radar | 6.924692 ± 0.087048 | 9.958049 ± 0.098592 | 5.892616 ± 0.098806 |
| noise | matched_radar | 7.236954 ± 0.147505 | 10.052383 ± 0.126829 | 6.599567 ± 0.146725 |
| delay_500ms | frozen | 7.239052 ± 0.000000 | 8.946203 ± 0.000000 | 0.000000 ± 0.000000 |
| delay_500ms | sliding | 3.026487 ± 0.000000 | 6.175129 ± 0.000000 | 1.242635 ± 0.000000 |
| delay_500ms | filtered | 4.263374 ± 0.000000 | 7.792127 ± 0.000000 | 0.970587 ± 0.000000 |
| delay_500ms | original | 6.252575 ± 0.279339 | 9.592077 ± 0.592897 | 4.518073 ± 0.401356 |
| delay_500ms | radar | 6.023069 ± 0.080597 | 9.390537 ± 0.166650 | 4.075267 ± 0.309489 |
| delay_500ms | matched_radar | 6.267482 ± 0.087050 | 9.490576 ± 0.159069 | 4.153270 ± 0.144541 |
| delay_1000ms | frozen | 7.239052 ± 0.000000 | 8.946203 ± 0.000000 | 0.000000 ± 0.000000 |
| delay_1000ms | sliding | 3.879909 ± 0.000000 | 7.300231 ± 0.000000 | 1.268967 ± 0.000000 |
| delay_1000ms | filtered | 5.119225 ± 0.000000 | 8.813320 ± 0.000000 | 0.937831 ± 0.000000 |
| delay_1000ms | original | 6.614336 ± 0.344673 | 10.086715 ± 0.658577 | 4.713775 ± 0.431961 |
| delay_1000ms | radar | 6.327768 ± 0.080314 | 9.869957 ± 0.207557 | 4.253449 ± 0.364557 |
| delay_1000ms | matched_radar | 6.644946 ± 0.124796 | 10.044363 ± 0.215243 | 4.350688 ± 0.166147 |
| dropout | frozen | 7.239052 ± 0.000000 | 8.946203 ± 0.000000 | 0.000000 ± 0.000000 |
| dropout | sliding | 2.785395 ± 0.000000 | 5.844308 ± 0.000000 | 0.960678 ± 0.000000 |
| dropout | filtered | 4.228504 ± 0.000000 | 7.766158 ± 0.000000 | 0.954389 ± 0.000000 |
| dropout | original | 6.285151 ± 0.231848 | 9.559682 ± 0.572576 | 4.486626 ± 0.409968 |
| dropout | radar | 6.108900 ± 0.101157 | 9.435026 ± 0.187986 | 4.083562 ± 0.314521 |
| dropout | matched_radar | 6.320326 ± 0.100818 | 9.498583 ± 0.181594 | 4.104285 ± 0.092340 |

## 覆盖与fallback

| 条件 | 有回波更新 | 平均点数 | Radar fallback更新/seed |
| --- | ---: | ---: | ---: |
| clean | 88.3946% | 9.778 | 60 |
| noise | 90.1354% | 9.870 | 51 |
| delay_500ms | 90.1354% | 9.702 | 51 |
| delay_1000ms | 90.9091% | 9.658 | 47 |
| dropout | 89.3617% | 9.700 | 55 |

噪声/延迟时覆盖更高只说明门控内出现点，不说明关联更准。

## 配对场景差值（仅描述）

先对固定三seed场景sum取平均，再做2000次配对scene-cluster bootstrap，统计seed0。
仅2个场景，区间不能支持泛化声明；负值表示Radar更低。策略自身CI见report.json。

| 条件 | 对比 | ADE差值 [95%区间] | FDE差值 [区间] | jitter差值 [区间] |
| --- | --- | ---: | ---: | ---: |
| clean | radar-original | -0.154446 [-0.607629, 0.057984] | -0.139513 [-1.034798, 0.280152] | -0.370484 [-0.501168, -0.091691] |
| clean | radar-matched_radar | -0.182994 [-0.331629, -0.113322] | -0.009828 [-0.563030, 0.249485] | -0.031696 [-0.192955, 0.312322] |
| noise | radar-original | -0.184646 [-0.494700, -0.039308] | -0.099003 [-0.756648, 0.209269] | -0.436725 [-0.490735, -0.321504] |
| noise | radar-matched_radar | -0.312262 [-0.586993, -0.183482] | -0.094335 [-0.638237, 0.160619] | -0.706950 [-0.708866, -0.702864] |
| delay_500ms | radar-original | -0.229506 [-0.742492, 0.010957] | -0.201540 [-1.173261, 0.253955] | -0.442807 [-0.598557, -0.110538] |
| delay_500ms | radar-matched_radar | -0.244412 [-0.502712, -0.123334] | -0.100039 [-0.789686, 0.223233] | -0.078003 [-0.273245, 0.338511] |
| delay_1000ms | radar-original | -0.286569 [-0.868426, -0.013823] | -0.216758 [-1.207969, 0.247871] | -0.460326 [-0.629008, -0.100472] |
| delay_1000ms | radar-matched_radar | -0.317179 [-0.718166, -0.129216] | -0.174407 [-0.966857, 0.197055] | -0.097239 [-0.317395, 0.372428] |
| dropout | radar-original | -0.176251 [-0.692008, 0.065510] | -0.124656 [-1.066722, 0.316937] | -0.403064 [-0.521193, -0.151055] |
| dropout | radar-matched_radar | -0.211426 [-0.463038, -0.093483] | -0.063557 [-0.715682, 0.242127] | -0.020723 [-0.194480, 0.349961] |

## 丢点恢复

第7次重新交付后，ADE相对同策略clean条件差值<=.25m且连续两次才算恢复；
记录满足窗口起点距第7次的秒数，未恢复在2秒处右删失，不当成0秒。
141行=47条轨迹×3个固定seed，仅作描述，不当作141个独立试验。

| 策略 | 恢复/141 | 未恢复 | 已恢复者平均秒数 |
| --- | ---: | ---: | ---: |
| filtered | 135 | 6 | 0.955556 |
| frozen | 141 | 0 | 0.000000 |
| matched_radar | 141 | 0 | 0.088652 |
| original | 141 | 0 | 0.092199 |
| radar | 141 | 0 | 0.092199 |
| sliding | 141 | 0 | 0.361702 |

frozen忽略新位置，恢复=0只表示与自身clean预测相同，不代表准确。Radar与original
平均恢复时间相同，未观察到加速恢复；filtered有6/141行右删失。

## 单次真实输入与缓存路径计时

固定相同256个condition/sample/update key，每seed每路径16次warmup，CUDA同步、batch=1。
raw禁用解码缓存，计入位置replay、选帧、PCD读/解码、变换、门控聚合、特征/传输、
head及返回；不含metadata索引启动/来源核验，OS文件缓存可能已热，不称物理冷盘性能。
cached读取预计算统计。下表是三个seed各自百分位范围，不将百分位平均成新分布。

| 路径 | 总计p50范围/ms | 总计p95范围/ms | read/decode p50范围/ms |
| --- | ---: | ---: | ---: |
| cached_radar | 1.841–2.153 | 2.955–4.089 | 0.000–0.000 |
| raw_radar | 36.857–38.227 | 41.564–43.373 | 15.738–16.185 |

逐次及分段计时在latency*.csv；Radar总计还含路径检查、索引、查表等开销，
不等于列出阶段百分位的和。batch forward耗时另存summary，不是单次输入路径延迟。

## 结论与限制

Radar三seed平均ADE在五条件下均低于original/matched_radar，如noise为7.109338→6.924692m。
但clean、0.5s延迟、dropout的radar-original ADE场景区间跨0，各seed方向也不一致。
仅2个评估场景，不能据这些区间证明稳定泛化。
noise的filtered ADE=3.940704m，远低于Radar；其他四条件sliding ADE也明显更低。
learned模型FDE/jitter也落后于相应强基线，不宣称整体方案优越。
初始干净oracle历史/anchor、mini规模、混合退化训练预算、非目标专属回波、参数/
继承预训练不匹配和真实输入路径耗时限制保留；未根据mini_val结果反复调参。

P5工程与固定可行性实验结束。P6仍需独立细化，以P4已显示价值的地图和强CV/滤波
对照为依据；本结果不足以自动支持更复杂Radar编码或学习门控。

## 复现

正式训练和评估代码revision c93b800；报告代码0202e1e。完整来源hash在metadata。
CPU单epoch三seed短训练另存outputs/context/p5/smoke。

```powershell
$python = (Resolve-Path ../trajectory-mvp/.venv/Scripts/python.exe).Path
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
& $python scripts/prepare_radar_context.py --trajectory-root ../../data/cache/mini --prediction-root ../../outputs/mini --dataroot ../../data/nuscenes --output-root outputs/context/p5/cache
& $python scripts/train_radar_corrector.py --trajectory-root ../../data/cache/mini --prediction-root ../../outputs/mini --dataroot ../../data/nuscenes --context-root outputs/context/p5/cache --output-root outputs/context/p5/training --device cuda
& $python scripts/evaluate_radar_online.py --trajectory-root ../../data/cache/mini --prediction-root ../../outputs/mini --dataroot ../../data/nuscenes --context-root outputs/context/p5/cache --training-root outputs/context/p5/training --output-root outputs/context/p5/evaluation --device cuda
& $python scripts/report_radar_ablation.py --runs outputs/context/p5/evaluation/seed-2027 outputs/context/p5/evaluation/seed-2028 outputs/context/p5/evaluation/seed-2029 --output outputs/context/p5/report
```

## 来源

| 产物 | SHA256 |
| --- | --- |
| mini_train Radar NPZ | `6552265d35a1e3af4bea343cb41c25baeb03ef4710b63ae96f013008bc303529` |
| mini_val Radar NPZ | `942ed4531e71013f7dbcacfe6dd7eed5e8fe3948f04e11e8c65a4dd5fd5cda61` |
| seed2027 best.pt | `6d59ae79503874ae29f2048eeb1c6500279ec710601a225ef7bb1f1af6b4193b` |
| seed2028 best.pt | `5b09e1b394ecb9853011bc76a3b03bc99a39e92cee47ecc83b8bf41347c79812` |
| seed2029 best.pt | `543da490a6199112c6007e680ff19a3ab17ddb0bc3d2c8e704e8b0228ad387f4` |
| 固定GRU checkpoint_sha256 | `d0af99801477de295b11fe6eeae1e8dada4ad74eb2cec57e6dfb8cb521da5ddf` |
| 固定GRU data_config_sha256 | `dcf12189c5e5ddfb05a5e40dd10eefb0c41fa0aa3b6b9f847a8c745c2a5ff03b` |
| 固定GRU model_config_sha256 | `88d7d5b77bcdc950b53345720f04a8bc132812c182574de447e883ac74974d24` |
| 三seed相同latency keys | `01ded4c8231748c75817bdca6a07dc1cf89eb8e5c85a91b6751b0f8f182a3ab9` |
| 完整report.json | `20d6b07e4c58b412626027000f89a126bd1543743444b98ef40860da49f2241d` |

## 独立审查与验收

审查覆盖sensor/state/cache/training/evaluation/report，并复核以下回归修复：

- 必需metadata/已选PCD集合独立确定，拒绝source遗漏。
- 由原轨迹重建所有退化/滤波状态和帧选择，拒绝错误或泄漏中心。
- 子集恢复报告按真实sample_indices标识。
- 单次计时完整总计及六阶段强制校验，百分位与逐次测量一致。
- 各策略、各seed的逐行传感器诊断一致，fallback与coverage相符。

均先复现失败再修复并通过回归，独立复核未发现其他重要问题。
最终420项pytest通过；Ruff、公开仓库扫描、diff检查通过。

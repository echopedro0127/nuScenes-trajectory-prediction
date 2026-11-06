# P6.6C/1 独立分组划分与GRU上游重训

日期2026-11-07，父提交c88cc3f。用户要求继续独立验证方向。

## 本轮范围

旧GRU在原train拟合，再把原train中的场景称为未见holdout会产生上游暴露。本轮先
固定新训练链的fit/select/holdout分组，生成独立缓存，审计旧模型暴露，并从头重训
一份GRU。地图修正、质量门控和一致性head的重训是后续阶段，本轮不评估holdout。

这不是全新采集的数据盲测：原数据参与过旧研究，但新模型链可以在重划分上严格排除
holdout。当前只完成新GRU的隔离，不代表整个预测链已经完成独立验证。

## 固定划分

仅使用原官方train缓存的有效轨迹，不混入旧train_val或val。用scene.json解析每个
scene对应的log_token。同一log的所有有效scene必须进入同一role，避免相邻采集片段
跨集合。按SHA256('p66c:20271107:'+log_token)、token排序，前floor(0.70*G)个log为
independent_fit，随后floor(0.15*G)个为independent_select，其余为independent_holdout。
G为有效源场景的log数，至少7；不根据轨迹标签、指标或划分后的数量调整seed/边界。
比例针对log数，不保证scene/轨迹数量恰好70/15/15。位置分布仅描述，不用于事后重划。

保留源缓存行序；scene/log/instance+sample key三层均不得跨role重复，所有源行恰好
分配一次。原缓存按旧checksum/schema校验，重复key、缺失scene、冲突log映射均拒绝。

## 来源与暴露审计

旧GRUcheckpoint的data_config_sha256必须等于原缓存预处理哈希，train_split为train。
其文件SHA256还须等于原P6缓存base_hashes.model_identity中的checkpoint_sha256，
并核对model/data config身份，确认当前链使用的确是该旧GRU。

新holdout来自旧GRU的训练集合，因此旧GRU及使用其预测的旧map/gate/RH/consistency
链不能用于新独立评估。暴露计数来自数据/模型来源，不执行旧模型的holdout预测。
纯静态地图与原始标注可以重用，但必须按新key/时间/配置重建上下文契约。

新增独立data/independent_split.py与prepare_independent_validation.py，不改旧schema。
输出新cache目录及data.yaml；cache manifest保存父manifest/NPZ、scene.json、旧GRU/
P6来源SHA256、分组规则/清单哈希。holdout标签只机械复制/校验，不计算误差统计。
所有split的预处理哈希绑定同一data.yaml及分组身份。原cache和旧实验产物保持不变。
完成marker最后写，已有输出目录拒绝覆盖。

## GRU重训预算

复用原scripts/train.py与configs/models/gru.yaml；新数据配置仅列fit/select/holdout。
训练只传independent_fit与independent_select；不用--resume，不读旧模型权重。
seed2027、原单层GRU hidden64、80epoch上限、batch256、lr.001、weight_decay.0001、
patience12；按select ADE选best。CUDA/CuBLAS和seed按原训练路径记录，不新增搜索。
新输出目录独立，data.yaml哈希将分组与训练身份绑定。

训练后校验best metadata的数据/模型配置哈希、split名、history_steps、epoch与有限指标，
加载新fit/select/holdout manifest验证scene/log/key隔离，证明本轮GRU未拟合或选模于
holdout。该证明依赖实际执行的原train.py和固定命令，不等于全部下游已隔离。

## 验收和后续

测试分组确定性、输入顺序不变性、同log闭包、无重复key、父模型身份拒绝、缓存逐行
来源、旧上游暴露拒绝、数据配置可被原训练CLI读取、完成文件哈希与禁止覆盖。
全量pytest/Ruff/公开扫描通过，冻结代码后生成真实划分并训练一个新GRU。
报告实际log/scene/轨迹数量、暴露计数和训练预算，不做新旧不同split的性能对比。
下一阶段重建新GRU预测与map/gate训练契约；本轮不推送、不下载、不改简历或开启P7。

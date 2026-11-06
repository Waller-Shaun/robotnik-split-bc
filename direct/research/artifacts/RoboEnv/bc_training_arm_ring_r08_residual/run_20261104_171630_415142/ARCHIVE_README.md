# bc_training_arm_ring_r08_residual / run_20261104_171630_415142

分类：历史正式训练运行／模型快照。

训练结束元数据见 [training_finished.json](training_finished.json)，以该快照为准。

| 检查点 | 用途 | SHA256 |
|---|---|---|
| [baseline_model.pt](baseline_model.pt) | 冻结父模型对照，不是新模型 | `87b76efbc0f5a7c2d6b668e88b5e765771d3e0c496e9bb838c30e4fd54c7b5e2` |
| [best_model.pt](best_model.pt) | 按该运行实际选模规则保存；不自动代表超过父模型 | `80ebc384d64be494fe5f93cecc6a08fd36177f0d998104bb1f847984a8712121` |
| [last_model.pt](last_model.pt) | 最后训练状态，部分包含优化器，供续训；不等于最佳 | `95782c6c118837688f325a9efb16a7e70a0237ef24f76ac1e1eeb7e1cb85b13c` |

本目录保持源文件字节不变。配置和检查点内部可能保留历史绝对路径；这是归档，不代表解压后能在任意目录直接运行。配套策略类、父模型、归一化与导航均需匹配，不能只加载某个同名best_model.pt。原始大体积rollout NPZ未随本分支全部提交，summary/CSV及子目录小型诊断记录已保留。

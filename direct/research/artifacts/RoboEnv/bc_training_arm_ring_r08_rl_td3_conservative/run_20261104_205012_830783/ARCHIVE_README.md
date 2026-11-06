# bc_training_arm_ring_r08_rl_td3_conservative / run_20261104_205012_830783

分类：历史正式训练运行／模型快照。

训练结束元数据见 [training_finished.json](training_finished.json)，以该快照为准。

| 检查点 | 用途 | SHA256 |
|---|---|---|
| [best_model.pt](best_model.pt) | 本次最终选中初始零残差策略；不是RL提升 | `d98b4d77eb7b1559769ca7a374d7253458dd99cacf613709b94c7bb66e63de0f` |
| [initial_model.pt](initial_model.pt) | 训练前初始策略／零残差，不能作为新训练成果 | `be22d3e68f9973fb4fce1b7a7b8e60aefd3f9fef92ae0b1bf0f56383a6f5e3b8` |
| [last_model.pt](last_model.pt) | 最后训练状态，部分包含优化器，供续训；不等于最佳 | `a1b8408b9b1aeea844af5e666d82578628a0f9fc7b17ffe6077883c9b1a65503` |

本目录保持源文件字节不变。配置和检查点内部可能保留历史绝对路径；这是归档，不代表解压后能在任意目录直接运行。配套策略类、父模型、归一化与导航均需匹配，不能只加载某个同名best_model.pt。原始大体积rollout NPZ未随本分支全部提交，summary/CSV及子目录小型诊断记录已保留。

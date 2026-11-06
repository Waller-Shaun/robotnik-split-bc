# bc_training_arm_ring_r08_rl_td3 / run_20261104_200609_974210

分类：历史正式训练运行／模型快照。

训练结束元数据见 [training_finished.json](training_finished.json)，以该快照为准。

| 检查点 | 用途 | SHA256 |
|---|---|---|
| [best_model.pt](best_model.pt) | 固定预算末轮策略；未通过验证选优，文件名best不代表最佳 | `cd9dcb3ec1c26662a8d4f1a04f1845a4c09b70d1b15bdd9bfb6ef943e5753aa5` |
| [initial_model.pt](initial_model.pt) | 训练前初始策略／零残差，不能作为新训练成果 | `a978c80c1af98074d9eddd75313f7dfcd911692409ec60b5014a43a58b40dfeb` |
| [last_model.pt](last_model.pt) | 最后训练状态，部分包含优化器，供续训；不等于最佳 | `cd9dcb3ec1c26662a8d4f1a04f1845a4c09b70d1b15bdd9bfb6ef943e5753aa5` |

本目录保持源文件字节不变。配置和检查点内部可能保留历史绝对路径；这是归档，不代表解压后能在任意目录直接运行。配套策略类、父模型、归一化与导航均需匹配，不能只加载某个同名best_model.pt。原始大体积rollout NPZ未随本分支全部提交，summary/CSV及子目录小型诊断记录已保留。

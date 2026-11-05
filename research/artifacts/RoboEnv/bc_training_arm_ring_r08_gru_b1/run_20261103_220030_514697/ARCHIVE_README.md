# bc_training_arm_ring_r08_gru_b1 / run_20261103_220030_514697

分类：历史正式训练运行／模型快照。

训练结束元数据见 [training_finished.json](training_finished.json)，以该快照为准。

| 检查点 | 用途 | SHA256 |
|---|---|---|
| [best_model.pt](best_model.pt) | 按该运行实际选模规则保存；不自动代表超过父模型 | `2bb5ebd5b38be599990abb9e1e8d50710db1c9a2f704cc576c6d91fdf8a8568b` |
| [last_model.pt](last_model.pt) | 最后训练状态，部分包含优化器，供续训；不等于最佳 | `aa1ce1a7dc15fafcc90d526bad37265a73df8f60e46b1a67b37847700111d4f9` |

本目录保持源文件字节不变。配置和检查点内部可能保留历史绝对路径；这是归档，不代表解压后能在任意目录直接运行。配套策略类、父模型、归一化与导航均需匹配，不能只加载某个同名best_model.pt。原始大体积rollout NPZ未随本分支全部提交，summary/CSV及子目录小型诊断记录已保留。

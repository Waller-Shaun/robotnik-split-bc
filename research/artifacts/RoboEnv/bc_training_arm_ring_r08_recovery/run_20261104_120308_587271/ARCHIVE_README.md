# bc_training_arm_ring_r08_recovery / run_20261104_120308_587271

分类：历史正式训练运行／模型快照。

训练结束元数据见 [training_finished.json](training_finished.json)，以该快照为准。

| 检查点 | 用途 | SHA256 |
|---|---|---|
| [best_model.pt](best_model.pt) | 第0轮父模型；旧选择规则未接受微调轮次，并非训练未执行 | `3526f63813a9322baa476631a1b37125b91d40dfa699ed398389c184dbc578c7` |
| [last_model.pt](last_model.pt) | 最后训练状态，部分包含优化器，供续训；不等于最佳 | `2a1972e2f9a63b46b150ce1b0bfdb4540d5d02cde35f54fdff9ec7dd01c9e725` |

本目录保持源文件字节不变。配置和检查点内部可能保留历史绝对路径；这是归档，不代表解压后能在任意目录直接运行。配套策略类、父模型、归一化与导航均需匹配，不能只加载某个同名best_model.pt。原始大体积rollout NPZ未随本分支全部提交，summary/CSV及子目录小型诊断记录已保留。

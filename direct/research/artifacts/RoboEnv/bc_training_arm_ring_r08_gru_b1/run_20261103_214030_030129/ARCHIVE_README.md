# bc_training_arm_ring_r08_gru_b1 / run_20261103_214030_030129

分类：助手执行的续训探索（最佳55轮）；保留并标注，非新的用户完整评估。

训练结束元数据见 [training_finished.json](training_finished.json)，以该快照为准。

| 检查点 | 用途 | SHA256 |
|---|---|---|
| [best_model.pt](best_model.pt) | 按该运行实际选模规则保存；不自动代表超过父模型 | `d37899d31422c6f2a901939bfbc90ada9df75e3ccd4b1facc495d12255a4bad1` |
| [last_model.pt](last_model.pt) | 最后训练状态，部分包含优化器，供续训；不等于最佳 | `f162a7fcdd7d89a2fbe4cca3aa8cc0795a4a15d6a1e57106b956ac4a9dfe8074` |

本目录保持源文件字节不变。配置和检查点内部可能保留历史绝对路径；这是归档，不代表解压后能在任意目录直接运行。配套策略类、父模型、归一化与导航均需匹配，不能只加载某个同名best_model.pt。原始大体积rollout NPZ未随本分支全部提交，summary/CSV及子目录小型诊断记录已保留。

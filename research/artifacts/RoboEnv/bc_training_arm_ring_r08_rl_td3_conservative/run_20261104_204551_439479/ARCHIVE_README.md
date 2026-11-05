# bc_training_arm_ring_r08_rl_td3_conservative / run_20261104_204551_439479

分类：程序冒烟检查，不计正式成绩。

训练结束元数据见 [training_finished.json](training_finished.json)，以该快照为准。

| 检查点 | 用途 | SHA256 |
|---|---|---|
| [best_model.pt](best_model.pt) | 本次最终选中初始零残差策略；不是RL提升 | `3fb51ba99b37616357dc6b49966cdd25e0d5bc7d4b86f9e0e25a404729f0ea94` |
| [initial_model.pt](initial_model.pt) | 训练前初始策略／零残差，不能作为新训练成果 | `4cb6c99c8354a0ae1831a0a9016bc7928b3e4903200344f025e406c82936f701` |
| [last_model.pt](last_model.pt) | 最后训练状态，部分包含优化器，供续训；不等于最佳 | `c2967d6cd3af9a27c700fb7da23a1da9037abbc618aabf81f9630cb5f9c2a2ac` |

本目录保持源文件字节不变。配置和检查点内部可能保留历史绝对路径；这是归档，不代表解压后能在任意目录直接运行。配套策略类、父模型、归一化与导航均需匹配，不能只加载某个同名best_model.pt。原始大体积rollout NPZ未随本分支全部提交，summary/CSV及子目录小型诊断记录已保留。

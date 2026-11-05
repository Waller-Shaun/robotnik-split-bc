# bc_training_arm_ring_r08_residual120 / run_20261104_174723_237778

分类：历史正式训练运行／模型快照。

训练结束元数据见 [training_finished.json](training_finished.json)，以该快照为准。

| 检查点 | 用途 | SHA256 |
|---|---|---|
| [baseline_model.pt](baseline_model.pt) | 冻结父模型对照，不是新模型 | `15d441903d77e6951e99e6b957ed6eaa5c5aa6e30d20b139a8cc0392470e5519` |
| [best_model.pt](best_model.pt) | 按该运行实际选模规则保存；不自动代表超过父模型 | `503417db306f21baf68b287107d7acd87348eaf406c6240d05bc359260bc3518` |
| [last_model.pt](last_model.pt) | 最后训练状态，部分包含优化器，供续训；不等于最佳 | `e704de132c02c6982dc8f94865cbddfe325be7d12ebe5e5665970d2fd9dd98ef` |

本目录保持源文件字节不变。配置和检查点内部可能保留历史绝对路径；这是归档，不代表解压后能在任意目录直接运行。配套策略类、父模型、归一化与导航均需匹配，不能只加载某个同名best_model.pt。原始大体积rollout NPZ未随本分支全部提交，summary/CSV及子目录小型诊断记录已保留。

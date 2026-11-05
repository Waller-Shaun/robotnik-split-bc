# bc_training_arm_ring_r08_aux_b2_control / run_20261104_025706_206181

分类：历史正式训练运行／模型快照。

训练结束元数据见 [training_finished.json](training_finished.json)，以该快照为准。

| 检查点 | 用途 | SHA256 |
|---|---|---|
| [best_model.pt](best_model.pt) | 按该运行实际选模规则保存；不自动代表超过父模型 | `f3374c43fa37d65df682ced14885181e0b2654bfd1ca28f1acb51155b8ff7168` |
| [last_model.pt](last_model.pt) | 最后训练状态，部分包含优化器，供续训；不等于最佳 | `8202a4fb3f2e5c4dc4a887b4500cbd05536b429a64a200cea8baa9135501c14c` |

本目录保持源文件字节不变。配置和检查点内部可能保留历史绝对路径；这是归档，不代表解压后能在任意目录直接运行。配套策略类、父模型、归一化与导航均需匹配，不能只加载某个同名best_model.pt。原始大体积rollout NPZ未随本分支全部提交，summary/CSV及子目录小型诊断记录已保留。
